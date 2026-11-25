"""V2: source-compatible backbone with zero-initialized sparse corrections."""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from ..graph_learner.simlearner import SimLearner
from ..graph_learner.provenance_relation import provenance_mass
from .stconv import STConv


def sparse_relation(history, fc, provenance=None, k=4, virtual_budget=1/3):
    """Bound neighbors and actual virtual mass in the correction branch only."""
    b, n, _, _ = history.shape
    real = n if provenance is None else provenance.shape[1]
    centered = (history - history.mean(-1, keepdim=True)).reshape(b, n, -1)
    raw = F.normalize(centered, dim=-1, eps=1e-8)
    encoded = F.normalize(F.linear(centered, fc.weight, None), dim=-1, eps=1e-8)
    score = (0.5 * (raw @ raw.transpose(-1, -2)
                    + encoded @ encoded.transpose(-1, -2))).clamp(0, 1)
    score = score.masked_fill(torch.eye(n, dtype=torch.bool, device=history.device)[None], 0)

    def select(weights):
        if weights.shape[-1] == 0:
            return weights, weights.sum(-1, keepdim=True)
        values, indices = weights.topk(min(k, weights.shape[-1]), dim=-1)
        sparse = torch.zeros_like(weights).scatter(-1, indices, values)
        total = sparse.sum(-1, keepdim=True)
        return sparse / total.clamp_min(1e-8), total

    qr, real_sum = select(score[..., :real])
    fallback = (torch.eye(n, device=history.device, dtype=history.dtype)
                if provenance is None else provenance.to(history))
    qr = torch.where(real_sum > 1e-8, qr, fallback[None])
    if real == n:
        return qr
    mu = provenance_mass(provenance, n, 0.5, dtype=history.dtype, device=history.device)
    qv, virtual_sum = select(score[..., real:] * mu[None, None, real:])
    beta = virtual_budget * (virtual_sum > 1e-8).to(history.dtype)
    return torch.cat(((1 - beta) * qr, beta * qv), dim=-1)


class FewNodeSTConv(STConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_map = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        nn.init.xavier_uniform_(self.residual_map.weight, gain=0.1)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x, edge_index, edge_weight, q):
        t0 = self._temporal_conv1(x)
        b, t, n, f = t0.shape
        flat = t0.permute(1, 0, 2, 3).reshape(t, b * n, f)
        base = self._graph_conv(flat, edge_index, edge_weight)
        base = base.reshape(t, b, n, f).permute(1, 0, 2, 3)
        contrast = torch.einsum('bij,btjf->btif', q, t0) - t0
        spatial = base + torch.tanh(self.residual_scale) * self.residual_map(contrast)
        result = self._temporal_conv2(F.relu(spatial))
        if self.cfg['batch_norm']:
            result = self._batch_norm(result.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.dropout(result)


class FewNodeSTFiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        spec = cfg['model']
        nodes = [int(cfg['train_node_ratio'] * spec['num_nodes'])] * 2 + [spec['num_nodes']]
        args = (spec['hidden_channels'], spec['hidden_channels'], spec['kernel_size'],
                spec['K'], cfg, spec['normalization'], spec['bias'])
        self.st_block_1 = FewNodeSTConv(nodes, spec['in_channels'], *args)
        self.st_block_2 = FewNodeSTConv(nodes, spec['hidden_channels'], *args)
        self.output_layer = nn.Linear(spec['hidden_channels'] * 4, spec['out_channels'])
        self.graph_learner = SimLearner(cfg)

    def forward(self, history: Tensor, provenance=None, *, return_aux=False):
        b, n, _, _ = history.shape
        graph = self.graph_learner(history)
        sample, row, col = (graph > 0).nonzero().t()
        edges = torch.stack((row + sample * n, col + sample * n))
        weights = graph[sample, row, col]
        q = sparse_relation(history, self.graph_learner.fc, provenance)
        x = history.permute(0, 3, 1, 2).contiguous()
        x = self.st_block_1(x, edges, weights, q)
        x = self.st_block_2(x, edges, weights, q)
        prediction = self.output_layer(x.permute(0, 2, 1, 3).reshape(b, n, -1))
        return (prediction, q) if return_aux else prediction


def load_transferable_state(model, source_state):
    """Strict complete backbone load, including ChebConv and source graph."""
    prefixes = ('graph_learner.', 'st_block_1.', 'st_block_2.', 'output_layer.')
    target = model.state_dict()
    expected = {k for k in target if k.startswith(prefixes) and '.residual_' not in k}
    source = {k.removeprefix('module.'): v for k, v in source_state.items()}
    missing = sorted(expected - source.keys())
    if missing:
        raise ValueError(f'Missing source backbone tensors: {missing}')
    for key in expected:
        value = source[key]
        if value.shape != target[key].shape or not torch.isfinite(value).all():
            raise ValueError(f'Incompatible source backbone: {key}')
        target[key] = value.to(target[key]).clone()
    model.load_state_dict(target, strict=True)
    return {'loaded_tensor_count': len(expected), 'loaded_keys': sorted(expected),
            'loaded_parameter_elements': sum(source[k].numel() for k in expected),
            'skipped_keys': [], 'cheb_tensors': sorted(k for k in expected if '_graph_conv' in k),
            'initial_residual_scale': 0.0}
