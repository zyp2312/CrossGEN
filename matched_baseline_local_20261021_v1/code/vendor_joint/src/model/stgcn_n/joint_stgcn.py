"""V10-compatible wrapper for the isolated joint graph/predictor candidate.

The pretrained VAE remains unchanged. During training its virtual nodes carry
an explicit two-parent provenance matrix into the new relation learner; at
validation and inference only real nodes are passed to the predictor.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..vae.vae import VAE
from ..graph_learner.provenance_relation import build_provenance
from .fewnode_stgcn import FewNodeSTFiT


# runner.model_from_fixed_split temporarily swaps this symbol for baseline and
# transfer models. The joint candidate is independent of DataTrimer.
DataTrimer = None


class STGCN_n(FewNodeSTFiT):
    """Preserve the historical ``(prediction, future, embedding, vae_loss)`` API."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_num_nodes = int(cfg["train_node_ratio"] * cfg["model"]["num_nodes"])
        self.vae = VAE(cfg)

    def forward(self, history_data: Tensor, future_data: Tensor | None = None, **kwargs):
        embedding = None
        vae_loss = history_data.new_zeros(())
        provenance = None
        if self.cfg.get("aug_node", False) and self.training:
            if future_data is None:
                raise ValueError("VAE augmentation requires training future_data")
            full_data = torch.cat((history_data, future_data), dim=-1)
            batch, n_real, channels, window = full_data.shape
            if window != int(self.cfg["window_size"]):
                raise ValueError("VAE input window does not match configured window_size")
            # Keep V10's dynamic replacement sampling and latent-space mixup.
            sampling_weights = torch.ones(n_real, device=history_data.device)
            mix_pair = torch.stack((
                torch.multinomial(sampling_weights, n_real, replacement=True),
                torch.multinomial(sampling_weights, n_real, replacement=True),
            ), dim=0)
            self.mix_pair = mix_pair
            transformed = full_data.transpose(-1, -2).reshape(-1, window, channels)
            mu, logvar = self.vae.encoder(transformed)
            embedding = self.vae.reparameterize(mu, logvar)
            vae_loss = torch.mean(
                -self.cfg.get("vae_loss_weight", 2)
                * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
            )
            lam = float(self.cfg["anchor_lambda"])
            mixed = (lam * embedding.reshape(batch, n_real, -1)[:, mix_pair[0]]
                     + (1.0 - lam) * embedding.reshape(batch, n_real, -1)[:, mix_pair[1]])
            decoded = self.vae.decoder(mixed.reshape(-1, mixed.shape[-1]))
            virtual = decoded.transpose(-1, -2).reshape(batch, n_real, channels, window)
            history_data = torch.cat((history_data, virtual[..., :history_data.shape[-1]]), dim=1)
            future_data = torch.cat((future_data, virtual[..., history_data.shape[-1]:]), dim=1)
            provenance = build_provenance(n_real, mix_pair.transpose(0, 1), lam).to(history_data.device)

            if self.cfg.get("sim_loss", False):
                left = full_data[:, mix_pair[0]]
                right = full_data[:, mix_pair[1]]
                if self.cfg.get("aug_loss", "MSE") == "MSE":
                    vae_loss = vae_loss + torch.nn.functional.mse_loss(left, virtual)
                    vae_loss = vae_loss + torch.nn.functional.mse_loss(right, virtual)
                elif self.cfg.get("aug_loss") == "Contrastive":
                    vae_loss = vae_loss + torch.nn.functional.cosine_similarity(left, virtual).sum()
                    vae_loss = vae_loss + torch.nn.functional.cosine_similarity(right, virtual).sum()

        prediction = super().forward(history_data, provenance=provenance)
        return prediction, future_data, embedding, vae_loss
