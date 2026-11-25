"""Transferable relation learner with VAE-parent provenance weighting.

This file belongs to the isolated candidate copy. It does not modify the
historical ``SimLearner`` or any active experiment source.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RelationGraph(NamedTuple):
    kernel: Tensor
    mu: Tensor
    degree: Tensor
    q: Tensor
    confidence: Tensor
    provenance: Tensor | None


def build_provenance(n_real: int, parents: Tensor, lam: Tensor | float) -> Tensor:
    """Return P for real nodes first, then virtual nodes from two real parents."""
    if n_real < 1 or parents.ndim != 2 or parents.shape[1] != 2:
        raise ValueError("expected n_real >= 1 and parents [M,2]")
    if parents.dtype not in (torch.int32, torch.int64):
        raise TypeError("parent indices must be integer")
    if parents.numel() and ((parents < 0).any() or (parents >= n_real).any()):
        raise ValueError("parent index out of range")
    m = parents.shape[0]
    lam = torch.as_tensor(lam, dtype=torch.float64, device=parents.device)
    if lam.ndim == 0:
        lam = lam.expand(m)
    if lam.shape != (m,) or not torch.isfinite(lam).all() or ((lam < 0) | (lam > 1)).any():
        raise ValueError("lam must be finite in [0,1] with shape [M]")
    p = torch.zeros((n_real + m, n_real), dtype=torch.float64, device=parents.device)
    p[:n_real] = torch.eye(n_real, dtype=p.dtype, device=p.device)
    if m:
        rows = torch.arange(m, device=p.device) + n_real
        p[rows, parents[:, 0]] += lam
        p[rows, parents[:, 1]] += 1.0 - lam
    return p


def provenance_mass(p: Tensor | None, n_nodes: int, alpha: float,
                    *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Compute source-balanced node masses; P=None is the real-only test path."""
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if p is None:
        return torch.full((n_nodes,), 1.0 / n_nodes, dtype=dtype, device=device)
    if p.ndim != 2 or p.shape[0] != n_nodes or p.shape[1] < 1:
        raise ValueError("P must have shape [n+M,n]")
    n_real = p.shape[1]
    p64 = p.to(device=device, dtype=torch.float64)
    if not torch.isfinite(p64).all() or (p64 < 0).any():
        raise ValueError("P must be finite and nonnegative")
    if not torch.allclose(p64.sum(-1), torch.ones(n_nodes, device=device, dtype=p64.dtype),
                          rtol=0.0, atol=1e-12):
        raise ValueError("every P row must sum to one")
    if not torch.allclose(p64[:n_real], torch.eye(n_real, device=device, dtype=p64.dtype),
                          rtol=0.0, atol=1e-12):
        raise ValueError("the first n rows of P must be real-node identities")
    m = n_nodes - n_real
    if not m:
        return torch.full((n_nodes,), 1.0 / n_real, dtype=dtype, device=device)
    if alpha <= 0:
        raise ValueError("alpha must be positive when virtual nodes are present")
    virtual_p = p64[n_real:]
    coverage = virtual_p.sum(0)
    inverse_coverage = torch.where(coverage > 0, coverage.reciprocal(), torch.zeros_like(coverage))
    virtual_score = (virtual_p * inverse_coverage[None, :]).sum(-1)
    if not torch.isfinite(virtual_score).all() or (virtual_score <= 0).any():
        raise ValueError("every virtual node must have positive real-source provenance")
    # Reserve exact total mass for real and virtual populations, then distribute
    # the virtual budget inversely to each parent source's augmentation coverage.
    real_mu = torch.full((n_real,), 1.0 / (1.0 + alpha) / n_real,
                         dtype=p64.dtype, device=device)
    virtual_mu = (alpha / (1.0 + alpha)) * virtual_score / virtual_score.sum()
    mu = torch.cat((real_mu, virtual_mu))
    if not torch.isfinite(mu).all() or (mu < 0).any():
        raise FloatingPointError("invalid provenance mass")
    return mu.to(dtype=dtype)


class ProvenanceRelationLearner(nn.Module):
    """Source-initializable history encoder and positive, set-stable relation."""

    def __init__(self, input_size: int, history_steps: int, hidden_dim: int,
                 tau: float = 0.5, eps: float = 1e-12):
        super().__init__()
        if input_size < 1 or history_steps < 1 or hidden_dim < 1:
            raise ValueError("relation dimensions must be positive")
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("tau and eps must be positive finite values")
        self.fc = nn.Linear(input_size * history_steps, hidden_dim)
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(self, history: Tensor, provenance: Tensor | None = None,
                alpha: float = 0.5) -> RelationGraph:
        """history [B,N,C,L], P [N,n] (real rows first) or None for inference."""
        if history.ndim != 4 or history.shape[1] < 1 or not torch.isfinite(history).all():
            raise ValueError("history must be finite [B,N,C,L]")
        b, n, c, length = history.shape
        if c * length != self.fc.in_features:
            raise ValueError("history channel/time size does not match source encoder")
        if provenance is not None and (provenance.shape[0] != n or provenance.device != history.device):
            raise ValueError("P node count/device does not match history")
        encoded = self.fc(history.reshape(b, n, -1))
        norms = encoded.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        encoded = encoded / norms
        cosine = torch.matmul(encoded, encoded.transpose(-1, -2)).clamp(-1.0, 1.0)
        kernel = torch.exp((cosine - 1.0) / self.tau)
        if not torch.isfinite(kernel).all() or (kernel <= 0).any():
            raise FloatingPointError("positive relation kernel underflowed; increase tau")

        mu = provenance_mass(provenance, n, alpha, dtype=history.dtype, device=history.device)
        # Symmetric importance-weighted kernel. Using sqrt(mu_i * mu_j)
        # keeps Q symmetric while correcting the virtual-node multiplicity.
        sqrt_mu = mu.clamp_min(0).sqrt()
        weighted_kernel = (sqrt_mu.view(1, n, 1) * kernel
                           * sqrt_mu.view(1, 1, n))
        degree = weighted_kernel.sum(-1)
        if not torch.isfinite(degree).all() or (degree <= 0).any():
            raise FloatingPointError("relation degree must be positive and finite")
        inv_sqrt = degree.rsqrt()
        q = inv_sqrt.unsqueeze(-1) * weighted_kernel * inv_sqrt.unsqueeze(-2)

        diagonal = torch.diagonal(weighted_kernel, dim1=-2, dim2=-1)
        off_diagonal_mass = (degree - diagonal).clamp_min(0.0)
        confidence = off_diagonal_mass / degree.clamp_min(self.eps)
        return RelationGraph(kernel, mu, degree, q, confidence, provenance)
