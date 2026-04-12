"""
Symmetry-based compression via group theory.

Neural nets have permutation/scale symmetries — many weight configs encode
the SAME function.  Canonicalizing to a unique representative per orbit
reduces entropy and improves downstream compressibility.
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class CanonicalResult:
    weights: torch.Tensor       # canonicalized weights
    scales: torch.Tensor        # per-neuron scale factors (for ScaleCanonicalizer)
    perm: torch.Tensor          # permutation indices applied
    entropy_before: float       # Shannon entropy estimate (binned) of raw
    entropy_after: float        # Shannon entropy estimate of canonicalized


def _binned_entropy(t: torch.Tensor, bins: int = 256) -> float:
    """Estimate entropy via histogram binning."""
    h = torch.histc(t.float(), bins=bins)
    p = h / h.sum()
    p = p[p > 0]
    return -(p * p.log2()).sum().item()


class PermutationCanonicalizer:
    """Sort neurons (rows) by L2 norm descending — canonical representative
    of the permutation symmetry group S_n acting on hidden units."""

    def __call__(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norms = W.norm(dim=1)
        perm = norms.argsort(descending=True)
        return W[perm], perm


class ScaleCanonicalizer:
    """Normalize each neuron to unit L2 norm, factor out scales.
    Quotients by the positive-reals scaling symmetry per neuron."""

    def __call__(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norms = W.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return W / norms, norms.squeeze(1)


class SymmetryExploiter:
    """Full canonicalization pipeline: permute then scale-normalize.
    Reports entropy reduction as a proxy for compressibility gain."""

    def __init__(self):
        self.perm = PermutationCanonicalizer()
        self.scale = ScaleCanonicalizer()

    def canonicalize(self, W: torch.Tensor) -> CanonicalResult:
        ent_before = _binned_entropy(W)
        W_sorted, perm = self.perm(W)
        W_canon, scales = self.scale(W_sorted)
        ent_after = _binned_entropy(W_canon)
        return CanonicalResult(W_canon, scales, perm, ent_before, ent_after)

    def compressibility_gain(self, W: torch.Tensor) -> float:
        r = self.canonicalize(W)
        return (r.entropy_before - r.entropy_after) / max(r.entropy_before, 1e-12)
