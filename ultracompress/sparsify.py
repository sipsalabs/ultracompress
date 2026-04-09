"""
Stage 5: N:M Structured Sparsity

Apply structured sparsity on top of binarized/codebook-compressed factors.
For every M elements, keep only the N with largest scale values, zero the rest.

Sparse-BitNet (March 2026) showed that 1.58-bit models tolerate N:M sparsity
much better than full-precision models. We exploit this for extra compression.

Common patterns:
  2:4 — keep 2 of every 4 (50% sparse, NVIDIA hardware accelerated)
  1:4 — keep 1 of every 4 (75% sparse, aggressive)
  2:8 — keep 2 of every 8 (75% sparse)
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class SparseRepresentation:
    """N:M sparse representation with bitmask."""
    values: torch.Tensor     # Only the non-zero values (packed)
    bitmask: torch.Tensor    # Bool tensor indicating which positions are kept
    n: int                   # Keep N
    m: int                   # Out of M
    original_shape: tuple

    def storage_bytes(self) -> int:
        value_bytes = self.values.numel() // 8  # Binary values = 1 bit each
        mask_bytes = self.bitmask.numel() // 8   # 1 bit per position
        return value_bytes + mask_bytes

    def decompress(self) -> torch.Tensor:
        """Reconstruct the sparse tensor."""
        result = torch.zeros(self.bitmask.numel(), device=self.values.device)
        result[self.bitmask.reshape(-1)] = self.values.float()
        return result.reshape(self.original_shape)


def apply_nm_sparsity(
    tensor: torch.Tensor,
    n: int = 2,
    m: int = 4,
) -> SparseRepresentation:
    """
    Apply N:M structured sparsity.

    For every group of M consecutive elements, keep the N largest by magnitude
    and zero the rest.
    """
    original_shape = tuple(tensor.shape)
    device = tensor.device
    flat = tensor.float().reshape(-1)

    # Pad to multiple of M
    remainder = flat.numel() % m
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(m - remainder, device=device)])

    groups = flat.reshape(-1, m)  # (n_groups, M)

    # Find top-N indices per group by magnitude
    _, topk_indices = groups.abs().topk(n, dim=1)  # (n_groups, N)

    # Build bitmask
    bitmask = torch.zeros_like(groups, dtype=torch.bool)
    bitmask.scatter_(1, topk_indices, True)

    # Extract kept values
    values = groups[bitmask]  # Flat tensor of kept values

    # Trim bitmask to original size
    bitmask_flat = bitmask.reshape(-1)[:tensor.numel()]

    return SparseRepresentation(
        values=values,
        bitmask=bitmask_flat.reshape(original_shape),
        n=n,
        m=m,
        original_shape=original_shape,
    )


def sparsity_ratio(n: int, m: int) -> float:
    """What fraction of weights are zeroed."""
    return 1.0 - (n / m)


def estimate_sparse_savings(original_bytes: int, n: int, m: int) -> int:
    """Estimate bytes after N:M sparsity (values + bitmask)."""
    kept_fraction = n / m
    value_bytes = int(original_bytes * kept_fraction)
    mask_bytes = original_bytes // 8  # 1 bit per original element
    return value_bytes + mask_bytes
