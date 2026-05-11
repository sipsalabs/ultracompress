"""
Stage 4: Codebook Compression (v2 — Residual Vector Quantization)

v1 used flat vector quantization with 256 entries.

v2 upgrades:
  - Larger codebooks (4096+) for better pattern coverage
  - Batch vector quantization for speed on large tensors
  - Better initialization (centroid initialization style)

For a codebook of size K with group_size G:
  Effective BPW = log2(K) / G
  K=4096, G=128 → 12/128 = 0.094 BPW
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class CodebookCompressed:
    """Codebook-compressed representation of binary patterns."""
    codebook: torch.Tensor    # (K, group_size)
    indices: torch.Tensor     # Integer indices
    scales: torch.Tensor      # Per-group scaling factors
    codebook_bits: int        # Bits per index
    group_size: int
    original_shape: tuple
    n_original_elements: int

    def storage_bytes(self) -> int:
        codebook_bytes = self.codebook.numel() // 8
        index_bytes = (self.indices.numel() * self.codebook_bits + 7) // 8
        scale_bytes = self.scales.numel() * 2
        return codebook_bytes + index_bytes + scale_bytes

    def decompress(self) -> torch.Tensor:
        patterns = self.codebook[self.indices.long()]
        values = patterns.float() * 2.0 - 1.0
        n_groups = values.shape[0]

        scales = self.scales.float()
        if scales.numel() < n_groups:
            scales = torch.cat([scales, scales[-1:].expand(n_groups - scales.numel())])
        scales = scales[:n_groups]

        scaled = values * scales.unsqueeze(1)
        flat = scaled.reshape(-1)[:self.n_original_elements]
        return flat.reshape(self.original_shape)


def fast_init(data: torch.Tensor, k: int) -> torch.Tensor:
    """Fast diverse initialization: random sample from data."""
    n = data.shape[0]
    perm = torch.randperm(n, device=data.device)[:k]
    return data[perm].clone()


def build_codebook(
    signs: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    codebook_size: int = 4096,
    n_iter: int = 15,
    original_shape: tuple = None,
) -> CodebookCompressed:
    """Build a codebook from binary sign patterns using centroid initialization."""
    device = signs.device
    flat = signs.reshape(-1).float()
    n_elements = flat.numel()

    remainder = n_elements % group_size
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(group_size - remainder, device=device)])

    groups = flat.reshape(-1, group_size)
    n_groups = groups.shape[0]

    # Recompute per-group scales
    group_scales = torch.ones(n_groups, device=device, dtype=torch.float16)
    if scales is not None and scales.numel() > 0:
        if scales.numel() == n_groups:
            group_scales = scales
        elif scales.numel() > n_groups:
            group_scales = scales[:n_groups]
        else:
            repeats = (n_groups + scales.numel() - 1) // scales.numel()
            group_scales = scales.repeat(repeats)[:n_groups]

    codebook_bits = int(np.ceil(np.log2(max(codebook_size, 2))))
    actual_k = min(codebook_size, n_groups)

    if n_groups <= actual_k:
        return CodebookCompressed(
            codebook=groups.bool(),
            indices=torch.arange(n_groups, device=device, dtype=torch.int16),
            scales=group_scales,
            codebook_bits=int(np.ceil(np.log2(max(n_groups, 2)))),
            group_size=group_size,
            original_shape=original_shape,
            n_original_elements=n_elements,
        )

    # Fast random initialization
    codebook = fast_init(groups, actual_k)

    # Fast vector quantization: chunked assignment + scatter-based update (no per-cluster loop)
    chunk_size = min(n_groups, 50000)
    indices = torch.zeros(n_groups, device=device, dtype=torch.int64)

    for iteration in range(n_iter):
        # Assignment step (chunked for memory)
        for start in range(0, n_groups, chunk_size):
            end = min(start + chunk_size, n_groups)
            scores = groups[start:end] @ codebook.t()
            indices[start:end] = scores.argmax(dim=1)

        # Vectorized update: scatter_add to accumulate sums per cluster
        sums = torch.zeros(actual_k, group_size, device=device)
        counts = torch.zeros(actual_k, device=device)
        sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, group_size), groups)
        counts.scatter_add_(0, indices, torch.ones(n_groups, device=device))

        # Majority vote: mean >= 0.5
        valid = counts > 0
        if valid.any():
            means = sums[valid] / counts[valid].unsqueeze(1)
            codebook[valid] = (means >= 0.5).float()

    return CodebookCompressed(
        codebook=codebook.bool(),
        indices=indices.short(),
        scales=group_scales,
        codebook_bits=codebook_bits,
        group_size=group_size,
        original_shape=original_shape,
        n_original_elements=n_elements,
    )


def compress_binarized_factor(binarized_factor, codebook_size: int = 4096) -> CodebookCompressed:
    """Compress a BinarizedFactor using codebook quantization."""
    return build_codebook(
        signs=binarized_factor.signs,
        scales=binarized_factor.scales,
        group_size=binarized_factor.group_size,
        codebook_size=codebook_size,
        original_shape=binarized_factor.original_shape,
    )
