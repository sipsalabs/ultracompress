"""
Product Quantization (PQ) — The Path to Sub-0.01 BPW

Standard VQ: each group of G weights maps to 1 of K codebook entries.
  BPW = log2(K) / G
  K=256, G=8 -> 1.0 BPW. To get 0.01 BPW you'd need G=800. Impossible.

Product Quantization: split each group of G weights into M sub-vectors
of G/M elements each. Each sub-vector gets its own codebook of K entries.
  Effective codebook size: K^M (combinatorial explosion!)
  BPW = M * log2(K) / G

  M=4, K=256, G=32: BPW = 4*8/32 = 1.0 (same BPW as standard VQ)
  BUT: effective codebook = 256^4 = 4.3 BILLION entries from 4*256 = 1024 stored entries

  M=8, K=16, G=64:  BPW = 8*4/64 = 0.5
  Effective codebook = 16^8 = 4.3 billion entries from 8*16 = 128 stored entries

  M=16, K=4, G=128: BPW = 16*2/128 = 0.25
  Effective codebook = 4^16 = 4.3 billion entries from 16*4 = 64 stored entries!

  M=32, K=4, G=256: BPW = 32*2/256 = 0.25
  M=64, K=4, G=512: BPW = 64*2/512 = 0.25

The insight: with M sub-codebooks, we can represent K^M distinct weight
patterns while only storing M*K codebook entries. The indices cost M*log2(K)
bits per group, but for large G this amortizes to nearly nothing.

For 0.016 BPW (10T -> 20GB target):
  M=8, K=4, G=1024: BPW = 8*2/1024 = 0.016 BPW !!!
  Effective codebook = 4^8 = 65,536 entries from 32 stored entries
  Each group of 1024 weights is represented by 8 indices into 4-entry codebooks.

This is how we flagship the industry.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProductQuantized:
    """Product-quantized representation of a weight tensor.

    The weight is divided into groups of G elements.
    Each group is split into M sub-vectors of G/M elements.
    Each sub-vector is quantized to the nearest of K codebook entries.

    Storage:
      - M codebooks of K entries, each of size G/M: M * K * (G/M) * dtype = M*K*(G/M)*2 bytes (FP16)
        Simplifies to: K * G * 2 bytes (same as one group in FP16!)
      - Per-group: M indices of log2(K) bits each: n_groups * M * log2(K) bits
      - Per-group scales: n_groups * 2 bytes (FP16)
    """
    codebooks: List[torch.Tensor]  # M codebooks, each (K, G/M)
    indices: torch.Tensor          # (n_groups, M) — sub-vector indices
    scales: torch.Tensor           # (n_groups,) — per-group scale
    n_subvectors: int              # M
    codebook_size: int             # K
    group_size: int                # G
    sub_vector_size: int           # G/M
    original_shape: tuple
    n_elements: int

    def storage_bytes(self) -> int:
        bits_per_index = int(np.ceil(np.log2(max(self.codebook_size, 2))))
        # Codebooks: M * K * sub_vector_size * 2 bytes (FP16)
        codebook_bytes = sum(cb.numel() * 2 for cb in self.codebooks)
        # Indices: n_groups * M * bits_per_index
        index_bits = self.indices.numel() * bits_per_index
        index_bytes = (index_bits + 7) // 8
        # Scales
        scale_bytes = self.scales.numel() * 2
        return codebook_bytes + index_bytes + scale_bytes

    @property
    def bits_per_weight(self) -> float:
        return (self.storage_bytes() * 8) / self.n_elements

    def decompress(self) -> torch.Tensor:
        """Reconstruct the weight tensor."""
        n_groups = self.indices.shape[0]
        M = self.n_subvectors

        # Lookup each sub-vector from its codebook
        parts = []
        for m in range(M):
            idx = self.indices[:, m].long()
            sub_vectors = self.codebooks[m][idx]  # (n_groups, sub_vector_size)
            parts.append(sub_vectors)

        # Concatenate sub-vectors to form full group vectors
        groups = torch.cat(parts, dim=1)  # (n_groups, group_size)

        # Apply per-group scale
        scaled = groups * self.scales.float().unsqueeze(1)

        # Flatten and trim
        flat = scaled.reshape(-1)[:self.n_elements]
        return flat.reshape(self.original_shape)


def product_quantize(
    weight: torch.Tensor,
    n_subvectors: int = 8,
    codebook_size: int = 16,
    group_size: int = 64,
    n_iter: int = 20,
) -> ProductQuantized:
    """Product Quantization with per-subvector k-means.

    Args:
        weight: Input tensor to quantize
        n_subvectors: M — number of sub-vectors per group
        codebook_size: K — entries per sub-codebook
        group_size: G — elements per group (must be divisible by M)
        n_iter: k-means iterations per sub-codebook

    The key insight: we train M independent codebooks, one per sub-vector
    position. Each codebook learns the distribution of that specific
    sub-vector position across all groups. This captures position-specific
    patterns in the weight matrix.
    """
    original_shape = tuple(weight.shape)
    n_elements = weight.numel()
    device = weight.device
    flat = weight.float().reshape(-1)

    assert group_size % n_subvectors == 0, \
        f"group_size ({group_size}) must be divisible by n_subvectors ({n_subvectors})"

    sub_vector_size = group_size // n_subvectors

    # Pad to group_size
    remainder = flat.numel() % group_size
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(group_size - remainder, device=device)])

    groups = flat.reshape(-1, group_size)  # (n_groups, G)
    n_groups = groups.shape[0]

    # Per-group normalization: extract scale, normalize groups
    scales = groups.norm(dim=1) / np.sqrt(group_size)
    scales = scales.clamp(min=1e-10)
    normalized = groups / scales.unsqueeze(1)

    # Split each group into M sub-vectors
    sub_vectors = normalized.reshape(n_groups, n_subvectors, sub_vector_size)
    # sub_vectors: (n_groups, M, sub_vector_size)

    actual_k = min(codebook_size, n_groups)

    # Train M independent codebooks via k-means
    codebooks = []
    all_indices = []

    for m in range(n_subvectors):
        data = sub_vectors[:, m, :]  # (n_groups, sub_vector_size)

        # Initialize codebook from random samples
        perm = torch.randperm(n_groups, device=device)[:actual_k]
        codebook = data[perm].clone()

        chunk_size = min(n_groups, 100000)

        for _ in range(n_iter):
            # Assignment: find nearest codebook entry for each group
            cb_sq = (codebook ** 2).sum(dim=1)  # (K,)
            indices = torch.zeros(n_groups, device=device, dtype=torch.int64)

            for start in range(0, n_groups, chunk_size):
                end = min(start + chunk_size, n_groups)
                chunk = data[start:end]
                chunk_sq = (chunk ** 2).sum(dim=1)
                dots = chunk @ codebook.t()
                dists = chunk_sq.unsqueeze(1) + cb_sq.unsqueeze(0) - 2 * dots
                indices[start:end] = dists.argmin(dim=1)

            # Update: new centroid = mean of assigned vectors
            sums = torch.zeros(actual_k, sub_vector_size, device=device)
            counts = torch.zeros(actual_k, device=device)
            sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, sub_vector_size), data)
            counts.scatter_add_(0, indices, torch.ones(n_groups, device=device))
            valid = counts > 0
            if valid.any():
                codebook[valid] = sums[valid] / counts[valid].unsqueeze(1)

        codebooks.append(codebook.half())
        all_indices.append(indices)

    # Stack indices: (n_groups, M)
    indices_tensor = torch.stack(all_indices, dim=1).short()

    return ProductQuantized(
        codebooks=codebooks,
        indices=indices_tensor,
        scales=scales.half(),
        n_subvectors=n_subvectors,
        codebook_size=actual_k,
        group_size=group_size,
        sub_vector_size=sub_vector_size,
        original_shape=original_shape,
        n_elements=n_elements,
    )


def adaptive_product_quantize(
    weight: torch.Tensor,
    target_bpw: float = 0.5,
    min_quality: float = 0.99,
    n_iter: int = 20,
) -> ProductQuantized:
    """Automatically choose PQ parameters to hit a target BPW.

    Searches the (M, K, G) parameter space for configurations that
    achieve the target BPW while maximizing quality.

    BPW formula: (M * log2(K) + overhead) / G
    Where overhead accounts for per-group scale storage.
    """
    from .metrics import compute_quality

    device = weight.device
    w = weight.float().to(device)

    # Generate candidate configurations sorted by BPW
    candidates = []
    for K in [4, 8, 16, 32, 64, 256]:
        for G in [32, 64, 128, 256, 512, 1024]:
            for M in [2, 4, 8, 16, 32]:
                if G % M != 0:
                    continue
                if G // M < 2:
                    continue  # sub-vectors too small
                bits_per_index = np.ceil(np.log2(max(K, 2)))
                bpw_est = (M * bits_per_index + 16) / G  # +16 for per-group scale
                if bpw_est > target_bpw * 3:
                    continue  # Way over budget
                candidates.append((bpw_est, K, G, M))

    candidates.sort()

    best_result = None
    best_quality = -1.0
    best_pq = None

    for bpw_est, K, G, M in candidates:
        if weight.numel() < G * 4:
            continue  # Not enough elements for meaningful groups

        try:
            pq = product_quantize(weight, n_subvectors=M, codebook_size=K,
                                  group_size=G, n_iter=n_iter)
            recon = pq.decompress().to(device)
            if recon.shape != w.shape:
                recon = recon.reshape(w.shape)
            quality = compute_quality(w, recon)

            actual_bpw = pq.bits_per_weight

            if quality["cosine_sim"] > best_quality:
                best_quality = quality["cosine_sim"]
                best_pq = pq
                best_result = quality

            if quality["cosine_sim"] >= min_quality and actual_bpw <= target_bpw * 1.5:
                return pq

        except Exception:
            continue

    if best_pq is not None:
        return best_pq

    raise RuntimeError(f"No PQ config found for target_bpw={target_bpw}")
