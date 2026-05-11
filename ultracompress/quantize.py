"""
Direct Low-Bit Group Quantization (no SVD needed)

For matrices with flat singular value spectra (like dequantized GGUF weights),
SVD doesn't help — the rank needed is too close to full rank.

Instead, directly quantize values into low-bit representations:
  - Group values into blocks of G elements
  - Per-group: compute scale (and optional zero-point)
  - Map each value to the nearest N-bit integer
  - Store: quantized integers + scales

This is the core technique behind GPTQ, AWQ, QuIP, etc.

At 2 bits: 4 levels per element → ~0.99 cosine sim
At 1 bit with sigma-delta: ~0.95-0.97 cosine sim
At mixed precision (outlier-aware): even better
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class QuantizedTensor:
    """Low-bit quantized tensor with per-group scales."""
    codes: torch.Tensor       # Quantized integer codes (uint8 packed)
    scales: torch.Tensor      # Per-group scale (float16)
    zeros: torch.Tensor       # Per-group zero-point (float16)
    bits: int                 # Bits per element
    group_size: int
    original_shape: tuple
    n_elements: int

    def storage_bytes(self) -> int:
        # Packed codes
        code_bits = self.codes.numel() * self.bits
        code_bytes = (code_bits + 7) // 8
        # Scales + zeros
        param_bytes = (self.scales.numel() + self.zeros.numel()) * 2
        return code_bytes + param_bytes

    @property
    def bits_per_weight(self) -> float:
        return (self.storage_bytes() * 8) / self.n_elements

    def decompress(self) -> torch.Tensor:
        """Reconstruct the float tensor."""
        n_groups = self.scales.numel()
        codes = self.codes.float().reshape(n_groups, -1)
        scales = self.scales.float().unsqueeze(1)
        zeros = self.zeros.float().unsqueeze(1)

        # Dequantize: value = code * scale + zero
        values = codes * scales + zeros
        flat = values.reshape(-1)[:self.n_elements]
        return flat.reshape(self.original_shape)


def quantize_absmax(
    weight: torch.Tensor,
    bits: int = 2,
    group_size: int = 128,
) -> QuantizedTensor:
    """
    Absmax symmetric quantization.

    Maps values to [-max, +max] range using `bits` levels.
    Symmetric around zero — good for normally distributed weights.
    """
    original_shape = tuple(weight.shape)
    n_elements = weight.numel()
    device = weight.device
    flat = weight.float().reshape(-1)

    # Pad to group_size
    remainder = flat.numel() % group_size
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(group_size - remainder, device=device)])

    groups = flat.reshape(-1, group_size)
    n_groups = groups.shape[0]

    n_levels = 2 ** bits
    half_levels = n_levels // 2  # e.g., 2 bits -> 4 levels -> half = 2

    # Per-group max absolute value
    max_abs = groups.abs().max(dim=1).values.clamp(min=1e-10)

    # Scale: maps [-max, +max] to [-half_levels, +half_levels-1]
    scales = max_abs / (half_levels - 0.5)

    # Quantize: round to nearest integer in [-half, +half-1]
    normalized = groups / scales.unsqueeze(1)
    codes = torch.clamp(torch.round(normalized), -half_levels, half_levels - 1)

    # Shift to unsigned for storage: [-half, half-1] -> [0, n_levels-1]
    codes_unsigned = (codes + half_levels).to(torch.uint8)

    # For decompress: value = (code - half_levels) * scale
    # So zeros = -half_levels * scale
    zeros = -half_levels * scales

    return QuantizedTensor(
        codes=codes_unsigned,
        scales=scales.half(),
        zeros=zeros.half(),
        bits=bits,
        group_size=group_size,
        original_shape=original_shape,
        n_elements=n_elements,
    )


def quantize_minmax(
    weight: torch.Tensor,
    bits: int = 2,
    group_size: int = 128,
) -> QuantizedTensor:
    """
    Min-max asymmetric quantization.

    Maps [min, max] range to [0, 2^bits - 1]. Better for skewed distributions.
    """
    original_shape = tuple(weight.shape)
    n_elements = weight.numel()
    device = weight.device
    flat = weight.float().reshape(-1)

    remainder = flat.numel() % group_size
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(group_size - remainder, device=device)])

    groups = flat.reshape(-1, group_size)
    n_groups = groups.shape[0]
    n_levels = 2 ** bits

    # Per-group min and max
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values

    # Scale and zero-point
    ranges = (g_max - g_min).clamp(min=1e-10)
    scales = ranges / (n_levels - 1)
    zeros = g_min

    # Quantize
    normalized = (groups - zeros.unsqueeze(1)) / scales.unsqueeze(1)
    codes = torch.clamp(torch.round(normalized), 0, n_levels - 1).to(torch.uint8)

    return QuantizedTensor(
        codes=codes,
        scales=scales.half(),
        zeros=zeros.half(),
        bits=bits,
        group_size=group_size,
        original_shape=original_shape,
        n_elements=n_elements,
    )


def quantize_outlier_aware(
    weight: torch.Tensor,
    bits: int = 2,
    group_size: int = 128,
    outlier_threshold: float = 3.0,
    outlier_bits: int = 8,
) -> tuple:
    """
    Outlier-aware quantization: handle outliers separately at higher precision.

    1. Identify outlier elements (> threshold * std)
    2. Store outliers at higher precision (8-bit)
    3. Quantize the rest at target bits

    Returns: (QuantizedTensor for main, outlier_indices, outlier_values)
    """
    original_shape = tuple(weight.shape)
    device = weight.device
    flat = weight.float().reshape(-1)

    # Identify outliers
    std = flat.std()
    mean = flat.mean()
    outlier_mask = (flat - mean).abs() > outlier_threshold * std
    n_outliers = outlier_mask.sum().item()

    # Store outliers separately
    outlier_indices = outlier_mask.nonzero(as_tuple=True)[0]
    outlier_values = flat[outlier_indices].half()

    # Replace outliers with group mean for main quantization
    flat_clean = flat.clone()
    if n_outliers > 0:
        # Replace outliers with 0 (will be restored from outlier store)
        flat_clean[outlier_mask] = 0.0

    # Quantize the cleaned tensor
    cleaned_weight = flat_clean.reshape(original_shape)
    quantized = quantize_absmax(cleaned_weight, bits=bits, group_size=group_size)

    # Total storage
    outlier_bytes = n_outliers * (4 + 2)  # 4 bytes index + 2 bytes value

    return quantized, outlier_indices, outlier_values, outlier_bytes


def quantize_vector_codebook(
    weight: torch.Tensor,
    codebook_size: int = 256,
    group_size: int = 8,
    n_iter: int = 10,
    n_residual_levels: int = 1,
) -> 'VectorQuantized':
    """
    Vector Quantization (VQ) on raw weights — achieves fractional BPW.

    Groups weight elements into vectors of size G, then quantizes each
    vector to the nearest codebook entry. This is the core technique
    behind AQLM, QuIP#, etc.

    BPW = log2(codebook_size) / group_size * n_residual_levels
    Examples:
      K=256 (8-bit), G=8  → 8/8  = 1.0 BPW
      K=256 (8-bit), G=16 → 8/16 = 0.5 BPW
      K=4096 (12-bit), G=8 → 12/8 = 1.5 BPW
      K=256, G=8, 2 residual levels → 2.0 BPW

    With residual VQ (RVQ):
      Level 1: quantize to nearest codebook entry
      Level 2: quantize the RESIDUAL to a second codebook
      This captures fine detail at sub-linear cost.
    """
    original_shape = tuple(weight.shape)
    n_elements = weight.numel()
    device = weight.device
    flat = weight.float().reshape(-1)

    remainder = flat.numel() % group_size
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(group_size - remainder, device=device)])

    groups = flat.reshape(-1, group_size)  # (n_groups, G)
    n_groups = groups.shape[0]

    all_codebooks = []
    all_indices = []
    residual = groups.clone()

    for level in range(n_residual_levels):
        actual_k = min(codebook_size, n_groups)

        # Random initialization from diverse data samples
        perm = torch.randperm(n_groups, device=device)[:actual_k]
        codebook = residual[perm].clone()

        chunk_size = min(n_groups, 50000)  # Process in chunks for memory

        # vector quantization with fast L2 via dot product trick:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        for _ in range(n_iter):
            cb_sq = (codebook ** 2).sum(dim=1)  # (K,)

            indices = torch.zeros(n_groups, device=device, dtype=torch.int64)
            for start in range(0, n_groups, chunk_size):
                end = min(start + chunk_size, n_groups)
                chunk = residual[start:end]
                chunk_sq = (chunk ** 2).sum(dim=1)  # (chunk,)
                # dists = chunk_sq + cb_sq - 2 * chunk @ codebook^T
                dots = chunk @ codebook.t()  # (chunk, K)
                dists = chunk_sq.unsqueeze(1) + cb_sq.unsqueeze(0) - 2 * dots
                indices[start:end] = dists.argmin(dim=1)

            # Vectorized update via scatter
            sums = torch.zeros(actual_k, group_size, device=device)
            counts = torch.zeros(actual_k, device=device)
            sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, group_size), residual)
            counts.scatter_add_(0, indices, torch.ones(n_groups, device=device))
            valid = counts > 0
            if valid.any():
                codebook[valid] = sums[valid] / counts[valid].unsqueeze(1)

        all_codebooks.append(codebook)
        all_indices.append(indices.short())

        # Compute residual for next level
        quantized = codebook[indices]
        residual = residual - quantized

    return VectorQuantized(
        codebooks=all_codebooks,
        indices=all_indices,
        group_size=group_size,
        codebook_size=codebook_size,
        original_shape=original_shape,
        n_elements=n_elements,
    )


@dataclass
class VectorQuantized:
    """Residual vector quantized representation."""
    codebooks: list       # List of (K, G) tensors
    indices: list         # List of (n_groups,) index tensors
    group_size: int
    codebook_size: int
    original_shape: tuple
    n_elements: int

    def storage_bytes(self) -> int:
        bits_per_index = int(np.ceil(np.log2(max(self.codebook_size, 2))))
        total = 0
        for cb, idx in zip(self.codebooks, self.indices):
            total += cb.numel() * 2  # codebook in FP16
            total += (idx.numel() * bits_per_index + 7) // 8  # packed indices
        return total

    @property
    def bits_per_weight(self) -> float:
        return (self.storage_bytes() * 8) / self.n_elements

    def decompress(self) -> torch.Tensor:
        result = torch.zeros_like(self.codebooks[0][self.indices[0].long()])
        for cb, idx in zip(self.codebooks, self.indices):
            result = result + cb[idx.long()]
        flat = result.reshape(-1)[:self.n_elements]
        return flat.reshape(self.original_shape)


def smart_quantize(
    weight: torch.Tensor,
    target_bpw: float = 0.5,
    group_size: int = 128,
) -> QuantizedTensor:
    """Automatically choose bits and method based on target BPW."""
    n_elements = weight.numel()
    n_groups = (n_elements + group_size - 1) // group_size
    overhead_bits = n_groups * 32
    overhead_bpw = overhead_bits / n_elements
    effective_bits = target_bpw - overhead_bpw
    if effective_bits <= 0:
        effective_bits = 0.5

    if effective_bits >= 3.5:
        bits = 4
    elif effective_bits >= 1.5:
        bits = 2
    else:
        bits = 1

    return quantize_absmax(weight, bits=bits, group_size=group_size)
