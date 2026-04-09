"""
Spectral Weight Compression — DCT/Frequency Domain Coding

Neural network weights have strong spatial correlations — neighboring elements
tend to be similar. This is exactly what frequency-domain transforms exploit.

The Discrete Cosine Transform (DCT) is the core of JPEG compression.
Apply it to weight matrices:
  1. Reshape weight into 2D blocks
  2. Apply 2D DCT to each block
  3. Most energy concentrates in low-frequency coefficients
  4. Quantize/threshold the DCT coefficients
  5. Store only the significant coefficients (sparse)

This is "JPEG for neural network weights."

Combined with Product Quantization:
  - DCT concentrates energy into fewer coefficients
  - PQ compresses those coefficients with massive codebook expressiveness
  - The combination can push quality higher at the same BPW
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.fft import dctn, idctn


@dataclass
class SpectralCompressed:
    """Spectral (DCT) compressed representation."""
    coefficients: torch.Tensor  # Non-zero DCT coefficients
    indices: torch.Tensor       # Positions of non-zero coefficients
    shape: tuple               # Original matrix shape
    block_size: int
    n_blocks: tuple            # (n_block_rows, n_block_cols)
    threshold: float           # Coefficient threshold used
    n_elements: int
    original_bytes: int

    def storage_bytes(self) -> int:
        # Coefficients in FP16 + indices as int16
        return self.coefficients.numel() * 2 + self.indices.numel() * 2

    @property
    def bits_per_weight(self) -> float:
        return (self.storage_bytes() * 8) / self.n_elements

    @property
    def sparsity(self) -> float:
        return 1.0 - (self.coefficients.numel() / self.n_elements)


def spectral_compress(
    weight: torch.Tensor,
    block_size: int = 8,
    energy_keep: float = 0.99,
    device: str = "cuda",
) -> SpectralCompressed:
    """Compress a weight matrix using block DCT.

    1. Divide weight into block_size x block_size blocks
    2. Apply 2D DCT to each block
    3. Keep only coefficients that explain energy_keep fraction of energy
    4. Store as sparse representation

    Args:
        weight: 2D weight matrix
        block_size: DCT block size (8 = standard JPEG, larger = more compression)
        energy_keep: Fraction of energy to retain (0.99 = 99%)
    """
    w = weight.float().cpu().numpy()
    original_shape = w.shape
    n_elements = w.size

    if w.ndim == 1:
        w = w.reshape(1, -1)

    m, n = w.shape

    # Pad to block_size multiples
    pad_m = (block_size - m % block_size) % block_size
    pad_n = (block_size - n % block_size) % block_size
    if pad_m > 0 or pad_n > 0:
        w = np.pad(w, ((0, pad_m), (0, pad_n)), mode='constant')

    pm, pn = w.shape
    n_bm = pm // block_size
    n_bn = pn // block_size

    # Reshape into blocks and apply DCT
    blocks = w.reshape(n_bm, block_size, n_bn, block_size)
    blocks = blocks.transpose(0, 2, 1, 3)  # (n_bm, n_bn, bs, bs)

    # Apply 2D DCT to each block
    dct_blocks = np.zeros_like(blocks)
    for i in range(n_bm):
        for j in range(n_bn):
            dct_blocks[i, j] = dctn(blocks[i, j], type=2, norm='ortho')

    # Find energy-based threshold
    all_coeffs = dct_blocks.flatten()
    energies = all_coeffs ** 2
    total_energy = energies.sum()

    # Sort by magnitude (descending)
    sorted_idx = np.argsort(np.abs(all_coeffs))[::-1]
    cumulative = np.cumsum(energies[sorted_idx])

    # Find how many coefficients needed for energy_keep
    n_keep = np.searchsorted(cumulative, total_energy * energy_keep) + 1
    n_keep = max(1, min(n_keep, len(all_coeffs)))

    threshold = np.abs(all_coeffs[sorted_idx[n_keep - 1]])

    # Threshold: keep only significant coefficients
    mask = np.abs(dct_blocks) >= threshold
    significant_values = dct_blocks[mask]
    significant_indices = np.argwhere(mask)

    # Pack indices efficiently
    # Each index is (block_row, block_col, coeff_row, coeff_col)
    # Pack into a single integer for storage
    flat_indices = (significant_indices[:, 0] * n_bn * block_size * block_size +
                   significant_indices[:, 1] * block_size * block_size +
                   significant_indices[:, 2] * block_size +
                   significant_indices[:, 3])

    return SpectralCompressed(
        coefficients=torch.tensor(significant_values, dtype=torch.float16),
        indices=torch.tensor(flat_indices, dtype=torch.int32),
        shape=original_shape,
        block_size=block_size,
        n_blocks=(n_bm, n_bn),
        threshold=threshold,
        n_elements=n_elements,
        original_bytes=n_elements * 2,
    )


def spectral_decompress(compressed: SpectralCompressed) -> torch.Tensor:
    """Reconstruct weight matrix from spectral representation."""
    n_bm, n_bn = compressed.n_blocks
    bs = compressed.block_size

    # Reconstruct DCT blocks
    dct_blocks = np.zeros((n_bm, n_bn, bs, bs))

    values = compressed.coefficients.float().numpy()
    flat_indices = compressed.indices.numpy()

    for val, idx in zip(values, flat_indices):
        bi = idx // (n_bn * bs * bs)
        remainder = idx % (n_bn * bs * bs)
        bj = remainder // (bs * bs)
        remainder = remainder % (bs * bs)
        ci = remainder // bs
        cj = remainder % bs
        if bi < n_bm and bj < n_bn:
            dct_blocks[bi, bj, ci, cj] = val

    # Inverse DCT
    blocks = np.zeros_like(dct_blocks)
    for i in range(n_bm):
        for j in range(n_bn):
            blocks[i, j] = idctn(dct_blocks[i, j], type=2, norm='ortho')

    # Reassemble
    blocks = blocks.transpose(0, 2, 1, 3)  # (n_bm, bs, n_bn, bs)
    w = blocks.reshape(n_bm * bs, n_bn * bs)

    # Trim to original shape
    orig_shape = compressed.shape
    if len(orig_shape) == 1:
        w = w.flatten()[:orig_shape[0]]
    else:
        w = w[:orig_shape[0], :orig_shape[1]]

    return torch.tensor(w, dtype=torch.float32)


def spectral_analysis(weight: torch.Tensor, block_size: int = 8) -> dict:
    """Analyze how well a weight matrix compresses in frequency domain.

    Returns the energy distribution across DCT coefficients.
    """
    w = weight.float().cpu().numpy()
    if w.ndim == 1:
        w = w.reshape(1, -1)

    m, n = w.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_n = (block_size - n % block_size) % block_size
    if pad_m > 0 or pad_n > 0:
        w = np.pad(w, ((0, pad_m), (0, pad_n)), mode='constant')

    pm, pn = w.shape
    n_bm = pm // block_size
    n_bn = pn // block_size

    blocks = w.reshape(n_bm, block_size, n_bn, block_size).transpose(0, 2, 1, 3)

    # DCT all blocks
    all_energies = np.zeros((block_size, block_size))
    for i in range(n_bm):
        for j in range(n_bn):
            dct_block = dctn(blocks[i, j], type=2, norm='ortho')
            all_energies += dct_block ** 2

    total_energy = all_energies.sum()
    all_energies /= total_energy + 1e-10

    # What fraction of coefficients needed for various energy levels
    flat_e = all_energies.flatten()
    sorted_e = np.sort(flat_e)[::-1]
    cumsum = np.cumsum(sorted_e)

    result = {
        "dc_energy_fraction": all_energies[0, 0],
        "top4_energy_fraction": float(np.sort(flat_e)[-4:].sum()),
        "block_size": block_size,
    }

    for target in [0.9, 0.95, 0.99, 0.999]:
        n_needed = np.searchsorted(cumsum, target) + 1
        result[f"coeffs_for_{target}"] = n_needed
        result[f"fraction_for_{target}"] = n_needed / (block_size * block_size)

    return result
