"""Hadamard Rotation for Incoherent Quantization (QuIP#, 2402.04396).
LOSSLESS transform making weights uniform -> better quantization at same bits.
Pipeline: Hadamard rotate (lossless) -> Quantize (lossy) -> Unrotate on decompress"""

import torch
from dataclasses import dataclass
from .quantize import quantize_absmax, QuantizedTensor


def _hadamard_matrix(n: int) -> torch.Tensor:
    """Sylvester construction: n must be power of 2."""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)  # Normalize so H @ H.T = I


class HadamardTransform:
    """Randomized Hadamard rotation — lossless and invertible."""

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.seed = seed
        # Pad to next power of 2
        self.padded = 1 << (dim - 1).bit_length() if dim > 0 else 1
        H = _hadamard_matrix(self.padded)
        # Random sign flip for incoherence (Diagonal * Hadamard)
        rng = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (self.padded,), generator=rng) * 2 - 1
        self.Q = H * signs.unsqueeze(0)  # columns flipped by random signs

    def rotate(self, W: torch.Tensor) -> torch.Tensor:
        """W @ Q — rotate columns. Pad if needed, slice after."""
        rows, cols = W.shape
        Q = self.Q.to(W.device, W.dtype)
        if cols < self.padded:
            W = torch.nn.functional.pad(W, (0, self.padded - cols))
        return W @ Q

    def unrotate(self, W_rot: torch.Tensor) -> torch.Tensor:
        """W_rot @ Q.T — exact inverse (Q is orthogonal)."""
        Q = self.Q.to(W_rot.device, W_rot.dtype)
        out = W_rot @ Q.T
        return out[:, :self.dim]


@dataclass
class IncoherentQuantized:
    """Hadamard-rotated + quantized weights."""
    quantized: QuantizedTensor
    seed: int
    original_cols: int
    padded_rows: int  # rows of the rotated matrix

    def storage_bytes(self) -> int:
        return self.quantized.storage_bytes() + 12

    def decompress(self) -> torch.Tensor:
        """Dequantize then unrotate -> original weights."""
        W_rot = self.quantized.decompress()
        ht = HadamardTransform(self.original_cols, seed=self.seed)
        return ht.unrotate(W_rot)


def incoherent_quantize(
    weight: torch.Tensor, bits: int = 2, group_size: int = 128, seed: int = 42,
) -> IncoherentQuantized:
    """Rotate weights for incoherence, then quantize. Key insight from QuIP#."""
    rows, cols = weight.shape
    ht = HadamardTransform(cols, seed=seed)
    rotated = ht.rotate(weight)
    qt = quantize_absmax(rotated, bits=bits, group_size=group_size)
    return IncoherentQuantized(
        quantized=qt, seed=seed, original_cols=cols, padded_rows=rows,
    )
