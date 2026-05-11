"""Error-Correcting Codes for Weight Compression — E8 lattice quantization + parity correction."""

import torch
import torch.nn.functional as F


class LatticeQuantizer:
    """Quantize weight vectors using a learned 8D codebook approximating E8 lattice structure."""

    def __init__(self, codebook_size=256, dim=8, device="cpu"):
        self.dim = dim
        # Init codebook from E8-like half-integer lattice points (all coords ±0.5 or ints, even sum)
        self.codebook = torch.randn(codebook_size, dim, device=device) * 0.5
        self.codebook = torch.round(self.codebook * 2) / 2  # snap to half-integers

    def fit(self, weights: torch.Tensor, iters=20):
        """vector quantization refinement on weight data to learn E8-adapted codebook."""
        groups = weights.reshape(-1, self.dim)
        for _ in range(iters):
            indices = self.quantize(groups)
            for i in range(self.codebook.shape[0]):
                mask = indices == i
                if mask.any():
                    self.codebook[i] = groups[mask].mean(0)

    def quantize(self, weights: torch.Tensor) -> torch.Tensor:
        flat = weights.reshape(-1, self.dim)
        dists = torch.cdist(flat, self.codebook)
        return dists.argmin(dim=1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook[indices.long()]


class RedundantCoding:
    """Add parity-based redundancy that enables error correction after quantization."""

    def __init__(self, block_size=4):
        self.block_size = block_size  # num groups per parity block

    def encode(self, weights: torch.Tensor, quantized: torch.Tensor):
        """Compute parity (original - quantized residual) per block for later correction."""
        residual = weights - quantized
        # Store mean residual per block — cheap correction signal
        n = residual.shape[0]
        pad = (self.block_size - n % self.block_size) % self.block_size
        if pad:
            residual = F.pad(residual, (0, 0, 0, pad))
        blocks = residual.reshape(-1, self.block_size, residual.shape[-1])
        parity = blocks.mean(dim=1)  # compact summary per block
        return parity  # overhead: 1/block_size of original

    def correct(self, quantized: torch.Tensor, parity: torch.Tensor) -> torch.Tensor:
        """Apply parity correction to reconstructed weights."""
        n = quantized.shape[0]
        pad = (self.block_size - n % self.block_size) % self.block_size
        out = F.pad(quantized, (0, 0, 0, pad)) if pad else quantized.clone()
        blocks = out.reshape(-1, self.block_size, out.shape[-1])
        blocks += parity.unsqueeze(1)  # broadcast correction across block
        return blocks.reshape(-1, out.shape[-1])[:n]

    def overhead_ratio(self) -> float:
        return 1.0 / self.block_size
