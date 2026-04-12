"""
Dynamic Precision — Input-adaptive precision selection per token.

Easy tokens (predictable) use 2-bit weights (fast, tiny). Hard tokens
(surprising) use 8-bit weights (accurate). A lightweight router scores
input difficulty and selects precision on the fly. Average BPW adapts
to the actual difficulty of the input stream.
"""

import torch
import torch.nn as nn
from ultracompress.quantize import quantize_absmax


class DynamicPrecisionLayer(nn.Module):
    """Wraps a weight matrix with 2/4/8-bit copies and a difficulty router."""

    def __init__(self, weight: torch.Tensor, group_size: int = 128):
        super().__init__()
        # Quantize at three precision levels and store dequantized copies
        self.register_buffer("w2", quantize_absmax(weight, bits=2, group_size=group_size).decompress())
        self.register_buffer("w4", quantize_absmax(weight, bits=4, group_size=group_size).decompress())
        self.register_buffer("w8", quantize_absmax(weight, bits=8, group_size=group_size).decompress())
        # Lightweight router: input features -> 3 precision logits
        in_dim = weight.shape[1] if weight.ndim == 2 else weight.shape[0]
        self.router = nn.Linear(in_dim, 3, bias=False)
        nn.init.zeros_(self.router.weight)
        self._avg_precision = 4.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, dim) or (batch, dim)."""
        logits = self.router(x.detach().float())          # (..., 3)
        probs = torch.softmax(logits, dim=-1)
        # Weighted blend of precisions (differentiable)
        out2 = x @ self.w2.T
        out4 = x @ self.w4.T
        out8 = x @ self.w8.T
        p = probs.unsqueeze(-1)                            # (..., 3, 1)
        result = p[..., 0, :] * out2 + p[..., 1, :] * out4 + p[..., 2, :] * out8
        # Track average effective precision
        bits = torch.tensor([2.0, 4.0, 8.0], device=x.device)
        self._avg_precision = (probs.detach() * bits).sum(-1).mean().item()
        return result

    @property
    def avg_bits(self) -> float:
        return self._avg_precision
