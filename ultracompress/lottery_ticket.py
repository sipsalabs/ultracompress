"""
Iterative Magnitude Pruning — Lottery Ticket Hypothesis

Prune smallest weights, rewind survivors to initial values, repeat.
The surviving mask IS the winning ticket. Storage: 1-bit mask + fp16 values.
"""

import torch
from dataclasses import dataclass


@dataclass
class WinningTicket:
    mask: torch.Tensor        # bool — 1 bit per weight position
    values: torch.Tensor      # float16 surviving weights (flat)
    original_shape: tuple
    survival_ratio: float

    def storage_bytes(self) -> int:
        return (self.mask.numel() + 7) // 8 + self.values.numel() * 2

    @property
    def compression_ratio(self) -> float:
        return 1.0 / max(self.survival_ratio, 1e-9)

    def decompress(self) -> torch.Tensor:
        out = torch.zeros(self.mask.numel(), dtype=torch.float32)
        out[self.mask.reshape(-1)] = self.values.float()
        return out.reshape(self.original_shape)


class LotteryTicketFinder:
    """Iterative magnitude pruning to find winning tickets."""

    def find_ticket(self, weights: torch.Tensor, prune_ratio: float = 0.2,
                    n_rounds: int = 5) -> WinningTicket:
        init_weights = weights.clone()
        mask = torch.ones_like(weights, dtype=torch.bool)
        for _ in range(n_rounds):
            alive = weights[mask].abs()
            n_prune = int(alive.numel() * prune_ratio)
            if n_prune == 0:
                break
            threshold = alive.kthvalue(n_prune).values.item()
            mask &= weights.abs() > threshold
            weights = init_weights.clone()  # rewind to initial values
        survival = mask.sum().item() / mask.numel()
        values = init_weights[mask].to(torch.float16)
        return WinningTicket(mask, values, weights.shape, survival)
