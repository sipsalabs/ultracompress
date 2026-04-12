"""
Progressive Model Loading — Coarse-to-Fine Decompression

Level 0: 2-bit (instant, usable immediately). Level 1: 4-bit residual
correction (background). Level 2: fp16 residual (final quality).
"""

import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class ProgressiveModel:
    levels: List[torch.Tensor] = field(default_factory=list)
    original_shape: tuple = ()
    current_level: int = 0

    @staticmethod
    def encode(weights: torch.Tensor) -> "ProgressiveModel":
        shape, flat = weights.shape, weights.flatten().float()
        mn, mx = flat.min(), flat.max()
        s0 = (mx - mn) / 3.0
        q0 = ((flat - mn) / s0).clamp(0, 3).round()
        r0 = q0 * s0 + mn                          # Level 0 reconstruction
        res1 = flat - r0
        mn1, mx1 = res1.min(), res1.max()
        s1 = (mx1 - mn1) / 15.0 if mx1 > mn1 else torch.tensor(1.0)
        q1 = ((res1 - mn1) / s1).clamp(0, 15).round()
        r1 = q1 * s1 + mn1                         # Level 1 reconstruction
        res2 = (flat - r0 - r1).to(torch.float16)  # Level 2 residual
        L0 = torch.cat([torch.tensor([mn, s0]), q0])
        s1v = s1 if isinstance(s1, float) else s1.item()
        L1 = torch.cat([torch.tensor([mn1, s1v]), q1])
        return ProgressiveModel(levels=[L0, L1, res2], original_shape=shape)

    def decompress(self, up_to_level: int = None) -> torch.Tensor:
        lvl = min(up_to_level if up_to_level is not None else 2, len(self.levels) - 1)
        n = 1
        for d in self.original_shape:
            n *= d
        out = torch.zeros(n, dtype=torch.float32)
        if lvl >= 0:
            out += self.levels[0][2:2+n] * self.levels[0][1] + self.levels[0][0]
        if lvl >= 1:
            out += self.levels[1][2:2+n] * self.levels[1][1] + self.levels[1][0]
        if lvl >= 2:
            out += self.levels[2][:n].float()
        self.current_level = lvl
        return out.reshape(self.original_shape)
