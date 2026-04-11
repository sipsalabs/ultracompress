"""
Genome V2 — More expressive layer replacement.

V1 problem: single bottleneck (1024→128) throws away too much info.
The attention and FFN in the tiny space can't recover what was lost.

V2 approach: Multiple parallel projections ("views") that each
capture different aspects of the input, process independently,
then combine. Like multiple experts but at the projection level.

Same parameter count as V1 but more expressive because:
- 4 views of 32-dim each = 128 total dim
- Each view specializes on different input subspaces
- Combination is learned, not fixed

Also adds a direct residual path (skip the bottleneck entirely)
with a learned scaling factor. This lets the genome learn to
pass through information it can't improve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiViewGenomeLayer(nn.Module):
    """V2 genome layer with multiple projection views + direct path."""

    def __init__(self, big_dim, small_dim, n_views=4, ff_mult=2):
        super().__init__()
        self.big_dim = big_dim
        self.small_dim = small_dim
        self.n_views = n_views
        self.view_dim = small_dim // n_views

        # Multiple down projections (different "views" of the input)
        self.down_projs = nn.ModuleList([
            nn.Linear(big_dim, self.view_dim, bias=False)
            for _ in range(n_views)
        ])

        # Per-view processing (simple but effective: linear + nonlinearity)
        self.view_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.view_dim, self.view_dim * ff_mult, bias=False),
                nn.SiLU(),
                nn.Linear(self.view_dim * ff_mult, self.view_dim, bias=False),
            )
            for _ in range(n_views)
        ])

        # Combine views back to big_dim
        self.up = nn.Linear(small_dim, big_dim, bias=False)

        # Cross-view mixing (lets views share information)
        self.mix = nn.Linear(small_dim, small_dim, bias=False)
        self.mix_norm = nn.RMSNorm(small_dim)

        # Direct residual scaling (learnable bypass)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """x: (batch, seq, big_dim) -> delta (batch, seq, big_dim)"""
        # Multiple views of the input
        views = []
        for i in range(self.n_views):
            v = self.down_projs[i](x)  # (B, T, view_dim)
            v = v + self.view_transforms[i](v)  # residual FFN per view
            views.append(v)

        # Concatenate views
        combined = torch.cat(views, dim=-1)  # (B, T, small_dim)

        # Cross-view mixing
        combined = combined + self.mix(self.mix_norm(combined))

        # Project back up
        delta = self.up(combined)

        # Learnable residual bypass
        return delta * self.residual_scale


class LoRAGenomeLayer(nn.Module):
    """V2 alternative: LoRA-style low-rank transformation.

    Instead of down→process→up, directly learn a low-rank
    transformation: delta = x @ A @ B where A is (big, r), B is (r, big).

    Plus a small nonlinear correction term.
    This is the most parameter-efficient approach.
    """

    def __init__(self, big_dim, rank, ff_mult=2):
        super().__init__()
        self.big_dim = big_dim
        self.rank = rank

        # Low-rank linear path: delta_linear = x @ A @ B
        self.A = nn.Linear(big_dim, rank, bias=False)
        self.B = nn.Linear(rank, big_dim, bias=False)

        # Nonlinear correction: delta_nonlinear = MLP(x_projected)
        self.correct_down = nn.Linear(big_dim, rank, bias=False)
        self.correct_act = nn.SiLU()
        self.correct_up = nn.Linear(rank, big_dim, bias=False)

        # Scaling
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Initialize B to zero so genome starts as identity
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.correct_up.weight)

    def forward(self, x):
        # Linear low-rank path
        linear = self.B(self.A(x))

        # Nonlinear correction
        nonlinear = self.correct_up(self.correct_act(self.correct_down(x)))

        return (linear + nonlinear) * self.scale
