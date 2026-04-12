"""Cellular Automata Weight Generation — grow weight matrices from tiny seeds + rules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellularWeightGenerator(nn.Module):
    """Learned rules + seed → iterate → weight matrix. Like Conway's Game of Life for weights."""

    def __init__(self, seed_h=8, seed_w=8, hidden=16):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, seed_h, seed_w) * 0.1)
        self.rules = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def grow(self, rows, cols, n_iter=64):
        x = F.interpolate(self.seed, size=(rows, cols), mode="bilinear", align_corners=False)
        for _ in range(n_iter):
            x = x + self.rules(x) * 0.1          # residual update per step
        return x.squeeze(0).squeeze(0)

    def compression_ratio(self, rows, cols):
        param_count = sum(p.numel() for p in self.parameters())
        return (rows * cols) / param_count


class NCAWeightGenerator(nn.Module):
    """Neural Cellular Automata — each cell carries a hidden state vector, updated by a small convnet."""

    def __init__(self, seed_h=8, seed_w=8, state_dim=8, hidden=32):
        super().__init__()
        self.state_dim = state_dim
        self.seed = nn.Parameter(torch.randn(1, state_dim, seed_h, seed_w) * 0.1)
        self.update = nn.Sequential(
            nn.Conv2d(state_dim, hidden, 3, padding=1, groups=1), nn.GELU(),
            nn.Conv2d(hidden, state_dim, 1),
        )
        self.project = nn.Linear(state_dim, 1)   # state → scalar weight

    def grow(self, rows, cols, n_iter=64):
        x = F.interpolate(self.seed, size=(rows, cols), mode="bilinear", align_corners=False)
        for _ in range(n_iter):
            dx = self.update(x) * 0.1
            mask = (torch.rand(1, 1, rows, cols, device=x.device) > 0.5).float()  # stochastic update
            x = x + dx * mask
        # project multi-channel state to scalar weight per cell
        return self.project(x.squeeze(0).permute(1, 2, 0)).squeeze(-1)

    def compression_ratio(self, rows, cols):
        param_count = sum(p.numel() for p in self.parameters())
        return (rows * cols) / param_count


def fit_ca(target, generator=None, lr=1e-3, steps=2000, seed_h=8, seed_w=8):
    """Fit a cellular automata generator to reproduce a target weight matrix."""
    rows, cols = target.shape
    if generator is None:
        generator = NCAWeightGenerator(seed_h, seed_w).to(target.device)
    opt = torch.optim.Adam(generator.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        out = generator.grow(rows, cols)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            cr = generator.compression_ratio(rows, cols)
            print(f"  step {step}: loss={loss.item():.6f}  compression={cr:.1f}x")
    return generator
