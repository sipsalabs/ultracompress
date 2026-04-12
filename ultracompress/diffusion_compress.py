"""
Diffusion Model Compression via FRR

Diffusion U-Nets/DiTs have many repeated blocks — perfect for FRR sharing.
One shared denoising block applied at different noise levels with
per-timestep modulation (like per-scale in FRR).
"""

import torch
import torch.nn as nn


class DiffusionFRR(nn.Module):
    """Shared denoising block applied across timesteps with per-step modulation."""

    def __init__(self, dim: int, n_timesteps: int = 1000, n_iterations: int = 4):
        super().__init__()
        self.n_iterations = n_iterations
        self.shared_block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.time_embed = nn.Sequential(
            nn.Embedding(n_timesteps, dim), nn.Linear(dim, dim),
        )
        self.iter_scales = nn.ParameterList([
            nn.Parameter(torch.ones(dim)) for _ in range(n_iterations)
        ])
        self.iter_shifts = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(n_iterations)
        ])

    def denoise_step(self, x_noisy: torch.Tensor, timestep: int) -> torch.Tensor:
        """Single denoising step: shared block modulated by timestep and iteration."""
        t_emb = self.time_embed(torch.tensor(timestep, device=x_noisy.device))
        x = x_noisy + t_emb
        for i in range(self.n_iterations):
            residual = self.shared_block(x)
            x = x + residual * self.iter_scales[i] + self.iter_shifts[i]
        return x
