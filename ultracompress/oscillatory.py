"""
OSCILLATORY BINDING — Frequency-based feature binding (#9).

The brain binds features through synchronized oscillations.
Neurons that fire together at the same frequency = bound.

Applied to AI: represent features as oscillations at different
frequencies. Features at the same frequency are "bound" (related).
Cross-frequency coupling = hierarchical relationships.

This gives the model a natural way to group and relate information
without attention — through resonance, not pairwise comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OscillatoryLayer(nn.Module):
    """Bind features through learned oscillation frequencies."""

    def __init__(self, dim, n_oscillators=16):
        super().__init__()
        # Each oscillator has a frequency and phase
        self.frequencies = nn.Parameter(torch.randn(n_oscillators) * 0.1 + 1.0)
        self.phases = nn.Parameter(torch.zeros(n_oscillators))
        # Project to/from oscillator space
        self.to_osc = nn.Linear(dim, n_oscillators, bias=False)
        self.from_osc = nn.Linear(n_oscillators, dim, bias=False)
        nn.init.zeros_(self.from_osc.weight)

    def forward(self, x, t=None):
        """x: (B, S, D), t: time step (cycle number)"""
        if t is None:
            t = torch.zeros(1, device=x.device)
        # Project to oscillator amplitudes
        amps = self.to_osc(x)  # (B, S, n_osc)
        # Apply oscillation: amplitude * sin(freq * t + phase)
        osc = amps * torch.sin(self.frequencies * t + self.phases)
        # Project back
        return x + self.from_osc(osc)
