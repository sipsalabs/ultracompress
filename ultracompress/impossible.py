"""
IMPOSSIBLE IDEAS — Things nobody would try. Why not?

"Even if it doesn't sound good on paper it might be insane in practice."

1. WEIGHT TELEPORTATION: What if weights don't need to travel through
   the network? Quantum-inspired: each weight exists in superposition
   across all layers, collapsing to a specific value based on context.

2. BACKWARDS CAUSALITY: What if later layers influence earlier layers?
   Not just residual connections — actual backward information flow
   during the forward pass. Like the brain's feedback connections.

3. ZERO-PARAMETER TRANSFORM: What if a layer could transform without
   ANY parameters? Just rearrange, permute, and interfere existing
   activations. The information is already there — just reorganize it.

4. COMPRESSION BY FORGETTING: What if the best compression is knowing
   what to FORGET? A model that actively discards irrelevant information
   compresses better than one that tries to keep everything.

5. SELF-MODIFYING WEIGHTS: What if weights change DURING inference
   based on the input? Not fixed weights, not generated weights —
   weights that ADAPT in real-time via a Hebbian-like rule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WeightTeleporter(nn.Module):
    """Quantum-inspired: weights exist in superposition across all layers.

    Instead of per-layer weights, a small "wave function" encodes the
    probability distribution of weight values. Each layer "measures"
    the wave function at its position, collapsing it to specific values.

    The wave function is shared — the measurements are different.
    Like FRR but continuous instead of discrete scale modulation.
    """
    def __init__(self, hidden_dim, n_waves=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_waves = n_waves

        # Wave function: encodes weight distributions
        # Each wave has amplitude and frequency
        self.amplitudes = nn.Parameter(torch.randn(n_waves, hidden_dim) * 0.01)
        self.frequencies = nn.Parameter(torch.randn(n_waves) * 0.1 + 1.0)
        self.phases = nn.Parameter(torch.zeros(n_waves))

        # Measurement basis per position
        self.measure = nn.Linear(1, n_waves, bias=False)

    def collapse(self, position_frac):
        """Collapse wave function at a given position -> weight-like values."""
        # Position as continuous [0, 1]
        pos = torch.tensor([[position_frac]], device=self.amplitudes.device)
        measurement = torch.sigmoid(self.measure(pos)).squeeze()  # (n_waves,)

        # Interfere waves at this position
        t = position_frac * 2 * math.pi
        wave_values = torch.stack([
            self.amplitudes[i] * torch.sin(self.frequencies[i] * t + self.phases[i])
            for i in range(self.n_waves)
        ])  # (n_waves, hidden_dim)

        # Weighted sum based on measurement
        collapsed = (wave_values * measurement.unsqueeze(-1)).sum(dim=0)  # (hidden_dim,)
        return collapsed

    def forward(self, x, position_frac):
        """Apply collapsed wave function as a transformation."""
        w = self.collapse(position_frac)
        # Use collapsed values as a diagonal transform + learned offset
        return x * (1 + w.unsqueeze(0).unsqueeze(0))


class BackwardsCausalBlock(nn.Module):
    """Later layers influence earlier layers DURING forward pass.

    Like the brain's feedback connections: higher cortical areas
    send predictions back to lower areas.

    Implementation: run forward, capture output, use it to modulate
    the NEXT forward pass through the same block (iterative refinement
    with feedback).
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Forward path (standard)
        self.forward_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.RMSNorm(hidden_dim)

        # Feedback path (backward influence)
        self.feedback_compress = nn.Linear(hidden_dim, hidden_dim // 4, bias=False)
        self.feedback_expand = nn.Linear(hidden_dim // 4, hidden_dim, bias=False)

        nn.init.zeros_(self.feedback_expand.weight)  # Start with no feedback

    def forward(self, x, future_context=None):
        """
        x: current hidden state
        future_context: information from later layers (if available)
        """
        h = self.norm(x)

        # Forward transformation
        out = F.silu(self.forward_proj(h))

        # If we have feedback from future, use it to modulate
        if future_context is not None:
            feedback = self.feedback_expand(F.silu(self.feedback_compress(future_context)))
            out = out + feedback * 0.1  # Gentle feedback influence

        return x + out


class ZeroParamTransform(nn.Module):
    """Transform without ANY learnable parameters.

    Uses only: permutation, sign flip, and addition of existing features.
    The information is already in the activations — just rearrange it.

    This is the ultimate compression: 0 parameters per layer.
    Only thing learned: which permutation to use (a single index tensor).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The only "parameters": permutation indices + sign flips
        # These are integers, not floating point — essentially free storage
        self.register_buffer('perm', torch.randperm(hidden_dim))
        self.register_buffer('signs', (torch.randint(0, 2, (hidden_dim,)) * 2 - 1).float())

        # Learnable: just a single scalar per layer (mixing ratio)
        self.mix = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """Transform by permuting and sign-flipping features."""
        # Permute features
        x_perm = x[..., self.perm]
        # Flip signs of half the features
        x_flip = x_perm * self.signs
        # Mix with original
        return x + self.mix * (x_flip - x)


class SelectiveForgetter(nn.Module):
    """Compression by FORGETTING irrelevant information.

    At each layer, identify which features are irrelevant for the task
    and zero them out. The model gets SMALLER as it processes.

    Like attention dropout but learned and content-dependent.
    The compression IS the inference.
    """
    def __init__(self, hidden_dim, keep_ratio=0.5):
        super().__init__()
        self.keep_ratio = keep_ratio

        # Importance scorer: which features matter?
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

    def forward(self, x):
        """Score features, keep only the important ones."""
        scores = torch.sigmoid(self.scorer(x.detach()))  # Detach to not affect main gradient

        # Soft top-k: keep top keep_ratio features
        k = int(self.keep_ratio * x.shape[-1])
        threshold = scores.topk(k, dim=-1).values[..., -1:]
        mask = (scores >= threshold).float()

        # Apply mask — forgotten features become zero
        return x * mask


class HebbianAdapter(nn.Module):
    """Self-modifying weights via Hebbian learning DURING inference.

    "Neurons that fire together wire together."

    The weights update themselves based on the input pattern —
    no gradient descent, just correlation-based adaptation.
    The model literally learns from each input in real-time.
    """
    def __init__(self, hidden_dim, rank=16, hebbian_lr=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.hebbian_lr = hebbian_lr

        # Base transformation (learned by gradient descent)
        self.base_down = nn.Linear(hidden_dim, rank, bias=False)
        self.base_up = nn.Linear(rank, hidden_dim, bias=False)

        # Hebbian state: accumulated correlation matrix
        # This CHANGES during inference
        self.register_buffer('hebbian_state', torch.zeros(rank, rank))

        nn.init.zeros_(self.base_up.weight)

    def forward(self, x):
        """Apply base transform + Hebbian modification."""
        # Project down
        h = self.base_down(x)  # (B, T, rank)

        # Hebbian update: correlate projected features
        if self.training or True:  # Always adapt
            with torch.no_grad():
                # Correlation of projected features (averaged over batch and seq)
                h_mean = h.reshape(-1, self.rank)
                correlation = (h_mean.T @ h_mean) / h_mean.shape[0]
                # Update Hebbian state (exponential moving average)
                self.hebbian_state = 0.99 * self.hebbian_state + 0.01 * correlation

        # Apply Hebbian-modified transformation
        h_modified = h + F.linear(h, self.hebbian_state * self.hebbian_lr)

        # Project back up
        return x + self.base_up(h_modified)
