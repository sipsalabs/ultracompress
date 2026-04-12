"""
NEURO-ADVANCED MODULES — Neuroscience-inspired computation for FRR.

Three modules that enhance the shared block with brain-inspired mechanisms:

1. PhaseBasedLayer (PRISM-inspired): Complex-valued weights where phase
   encodes relationships and magnitude encodes strength. 2x information
   per parameter via complex arithmetic.

2. AstrocyteModulator: A tiny meta-network that watches activation
   statistics and globally modulates synaptic strength. Applied between
   FRR iterations like glial cells modulating neural circuits.

3. OscillatoryBinding: Learned oscillation frequencies per feature
   dimension. Features that should bind get similar frequencies,
   adding temporal structure to the residual stream.

All drop-in compatible with FractalModel's shared block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ================================================================
# MODULE 1: PHASE-BASED LAYER (PRISM-inspired, Yildirim 2025)
# ================================================================

class PhaseBasedLayer(nn.Module):
    """Complex-valued linear where phase = relationship, magnitude = strength.

    Converts real input to complex (first half = real, second half = imag),
    applies complex linear transform, returns magnitude of result.
    Same param count as real linear but 2x information capacity.
    """
    def __init__(self, dim, bias=False):
        super().__init__()
        half = dim // 2
        self.half = half
        # Complex weight stored as real + imag parts
        self.weight_real = nn.Parameter(torch.randn(half, half) * (2.0 / dim) ** 0.5)
        self.weight_imag = nn.Parameter(torch.randn(half, half) * (2.0 / dim) ** 0.5)
        # Phase coherence prior: init imag small so phases start near zero
        self.weight_imag.data *= 0.1
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(half))
            self.bias_imag = nn.Parameter(torch.zeros(half))
        else:
            self.bias_real = self.bias_imag = None
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """x: (..., dim) real -> (..., dim) real via complex transform."""
        # Split real input into complex: first half = real, second half = imag
        x_real, x_imag = x[..., :self.half], x[..., self.half:2*self.half]

        # Complex matmul: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = F.linear(x_real, self.weight_real) - F.linear(x_imag, self.weight_imag)
        out_imag = F.linear(x_real, self.weight_imag) + F.linear(x_imag, self.weight_real)

        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag

        # Output: magnitude preserves information, phase was the computation
        magnitude = (out_real.pow(2) + out_imag.pow(2) + 1e-8).sqrt()
        # Reconstruct full-dim output: magnitude + phase-rotated components
        return torch.cat([magnitude, out_real], dim=-1) * self.output_scale


# ================================================================
# MODULE 2: ASTROCYTE MODULATOR
# ================================================================

class AstrocyteModulator(nn.Module):
    """Meta-network that globally modulates synaptic strength.

    Reads running activation statistics (mean, variance) and outputs
    per-dimension gain modulation. Like astrocytes watching neural
    activity and adjusting synaptic efficacy.

    Tiny: 2-layer MLP from 2*hidden_dim stats -> hidden_dim gains.
    Applied BETWEEN FRR iterations.
    """
    def __init__(self, hidden_dim, bottleneck=32):
        super().__init__()
        # Input: mean + variance per dimension = 2 * hidden_dim
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_dim, bias=False),
        )
        # Init to output near-1 gains (no modulation at start)
        nn.init.zeros_(self.net[2].weight)
        self.baseline = nn.Parameter(torch.ones(hidden_dim))
        # EMA momentum for running stats
        self.momentum = 0.9

    def forward(self, x, running_mean=None, running_var=None):
        """x: (B, T, D) -> gain: (1, 1, D), updated stats."""
        with torch.no_grad():
            batch_mean = x.mean(dim=(0, 1))  # (D,)
            batch_var = x.var(dim=(0, 1))     # (D,)
        if running_mean is None:
            running_mean = batch_mean
            running_var = batch_var
        else:
            running_mean = self.momentum * running_mean + (1 - self.momentum) * batch_mean
            running_var = self.momentum * running_var + (1 - self.momentum) * batch_var

        # Feed stats to tiny MLP
        stats = torch.cat([running_mean, running_var], dim=-1).unsqueeze(0)  # (1, 2D)
        delta = self.net(stats)  # (1, D)
        gain = (self.baseline + delta).unsqueeze(0)  # (1, 1, D)

        return gain, running_mean, running_var


# ================================================================
# MODULE 3: OSCILLATORY BINDING
# ================================================================

class OscillatoryBinding(nn.Module):
    """Neurons communicate via frequency, not just amplitude.

    Each feature dimension has a learned oscillation frequency.
    Hidden states are modulated by sin(freq * layer_idx), adding
    temporal/depth structure. Features that should bind learn
    similar frequencies, creating coherent oscillatory groups.
    """
    def __init__(self, hidden_dim, max_freq=10.0):
        super().__init__()
        # Learned frequency per feature dimension
        self.frequencies = nn.Parameter(torch.randn(hidden_dim) * 0.5)
        # Learned phase offset per dimension
        self.phases = nn.Parameter(torch.zeros(hidden_dim))
        # How strongly oscillation modulates (starts subtle)
        self.strength = nn.Parameter(torch.tensor(0.1))
        self.max_freq = max_freq

    def forward(self, x, layer_idx):
        """x: (B, T, D) -> modulated x with oscillatory binding.

        layer_idx: int, the current depth/iteration index.
        """
        # Clamp frequencies to prevent explosion
        freq = self.frequencies.clamp(-self.max_freq, self.max_freq)
        # Oscillation: sin(freq * layer_idx + phase)
        osc = torch.sin(freq * layer_idx + self.phases)  # (D,)
        # Modulate: x * (1 + strength * oscillation)
        return x * (1.0 + self.strength * osc)


# ================================================================
# COMBINED: NeuroFractalBlock
# ================================================================

class NeuroFractalBlock(nn.Module):
    """FRR shared block enhanced with all three neuro modules.

    Standard attention + FFN backbone, augmented with:
    - PhaseBasedLayer replacing the output projection (2x info capacity)
    - AstrocyteModulator for global gain control between iterations
    - OscillatoryBinding for depth-dependent feature modulation

    Drop-in replacement for FractalBlock in FractalModel.
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2, astro_bottleneck=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Standard attention
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Phase-based layer in FFN (replaces standard gate projection)
        ff_dim = hidden_dim * ff_mult
        self.phase_gate = PhaseBasedLayer(hidden_dim)
        self.up = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, hidden_dim, bias=False)

        # Neuro modules
        self.astrocyte = AstrocyteModulator(hidden_dim, astro_bottleneck)
        self.oscillator = OscillatoryBinding(hidden_dim)

        # Norms
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None,
                layer_idx=0, astro_state=None):
        """Forward with neuro enhancements.

        Args:
            x: (B, T, D) hidden states
            scale_gamma/beta: per-scale modulation from FractalModel
            layer_idx: current iteration index (for oscillatory binding)
            astro_state: tuple of (running_mean, running_var) or None

        Returns:
            x: (B, T, D) updated hidden states
            astro_state: updated (running_mean, running_var)
        """
        B, T, D = x.shape

        # Oscillatory binding: depth-dependent feature modulation
        x = self.oscillator(x, layer_idx)

        # Astrocyte modulation: global gain from activation statistics
        running_mean = running_var = None
        if astro_state is not None:
            running_mean, running_var = astro_state
        gain, running_mean, running_var = self.astrocyte(x, running_mean, running_var)
        x = x * gain

        # --- Standard attention ---
        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        # --- Phase-enhanced FFN ---
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # Phase layer adds complex-valued relationship encoding to input
        h_phase = h + self.phase_gate(h)
        x = x + self.down(F.silu(self.gate(h_phase)) * self.up(h))

        return x, (running_mean, running_var)
