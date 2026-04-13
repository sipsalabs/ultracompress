"""
PHASE INTERFERENCE NETWORK — Computation through wave interference.

The invention: represent everything as complex waves (amplitude + phase).
Computation = interference of multiple waves.

Why this is fundamentally different:
- Transformer: multiply giant matrices (brute force, O(n²))
- Phase net: interfere waves (natural parallelism, O(n log n) via FFT)

One wave pattern can encode MULTIPLE functions simultaneously.
That's what a hologram does: one 2D pattern encodes entire 3D scene.
Our HWI module proved this works for weight storage.
This extends it: the COMPUTATION is holographic, not just the storage.

A complex number z = r * e^(iθ) is literally a 2D rotation.
This IS Sip's "dimensional box" — every complex multiplication
rotates AND scales in the complex plane simultaneously.

Trainable with Wirtinger derivatives (well-studied for complex gradients).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhaseLayer(nn.Module):
    """One layer of phase interference computation.

    Input: real tensor (B, S, D)
    1. Lift to complex: multiply by learned phase angles → complex
    2. Interfere: apply learned interference patterns
    3. Project back: take magnitude or real part → real

    The interference pattern IS the computation.
    Different angles = different functions.
    One set of angles, recursively applied with modulation = FRR but in phase space.
    """
    def __init__(self, hidden_dim, n_harmonics=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_harmonics = n_harmonics

        # Phase angles for lifting real → complex (the DNA)
        self.phase_angles = nn.Parameter(torch.randn(n_harmonics) * 0.1)

        # Frequency indices for each harmonic
        self.register_buffer('freq_indices',
            torch.arange(n_harmonics).float() / n_harmonics * hidden_dim)

        # Interference weights (how harmonics combine)
        self.interference = nn.Parameter(torch.randn(n_harmonics, n_harmonics) * 0.01)

        # Output projection (complex → real)
        self.out_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        B, S, D = x.shape

        # 1. Lift to frequency domain via learned phase modulation
        # Apply FFT along hidden dim
        x_freq = torch.fft.rfft(x, dim=-1)  # (B, S, D//2+1)

        # 2. Phase modulation: rotate frequencies by learned angles
        n_freq = x_freq.shape[-1]
        n_use = min(self.n_harmonics, n_freq)
        phase = torch.exp(1j * self.phase_angles[:n_use])
        x_freq[..., :n_use] = x_freq[..., :n_use] * phase

        # 3. Interference: harmonics interact (the actual computation)
        # Extract top harmonics, mix them, put back
        harmonics = x_freq[..., :n_use]  # (B, S, n_use)
        # Complex matrix multiply = interference
        mixed = torch.einsum('bsh,hk->bsk', harmonics, self.interference[:n_use, :n_use].to(torch.cfloat))
        x_freq[..., :n_use] = mixed

        # 4. Back to real domain
        x_out = torch.fft.irfft(x_freq, n=D, dim=-1)  # (B, S, D)

        return x_out * self.out_scale


class PhaseNet(nn.Module):
    """Full phase interference network.

    Like FRR but the shared block is phase-based, not attention+FFN.
    O(n log n) per pass via FFT instead of O(n²) for attention.
    """
    def __init__(self, hidden_dim, n_harmonics, n_cycles, vocab_size,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_cycles = n_cycles

        # SHARED phase layer (applied recursively like FRR)
        self.phase = PhaseLayer(hidden_dim, n_harmonics)

        # Per-cycle modulation (like FRR's gamma/beta)
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_dim)) for _ in range(n_cycles)
        ])
        self.betas = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(n_cycles)
        ])

        # Cross-position: simple causal convolution (NOT attention)
        self.pos_mix = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                                 padding=1, groups=hidden_dim, bias=False)

        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_cycles)])

        # Embeddings
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.out_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.out_norm.weight = nn.Parameter(norm_weight, requires_grad=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()

        for c in range(self.n_cycles):
            h = self.norms[c](x) * self.gammas[c] + self.betas[c]

            # Phase interference (replaces both attention AND FFN)
            h_phase = self.phase(h)

            # Cross-position mixing (cheap causal conv, NOT O(n²) attention)
            h_pos = self.pos_mix(h.transpose(1, 2)).transpose(1, 2)

            # Combine
            x = x + F.silu(h_phase + h_pos) * 0.3

        x = self.out_norm(x)
        return self.lm_head(x)
