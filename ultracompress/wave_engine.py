"""
WAVE INTERFERENCE ENGINE — Computation through wave propagation.

NOT sequential. NOT attention. NOT matrix multiply.

Every token emits a wave in hidden space.
The waves propagate and INTERFERE with each other.
The interference pattern IS the output.

Like dropping stones in a pond:
- Each stone = one token
- The ripples spread and overlap
- Where ripples meet = interference = computation
- The final pattern encodes the answer

The "model" is the MEDIUM — how waves propagate.
A tiny set of parameters defines propagation speed,
dampening, and reflection. That's the DNA.

O(n log n) via FFT. Naturally parallel.
No attention. No sequential scan. Just waves.

This is Sip's "unlimited connections" realized:
every token connects to every other token through
wave propagation, simultaneously, at the speed of FFT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WaveMedium(nn.Module):
    """The medium through which waves propagate.

    Defines propagation characteristics:
    - Speed: how fast info travels (frequency-dependent)
    - Dampening: how quickly waves decay
    - Reflection: how waves bounce off boundaries
    - Nonlinearity: how waves interact (the key to computation)

    All encoded in a tiny parameter set = the DNA.
    """
    def __init__(self, hidden_dim, n_freqs=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_freqs = n_freqs

        # Wave propagation parameters (THE DNA)
        # Speed per frequency: how fast each frequency component travels
        self.speed = nn.Parameter(torch.ones(n_freqs) * 0.5)
        # Dampening per frequency: how quickly each dies out
        self.dampen = nn.Parameter(torch.zeros(n_freqs))
        # Phase shift per frequency: how the medium bends the wave
        self.phase_shift = nn.Parameter(torch.zeros(n_freqs))
        # Nonlinear coupling: how different frequencies interact
        # THIS is where computation happens — wave mixing
        self.coupling = nn.Parameter(torch.eye(n_freqs) * 0.1)

    def propagate(self, wave_spectrum):
        """Propagate waves through the medium.

        wave_spectrum: (B, S, n_freqs) complex tensor in frequency domain
        Returns: transformed spectrum after propagation
        """
        # 1. Frequency-dependent speed (shift phases based on speed)
        phase = torch.exp(1j * self.speed.unsqueeze(0).unsqueeze(0) * math.pi)
        wave_spectrum = wave_spectrum * phase

        # 2. Dampening (attenuate high frequencies more)
        dampen = torch.exp(-F.softplus(self.dampen).unsqueeze(0).unsqueeze(0))
        wave_spectrum = wave_spectrum * dampen

        # 3. Phase shift (the medium bends the wave)
        shift = torch.exp(1j * self.phase_shift.unsqueeze(0).unsqueeze(0))
        wave_spectrum = wave_spectrum * shift

        # 4. Nonlinear frequency coupling (THIS IS THE COMPUTATION)
        # Different frequencies interact → new frequencies emerge
        # Like how ocean waves create harmonics when they collide
        coupling = self.coupling.to(wave_spectrum.dtype)
        wave_spectrum = torch.einsum('bsf,fg->bsg', wave_spectrum, coupling)

        return wave_spectrum


class WaveEngine(nn.Module):
    """Full wave interference engine for language.

    Pipeline:
    1. Embed tokens → initial wave amplitudes
    2. FFT → frequency domain (each position's contribution to each frequency)
    3. Propagate through medium (waves interact, interfere, compute)
    4. IFFT → back to position domain
    5. Repeat (multiple propagation steps = deeper processing)
    6. Read out → logits

    The SAME medium is used for every propagation step (like FRR's shared block).
    The medium IS the model. Everything else is just signal processing.
    """
    def __init__(self, hidden_dim, n_freqs=64, n_steps=28, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_freqs = n_freqs
        self.n_steps = n_steps

        # THE MEDIUM (shared across all propagation steps)
        self.medium = WaveMedium(hidden_dim, n_freqs)

        # Lift hidden state to wave space
        self.to_wave = nn.Linear(hidden_dim, n_freqs * 2, bias=False)  # real + imag
        self.from_wave = nn.Linear(n_freqs, hidden_dim, bias=False)
        nn.init.zeros_(self.from_wave.weight)

        # Per-step modulation (like FRR's gamma/beta)
        self.step_scale = nn.Parameter(torch.ones(n_steps))

        # Position encoding via wave (natural — position IS phase)
        self.pos_freq = nn.Parameter(torch.randn(n_freqs) * 0.1)

        # Norms for stability
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_steps)])

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
        B, S, D = x.shape

        for step in range(self.n_steps):
            h = self.norms[step](x)

            # Lift to wave space (complex-valued)
            wave_components = self.to_wave(h)  # (B, S, n_freqs*2)
            wave_real = wave_components[..., :self.n_freqs]
            wave_imag = wave_components[..., self.n_freqs:]
            waves = torch.complex(wave_real, wave_imag)  # (B, S, n_freqs)

            # Add position encoding as phase
            positions = torch.arange(S, device=x.device).float().unsqueeze(1)  # (S, 1)
            pos_phase = torch.exp(1j * positions * self.pos_freq.unsqueeze(0))  # (S, n_freqs)
            waves = waves * pos_phase.unsqueeze(0)

            # FFT along SEQUENCE dimension → interference between positions
            # This is the key: FFT makes ALL positions interact simultaneously
            wave_freq = torch.fft.fft(waves, dim=1)  # (B, S, n_freqs)

            # PROPAGATE through medium (the actual computation)
            wave_freq = self.medium.propagate(wave_freq)

            # IFFT back to position domain
            waves_out = torch.fft.ifft(wave_freq, dim=1)  # (B, S, n_freqs)

            # Convert back to real hidden state
            wave_real_out = waves_out.real  # (B, S, n_freqs)
            delta = self.from_wave(wave_real_out)  # (B, S, D)

            # Residual with step scaling
            x = x + delta * self.step_scale[step] * 0.1

        x = self.out_norm(x)
        return self.lm_head(x)

    def dna_size(self):
        """The medium parameters = the DNA of the wave engine."""
        medium_params = sum(p.numel() for p in self.medium.parameters())
        lift_params = sum(p.numel() for p in [self.to_wave.weight, self.from_wave.weight])
        mod_params = self.step_scale.numel() + self.pos_freq.numel()
        norm_params = sum(p.numel() for n in self.norms for p in n.parameters())
        return {
            'medium': medium_params,
            'lift': lift_params,
            'modulation': mod_params,
            'norms': norm_params,
            'total': medium_params + lift_params + mod_params + norm_params,
        }
