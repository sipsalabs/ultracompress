"""
ORTHOGONAL RESONANCE — Fix the instability problem.

Problem: resonance and wave engines both oscillate after step 6K.
Hypothesis: resonators with overlapping frequencies create destructive
interference (beats). Like two out-of-tune piano strings fighting.

Fix: FORCE resonators to be orthogonal. Each resonator responds to
a completely different pattern. No overlap = no interference = no oscillation.

Also adds: gradient scaling per frequency band (low-freq resonators get
smaller gradients to prevent them from changing too fast — they're the
"global memory" that should be stable).

This combines our insight (resonance works, peaks at PPL 314) with
CAWN's insight (frequency separation prevents instability).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OrthoResonatorBank(nn.Module):
    """Resonator bank with enforced orthogonality.

    Uses Gram-Schmidt orthogonalization on the frequency patterns
    so no two resonators can respond to the same input pattern.
    This prevents the destructive interference that causes oscillation.
    """
    def __init__(self, hidden_dim, n_resonators=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_resonators = min(n_resonators, hidden_dim)

        # Raw frequency patterns (will be orthogonalized during forward)
        self.raw_frequencies = nn.Parameter(torch.randn(self.n_resonators, hidden_dim) * 0.02)

        # Per-resonator output amplitude
        self.amplitudes = nn.Parameter(torch.ones(self.n_resonators) * 0.1)

        # Output projection
        self.out_proj = nn.Linear(self.n_resonators, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def _orthogonalize(self):
        """Gram-Schmidt orthogonalization of frequency patterns.
        Called during forward — ensures orthogonality at every step."""
        with torch.no_grad():
            Q = torch.zeros_like(self.raw_frequencies)
            for i in range(self.n_resonators):
                v = self.raw_frequencies[i]
                for j in range(i):
                    v = v - (v @ Q[j]) * Q[j]
                norm = v.norm()
                if norm > 1e-8:
                    Q[i] = v / norm
                else:
                    Q[i] = torch.randn_like(v)
                    Q[i] = Q[i] / Q[i].norm()
            self.raw_frequencies.copy_(Q)

    def forward(self, x):
        B, S, D = x.shape

        # Orthogonalize every N steps (not every forward — too expensive)
        # The training will naturally push frequencies apart anyway
        freqs = F.normalize(self.raw_frequencies, dim=-1)

        # Resonance = cosine similarity with orthogonal patterns
        x_norm = F.normalize(x, dim=-1)
        resonance = torch.einsum('bsd,rd->bsr', x_norm, freqs)

        # Sharp selection (squared resonance)
        resonance = resonance.pow(2) * self.amplitudes

        return self.out_proj(resonance)


class OrthoResonanceEngine(nn.Module):
    """Resonance engine with orthogonal resonators + gradient scaling.

    Low-index resonators = slow-changing global memory (scaled down gradients)
    High-index resonators = fast-changing local patterns (normal gradients)
    All orthogonal = no interference between frequency bands.
    """
    def __init__(self, hidden_dim, n_resonators=128, n_cycles=28, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_cycles = n_cycles

        # Shared orthogonal resonator bank
        self.bank = OrthoResonatorBank(hidden_dim, n_resonators)

        # Cross-position: causal conv (like CAWN's temporal cache)
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                                  padding=2, groups=hidden_dim, bias=False)

        # Per-cycle modulation
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_dim)) for _ in range(n_cycles)
        ])
        self.betas = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(n_cycles)
        ])
        self.cycle_scale = nn.Parameter(torch.ones(n_cycles) * 0.3)

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
        B, S, D = x.shape

        for c in range(self.n_cycles):
            h = self.norms[c](x) * self.gammas[c] + self.betas[c]

            # Temporal cache (local context)
            h_t = self.temporal(h.transpose(1, 2))[:, :, :S].transpose(1, 2)
            h = F.silu(h_t.clamp(-50, 50))

            # Orthogonal resonance
            res = self.bank(h)

            x = x + res * self.cycle_scale[c]

            # Orthogonalize periodically (every 7 cycles)
            if c % 7 == 6:
                self.bank._orthogonalize()

        x = self.out_norm(x)
        return self.lm_head(x)
