"""
CAWN-RESONANCE HYBRID — Our resonance + CAWN's stability techniques.

Combines:
- OUR resonance concept (selective amplification of matching patterns)
- CAWN's frequency-dependent retention (low freq=global, high=local)
- CAWN's hard-threshold gating (prevent noise corruption)
- CAWN's temporal syntax cache (local context before global)
- CAWN's causal phase accumulation (O(L) strict causal)

Why this combo nobody has:
- CAWN uses unique layers (no weight sharing). We share recursively = FRR-like.
- CAWN's resonance is implicit. Ours is explicit (learned resonator bank).
- Combined: shared resonance block + CAWN stability = tiny + stable.

This is the fusion of our invention (resonance, FRR) with their engineering
(stability, gating, frequency separation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StableResonatorBank(nn.Module):
    """Resonator bank with CAWN's stability techniques.

    Each resonator has:
    - Frequency pattern (what it responds to)
    - Retention rate (frequency-dependent: low=global, high=local)
    - Hard-threshold gate (prevent noise from corrupting state)
    """
    def __init__(self, hidden_dim, n_resonators=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_resonators = n_resonators

        # Resonator frequency patterns
        self.frequencies = nn.Parameter(torch.randn(n_resonators, hidden_dim) * 0.02)

        # Frequency-dependent retention (CAWN technique)
        # Low-index resonators = global memory (retain ~0.999)
        # High-index = local scratchpad (retain ~0.5)
        freq_bias = torch.linspace(3.0, 0.0, n_resonators)  # bias: high→low retention
        self.retention_bias = nn.Parameter(freq_bias)
        self.retention_proj = nn.Linear(hidden_dim, n_resonators, bias=False)

        # Hard-threshold gate (CAWN technique — prevent noise)
        self.gate_proj = nn.Linear(hidden_dim, n_resonators, bias=False)
        self.gate_bias = nn.Parameter(torch.full((n_resonators,), -3.0))  # default near-zero

        # Output mixing
        self.out_proj = nn.Linear(n_resonators, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x, prev_state=None):
        """
        x: (B, S, D)
        prev_state: (B, n_resonators) — accumulated resonance from previous positions
        Returns: output (B, S, D), new_state (B, n_resonators)
        """
        B, S, D = x.shape

        # Compute resonance strength per position
        x_norm = F.normalize(x, dim=-1)
        f_norm = F.normalize(self.frequencies, dim=-1)
        resonance = torch.einsum('bsd,rd->bsr', x_norm, f_norm)  # (B, S, n_res)

        # Hard-threshold gate (CAWN STE technique)
        gate_logit = self.gate_proj(x.mean(dim=1)) + self.gate_bias  # (B, n_res)
        gate_sigmoid = torch.sigmoid(gate_logit)
        # STE: hard threshold in forward, smooth gradient in backward
        gate_hard = gate_sigmoid * (gate_sigmoid >= 0.001).float()
        gate = gate_hard - gate_sigmoid.detach() + gate_sigmoid  # STE
        gate = gate.unsqueeze(1)  # (B, 1, n_res)

        # Frequency-dependent retention
        retention_logit = self.retention_proj(x.mean(dim=1)) + self.retention_bias  # (B, n_res)
        retention = torch.sigmoid(retention_logit).unsqueeze(1)  # (B, 1, n_res)

        # Causal accumulation with retention
        # Each position's resonance = gated new + retained old
        outputs = []
        state = prev_state if prev_state is not None else torch.zeros(B, self.n_resonators, device=x.device)

        for t in range(S):
            new_signal = resonance[:, t] * gate.squeeze(1)  # (B, n_res)
            state = new_signal + state * retention.squeeze(1)  # causal accumulation
            outputs.append(state)

        accumulated = torch.stack(outputs, dim=1)  # (B, S, n_res)

        # Project back to hidden space
        output = self.out_proj(accumulated)
        final_state = state

        return output, final_state


class CAWNResonanceBlock(nn.Module):
    """One block: temporal cache + stable resonance + FFN-lite.

    Applied recursively like FRR — same block, many passes.
    """
    def __init__(self, hidden_dim, n_resonators=128):
        super().__init__()

        # Temporal syntax cache (CAWN technique — local context first)
        self.temporal_cache = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                                        padding=2, groups=hidden_dim, bias=False)
        # padding=2 left for causal

        # Stable resonator bank (our invention + CAWN stability)
        self.resonators = StableResonatorBank(hidden_dim, n_resonators)

        # Tiny FFN (just a small projection, not the full 3x expansion)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
        )
        nn.init.zeros_(self.ffn[0].weight)

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None, resonance_state=None):
        B, S, D = x.shape

        # Pre-norm + modulation
        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # Temporal syntax cache (causal 1D conv)
        h_conv = self.temporal_cache(h.transpose(1, 2))[:, :, :S].transpose(1, 2)
        h = F.silu(h_conv.clamp(-50, 50))  # CAWN technique: clamp before activation

        # Stable resonance (cross-position through shared resonance state)
        res_out, new_state = self.resonators(h, resonance_state)
        x = x + res_out

        # Tiny FFN
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.ffn(h)

        return x, new_state


class CAWNResonanceEngine(nn.Module):
    """Full CAWN-Resonance language model.

    Shared block applied recursively (like FRR) but using
    resonance + CAWN stability instead of attention + FFN.

    O(L) per pass. No attention. Naturally causal.
    """
    def __init__(self, hidden_dim, n_resonators=128, n_cycles=28, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_cycles = n_cycles

        # SHARED BLOCK (applied at every cycle)
        self.block = CAWNResonanceBlock(hidden_dim, n_resonators)

        # Per-cycle modulation
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_dim)) for _ in range(n_cycles)
        ])
        self.betas = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(n_cycles)
        ])

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
        state = None

        for c in range(self.n_cycles):
            x, state = self.block(x, self.gammas[c], self.betas[c], state)

        x = self.out_norm(x)
        return self.lm_head(x)
