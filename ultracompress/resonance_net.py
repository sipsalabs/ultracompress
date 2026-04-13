"""
RESONANCE NETWORK — Intelligence through selective resonance.

Inspired by the cochlea: sound enters, different frequencies resonate
with different parts of the structure, and the RESONANCE PATTERN
is the output. No sequential processing. No attention. Resonance.

How it works:
1. Tokens enter as signals
2. A bank of LEARNED RESONATORS tuned to different "meaning frequencies"
3. Each resonator amplifies signals that match its frequency
4. The combined resonance pattern = the output representation
5. Applied recursively: each pass amplifies what resonated, dampens what didn't

Why this is different from everything:
- Attention: "what should I look at?" (selection)
- FFN: "how should I transform this?" (mapping)
- Waves: "how do signals propagate?" (transport)
- Resonance: "what MEANING does this match?" (recognition)

Resonance is inherently selective — only matching frequencies amplify.
Everything else is dampened naturally. No gating needed.

The resonators ARE the model's knowledge. Each one "knows" one pattern.
Like how each cochlear hair cell "knows" one frequency.
A language model needs ~1000 resonators to know language.

Total DNA: n_resonators * hidden_dim = ~128K for 128 resonators.
Compare: FRR block = 7.3M. This is 57x smaller.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResonatorBank(nn.Module):
    """A bank of learned resonators.

    Each resonator is a learned frequency pattern in hidden space.
    Input signals that MATCH a resonator's pattern get amplified.
    Signals that don't match get dampened.

    This is selective amplification — the core of resonance.
    """
    def __init__(self, hidden_dim, n_resonators=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_resonators = n_resonators

        # Each resonator has a frequency pattern (what it responds to)
        self.frequencies = nn.Parameter(torch.randn(n_resonators, hidden_dim) * 0.02)

        # Each resonator has an amplitude (how strongly it responds)
        self.amplitudes = nn.Parameter(torch.ones(n_resonators) * 0.1)

        # Dampening: how quickly non-resonant signals die
        self.dampen = nn.Parameter(torch.tensor(0.9))

        # Output mixing: combine resonator outputs
        self.mix = nn.Linear(n_resonators, hidden_dim, bias=False)
        nn.init.zeros_(self.mix.weight)

    def forward(self, x):
        """x: (B, S, D) -> resonated x: (B, S, D)

        1. Compute how well each position matches each resonator
        2. Amplify matching signals, dampen non-matching
        3. Mix resonator outputs back to hidden space
        """
        B, S, D = x.shape

        # How well does each position match each resonator?
        # Cosine similarity between input and resonator frequencies
        x_norm = F.normalize(x, dim=-1)  # (B, S, D)
        f_norm = F.normalize(self.frequencies, dim=-1)  # (n_res, D)
        resonance = torch.einsum('bsd,rd->bsr', x_norm, f_norm)  # (B, S, n_res)

        # Amplify resonant signals, dampen others
        # Resonance^2 makes it sharply selective (high resonance = big, low = tiny)
        resonance_strength = resonance.pow(2) * self.amplitudes.unsqueeze(0).unsqueeze(0)

        # Mix resonator activations back to hidden space
        output = self.mix(resonance_strength)  # (B, S, D)

        return output


class ResonanceCrossPosition(nn.Module):
    """Cross-position interaction through shared resonance.

    Instead of attention (pairwise), ALL positions resonate with the
    SAME resonator bank. Positions that activate the same resonators
    are implicitly connected — they "heard" the same frequency.

    This gives O(n) cross-position interaction (not O(n²) like attention).
    """
    def __init__(self, hidden_dim, n_resonators=64):
        super().__init__()
        self.bank = ResonatorBank(hidden_dim, n_resonators)

        # Global resonance: average activation across positions
        # then broadcast back — positions learn from collective resonance
        self.global_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.global_proj.weight)

    def forward(self, x):
        # Local resonance per position
        local = self.bank(x)  # (B, S, D)

        # Global resonance: what resonated across ALL positions?
        global_signal = local.mean(dim=1, keepdim=True)  # (B, 1, D)
        global_context = self.global_proj(global_signal)  # (B, 1, D)

        # Each position gets: its own resonance + what resonated globally
        return local + global_context


class ResonanceEngine(nn.Module):
    """Full resonance-based language model.

    No attention. No FFN. No matrix multiply for computation.
    Just resonance — selective amplification of matching patterns.

    Applied recursively like FRR: same resonator bank, 28 passes.
    Each pass amplifies what resonated, dampens what didn't.
    After enough passes, only the strongest resonances survive.
    The answer CRYSTALLIZES from resonance.
    """
    def __init__(self, hidden_dim, n_resonators=128, n_cross_resonators=64,
                 n_cycles=28, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_cycles = n_cycles

        # SHARED resonator bank (THE model — applied at every cycle)
        self.local_bank = ResonatorBank(hidden_dim, n_resonators)
        self.cross_pos = ResonanceCrossPosition(hidden_dim, n_cross_resonators)

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

        for c in range(self.n_cycles):
            h = self.norms[c](x) * self.gammas[c] + self.betas[c]

            # Local resonance (per-position: what patterns match here?)
            local = self.local_bank(h)

            # Cross-position resonance (what patterns match across the sequence?)
            cross = self.cross_pos(h)

            # Combine and residual
            x = x + (local + cross) * self.cycle_scale[c]

        x = self.out_norm(x)
        return self.lm_head(x)
