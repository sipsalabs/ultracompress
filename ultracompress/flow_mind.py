"""
FLOW MIND — Continuous computation. Everything emerges.

NOT discrete layers. NOT sequential passes. A CONTINUOUS FLOW.

Like a river: water flows, eddies form, patterns emerge.
The river doesn't have "layers" — it has dynamics.
The dynamics ARE the computation.

This unifies EVERY invention:
- Weight sharing: ONE flow equation, applied continuously (like FRR)
- Resonance: certain patterns amplify naturally in the flow
- Waves: information propagates through the flow as disturbances
- Crystallization: the flow converges to stable attractors (equilibria)
- Compression: the flow naturally discards noise, keeping signal
- Growth: the flow complexity adapts to what's needed
- Living: the flow never stops — inference IS continued computation

Implemented as a Neural ODE with a shared dynamics function.
Instead of x_{n+1} = f(x_n) (discrete, 28 steps),
we have dx/dt = f(x, t) (continuous, infinite resolution).

The "model" is f — the dynamics function. Tiny. Shared.
The "computation" is solving the ODE — the flow.
Intelligence emerges from the dynamics, not from layers.

Euler method for simplicity. Adaptive step size for efficiency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlowDynamics(nn.Module):
    """The dynamics function: dx/dt = f(x, t).

    This defines HOW the hidden state flows through time.
    The same function at every time point (like FRR's shared block).
    But continuous — not discrete layers.

    t is continuous time: t=0 is input, t=1 is output.
    The function can behave differently at different times
    because t is an input (like per-layer modulation, but smooth).
    """
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Time-conditioned transformation
        # Instead of per-layer gamma/beta, we have time-dependent modulation
        self.time_proj = nn.Linear(1, hidden_dim, bias=True)

        # Cross-position interaction (like attention but lighter)
        self.qk = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.out.weight)

        # Per-position transformation (like FFN but smaller)
        self.local = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        nn.init.zeros_(self.local[-1].weight)

        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, x, t):
        """x: (B, S, D), t: scalar in [0, 1] → dx/dt: (B, S, D)"""
        B, S, D = x.shape

        # Time-dependent modulation (smooth version of per-layer gamma/beta)
        t_emb = self.time_proj(torch.tensor([[t]], device=x.device, dtype=x.dtype))
        gamma = 1.0 + t_emb.unsqueeze(0)  # (1, 1, D)

        h = self.norm(x) * gamma

        # Cross-position flow (information transport)
        qk = self.qk(h).reshape(B, S, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(h).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if S > 1:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        cross_flow = (attn @ v).transpose(1, 2).reshape(B, S, D)
        cross_flow = self.out(cross_flow)

        # Local flow (per-position dynamics)
        local_flow = self.local(h)

        # Combined: the rate of change of x
        # Scale down to prevent explosion (the ODE solver accumulates these)
        dx_dt = (cross_flow + local_flow) * 0.3

        return dx_dt


class FlowMind(nn.Module):
    """Language model as continuous flow.

    Instead of 28 discrete passes, we solve an ODE from t=0 to t=1.
    The dynamics function is shared (like FRR).
    The number of "steps" is adaptive (like the growing rotation).

    More steps = higher precision = better output (but slower).
    Fewer steps = faster but rougher approximation.
    The model naturally trades compute for quality.
    """
    def __init__(self, hidden_dim, n_heads=8, n_steps=28, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_steps = n_steps

        # THE DYNAMICS (shared across all time)
        self.dynamics = FlowDynamics(hidden_dim, n_heads)

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

        # Solve ODE: dx/dt = f(x, t) from t=0 to t=1
        # Using Euler method with fixed step size
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            t = step * dt
            dx = self.dynamics(x, t)
            x = x + dx * dt

        x = self.out_norm(x)
        return self.lm_head(x)
