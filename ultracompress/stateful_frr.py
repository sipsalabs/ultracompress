"""
STATEFUL FRR — The shared block has MEMORY across applications.

The barrier: one block applied 28 times with gamma/beta modulation
caps at ~67% T10. The modulation space is too small (2048 params/layer).

The fix: give the block a STATE that carries across applications.
Each application reads the state, computes, then WRITES back.
The state accumulates context about what the block has already done.

This means application #14 KNOWS what applications #1-13 did.
Standard FRR: each application is independent (only gamma/beta differs).
Stateful FRR: each application builds on all previous ones through state.

The state is tiny — just a vector of size hidden_dim.
But it gives the block exponentially more expressivity because
the same weights + different state = fundamentally different behavior.

Like how the same CPU instructions produce different output
depending on what's in the registers. The registers are the state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StatefulBlock(nn.Module):
    """Shared transformer block WITH persistent state.

    Standard block: f(x, gamma, beta) → x'
    Stateful block: f(x, gamma, beta, state) → x', new_state

    The state is read/written at each application:
    - Read: state modulates how the block processes (beyond gamma/beta)
    - Write: block's output updates the state for the next application
    """
    def __init__(self, hidden_dim, n_heads, state_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.state_dim = state_dim

        # Standard transformer components
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn_gate = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.ffn_down = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

        # STATE: read and write mechanisms
        # Read: state → extra modulation (on top of gamma/beta)
        self.state_to_mod = nn.Linear(state_dim, hidden_dim * 2, bias=False)
        nn.init.zeros_(self.state_to_mod.weight)  # start as no-op

        # Write: hidden state → state update
        self.hidden_to_state = nn.Linear(hidden_dim, state_dim, bias=False)
        # Gate: how much to update state (prevents catastrophic overwrite)
        self.state_gate = nn.Linear(state_dim + state_dim, state_dim, bias=False)
        nn.init.constant_(self.state_gate.bias if hasattr(self.state_gate, 'bias') else
                          torch.zeros(1), -2.0)  # default: mostly retain old state

    def forward(self, x, gamma, beta, state):
        """
        x: (B, S, D), gamma/beta: (1, 1, D), state: (B, state_dim)
        Returns: x_out (B, S, D), new_state (B, state_dim)
        """
        B, S, D = x.shape

        # READ state: generate extra modulation
        state_mod = self.state_to_mod(state)  # (B, D*2)
        state_gamma = 1.0 + state_mod[:, :D].unsqueeze(1)  # (B, 1, D)
        state_beta = state_mod[:, D:].unsqueeze(1)  # (B, 1, D)

        # Combined modulation: gamma/beta + state-dependent
        full_gamma = gamma * state_gamma
        full_beta = beta + state_beta

        # Attention with combined modulation
        h = self.norm1(x) * full_gamma + full_beta
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if S > 1:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.o_proj(out)

        # FFN with combined modulation
        h = self.norm2(x) * full_gamma + full_beta
        gate = F.silu(self.ffn_gate(h))
        x = x + self.ffn_down(gate * self.ffn_up(h))

        # WRITE state: update based on what this application produced
        new_info = self.hidden_to_state(x.mean(dim=1))  # (B, state_dim)
        gate_input = torch.cat([state, new_info], dim=-1)
        update_gate = torch.sigmoid(self.state_gate(gate_input))
        new_state = state * (1 - update_gate) + new_info * update_gate

        return x, new_state


class StatefulFRR(nn.Module):
    """FRR with stateful shared block.

    The block accumulates state across all 28 applications.
    Each application has access to what ALL previous applications did.
    This should break the expressivity barrier without adding many params.

    Extra params vs standard FRR: ~state_dim * hidden_dim * 4 ≈ 260K
    That's <4% overhead for potentially much more expressivity.
    """
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale, vocab_size,
                 state_dim=64,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.state_dim = state_dim

        # THE STATEFUL BLOCK
        self.block = StatefulBlock(hidden_dim, n_heads, state_dim)

        # Per-layer modulation (same as standard FRR)
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_dim)) for _ in range(self.total_layers)
        ])
        self.betas = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(self.total_layers)
        ])
        self.iter_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.total_layers)
        ])

        # Initial state (learned)
        self.init_state = nn.Parameter(torch.zeros(1, state_dim))

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
        B = x.shape[0]

        # Initialize state
        state = self.init_state.expand(B, -1)

        layer_count = 0
        for scale in range(self.n_scales):
            for it in range(self.iters_per_scale):
                residual = x
                x, state = self.block(x, self.gammas[layer_count], self.betas[layer_count], state)
                x = residual + (x - residual) * torch.sigmoid(self.iter_scale[layer_count])
                layer_count += 1

        x = self.out_norm(x)
        return self.lm_head(x)
