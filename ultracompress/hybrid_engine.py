"""
HYBRID ENGINE — Stack ALL our best inventions into one system.

Not one module. ALL of them working together:
1. Weight Program generates weights on the fly (8.6K core)
2. Rotation handles geometric transformation
3. Mode routing selects behavior per cycle
4. Dual-objective loss (T10 broad + T1 sharp)
5. Causal scan for sequential structure

The idea: each invention handles what it's best at.
Weight program = WHAT to compute
Rotation = HOW to transform
Mode routing = WHEN to switch behavior
Causal scan = WHERE in the sequence
Dual loss = WHY (optimize for both T10 and T1)

Together they might break ceilings that each hits alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridCycle(nn.Module):
    """One cycle of the hybrid engine.
    Combines rotation + generated weights + mode selection."""

    def __init__(self, hidden_dim, n_planes=32, n_modes=4, program_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Rotation (geometric transformation)
        self.angles = nn.Parameter(torch.zeros(n_planes))
        torch.manual_seed(42)
        indices = torch.randperm(hidden_dim)
        self.register_buffer('plane_i', indices[:n_planes])
        self.register_buffer('plane_j', indices[n_planes:2*n_planes])

        # Mode routing (select behavior)
        self.n_modes = n_modes
        self.mode_scales = nn.Parameter(torch.ones(n_modes, hidden_dim))
        self.mode_shifts = nn.Parameter(torch.zeros(n_modes, hidden_dim))
        self.router = nn.Linear(hidden_dim, n_modes, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        # Mini weight generator (produces per-cycle transform)
        self.weight_gen = nn.Sequential(
            nn.Linear(hidden_dim, program_dim),
            nn.GELU(),
            nn.Linear(program_dim, hidden_dim),
        )
        nn.init.zeros_(self.weight_gen[-1].weight)

        # Norm
        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, x):
        B, S, D = x.shape

        # 1. Rotation (geometric)
        cos_a = torch.cos(self.angles)
        sin_a = torch.sin(self.angles)
        x_new = x.clone()
        xi = x[..., self.plane_i]
        xj = x[..., self.plane_j]
        x_new[..., self.plane_i] = xi * cos_a - xj * sin_a
        x_new[..., self.plane_j] = xi * sin_a + xj * cos_a

        # 2. Mode routing (select scale/shift)
        weights = F.softmax(self.router(x_new.mean(dim=1).detach()), dim=-1)
        gamma = (weights @ self.mode_scales).unsqueeze(1)
        beta = (weights @ self.mode_shifts).unsqueeze(1)
        x_mod = x_new * gamma + beta

        # 3. Weight-generated transform (dynamic)
        x_gen = self.weight_gen(x_mod)

        # 4. Combine with activation
        x_out = F.silu(x_gen)

        return x + self.norm(x_out - x) * 0.5


class HybridEngine(nn.Module):
    """The hybrid engine: rotation + modes + weight gen + causal, stacked."""

    def __init__(self, hidden_dim, n_cycles=28, n_planes=32, n_modes=4,
                 program_dim=32, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles

        # Shared hybrid cycles (like FRR — same cycle repeated)
        # But with n_shared shared cycles, each used multiple times
        n_shared = min(4, n_cycles)
        self.shared_cycles = nn.ModuleList([
            HybridCycle(hidden_dim, n_planes, n_modes, program_dim)
            for _ in range(n_shared)
        ])
        self.n_shared = n_shared

        # Causal scan (sequential structure)
        self.scan_decay = nn.Parameter(torch.tensor(0.9))
        self.scan_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.scan_gate.weight)

        # Per-cycle iteration scale
        self.cycle_scale = nn.Parameter(torch.ones(n_cycles))

        # Embedding and head
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.out_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.out_norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        B, S, D = x.shape

        for i in range(self.n_cycles):
            # Select shared cycle (round-robin)
            cycle = self.shared_cycles[i % self.n_shared]

            # Apply hybrid cycle
            x = x + (cycle(x) - x) * self.cycle_scale[i]

            # Causal scan every 7th cycle
            if i % 7 == 6:
                decay = torch.sigmoid(self.scan_decay)
                state = torch.zeros(B, 1, D, device=x.device)
                outputs = []
                for t in range(S):
                    state = decay * state + (1 - decay) * x[:, t:t+1, :]
                    outputs.append(state)
                causal = torch.cat(outputs, dim=1)
                x = x + self.scan_gate(causal)

        x = self.out_norm(x)
        return self.lm_head(x)
