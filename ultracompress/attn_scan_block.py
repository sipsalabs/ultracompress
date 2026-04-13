"""
ATTENTION + SELECTIVE SCAN HYBRID BLOCK

Third option for FFN replacement (after full FFN and rotation):
- Attention: cross-position reasoning (KEEP)
- Selective scan: position-aware sequential transform (REPLACES FFN)

Why scan might beat rotation as FFN replacement:
- Rotation is stateless: each position transforms independently
- Scan has STATE: carries info across positions within each pass
- FFN is also stateless, so rotation matches FFN's interface
- BUT: scan adds cheap sequential context that neither FFN nor rotation has

Mamba showed: selective scan + gating = strong language model
We're combining: attention (cross-position) + scan (sequential context)
in a SHARED RECURSIVE block.

Params:
- Full FFN: 9.4M
- Scan replacement: ~200K (47x smaller than FFN)
- Deep rotation: ~10K (940x smaller than FFN)
- Scan is between rotation and FFN in both size and expressivity

Nobody has: attention + scan in a shared recursive (FRR) architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveScanFFN(nn.Module):
    """Mamba-style selective scan replacing FFN.

    Instead of expand→gate→contract (FFN), does:
    input_proj → selective_scan → output_proj

    The scan carries a hidden state across positions,
    giving each position context from previous positions.
    Combined with attention (which sees all positions),
    this covers both local sequential and global reasoning.
    """
    def __init__(self, hidden_dim, scan_dim=64, state_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scan_dim = scan_dim
        self.state_dim = state_dim

        # Project to scan space
        self.in_proj = nn.Linear(hidden_dim, scan_dim, bias=False)
        self.out_proj = nn.Linear(scan_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # start as no-op

        # Scan parameters (learned dynamics)
        # A: state transition (how much to remember)
        self.A_log = nn.Parameter(torch.randn(scan_dim, state_dim) * 0.1)
        # B, C: input-dependent (selective — this is what makes Mamba work)
        self.B_proj = nn.Linear(scan_dim, state_dim, bias=False)
        self.C_proj = nn.Linear(scan_dim, state_dim, bias=False)
        # dt: step size (input-dependent)
        self.dt_proj = nn.Linear(scan_dim, scan_dim, bias=True)
        nn.init.constant_(self.dt_proj.bias, -3.0)  # small initial dt

        # Gate (like Mamba's z branch)
        self.gate_proj = nn.Linear(hidden_dim, scan_dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape

        # Project input
        z = self.in_proj(x)  # (B, S, scan_dim)
        gate = F.silu(self.gate_proj(x))  # (B, S, scan_dim)

        # Selective scan parameters
        A = -torch.exp(self.A_log)  # (scan_dim, state_dim) — negative for stability
        B_input = self.B_proj(z)  # (B, S, state_dim)
        C_input = self.C_proj(z)  # (B, S, state_dim)
        dt = F.softplus(self.dt_proj(z))  # (B, S, scan_dim) — positive step size

        # Discretize A: A_bar = exp(dt * A)
        # For efficiency, use first-order approximation: A_bar ≈ 1 + dt * A
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, S, scan_dim, state_dim)
        A_bar = 1 + dt_A  # first-order approx, stable

        # Sequential scan (the core computation)
        # state: (B, scan_dim, state_dim)
        state = torch.zeros(B, self.scan_dim, self.state_dim, device=x.device)
        outputs = []

        for t in range(S):
            # state = A_bar * state + B_bar * input
            state = A_bar[:, t] * state + dt[:, t].unsqueeze(-1) * (z[:, t].unsqueeze(-1) * B_input[:, t].unsqueeze(1))
            # output = C * state
            y = (state * C_input[:, t].unsqueeze(1)).sum(dim=-1)  # (B, scan_dim)
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # (B, S, scan_dim)

        # Gate and project back
        y = y * gate
        return self.out_proj(y)


class AttentionScanBlock(nn.Module):
    """Shared block: full attention + selective scan (no FFN).

    Attention: cross-position reasoning (global, bidirectional via causal mask)
    Scan: sequential pattern capture (local, carries state forward)

    Together they cover what FFN does (per-position transform)
    PLUS sequential context that FFN can't do.
    """
    def __init__(self, hidden_dim, n_heads, scan_dim=64, state_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # ATTENTION (unchanged)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # SELECTIVE SCAN (replaces FFN)
        self.scan_ffn = SelectiveScanFFN(hidden_dim, scan_dim, state_dim)

        # Norms
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None):
        B, T, D = x.shape

        # Pre-norm + modulation
        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # ATTENTION
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

        # SELECTIVE SCAN (replaces FFN)
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.scan_ffn(h)

        return x
