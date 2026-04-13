"""
ATTENTION + DEEP ROTATION HYBRID BLOCK

The invention: keep attention (cross-position reasoning) but replace FFN
with STACKED ROTATIONS (per-position transformation).

Why this might work:
- Rotation engine alone: 46% T10 at 5000x (no attention, just rotation)
- FRR attention carries most of the quality (CKA >0.9 across layers)
- FFN is 3.8M params doing per-position transform. Rotation does the same
  with ~10K params.
- A sequence of Givens rotations can approximate ANY orthogonal transform.
  Add scaling + nonlinearity between passes = universal function approximator.

Standard block:  attention (4.2M) + FFN (9.4M) = 7.3M shared + modulation
Hybrid block:    attention (4.2M) + deep_rotation (~10K) = 4.2M shared

Compression improvement: 7.3M -> 4.2M = 1.74x on top of FRR
With PHM on attention: 1.74 * 4 = 6.96x on top of FRR
FRR 48x * 6.96 = 334x base * Q2+entropy 48x = 16,000x

Nobody has tried this. Attention + rotation as a shared recursive block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeepRotationFFN(nn.Module):
    """Replaces FFN with stacked rotations + nonlinearities.

    Each pass rotates in different planes, covers different dims.
    3 passes of 256 planes = 768 dims rotated (75% coverage of 1024).
    Nonlinearity between passes makes this a universal approximator.

    Total params: ~10K (vs FFN's 9.4M = 940x smaller)
    """
    def __init__(self, hidden_dim, n_planes=256, n_passes=3):
        super().__init__()
        self.n_passes = n_passes

        # Each pass has its own rotation planes and angles
        self.angles = nn.ParameterList()
        self.plane_is = []
        self.plane_js = []

        for p in range(n_passes):
            self.angles.append(nn.Parameter(torch.zeros(n_planes)))
            # Different plane assignment per pass — maximum coverage
            torch.manual_seed(42 + p * 1000)
            indices = torch.randperm(hidden_dim)
            actual_planes = min(n_planes, hidden_dim // 2)
            self.register_buffer(f'plane_i_{p}', indices[:actual_planes])
            self.register_buffer(f'plane_j_{p}', indices[actual_planes:2*actual_planes])
            self.plane_is.append(f'plane_i_{p}')
            self.plane_js.append(f'plane_j_{p}')

        # Scaling between passes (like FFN's gating role)
        self.gate = nn.Parameter(torch.ones(n_passes, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(n_passes, hidden_dim))

    def forward(self, x):
        h = x
        for p in range(self.n_passes):
            angles = self.angles[p]
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            pi = getattr(self, self.plane_is[p])
            pj = getattr(self, self.plane_js[p])

            h_new = h.clone()
            hi = h[..., pi]
            hj = h[..., pj]
            h_new[..., pi] = hi * cos_a - hj * sin_a
            h_new[..., pj] = hi * sin_a + hj * cos_a

            # Gate + nonlinearity between passes
            h = F.silu(h_new * self.gate[p] + self.bias[p])

        return h


class AttentionRotationBlock(nn.Module):
    """Shared block: full attention + deep rotation (no FFN).

    Use like FRR's FractalModel block — apply 28 times with modulation.
    """

    def __init__(self, hidden_dim, n_heads, n_planes=256, n_rot_passes=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # ATTENTION (full, unchanged — the quality driver)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # DEEP ROTATION (replaces FFN — tiny, geometric, stacked)
        self.rotation_ffn = DeepRotationFFN(hidden_dim, n_planes, n_rot_passes)

        # Norms
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None):
        B, T, D = x.shape

        # Pre-norm + modulation
        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # ATTENTION (cross-position reasoning)
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

        # DEEP ROTATION (per-position transformation — replaces FFN)
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        x = x + self.rotation_ffn(h) - h  # residual

        return x
