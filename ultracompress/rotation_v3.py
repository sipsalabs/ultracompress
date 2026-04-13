"""
ROTATION V3 — Multiple rotation inventions to break the 46% ceiling.

V1: Fixed random planes + local mixing = 44-46%
V2: + global hub = 43% (didn't help)

V3 tries fundamentally different rotation approaches:

A) LEARNED PLANES: let the model learn WHICH dimensions to rotate
B) CAUSAL SCAN: cumulative state along sequence (like SSM)
C) DEEP ROTATION: multiple rotations per cycle (stacked, not parallel)
D) INPUT-DEPENDENT ANGLES: data steers geometry (stable because periodic)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnedPlaneRotation(nn.Module):
    """Rotation in LEARNED planes instead of random planes.
    The model discovers which dimension pairs matter most."""
    def __init__(self, dim, n_planes=64):
        super().__init__()
        # Learned mixing matrices that define the rotation subspace
        self.plane_proj = nn.Linear(dim, n_planes * 2, bias=False)
        self.angles = nn.Parameter(torch.zeros(n_planes))
        self.back_proj = nn.Linear(n_planes * 2, dim, bias=False)
        nn.init.zeros_(self.back_proj.weight)

    def forward(self, x):
        # Project to rotation subspace
        projected = self.plane_proj(x)  # (B, S, n_planes*2)
        n = projected.shape[-1] // 2
        a, b = projected[..., :n], projected[..., n:]

        # Rotate
        cos_a = torch.cos(self.angles)
        sin_a = torch.sin(self.angles)
        a_rot = a * cos_a - b * sin_a
        b_rot = a * sin_a + b * cos_a

        # Project back
        rotated = torch.cat([a_rot, b_rot], dim=-1)
        return x + self.back_proj(rotated)


class CausalScanMixer(nn.Module):
    """Causal cumulative mixing — each position sees all previous positions.
    Like a state space model's scan but simpler."""
    def __init__(self, dim, decay=0.9):
        super().__init__()
        self.decay = nn.Parameter(torch.tensor(decay))
        self.gate = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x):
        B, S, D = x.shape
        # Cumulative weighted average (causal: only sees past)
        state = torch.zeros(B, 1, D, device=x.device)
        outputs = []
        decay = torch.sigmoid(self.decay)  # bound 0-1
        for t in range(S):
            state = decay * state + (1 - decay) * x[:, t:t+1, :]
            outputs.append(state)
        causal = torch.cat(outputs, dim=1)  # (B, S, D)
        return x + self.gate(causal)


class RotationV3(nn.Module):
    """V3: Learned planes + causal scan + input-dependent angles."""
    def __init__(self, hidden_dim, n_planes=64, n_cycles=28,
                 vocab_size=151936, use_causal=True, use_input_angles=True,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles

        # Learned plane rotations (not random planes)
        self.rotations = nn.ModuleList([
            LearnedPlaneRotation(hidden_dim, n_planes) for _ in range(n_cycles)
        ])

        # Causal scan for sequential structure
        self.use_causal = use_causal
        if use_causal:
            self.causal = CausalScanMixer(hidden_dim)

        # Input-dependent angle modulation (stable because periodic!)
        self.use_input_angles = use_input_angles
        if use_input_angles:
            self.angle_gen = nn.Linear(hidden_dim, n_planes, bias=False)
            nn.init.normal_(self.angle_gen.weight, std=0.01)

        # Per-cycle scale/shift
        self.scales = nn.Parameter(torch.ones(n_cycles, hidden_dim))
        self.shifts = nn.Parameter(torch.zeros(n_cycles, hidden_dim))
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_cycles)])

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

        for i in range(self.n_cycles):
            # Input-dependent angle adjustment (stable: wraps around 2π)
            if self.use_input_angles:
                extra_angles = self.angle_gen(x.mean(dim=1))  # (B, n_planes)
                # Add to rotation's static angles (periodic, can't explode)
                orig_angles = self.rotations[i].angles
                self.rotations[i].angles = nn.Parameter(
                    orig_angles + extra_angles.mean(dim=0).detach() * 0.1)

            # Causal mixing (sequential structure)
            if self.use_causal and i % 4 == 0:  # Every 4th cycle
                x = self.causal(x)

            # Learned-plane rotation
            x_rot = self.rotations[i](x)

            # Nonlinear activation
            x_act = F.silu(x_rot * self.scales[i] + self.shifts[i])

            # Normalize + residual
            x = x + self.norms[i](x_act - x)

        x = self.out_norm(x)
        return self.lm_head(x)
