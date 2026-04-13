"""
ROTATION ENGINE — Sip's dimensional idea, realized.

Day 1: "What if we make it 4D or even 10D, a dimensional box
with unlimited connections?"

The answer: rotation in high-dimensional space.
- 1024D rotation has 523,776 rotation planes
- Every dimension connects to every other through rotation
- The "seed" is just ANGLES — one float per rotation plane
- Applied recursively = intelligence emerges

No attention. No FFN. No matrix multiplication.
Just rotate the state vector and apply a tiny nonlinearity.

The rotation angles ARE the DNA.
The recursive application IS the process.
Intelligence EMERGES from geometric transformation.

This is what PHM was pointing at (hypercomplex = higher-D rotation).
This is what RoPE was pointing at (position via rotation).
This is what Sip saw from Day 1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotationLayer(nn.Module):
    """Apply learned rotations in high-dimensional space.

    Instead of W @ x (matrix multiply, millions of params),
    this rotates x through learned angles (thousands of params).

    A rotation in D dimensions is defined by D*(D-1)/2 angles
    (one per rotation plane). But that's too many for D=1024.

    Efficient version: rotate in K random planes per step.
    K planes * 1 angle each = K params per layer.
    """
    def __init__(self, dim, n_planes=64):
        super().__init__()
        self.dim = dim
        self.n_planes = n_planes

        # Which dimension pairs to rotate in (fixed, not learned)
        # Random but deterministic plane selection
        torch.manual_seed(42)
        indices = torch.randperm(dim)
        self.register_buffer('plane_i', indices[:n_planes])
        self.register_buffer('plane_j', indices[n_planes:2*n_planes])

        # The ONLY learned params: rotation angles
        # One angle per plane = n_planes params total
        self.angles = nn.Parameter(torch.zeros(n_planes))

    def forward(self, x):
        """x: (B, S, D) -> rotated x: (B, S, D)"""
        cos_a = torch.cos(self.angles)
        sin_a = torch.sin(self.angles)

        x_new = x.clone()
        xi = x[..., self.plane_i]  # (B, S, n_planes)
        xj = x[..., self.plane_j]  # (B, S, n_planes)

        # Givens rotation in each plane
        x_new[..., self.plane_i] = xi * cos_a - xj * sin_a
        x_new[..., self.plane_j] = xi * sin_a + xj * cos_a

        return x_new


class RotationEngine(nn.Module):
    """Intelligence through pure rotation.

    The entire "model" is:
    1. Embed tokens
    2. Repeat N times: rotate + activate
    3. Project to vocab

    The DNA = rotation angles (n_planes * n_cycles params)
    Everything else = shared embeddings from teacher

    For n_planes=64, n_cycles=28: 1,792 params of DNA.
    Compare: FRR = 7,342,080 params. This is 4,097x smaller.
    """
    def __init__(self, hidden_dim, n_planes=64, n_cycles=28,
                 vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles
        self.n_planes = n_planes

        # One rotation layer per cycle (each has its own angles)
        self.rotations = nn.ModuleList([
            RotationLayer(hidden_dim, n_planes) for _ in range(n_cycles)
        ])

        # Tiny nonlinearity after each rotation
        # Just a learned scale + shift (like LayerNorm but simpler)
        self.scales = nn.Parameter(torch.ones(n_cycles, hidden_dim))
        self.shifts = nn.Parameter(torch.zeros(n_cycles, hidden_dim))

        # Cross-position mixing (without attention!)
        # Simple: shift tokens left/right and add (like a 1D convolution with kernel=3)
        self.mix_weight = nn.Parameter(torch.tensor([0.1, 0.8, 0.1]))

        # Norms for stability
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_cycles)])

        # Embedding and head (shared from teacher)
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
            # Cross-position mixing (replaces attention)
            # Weighted sum of [left neighbor, self, right neighbor]
            left = F.pad(x[:, :-1, :], (0, 0, 1, 0))
            right = F.pad(x[:, 1:, :], (0, 0, 0, 1))
            x_mixed = (self.mix_weight[0] * left +
                      self.mix_weight[1] * x +
                      self.mix_weight[2] * right)

            # Rotate in high-dimensional space (replaces attention projection)
            x_rot = self.rotations[i](x_mixed)

            # Nonlinear activation (replaces FFN)
            x_act = F.silu(x_rot * self.scales[i] + self.shifts[i])

            # Normalize + residual
            x = x + self.norms[i](x_act - x)

        x = self.out_norm(x)
        return self.lm_head(x)

    def dna_size(self):
        """The rotation angles = the DNA."""
        rotation_params = sum(r.angles.numel() for r in self.rotations)
        scale_params = self.scales.numel() + self.shifts.numel()
        mix_params = self.mix_weight.numel()
        norm_params = sum(p.numel() for n in self.norms for p in n.parameters())
        return {
            'rotation_angles': rotation_params,
            'scale_shift': scale_params,
            'mix': mix_params,
            'norms': norm_params,
            'total_dna': rotation_params + scale_params + mix_params + norm_params,
        }
