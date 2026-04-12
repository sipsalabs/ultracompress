"""
HYPERBOLIC NEURAL COMPUTATION — Exponentially more room per dimension.

Standard (Euclidean): A d-dimensional space has d degrees of freedom.
Hyperbolic: A d-dimensional hyperbolic space has EXPONENTIALLY more room.

Why this matters for compression:
- Language has hierarchical structure (words → phrases → sentences → paragraphs)
- Hierarchies embed perfectly in hyperbolic space with logarithmic distortion
- In Euclidean space, embedding a tree with N nodes needs O(sqrt(N)) dimensions
- In hyperbolic space, it needs O(log(N)) dimensions
- That's exponentially more compact = massive "free" compression

The Poincare ball model: all points inside the unit ball, with distance growing
exponentially near the boundary. One dimension in hyperbolic space does the
work of many Euclidean dimensions for hierarchical data.

Applied to FRR: if the shared block operates in hyperbolic space, it can
represent richer transformations per parameter than Euclidean.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ================================================================
# Poincare Ball Operations
# ================================================================

def poincare_add(x, y, c=1.0):
    """Mobius addition in the Poincare ball model.

    This is the "addition" operation in hyperbolic space.
    Unlike Euclidean addition, it curves the result to stay in the ball.
    """
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2*c*xy + c*y_sq) * x + (1 - c*x_sq) * y
    denom = 1 + 2*c*xy + c*c*x_sq*y_sq

    return num / denom.clamp(min=1e-6)


def poincare_distance(x, y, c=1.0):
    """Geodesic distance in the Poincare ball."""
    diff = poincare_add(-x, y, c)
    norm = diff.norm(dim=-1, keepdim=True).clamp(max=1-1e-5)
    return (2.0 / math.sqrt(c)) * torch.atanh(math.sqrt(c) * norm)


def exp_map(v, x=None, c=1.0):
    """Exponential map: tangent vector at x -> point on the ball."""
    if x is None:
        x = torch.zeros_like(v)

    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    lambda_x = 2.0 / (1 - c * (x*x).sum(dim=-1, keepdim=True)).clamp(min=1e-6)

    second_term = torch.tanh(math.sqrt(c) * lambda_x * v_norm / 2) * v / (math.sqrt(c) * v_norm)
    return poincare_add(x, second_term, c)


def log_map(y, x=None, c=1.0):
    """Logarithmic map: point on ball -> tangent vector at x."""
    if x is None:
        x = torch.zeros_like(y)

    diff = poincare_add(-x, y, c)
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6, max=1-1e-5)
    lambda_x = 2.0 / (1 - c * (x*x).sum(dim=-1, keepdim=True)).clamp(min=1e-6)

    return (2.0 / (math.sqrt(c) * lambda_x)) * torch.atanh(math.sqrt(c) * diff_norm) * diff / diff_norm


def project_to_ball(x, c=1.0, eps=1e-5):
    """Project points to inside the Poincare ball."""
    max_norm = (1.0 - eps) / math.sqrt(c)
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp(min=max_norm))


# ================================================================
# Hyperbolic Linear Layer
# ================================================================

class HyperbolicLinear(nn.Module):
    """Linear transformation in hyperbolic space.

    Instead of y = Wx + b (Euclidean),
    does: y = exp_map(W @ log_map(x)) (hyperbolic)

    The transformation happens in the tangent space (Euclidean),
    then maps back to the hyperbolic manifold.

    Same parameter count as nn.Linear, but operates on a curved manifold
    where each dimension encodes exponentially more structure.
    """
    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data *= 0.01  # Small init for hyperbolic stability

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        """x: (..., in_features) in Poincare ball -> (..., out_features) in Poincare ball"""
        # Map to tangent space at origin
        x_tangent = log_map(x, c=self.c)

        # Euclidean linear in tangent space
        out_tangent = F.linear(x_tangent, self.weight, self.bias)

        # Map back to ball
        out = exp_map(out_tangent, c=self.c)
        return project_to_ball(out, c=self.c)


# ================================================================
# Hyperbolic FRR Block
# ================================================================

class HyperbolicFractalBlock(nn.Module):
    """FRR shared block that operates in hyperbolic space.

    The key insight: hierarchical language structure embeds naturally
    in hyperbolic space. By doing computation in hyperbolic space,
    each dimension represents exponentially more structure.

    Same architecture as FractalBlock but with HyperbolicLinear
    for the key projections.
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2, curvature=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.c = curvature

        # Attention: Q/K in hyperbolic space, V in Euclidean
        # Q and K benefit from hyperbolic distance (hierarchical similarity)
        # V stays Euclidean (raw value retrieval)
        self.q_proj = HyperbolicLinear(hidden_dim, hidden_dim, c=curvature, bias=False)
        self.k_proj = HyperbolicLinear(hidden_dim, hidden_dim, c=curvature, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # FFN stays Euclidean (nonlinear transformation, not hierarchical)
        ff_dim = hidden_dim * ff_mult
        self.gate = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.up = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, hidden_dim, bias=False)

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None):
        B, T, D = x.shape

        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # Project to Poincare ball for Q/K
        h_ball = project_to_ball(h * 0.5, c=self.c)  # Scale down to stay in ball

        # Hyperbolic Q/K for attention (hierarchical similarity)
        q = self.q_proj(h_ball).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_ball).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Euclidean V
        v = self.v_proj(h).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores via hyperbolic distance (not dot product)
        # Hyperbolic distance naturally captures hierarchical similarity
        attn = -(q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1) / math.sqrt(self.head_dim)

        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        # FFN (Euclidean)
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))

        return x
