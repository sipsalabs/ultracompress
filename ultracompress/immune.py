"""
IMMUNE REPERTOIRE COMPRESSION — Biology's own compression algorithm.

The adaptive immune system encodes responses to BILLIONS of antigens using
only a few hundred gene segments. How? Combinatorial assembly:
- V segments (variable): ~50 variants
- D segments (diversity): ~25 variants
- J segments (joining): ~6 variants
- Total combinations: 50 * 25 * 6 = 7,500 unique antibodies
- With somatic hypermutation: effectively infinite diversity

Applied to weights:
- Store a small set of "gene segments" (learned basis vectors)
- Each weight vector = combination of V + D + J segments
- A tiny index (which V, which D, which J) replaces the full weight vector
- 200 total segments produce 8M+ unique weight vectors

Storage: segment_bank (small) + indices (tiny) vs full weight matrix (huge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImmuneRepertoire(nn.Module):
    """Generate weight vectors via V-D-J recombination.

    Instead of storing weight matrices, store gene segments and
    recombination indices. Each "weight row" is assembled from
    selected V + D + J segments.
    """
    def __init__(self, output_dim, n_v=64, n_d=32, n_j=16, segment_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.n_v = n_v
        self.n_d = n_d
        self.n_j = n_j

        # Segment dimensions: V covers first third, D middle, J last
        if segment_dim is None:
            segment_dim = output_dim // 3

        self.v_dim = segment_dim
        self.d_dim = segment_dim
        self.j_dim = output_dim - 2 * segment_dim

        # Gene segment banks (THE compressed representation)
        self.v_bank = nn.Parameter(torch.randn(n_v, self.v_dim) * 0.02)
        self.d_bank = nn.Parameter(torch.randn(n_d, self.d_dim) * 0.02)
        self.j_bank = nn.Parameter(torch.randn(n_j, self.j_dim) * 0.02)

        # Somatic hypermutation: small per-row perturbation
        self.mutation_scale = nn.Parameter(torch.tensor(0.01))

    def recombine(self, v_idx, d_idx, j_idx):
        """Assemble a weight vector from V+D+J segments."""
        v = self.v_bank[v_idx]
        d = self.d_bank[d_idx]
        j = self.j_bank[j_idx]
        return torch.cat([v, d, j], dim=-1)

    def param_count(self):
        """How many params does the repertoire use?"""
        return self.v_bank.numel() + self.d_bank.numel() + self.j_bank.numel()


class ImmuneLinear(nn.Module):
    """Linear layer where each output neuron's weights come from V-D-J recombination.

    Instead of storing W (out_features x in_features),
    stores: gene banks + per-row (V,D,J) selection indices.
    """
    def __init__(self, in_features, out_features, n_v=64, n_d=32, n_j=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.repertoire = ImmuneRepertoire(in_features, n_v, n_d, n_j)

        # Soft selection: each output row selects V, D, J via learned logits
        self.v_logits = nn.Parameter(torch.randn(out_features, n_v) * 0.1)
        self.d_logits = nn.Parameter(torch.randn(out_features, n_d) * 0.1)
        self.j_logits = nn.Parameter(torch.randn(out_features, n_j) * 0.1)

        # Per-row scale (like BitNet's per-channel scale)
        self.row_scale = nn.Parameter(torch.ones(out_features) * 0.1)

    def forward(self, x):
        """Assemble weight matrix from gene segments and apply."""
        # Soft V-D-J selection (differentiable)
        v_weights = F.softmax(self.v_logits, dim=-1)  # (out, n_v)
        d_weights = F.softmax(self.d_logits, dim=-1)  # (out, n_d)
        j_weights = F.softmax(self.j_logits, dim=-1)  # (out, n_j)

        # Assemble weight matrix
        v_part = v_weights @ self.repertoire.v_bank  # (out, v_dim)
        d_part = d_weights @ self.repertoire.d_bank  # (out, d_dim)
        j_part = j_weights @ self.repertoire.j_bank  # (out, j_dim)

        W = torch.cat([v_part, d_part, j_part], dim=-1)  # (out, in)
        W = W * self.row_scale.unsqueeze(-1)

        return F.linear(x, W)

    def param_count(self):
        return (self.repertoire.param_count() +
                self.v_logits.numel() + self.d_logits.numel() +
                self.j_logits.numel() + self.row_scale.numel())

    def compression_vs_standard(self):
        standard = self.out_features * self.in_features
        immune = self.param_count()
        return standard / immune


class ImmuneFractalBlock(nn.Module):
    """FRR block where linear layers use immune repertoire compression."""
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2,
                 n_v=64, n_d=32, n_j=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Immune attention
        self.qkv = ImmuneLinear(hidden_dim, 3 * hidden_dim, n_v, n_d, n_j)
        self.o_proj = ImmuneLinear(hidden_dim, hidden_dim, n_v, n_d, n_j)

        # Immune FFN
        ff_dim = hidden_dim * ff_mult
        self.gate = ImmuneLinear(hidden_dim, ff_dim, n_v, n_d, n_j)
        self.up = ImmuneLinear(hidden_dim, ff_dim, n_v, n_d, n_j)
        self.down = ImmuneLinear(ff_dim, hidden_dim, n_v, n_d, n_j)

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None):
        B, T, D = x.shape
        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

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

        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x
