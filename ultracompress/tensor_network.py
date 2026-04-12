"""
TENSOR NETWORK COMPRESSION — From quantum physics to neural networks.

Matrix Product States (MPS) decompose a matrix into a chain of small 3-index tensors.
Originally invented to compress quantum states with 10^80 parameters.
Neural network weights have similar structure — most information lives on a
low-dimensional manifold.

For a (rows, cols) matrix:
  Standard: rows * cols parameters
  MPS: chain of tensors with bond dimension D
  Total params: ~(rows + cols) * D^2 << rows * cols for small D

Bond dimension D controls accuracy/compression tradeoff.
D=1: maximum compression, minimum quality
D=full_rank: exact, no compression
D=16-64: sweet spot for neural networks

This is PROVEN in physics. Just never applied to LLM weight compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MPSLayer(nn.Module):
    """Matrix Product State decomposition of a weight matrix.

    Decomposes W (out, in) into a chain of small tensors:
    W[i,j] = sum over bonds: A1[i1,b1] * A2[b1,i2,b2] * ... * An[bn-1,in]

    Where i = (i1, i2, ..., in) and j = (j1, j2, ..., jm) are index factorizations.
    """
    def __init__(self, in_features, out_features, bond_dim=16, n_sites=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bond_dim = bond_dim

        # Factorize dimensions into small indices
        # For 1024: [4, 4, 4, 4, 4] = 1024 (5 sites, each physical dim 4)
        if n_sites is None:
            n_sites = max(4, int(math.log2(max(in_features, out_features)) / math.log2(4)))

        self.n_sites = n_sites
        self.phys_dim_in = int(round(in_features ** (1.0 / n_sites)))
        self.phys_dim_out = int(round(out_features ** (1.0 / n_sites)))

        # Pad to exact factorization
        self.padded_in = self.phys_dim_in ** n_sites
        self.padded_out = self.phys_dim_out ** n_sites

        # MPS cores: each is (bond_left, phys_in, phys_out, bond_right)
        self.cores = nn.ParameterList()
        for site in range(n_sites):
            bl = 1 if site == 0 else bond_dim
            br = 1 if site == n_sites - 1 else bond_dim
            core = nn.Parameter(
                torch.randn(bl, self.phys_dim_in, self.phys_dim_out, br) * 0.01
            )
            self.cores.append(core)

    def reconstruct_weight(self):
        """Contract MPS to get full weight matrix."""
        # Start from left
        result = self.cores[0]  # (1, p_in, p_out, D)

        for site in range(1, self.n_sites):
            core = self.cores[site]  # (D, p_in, p_out, D')
            # Contract bond dimension
            # result: (..., D), core: (D, p_in, p_out, D')
            result = torch.einsum('...d,dpqe->...pqe', result, core)

        # result shape: (1, p_in, p_out, p_in, p_out, ..., 1)
        # Reshape to (padded_out, padded_in)
        result = result.squeeze(0).squeeze(-1)  # Remove boundary bonds

        # Separate in and out indices
        shape = result.shape
        n_idx = len(shape)
        # Interleaved: (p_in1, p_out1, p_in2, p_out2, ...)
        # Need to permute to (p_out1, p_out2, ..., p_in1, p_in2, ...)
        in_idx = list(range(0, n_idx, 2))  # even positions
        out_idx = list(range(1, n_idx, 2))  # odd positions
        perm = out_idx + in_idx
        result = result.permute(*perm)

        result = result.reshape(self.padded_out, self.padded_in)
        return result[:self.out_features, :self.in_features]

    def forward(self, x):
        """Apply MPS-compressed linear transformation."""
        W = self.reconstruct_weight()
        return F.linear(x, W)

    def param_count(self):
        return sum(c.numel() for c in self.cores)

    def compression_ratio(self):
        standard = self.out_features * self.in_features
        mps = self.param_count()
        return standard / mps


class MPSFractalBlock(nn.Module):
    """FRR block with MPS-compressed linear layers.

    Each linear layer is decomposed into a tensor network,
    giving massive compression per layer on top of FRR weight sharing.
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2, bond_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # MPS attention
        self.qkv = MPSLayer(hidden_dim, 3 * hidden_dim, bond_dim)
        self.o_proj = MPSLayer(hidden_dim, hidden_dim, bond_dim)

        # MPS FFN
        ff_dim = hidden_dim * ff_mult
        self.gate = MPSLayer(hidden_dim, ff_dim, bond_dim)
        self.up = MPSLayer(hidden_dim, ff_dim, bond_dim)
        self.down = MPSLayer(ff_dim, hidden_dim, bond_dim)

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
