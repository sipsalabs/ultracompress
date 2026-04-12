"""
DENDRITIC COMPUTATION — Neurons that compute more per parameter.

Standard neuron: output = activation(sum(weight * input))
Dendritic neuron: output = activation(combine(dendrite1(input), dendrite2(input), ...))

Each dendrite does its own nonlinear computation on a SUBSET of inputs,
then the soma (cell body) combines the dendritic outputs. This means:
- One dendritic neuron ≈ multiple standard neurons
- The same parameter count does MORE computation
- Local feature detection happens in dendrites, global integration in soma

Applied to FRR: the shared block's neurons become dendritic, giving it
the computational power of a much larger block without more parameters.

Inspired by: Chavlis & Poirazi 2024, Numenta 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DendriticLinear(nn.Module):
    """A linear layer where each output neuron has multiple dendrites.

    Instead of: y = Wx + b
    Does: y = combine(d1(x_subset1), d2(x_subset2), ..., dk(x_subsetk))

    Each dendrite processes a DIFFERENT subset of inputs with its own
    nonlinear transformation. The soma combines via max or weighted sum.

    Parameter count: same as standard linear (or slightly more),
    but computational power: ~k times greater per neuron.
    """
    def __init__(self, in_features, out_features, n_dendrites=4, combine='max'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_dendrites = n_dendrites
        self.combine = combine

        # Each dendrite gets a portion of the input
        self.dendrite_dim = in_features // n_dendrites

        # Dendritic weights: each dendrite has its own projection
        # Total params: n_dendrites * dendrite_dim * out_features
        # ≈ in_features * out_features (same as standard linear!)
        self.dendrite_weights = nn.Parameter(
            torch.randn(n_dendrites, self.dendrite_dim, out_features) * (2.0 / in_features) ** 0.5
        )

        # Dendritic nonlinearity: per-dendrite threshold/gain
        self.dendrite_threshold = nn.Parameter(torch.zeros(n_dendrites, out_features))
        self.dendrite_gain = nn.Parameter(torch.ones(n_dendrites, out_features))

        # Soma combination weights (if using weighted sum)
        if combine == 'weighted':
            self.soma_weights = nn.Parameter(torch.ones(n_dendrites) / n_dendrites)

    def forward(self, x):
        """x: (..., in_features) -> (..., out_features)"""
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        B = x_flat.shape[0]

        # Split input across dendrites
        x_split = x_flat.reshape(B, self.n_dendrites, self.dendrite_dim)

        # Each dendrite computes its own transformation
        # x_split: (B, n_d, d_dim), weights: (n_d, d_dim, out)
        dendrite_outputs = torch.einsum('bnd,ndo->bno', x_split, self.dendrite_weights)

        # Dendritic nonlinearity (threshold + gain modulation)
        # Like biological dendrites: only fire if input exceeds threshold
        dendrite_outputs = self.dendrite_gain * F.silu(
            dendrite_outputs - self.dendrite_threshold
        )

        # Soma combines dendritic outputs
        if self.combine == 'max':
            output = dendrite_outputs.max(dim=1).values  # (B, out)
        elif self.combine == 'weighted':
            weights = F.softmax(self.soma_weights, dim=0)  # (n_d,)
            output = (dendrite_outputs * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        else:  # sum
            output = dendrite_outputs.sum(dim=1)

        return output.reshape(*batch_shape, self.out_features)


class DendriticFractalBlock(nn.Module):
    """FRR shared block with dendritic neurons.

    Same architecture as FractalBlock but linear layers are replaced
    with DendriticLinear. This gives the shared block more computational
    power per parameter — each application does more work.
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2, n_dendrites=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Dendritic attention projections
        self.qkv = DendriticLinear(hidden_dim, 3 * hidden_dim, n_dendrites)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Output stays standard

        # Dendritic FFN
        ff_dim = hidden_dim * ff_mult
        self.gate = DendriticLinear(hidden_dim, ff_dim, n_dendrites)
        self.up = DendriticLinear(hidden_dim, ff_dim, n_dendrites)
        self.down = nn.Linear(ff_dim, hidden_dim, bias=False)  # Down-projection stays standard

        # Norms
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None):
        B, T, D = x.shape

        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # Dendritic attention
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

        # Dendritic FFN
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))

        return x
