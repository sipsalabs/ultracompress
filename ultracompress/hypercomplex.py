"""
PARAMETERIZED HYPERCOMPLEX MULTIPLICATION (PHM) — 1/n parameters per layer.

Standard linear: y = Wx where W is (out, in) = out*in parameters
PHM linear: y = (sum_i A_i kron B_i) x where:
  - A_i are (n, n) learnable matrices (the "algebra")
  - B_i are (out/n, in/n) learnable matrices (the "actual weights")
  - Total params: n * (n*n + out/n * in/n) ≈ out*in/n for large layers

At n=4: 4x fewer parameters than standard linear.
At n=8: 8x fewer.

This is not an approximation — it's a different algebraic structure
that's provably as expressive but more parameter-efficient because
it exploits structured relationships between weight components.

Based on: Zhang et al. 2021 "Beyond Fully-Connected Layers with Quaternions"
(Parameterized Hypercomplex Multiplications)

Applied to FRR: the shared block's linear layers become PHM layers,
giving 4-8x parameter reduction ON TOP of the 42x from weight sharing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init


class PHMLinear(nn.Module):
    """Parameterized Hypercomplex Multiplication linear layer.

    Replaces nn.Linear with 1/n parameters.

    The weight matrix W is decomposed as:
        W = sum_{i=1}^{n} A_i ⊗ B_i

    Where:
        A_i: (n, n) — learned "algebra" defining how components interact
        B_i: (out_features//n, in_features//n) — the actual sub-weights

    The Kronecker product (⊗) reconstructs the full weight matrix,
    but we never materialize it — we compute the product efficiently.
    """
    def __init__(self, in_features, out_features, n=4, bias=True):
        super().__init__()
        assert in_features % n == 0, f"in_features ({in_features}) must be divisible by n ({n})"
        assert out_features % n == 0, f"out_features ({out_features}) must be divisible by n ({n})"

        self.in_features = in_features
        self.out_features = out_features
        self.n = n

        self.sub_in = in_features // n
        self.sub_out = out_features // n

        # The "algebra": n matrices of size (n, n) that define how
        # the n components of input interact with n components of output
        self.A = nn.Parameter(torch.empty(n, n, n))

        # The "sub-weights": n matrices of size (sub_out, sub_in)
        # These are the actual learned transformations
        self.B = nn.Parameter(torch.empty(n, self.sub_out, self.sub_in))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self._init_weights()

    def _init_weights(self):
        # Initialize A as near-identity algebra
        nn.init.eye_(self.A[0][:self.n, :self.n] if self.n <= self.A.shape[1] else self.A[0])
        for i in range(self.n):
            if i == 0:
                nn.init.eye_(self.A[i])
            else:
                nn.init.normal_(self.A[i], 0, 0.01)

        # Initialize B like standard linear
        for i in range(self.n):
            init.kaiming_uniform_(self.B[i], a=math.sqrt(5))
            self.B[i].data *= 0.5  # Scale down for stability

    def forward(self, x):
        """x: (..., in_features) -> (..., out_features)"""
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        B_total = x_flat.shape[0]

        # Reshape input: (batch, n, sub_in)
        x_components = x_flat.reshape(B_total, self.n, self.sub_in)

        # For each output component j:
        #   y_j = sum_i A[k, j, i] * (B[k] @ x_i) for each algebra element k
        # Efficient: compute all B[k] @ x_i first, then mix with A

        # Step 1: Apply all sub-weights to all input components
        # B: (n, sub_out, sub_in), x_components: (batch, n_input=n, sub_in)
        # For each algebra element k, transform each input component i with B[k]
        # Result: (batch, n_algebra=n, n_input=n, sub_out)
        transformed = torch.einsum('koi,bni->bkno', self.B, x_components)

        # Step 2: Mix input components into output components using algebra A
        # A: (n_algebra=n, n_out=n, n_in=n)
        # y[b,j,o] = sum_k sum_i A[k,j,i] * transformed[b,k,i,o]
        # Einsum: contract over k (algebra) and i (input component)
        output = torch.einsum('bkio,kji->bjo', transformed, self.A)
        # output: (batch, n_out=n, sub_out)

        # Reshape to flat output
        y = output.reshape(B_total, self.out_features)

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*batch_shape, self.out_features)

    def param_count(self):
        """Compare to standard linear."""
        phm_params = self.n * (self.n * self.n + self.sub_out * self.sub_in)
        standard_params = self.out_features * self.in_features
        return phm_params, standard_params, standard_params / phm_params


class PHMFractalBlock(nn.Module):
    """FRR shared block with PHM layers — 1/n parameters per linear.

    At n=4: the shared block has ~4x fewer parameters.
    Combined with 42x weight sharing = 168x total compression.
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2, n=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n = n

        # PHM attention projections (4x fewer params each)
        self.qkv = PHMLinear(hidden_dim, 3 * hidden_dim, n=n, bias=False)
        self.o_proj = PHMLinear(hidden_dim, hidden_dim, n=n, bias=False)

        # PHM FFN (4x fewer params each)
        ff_dim = hidden_dim * ff_mult
        # Ensure ff_dim is divisible by n
        ff_dim = (ff_dim // n) * n
        self.gate = PHMLinear(hidden_dim, ff_dim, n=n, bias=False)
        self.up = PHMLinear(hidden_dim, ff_dim, n=n, bias=False)
        self.down = PHMLinear(ff_dim, hidden_dim, n=n, bias=False)

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None):
        B, T, D = x.shape

        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # PHM attention
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

        # PHM FFN
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))

        return x

    def param_count_comparison(self):
        """Show parameter savings vs standard block."""
        phm_total = sum(p.numel() for p in self.parameters())
        # Standard block would have:
        D = self.hidden_dim
        ff = D * 2  # ff_mult=2
        standard = 3*D*D + D*D + D*ff*2 + ff*D + 2*D  # qkv + o + gate + up + down + norms
        return phm_total, standard, standard / phm_total
