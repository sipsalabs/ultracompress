"""
THALAMIC ROUTING — Brain-inspired attention modulation from TRC2 paper.

Three mechanisms stolen from neuroscience:
1. Query-biasing: modulate Q stream to steer what attention looks at
2. Surprise-gated pathway: dynamic modulation based on input novelty
3. TRN divisive competition: sparse competitive feature selection

Each can be used independently or combined.
All designed to drop into FRR's shared block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ThalamicRouter(nn.Module):
    """Thalamic routing module — modulates attention via Q-stream biasing.

    Instead of gamma/beta on FFN outputs, this modulates the QUERY stream
    of attention. Steers what the model attends to without touching K/V.
    Much more targeted modulation.

    Has two pathways (like the thalamus):
    - Local: current token through SiLU (fast, reactive)
    - Diffuse: running mean of past tokens (slow, contextual)
    Plus TRN-like divisive competition for sparsity.
    """
    def __init__(self, hidden_dim, bottleneck_dim=64, n_groups=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_groups = n_groups
        self.group_size = hidden_dim // n_groups

        # Compress to bottleneck (like thalamic relay)
        self.compress = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.expand = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        # Local pathway (fast, reactive)
        self.local_gate = nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)

        # Diffuse pathway (slow, contextual — running mean)
        self.diffuse_gate = nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)

        # Surprise detector: deviation from running mean
        self.surprise_scale = nn.Parameter(torch.tensor(1.0))

        # Output scaling
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.zeros_(self.expand.weight)  # Start as no-op

    def forward(self, x, running_mean=None):
        """x: (B, T, D) -> modulation signal (B, T, D) + updated running mean"""
        B, T, D = x.shape

        # Compress to bottleneck
        z = self.compress(x)  # (B, T, bottleneck)

        # Local pathway: current token
        local = F.silu(self.local_gate(z))

        # Diffuse pathway: running mean of past tokens
        if running_mean is None:
            running_mean = z.mean(dim=1, keepdim=True)  # (B, 1, bottleneck)

        # Surprise: how different is current from running mean
        surprise = (z - running_mean).abs() * self.surprise_scale
        diffuse = F.silu(self.diffuse_gate(running_mean.expand_as(z))) * torch.sigmoid(surprise)

        # Combine pathways
        combined = local + diffuse

        # Expand back to hidden dim
        modulation = self.expand(combined)  # (B, T, D)

        # TRN-like divisive competition: group features, divide by group mean
        # Forces sparse, competitive activation
        mod_grouped = modulation.reshape(B, T, self.n_groups, self.group_size)
        group_mean = mod_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        mod_competitive = mod_grouped / group_mean
        modulation = mod_competitive.reshape(B, T, D)

        # Update running mean (exponential moving average)
        new_mean = 0.9 * running_mean + 0.1 * z.mean(dim=1, keepdim=True)

        return modulation * self.output_scale, new_mean


class ThalamicFractalBlock(nn.Module):
    """FRR block with thalamic Q-stream modulation instead of gamma/beta.

    The thalamic router modulates the query stream of attention,
    steering what the model attends to at each depth level.
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2, bottleneck=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Standard attention (Q gets modulated by thalamus)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Thalamic router for Q modulation
        self.thalamus = ThalamicRouter(hidden_dim, bottleneck)

        # Standard FFN
        ff_dim = hidden_dim * ff_mult
        self.gate = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.up = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, hidden_dim, bias=False)

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None, running_mean=None):
        B, T, D = x.shape

        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # QKV projection
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # THALAMIC Q-STREAM MODULATION
        # Add thalamic signal to queries (steers attention)
        thal_mod, new_mean = self.thalamus(h, running_mean)
        thal_mod_heads = thal_mod.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q + thal_mod_heads

        # Standard attention from here
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        # Standard FFN
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))

        return x, new_mean


class PredictiveCodingLayer(nn.Module):
    """Predictive Coding — process errors, not raw input.

    The brain primarily processes PREDICTION ERRORS (surprises).
    This layer:
    1. Predicts the next layer's input
    2. Computes the error (actual - predicted)
    3. Only processes the error (much sparser signal)

    If predictions are good, the error is near-zero = massive compute savings.
    The prediction IS the compression.
    """
    def __init__(self, hidden_dim, rank=32):
        super().__init__()
        # Predictor: low-rank prediction of next state from current
        self.predict_down = nn.Linear(hidden_dim, rank, bias=False)
        self.predict_up = nn.Linear(rank, hidden_dim, bias=False)

        # Error processor: transform the prediction error
        self.error_down = nn.Linear(hidden_dim, rank, bias=False)
        self.error_act = nn.SiLU()
        self.error_up = nn.Linear(rank, hidden_dim, bias=False)

        # Blending: how much to trust prediction vs error
        self.blend = nn.Parameter(torch.tensor(0.5))

        nn.init.zeros_(self.predict_up.weight)
        nn.init.zeros_(self.error_up.weight)

    def forward(self, x, prev_prediction=None):
        """
        x: current hidden state
        prev_prediction: prediction from previous layer (if any)

        Returns: updated state, prediction for next layer
        """
        if prev_prediction is not None:
            # Compute prediction error
            error = x - prev_prediction
            # Process only the error (the surprise)
            error_processed = self.error_up(self.error_act(self.error_down(error)))
            # Blend prediction with error-corrected state
            blend = torch.sigmoid(self.blend)
            x = blend * prev_prediction + (1 - blend) * x + error_processed

        # Generate prediction for next layer
        prediction = x + self.predict_up(self.predict_down(x))

        return x, prediction


class ActivationSparsifier(nn.Module):
    """Activation sparsity — keep only top-K% of activations.

    Based on ProSparse: 89% activation sparsity with negligible loss.
    Applied after each layer to zero out unimportant activations.
    The remaining 11% does all the work.
    """
    def __init__(self, keep_ratio=0.1, learnable=True):
        super().__init__()
        self.keep_ratio = keep_ratio
        if learnable:
            self.threshold = nn.Parameter(torch.tensor(0.0))
        else:
            self.threshold = None

    def forward(self, x):
        if not self.training and self.threshold is not None:
            # During inference: use learned threshold
            mask = x.abs() > torch.sigmoid(self.threshold) * x.abs().mean()
            return x * mask

        # During training: soft top-k via straight-through estimator
        B, T, D = x.shape
        k = max(1, int(D * self.keep_ratio))

        # Find top-k threshold per token
        topk_vals = x.abs().topk(k, dim=-1).values
        threshold = topk_vals[:, :, -1:].detach()

        # Soft mask (differentiable)
        mask = torch.sigmoid(10 * (x.abs() - threshold))
        return x * mask
