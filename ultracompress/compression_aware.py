"""
Compression-Aware Training — train models DESIGNED to be compressed.

Instead of train-then-compress (lossy), train WITH compression in the loop
so the model learns weights that are inherently compressible:
  - Low effective rank (better SVD/factorization)
  - Few outliers (better quantization)
  - Cross-layer similarity (better FRR/delta compression)
"""

import torch
import torch.nn as nn


class CompressibleRegularizer:
    """Training regularizer that encourages compressible weight structure."""

    def __init__(self, rank_weight=0.01, outlier_weight=0.01, similarity_weight=0.01):
        self.rank_weight = rank_weight
        self.outlier_weight = outlier_weight
        self.similarity_weight = similarity_weight

    def loss_penalty(self, model):
        matrices, penalty = [], torch.tensor(0.0)
        for p in model.parameters():
            if p.ndim < 2:
                continue
            flat = p.view(p.shape[0], -1)
            matrices.append(flat)
            # Nuclear norm encourages low rank (sum of singular values)
            penalty = penalty + self.rank_weight * torch.linalg.matrix_norm(flat, ord='nuc') / flat.numel()
            # Outlier penalty: kurtosis-like — penalize heavy tails
            std = flat.std() + 1e-8
            penalty = penalty + self.outlier_weight * ((flat / std) ** 4).mean()
        # Cross-layer similarity: penalize variance of layer means/stds
        if len(matrices) > 1:
            means = torch.stack([m.mean() for m in matrices])
            stds = torch.stack([m.std() for m in matrices])
            penalty = penalty + self.similarity_weight * (means.var() + stds.var())
        return penalty


class _FakeQuantize(torch.autograd.Function):
    """Straight-through estimator: quantize forward, pass gradients backward."""

    @staticmethod
    def forward(ctx, x, bits):
        qmin, qmax = 0, (1 << bits) - 1
        scale = (x.max() - x.min()) / qmax
        scale = torch.clamp(scale, min=1e-8)
        quantized = torch.clamp(torch.round((x - x.min()) / scale), qmin, qmax)
        return quantized * scale + x.min()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # straight-through


class QuantizationAwareTraining:
    """Wrap a model so weights are fake-quantized during forward pass."""

    def __init__(self, bits=4):
        self.bits = bits

    def wrap_model(self, model):
        bits = self.bits
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                orig_forward = module.forward
                def make_qat_forward(mod, fwd):
                    def qat_forward(x):
                        mod.weight.data = _FakeQuantize.apply(mod.weight, bits)
                        return fwd(x)
                    return qat_forward
                module.forward = make_qat_forward(module, orig_forward)
        return model


class FRRReadyTrainer:
    """Encourage cross-layer weight similarity for better FRR compression."""

    def __init__(self, max_similarity_weight=0.1):
        self.max_weight = max_similarity_weight

    def similarity_loss(self, model, progress):
        """progress: 0→1 over training. Gradually increases sharing pressure."""
        weight = self.max_weight * min(progress, 1.0)
        layers = [p.view(-1) for p in model.parameters() if p.ndim >= 2]
        if len(layers) < 2:
            return torch.tensor(0.0)
        # Pad to same size, compute pairwise cosine distance from mean
        max_len = max(l.shape[0] for l in layers)
        padded = torch.stack([nn.functional.pad(l, (0, max_len - l.shape[0])) for l in layers])
        mean_layer = padded.mean(dim=0)
        return weight * sum(1 - nn.functional.cosine_similarity(p.unsqueeze(0), mean_layer.unsqueeze(0))
                           for p in padded).squeeze() / len(layers)
