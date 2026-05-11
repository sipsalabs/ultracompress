"""
Sparse MoE Weight Compression — Post-training dense-to-MoE conversion.

Instead of compressing values, restructure the weight matrix itself.
Cluster weight rows by similarity (vector quantization), each cluster becomes a
tiny expert. A learned router selects top-K experts per input token.
Inactive experts cost zero compute — massive savings at inference.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MoECompressed:
    """Sparse MoE representation of a formerly-dense weight matrix."""
    experts: torch.Tensor       # (n_experts, rows_per_expert, cols)
    centroids: torch.Tensor     # (n_experts, cols) — cluster centers as router keys
    row_assignments: torch.Tensor  # (orig_rows,) — which expert owns each row
    top_k: int
    orig_shape: Tuple[int, int]

    def storage_bytes(self) -> int:
        return (self.experts.numel() + self.centroids.numel()) * 2

    @property
    def bits_per_weight(self) -> float:
        orig_elements = self.orig_shape[0] * self.orig_shape[1]
        return (self.storage_bytes() * 8) / orig_elements


class DenseToMoE:
    """Convert a dense weight matrix into N sparse experts + router."""

    @staticmethod
    def convert(dense_weight: torch.Tensor, n_experts: int = 8,
                top_k: int = 2, n_iter: int = 20) -> MoECompressed:
        W = dense_weight.float()
        rows, cols = W.shape
        # vector quantization clustering on rows
        perm = torch.randperm(rows)[:n_experts]
        centroids = W[perm].clone()
        for _ in range(n_iter):
            dists = torch.cdist(W, centroids)
            assignments = dists.argmin(dim=1)
            for k in range(n_experts):
                mask = assignments == k
                if mask.any():
                    centroids[k] = W[mask].mean(dim=0)
        # Build expert weight blocks (ragged -> padded)
        max_rows = max(int((assignments == k).sum()) for k in range(n_experts))
        experts = torch.zeros(n_experts, max_rows, cols)
        for k in range(n_experts):
            rows_k = W[assignments == k]
            experts[k, :rows_k.shape[0]] = rows_k
        return MoECompressed(experts=experts.half(), centroids=centroids.half(),
                             row_assignments=assignments, top_k=top_k,
                             orig_shape=dense_weight.shape)

    @staticmethod
    def route(moe: MoECompressed, x: torch.Tensor) -> torch.Tensor:
        """Route input x through top-K experts. x: (..., cols)."""
        scores = F.cosine_similarity(x.unsqueeze(-2),
                                     moe.centroids.float().unsqueeze(0), dim=-1)
        topk = scores.topk(moe.top_k, dim=-1)
        weights = F.softmax(topk.values, dim=-1)
        out = torch.zeros_like(x)
        for i, k in enumerate(range(moe.top_k)):
            idx = topk.indices[..., k]
            expert_out = torch.stack([moe.experts[j].float() for j in idx.view(-1)])
            out += weights[..., k:k+1] * (x.unsqueeze(-2) @ expert_out.transpose(-1, -2)).squeeze(-2)
        return out
