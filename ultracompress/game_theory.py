"""Game-theoretic compression: Nash equilibrium between size and quality."""

import torch
import numpy as np

class CompressionGame:
    """Adversarial game: Compressor minimizes size, Quality preserves accuracy."""

    def compressor_move(self, weights, budget):
        """Compress by zeroing least-important weights within budget."""
        flat = weights.flatten()
        k = max(1, int(flat.numel() * budget))
        topk = flat.abs().topk(k).indices
        out = torch.zeros_like(flat)
        out[topk] = flat[topk]
        return out.view(weights.shape)

    def quality_move(self, compressed, original):
        """Return per-element importance = where compression hurt most."""
        err = (original - compressed).abs()
        return err / (err.max() + 1e-12)

    def play(self, weights, n_rounds=5, budget=0.5):
        """Iterate compressor/quality moves toward Nash equilibrium."""
        w = weights.clone().float()
        importance = torch.ones_like(w)
        for _ in range(n_rounds):
            effective = w * importance
            compressed = self.compressor_move(effective, budget)
            importance = 1.0 - self.quality_move(compressed, w) * 0.5
            budget = min(budget + 0.05, 0.95)
        return self.compressor_move(w * importance, budget)


class ParetoFrontier:
    """Find Pareto-optimal compression-quality tradeoff points."""

    @staticmethod
    def find_frontier(weight, methods, n_points=10):
        """Evaluate methods at various budgets, return Pareto-optimal pairs."""
        candidates = []
        for method in methods:
            for b in np.linspace(0.1, 0.95, n_points):
                c = method(weight, float(b))
                cos = torch.nn.functional.cosine_similarity(
                    weight.flatten().unsqueeze(0), c.flatten().unsqueeze(0)
                ).item()
                ratio = weight.numel() / max(1, (c != 0).sum().item())
                candidates.append((ratio, cos))
        candidates.sort(key=lambda x: x[0])
        frontier = []
        best_q = -1.0
        for comp, qual in candidates:
            if qual > best_q:
                best_q = qual
                frontier.append((comp, qual))
        return frontier
