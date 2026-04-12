"""Chaos theory & fractal dimension analysis for weight compression."""
import numpy as np

class FractalDimension:
    """Estimate fractal dimension of weight matrix via box-counting."""
    def estimate(self, W):
        W = np.asarray(W, dtype=np.float32)
        mn, mx = W.min(), W.max()
        if mx - mn < 1e-12:
            return 1.0
        norm = (W - mn) / (mx - mn)  # normalize to [0,1]
        counts, scales = [], []
        for exp in range(1, 7):
            n_boxes = 2 ** exp
            grid = np.clip((norm * n_boxes).astype(int), 0, n_boxes - 1)
            # count occupied boxes using row-col pairs
            flat = grid.reshape(-1) if W.ndim == 1 else (grid[:, :1] * n_boxes + grid[:, 1:] if W.ndim == 2 and W.shape[1] > 1 else grid.reshape(-1))
            if W.ndim >= 2 and W.shape[1] > 1:
                idx = grid[:, 0] * n_boxes + grid[:, 1] if W.ndim == 2 else grid.reshape(-1, 2)[:, 0] * n_boxes + grid.reshape(-1, 2)[:, 1]
                occupied = len(np.unique(idx))
            else:
                occupied = len(np.unique(grid))
            counts.append(np.log(max(occupied, 1)))
            scales.append(np.log(1.0 / n_boxes))
        # linear regression: log(count) = D * log(1/scale) + c
        scales, counts = np.array(scales), np.array(counts)
        D = np.polyfit(scales, counts, 1)[0]
        return float(np.clip(-D, 1.0, 2.0))

class LyapunovAnalyzer:
    """Measure sensitivity to perturbation via Lyapunov-like exponent."""
    def sensitivity(self, W, eps=1e-5, steps=10):
        W = np.asarray(W, dtype=np.float32).ravel()
        rng = np.random.default_rng(42)
        perturbed = W + rng.normal(0, eps, W.shape).astype(np.float32)
        lyap = 0.0
        for _ in range(steps):
            W = np.tanh(W)           # iterate a nonlinear map
            perturbed = np.tanh(perturbed)
            diff = np.abs(perturbed - W).mean()
            if diff < 1e-30:
                return float(-np.inf)
            lyap += np.log(diff / eps)
        return float(lyap / steps)

class AttractorCompressor:
    """Compress weights by finding attractor basins via iterative dynamics."""
    def compress(self, W, n_attractors=16, iters=20):
        W = np.asarray(W, dtype=np.float32)
        flat = W.ravel()
        # iterate logistic-like map to find natural attractors
        evolved = flat.copy()
        for _ in range(iters):
            evolved = np.tanh(evolved * 2.0)
        # cluster evolved points into attractor basins
        uniq = np.unique(np.round(evolved, decimals=3))
        rng = np.random.default_rng(0)
        centers = rng.choice(uniq, size=min(n_attractors, len(uniq)), replace=False)
        centers = np.sort(centers).astype(np.float32)
        # assign each weight to nearest attractor
        dists = np.abs(flat[:, None] - centers[None, :])
        labels = dists.argmin(axis=1).astype(np.uint8)
        return {"centers": centers, "labels": labels.reshape(W.shape),
                "shape": W.shape, "ratio": 32.0 * W.size / (32 * len(centers) + 8 * W.size)}

    def decompress(self, blob):
        return blob["centers"][blob["labels"]]
