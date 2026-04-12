"""
Compressed Sensing for Weights — Random projections with RIP guarantees.
Recovery via Iterative Hard Thresholding (IHT). The random Gaussian
measurement matrix satisfies RIP, so s-sparse signals recover exactly
from O(s * log(n/s)) measurements.
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class CompressedMeasurement:
    measurements: np.ndarray  # (m,) compressed measurements
    seed: int                 # RNG seed to reconstruct measurement matrix
    original_shape: tuple
    m: int
    n: int                    # original flattened size

class CompressedSensor:
    """Project weight matrices via random Gaussian measurements with RIP."""
    def __init__(self, seed: int = 42):
        self.seed = seed

    def sense(self, weight_matrix: np.ndarray, compression_ratio: float = 4.0) -> CompressedMeasurement:
        """Project weight matrix to lower dimensions via random Gaussian matrix."""
        flat = weight_matrix.ravel().astype(np.float32)
        n = flat.shape[0]
        m = max(1, int(n / compression_ratio))
        rng = np.random.RandomState(self.seed)
        phi = rng.randn(m, n).astype(np.float32) / np.sqrt(m)
        return CompressedMeasurement(phi @ flat, self.seed, weight_matrix.shape, m, n)

    def recover(self, cm: CompressedMeasurement, sparsity: int = 0, n_iters: int = 50) -> np.ndarray:
        """Recover weight matrix via Iterative Hard Thresholding (IHT)."""
        rng = np.random.RandomState(cm.seed)
        phi = rng.randn(cm.m, cm.n).astype(np.float32) / np.sqrt(cm.m)
        s = sparsity if sparsity > 0 else max(1, cm.n // 4)
        x = np.zeros(cm.n, dtype=np.float32)
        for _ in range(n_iters):
            x = x + phi.T @ (cm.measurements - phi @ x)
            idx = np.argpartition(np.abs(x), -s)[-s:]  # hard threshold: keep top-s
            mask = np.zeros_like(x)
            mask[idx] = 1.0
            x *= mask
        return x.reshape(cm.original_shape)
