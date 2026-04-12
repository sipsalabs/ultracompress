"""Information-theoretic analysis of neural network weights."""
import numpy as np
from typing import List, Tuple

def _H(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

class WeightEntropy:
    """Shannon entropy of weight distributions -- bits per weight."""
    def measure(self, weights: np.ndarray, bins: int = 256) -> float:
        counts, _ = np.histogram(weights.ravel().astype(np.float64), bins=bins)
        return _H(counts / counts.sum())

    def per_layer(self, layers: dict[str, np.ndarray], bins: int = 256) -> dict[str, float]:
        return {name: self.measure(w, bins) for name, w in layers.items()}

class MutualInformation:
    """Mutual information between layer weight distributions."""
    def mutual_info(self, a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
        af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        n = min(len(af), len(bf))
        joint, _, _ = np.histogram2d(af[:n], bf[:n], bins=bins)
        joint = joint / joint.sum()
        # I(A;B) = H(A) + H(B) - H(A,B)
        return max(0.0, _H(joint.sum(axis=1)) + _H(joint.sum(axis=0)) - _H(joint.ravel()))

    def redundancy_matrix(self, layers: dict[str, np.ndarray], bins: int = 64) -> dict:
        names = list(layers.keys())
        matrix = {}
        for i, na in enumerate(names):
            for nb in names[i + 1:]:
                matrix[(na, nb)] = self.mutual_info(layers[na], layers[nb], bins)
        return matrix

class RateDistortionEstimator:
    """Estimate rate-distortion curve -- theoretical compression limit at each quality."""
    def estimate_rd(self, weights: np.ndarray,
                    distortion_levels: List[float] = None) -> List[Tuple[float, float]]:
        var = float(np.var(weights.ravel().astype(np.float64))) or 1e-12
        if distortion_levels is None:
            distortion_levels = [var * f for f in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]]
        # Gaussian R-D bound: R(D) = 0.5 * log2(var / D) for D < var
        return [(0.5 * np.log2(var / max(D, 1e-15)) if D < var else 0.0, D)
                for D in distortion_levels]

class CompressionEfficiency:
    """Compare actual compression rate to theoretical optimum."""
    def efficiency(self, actual_bpw: float, theoretical_bpw: float) -> float:
        if actual_bpw <= 0:
            return 100.0
        if theoretical_bpw <= 0:
            return 0.0
        return (theoretical_bpw / actual_bpw) * 100.0

    def analyze(self, weights: np.ndarray, actual_bpw: float,
                target_distortion: float = None) -> dict:
        var = float(np.var(weights.astype(np.float64))) or 1e-12
        if target_distortion is None:
            target_distortion = var * 0.01
        theoretical = RateDistortionEstimator().estimate_rd(weights, [target_distortion])[0][0]
        return {
            "entropy_bpw": WeightEntropy().measure(weights),
            "theoretical_min_bpw": theoretical,
            "actual_bpw": actual_bpw,
            "efficiency_pct": self.efficiency(actual_bpw, theoretical),
            "headroom_bits": max(0, actual_bpw - theoretical),
            "variance": var,
        }
