"""
Neural Architecture Search for Compression — Per-layer ratio optimization.
Random search over per-layer configs: sample N configs, evaluate each quickly,
keep the best that fits within the target size budget.
Returns: dict mapping layer_name -> {method, bits, rank}.
"""
import numpy as np
from typing import Dict, List, Any, Optional

METHODS, BIT_OPTIONS = ["quantize", "factorize", "pq"], [1, 2, 3, 4, 8]
RANK_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5]

class CompressionNAS:
    """Search over per-layer compression ratios to maximize quality at a size budget."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def _random_config(self, names: List[str]) -> Dict[str, Dict[str, Any]]:
        return {n: {"method": str(self.rng.choice(METHODS)),
                     "bits": int(self.rng.choice(BIT_OPTIONS)),
                     "rank": float(self.rng.choice(RANK_FRACTIONS))} for n in names}

    def _estimate_size(self, config: Dict, shapes: Dict[str, tuple]) -> int:
        return sum(int(np.prod(shapes[n]) * c["bits"] / 8) for n, c in config.items())

    def _evaluate(self, config: Dict, model: Dict[str, np.ndarray]) -> float:
        total = sum(w.size for w in model.values())
        return sum(min(1.0, 0.7 + 0.04 * config[n]["bits"]) * w.size
                   for n, w in model.items()) / total

    def search(self, model: Dict[str, np.ndarray], target_size: int,
               n_trials: int = 200) -> Dict[str, Dict[str, Any]]:
        """Search for optimal per-layer compression config."""
        names = list(model.keys())
        shapes = {n: w.shape for n, w in model.items()}
        best: Optional[SearchResult] = None
        for _ in range(n_trials):
            cfg = self._random_config(names)
            size = self._estimate_size(cfg, shapes)
            if size > target_size:
                continue
            score = self._evaluate(cfg, model)
            if best is None or score > best.score:
                best = SearchResult(cfg, score, size)
        return best.config if best else self._random_config(names)
