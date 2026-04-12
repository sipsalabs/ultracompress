"""
Locality-Sensitive Hashing for Weights — Cross-Layer Deduplication

Neural networks reuse similar weight patterns across layers. LSH finds these
duplicates even when raw cosine similarity misses them (different scales,
slight rotations). Random hyperplane LSH hashes similar vectors into the
same bucket, then we store one representative per bucket + per-weight
bucket IDs. At decompression, each weight is replaced by its bucket centroid.

BPW: log2(n_buckets) per weight vector, plus one centroid per bucket.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class LSHCompressed:
    """LSH-deduplicated weight representation."""
    centroids: torch.Tensor   # (n_buckets, vec_dim) — one representative per bucket
    indices: torch.Tensor     # (n_vectors,) — bucket ID for each weight vector
    shapes: list              # original shapes per layer
    vec_dim: int
    n_vectors: int

    def storage_bytes(self) -> int:
        centroid_bytes = self.centroids.numel() * 2  # FP16
        index_bytes = self.indices.numel() * 4       # int32
        return centroid_bytes + index_bytes

    @property
    def bits_per_weight(self) -> float:
        total_weights = self.n_vectors * self.vec_dim
        return (self.storage_bytes() * 8) / total_weights


class WeightLSH:
    """Locality-Sensitive Hashing for weight vector deduplication."""

    def __init__(self, n_planes: int = 16, vec_dim: int = 64, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.planes = torch.tensor(rng.randn(n_planes, vec_dim), dtype=torch.float32)
        self.vec_dim = vec_dim
        self.n_planes = n_planes

    def _hash(self, vecs: torch.Tensor) -> torch.Tensor:
        proj = vecs @ self.planes.T  # (N, n_planes)
        bits = (proj > 0).long()
        powers = 2 ** torch.arange(self.n_planes)
        return (bits * powers).sum(dim=1)

    def compress(self, all_layer_weights: List[torch.Tensor]) -> LSHCompressed:
        shapes = [w.shape for w in all_layer_weights]
        flat = torch.cat([w.reshape(-1) for w in all_layer_weights])
        n_pad = (self.vec_dim - flat.numel() % self.vec_dim) % self.vec_dim
        if n_pad:
            flat = torch.cat([flat, torch.zeros(n_pad)])
        vecs = flat.reshape(-1, self.vec_dim).float()

        bucket_ids = self._hash(vecs)
        unique_ids = bucket_ids.unique()
        id_map = {uid.item(): i for i, uid in enumerate(unique_ids)}
        indices = torch.tensor([id_map[b.item()] for b in bucket_ids], dtype=torch.int32)

        centroids = torch.zeros(len(unique_ids), self.vec_dim)
        for i, uid in enumerate(unique_ids):
            centroids[i] = vecs[bucket_ids == uid].mean(dim=0)

        return LSHCompressed(
            centroids=centroids.half(), indices=indices,
            shapes=shapes, vec_dim=self.vec_dim, n_vectors=vecs.shape[0],
        )

    @staticmethod
    def decompress(comp: LSHCompressed) -> List[torch.Tensor]:
        flat = comp.centroids.float()[comp.indices.long()].reshape(-1)
        results, offset = [], 0
        for s in comp.shapes:
            n = 1
            for d in s:
                n *= d
            results.append(flat[offset:offset + n].reshape(s))
            offset += n
        return results
