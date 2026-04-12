"""
Topological Data Analysis for weight compression.

Persistent homology via SVD proxy: singular values ~ feature persistence.
Large singular values = load-bearing structure. Small = ephemeral noise.
"""

import torch
from dataclasses import dataclass

@dataclass
class PersistenceDiagram:
    birth: torch.Tensor   # feature appears at this threshold
    death: torch.Tensor   # feature vanishes at this threshold
    persistence: torch.Tensor  # death - birth (lifetime = importance)

class TopologicalAnalyzer:
    """Compute persistence diagram of a weight matrix via SVD proxy."""

    def analyze(self, W: torch.Tensor) -> PersistenceDiagram:
        M = W.reshape(W.shape[0], -1).float()
        S = torch.linalg.svdvals(M)
        death = S
        birth = torch.zeros_like(S)
        return PersistenceDiagram(birth=birth, death=death, persistence=death - birth)

@dataclass
class CompressedTopological:
    U: torch.Tensor
    S: torch.Tensor
    Vt: torch.Tensor
    orig_shape: tuple
    kept: int

class TopologicalCompressor:
    """Keep only persistent (high-singular-value) features."""

    def __init__(self):
        self.analyzer = TopologicalAnalyzer()

    def compress(self, W: torch.Tensor, persistence_threshold: float = 0.1) -> CompressedTopological:
        M = W.reshape(W.shape[0], -1).float()
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        mask = S > persistence_threshold * S[0]
        k = max(1, mask.sum().item())
        return CompressedTopological(U=U[:, :k], S=S[:k], Vt=Vt[:k], orig_shape=W.shape, kept=k)

    def decompress(self, c: CompressedTopological) -> torch.Tensor:
        return (c.U * c.S.unsqueeze(0)) @ c.Vt
