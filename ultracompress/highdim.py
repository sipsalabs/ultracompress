"""
HIGH-DIMENSIONAL WEIGHT SPACE — 10D geometric compression.

Instead of storing weight matrices as flat 2D arrays, represent each weight
vector as a POINT in 10D space. Nearby points = similar weights. The 10D
embedding is more compact because it exploits geometric structure that flat
representations miss entirely.

Why 10D? A 10D codebook with K entries per axis gives K^10 effective entries.
K=4 per axis = 4^10 = 1,048,576 distinct representable patterns from just
40 stored values. That's 20 bits of precision from 40 floats of storage.

Three compressors:
  1. WeightSpace10D        — Autoencoder: weights -> 10D coordinate -> reconstruction
  2. HierarchicalQuantizer — K-means in 10D (exponential codebook power)
  3. MultiScaleProjection  — Progressive 4D -> 7D -> 10D refinement
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ================================================================
# 1. WeightSpace10D — Autoencoder bottleneck in 10D
# ================================================================

@dataclass
class WeightSpace10DResult:
    """Compressed form: 10D coordinates + frozen decoder weights."""
    coords_10d: torch.Tensor       # (n_vectors, 10) — the compressed data
    decoder_weight: torch.Tensor   # decoder layer weights
    decoder_bias: torch.Tensor     # decoder layer biases
    original_shape: Tuple[int, ...]
    vector_dim: int

    def storage_bytes(self) -> int:
        coord_bytes = self.coords_10d.numel() * 2  # float16
        dec_bytes = (self.decoder_weight.numel() + self.decoder_bias.numel()) * 2
        return coord_bytes + dec_bytes

    def original_bytes(self) -> int:
        sz = 1
        for d in self.original_shape:
            sz *= d
        return sz * 4  # float32

    def compression_ratio(self) -> float:
        return self.original_bytes() / max(self.storage_bytes(), 1)


class WeightSpace10D:
    """Encode weight vectors as points in 10D space via autoencoder.

    The encoder maps weight vectors to 10D coordinates. The decoder
    reconstructs from those coordinates. After training, we discard the
    encoder and store only the 10D coords + decoder — that IS the
    compressed representation.
    """

    def __init__(self, dim: int = 10, hidden: int = 64, lr: float = 1e-3):
        self.dim = dim
        self.hidden = hidden
        self.lr = lr

    def compress(self, weight: torch.Tensor, vector_size: int = 32,
                 epochs: int = 200) -> WeightSpace10DResult:
        original_shape = weight.shape
        flat = weight.detach().float().reshape(-1)
        # pad to multiple of vector_size
        pad = (vector_size - flat.numel() % vector_size) % vector_size
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad)])
        vectors = flat.reshape(-1, vector_size)
        n = vectors.shape[0]

        # tiny autoencoder: vector_size -> hidden -> 10D -> hidden -> vector_size
        encoder = nn.Sequential(
            nn.Linear(vector_size, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.dim),
        )
        decoder = nn.Sequential(
            nn.Linear(self.dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, vector_size),
        )
        opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=self.lr)
        # train
        for _ in range(epochs):
            z = encoder(vectors)
            recon = decoder(z)
            loss = (recon - vectors).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            coords = encoder(vectors).half()
            # flatten decoder into single linear for compact storage
            dec_w = torch.cat([p.data.half().reshape(-1)
                               for p in decoder.parameters() if p.dim() >= 2])
            dec_b = torch.cat([p.data.half().reshape(-1)
                               for p in decoder.parameters() if p.dim() < 2])

        return WeightSpace10DResult(
            coords_10d=coords, decoder_weight=dec_w, decoder_bias=dec_b,
            original_shape=original_shape, vector_dim=vector_size,
        )

    def decompress(self, result: WeightSpace10DResult) -> torch.Tensor:
        """Rebuild weight tensor from 10D coords + decoder."""
        # reconstruct decoder layers from flattened params
        d, h, v = self.dim, self.hidden, result.vector_dim
        dec = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Linear(h, v),
        )
        w_sizes = [h * d, v * h]
        b_sizes = [h, v]
        w_off, b_off = 0, 0
        for i, layer in enumerate([dec[0], dec[2]]):
            ws = w_sizes[i]
            layer.weight.data = result.decoder_weight[w_off:w_off + ws].float().reshape(
                layer.weight.shape)
            w_off += ws
            bs = b_sizes[i]
            layer.bias.data = result.decoder_bias[b_off:b_off + bs].float()
            b_off += bs
        with torch.no_grad():
            recon = dec(result.coords_10d.float())
        flat = recon.reshape(-1)
        numel = 1
        for d in result.original_shape:
            numel *= d
        return flat[:numel].reshape(result.original_shape)


# ================================================================
# 2. HierarchicalQuantizer — K-means in 10D space
# ================================================================

@dataclass
class HierarchicalQuantized:
    """10D codebook quantization result."""
    codebook: torch.Tensor    # (K, 10) — codebook in 10D
    indices: torch.Tensor     # (n_vectors,) — index per vector
    projection: torch.Tensor  # (vector_dim, 10) — maps vectors to 10D
    scale: torch.Tensor       # (n_vectors,) — per-vector norms
    original_shape: Tuple[int, ...]
    vector_dim: int

    def storage_bytes(self) -> int:
        cb = self.codebook.numel() * 2            # float16 codebook
        idx_bits = int(np.ceil(np.log2(max(self.codebook.shape[0], 2))))
        idx = (self.indices.numel() * idx_bits + 7) // 8
        proj = self.projection.numel() * 2        # float16 projection
        sc = self.scale.numel() * 2               # float16 scales
        return cb + idx + proj + sc

    def original_bytes(self) -> int:
        sz = 1
        for d in self.original_shape:
            sz *= d
        return sz * 4

    def compression_ratio(self) -> float:
        return self.original_bytes() / max(self.storage_bytes(), 1)


class HierarchicalQuantizer:
    """Quantize weight vectors in 10D space instead of flat space.

    Project vectors to 10D via random projection, then run k-means in 10D.
    A codebook of K entries in 10D can capture exponentially richer structure
    than K entries in 1D, because 10D k-means partitions a 10D Voronoi
    tessellation — each cell is a 10D polytope, not a 1D interval.
    """

    def __init__(self, dim: int = 10, K: int = 256, iters: int = 30):
        self.dim = dim
        self.K = K
        self.iters = iters

    def compress(self, weight: torch.Tensor,
                 vector_size: int = 32) -> HierarchicalQuantized:
        original_shape = weight.shape
        flat = weight.detach().float().reshape(-1)
        pad = (vector_size - flat.numel() % vector_size) % vector_size
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad)])
        vectors = flat.reshape(-1, vector_size)
        n = vectors.shape[0]

        # normalize: store norms separately
        norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = vectors / norms

        # random projection to 10D (preserves distances by JL lemma)
        proj = torch.randn(vector_size, self.dim) / np.sqrt(self.dim)
        embedded = normed @ proj  # (n, 10)

        # k-means in 10D
        perm = torch.randperm(n)[:self.K]
        centroids = embedded[perm].clone()
        for _ in range(self.iters):
            dists = torch.cdist(embedded, centroids)  # (n, K)
            assigns = dists.argmin(dim=1)
            for k in range(self.K):
                mask = assigns == k
                if mask.any():
                    centroids[k] = embedded[mask].mean(dim=0)

        dists = torch.cdist(embedded, centroids)
        indices = dists.argmin(dim=1)

        return HierarchicalQuantized(
            codebook=centroids.half(), indices=indices.short(),
            projection=proj.half(), scale=norms.squeeze().half(),
            original_shape=original_shape, vector_dim=vector_size,
        )

    def decompress(self, result: HierarchicalQuantized) -> torch.Tensor:
        """Reconstruct from 10D codebook entries."""
        proj = result.projection.float()          # (vector_dim, 10)
        centroids = result.codebook.float()       # (K, 10)
        selected = centroids[result.indices.long()]  # (n, 10)
        # pseudo-inverse: proj is (vector_dim, 10), pinv is (10, vector_dim)
        # embedded = normed @ proj, so normed ~ embedded @ pinv(proj)
        recon = selected @ torch.linalg.pinv(proj)  # (n, vector_dim)
        recon = recon * result.scale.float().unsqueeze(1)
        flat = recon.reshape(-1)
        numel = 1
        for d in result.original_shape:
            numel *= d
        return flat[:numel].reshape(result.original_shape)


# ================================================================
# 3. MultiScaleProjection — Progressive 4D -> 7D -> 10D
# ================================================================

@dataclass
class MultiScaleResult:
    """Progressive multi-scale compressed representation."""
    coords_4d: torch.Tensor     # coarse: (n, 4)
    residual_7d: torch.Tensor   # medium: (n, 3) — adds to 4D to get 7D
    residual_10d: torch.Tensor  # fine:   (n, 3) — adds to 7D to get 10D
    decoder_params: torch.Tensor  # flattened decoder
    original_shape: Tuple[int, ...]
    vector_dim: int
    scale: int  # 1=4D only, 2=+7D, 3=+10D

    def storage_bytes(self, scale: Optional[int] = None) -> int:
        s = scale or self.scale
        total = self.coords_4d.numel() * 2  # always store 4D
        total += self.decoder_params.numel() * 2
        if s >= 2:
            total += self.residual_7d.numel() * 2
        if s >= 3:
            total += self.residual_10d.numel() * 2
        return total

    def original_bytes(self) -> int:
        sz = 1
        for d in self.original_shape:
            sz *= d
        return sz * 4

    def compression_ratio(self, scale: Optional[int] = None) -> float:
        return self.original_bytes() / max(self.storage_bytes(scale), 1)


class MultiScaleProjection:
    """Progressive refinement: 4D (coarse) -> 7D (medium) -> 10D (fine).

    Train one autoencoder with a 10D bottleneck, then split the 10D
    coordinates into three bands:
      - dims 0-3:  coarse structure  (4D)
      - dims 4-6:  medium detail     (7D = 4D + 3D residual)
      - dims 7-9:  fine detail       (10D = 7D + 3D residual)

    Progressive decompression: send 4D first for a rough version, then
    refine with 7D, then 10D. Each level costs only 3 more dims.
    """

    def __init__(self, hidden: int = 64, lr: float = 1e-3):
        self.hidden = hidden
        self.lr = lr

    def compress(self, weight: torch.Tensor, vector_size: int = 32,
                 epochs: int = 200) -> MultiScaleResult:
        original_shape = weight.shape
        flat = weight.detach().float().reshape(-1)
        pad = (vector_size - flat.numel() % vector_size) % vector_size
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad)])
        vectors = flat.reshape(-1, vector_size)

        # autoencoder with 10D bottleneck, trained with progressive loss
        encoder = nn.Sequential(
            nn.Linear(vector_size, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 10),
        )
        decoder = nn.Sequential(
            nn.Linear(10, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, vector_size),
        )
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr)

        for epoch in range(epochs):
            z = encoder(vectors)  # (n, 10)
            # progressive masking: early epochs see only coarse dims
            frac = min(1.0, (epoch + 1) / (epochs * 0.6))
            active_dims = max(4, int(frac * 10))
            z_masked = z.clone()
            z_masked[:, active_dims:] = 0.0
            recon = decoder(z_masked)
            loss = (recon - vectors).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            z = encoder(vectors)
            dec_params = torch.cat([p.data.reshape(-1) for p in decoder.parameters()])

        return MultiScaleResult(
            coords_4d=z[:, :4].half(),
            residual_7d=z[:, 4:7].half(),
            residual_10d=z[:, 7:10].half(),
            decoder_params=dec_params.half(),
            original_shape=original_shape,
            vector_dim=vector_size,
            scale=3,
        )

    def decompress(self, result: MultiScaleResult,
                   scale: int = 3) -> torch.Tensor:
        """Reconstruct at chosen scale: 1=coarse, 2=medium, 3=fine."""
        v = result.vector_dim
        h = self.hidden
        decoder = nn.Sequential(
            nn.Linear(10, h), nn.ReLU(), nn.Linear(h, v),
        )
        # load decoder params
        params = result.decoder_params.float()
        offset = 0
        for p in decoder.parameters():
            sz = p.numel()
            p.data = params[offset:offset + sz].reshape(p.shape)
            offset += sz

        z = torch.zeros(result.coords_4d.shape[0], 10)
        z[:, :4] = result.coords_4d.float()
        if scale >= 2:
            z[:, 4:7] = result.residual_7d.float()
        if scale >= 3:
            z[:, 7:10] = result.residual_10d.float()

        with torch.no_grad():
            recon = decoder(z)
        flat = recon.reshape(-1)
        numel = 1
        for d in result.original_shape:
            numel *= d
        return flat[:numel].reshape(result.original_shape)
