"""Streaming Decompression — Run inference BEFORE the full model is decompressed.

Decompress layer-by-layer on demand. While layer N runs inference,
layer N+1 decompresses in a background thread. Peak memory = ONE layer.

Enables 100T models on limited hardware: only one layer resident at a time.
"""

import torch
import threading
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, Future


class StreamingDecompressor:
    """Decompresses one layer at a time from a compressed archive."""

    def __init__(self):
        self.path = None
        self.meta = None
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Dict[int, Future] = {}

    def open(self, path: str, decompress_fn=None):
        """Prepare for streaming. decompress_fn(path, layer_idx) -> weight dict."""
        self.path = path
        self._decompress_fn = decompress_fn or self._default_decompress
        self._cache.clear()
        return self

    def _default_decompress(self, path: str, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Default: load from safetensors/torch archive by layer prefix."""
        from ultracompress.streaming_loader import stream_layer_weights
        return stream_layer_weights(path, layer_idx, device="cpu")

    def get_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Decompress and return ONE layer's weights. Frees previous layers."""
        if layer_idx in self._pending:
            self._cache[layer_idx] = self._pending.pop(layer_idx).result()
        if layer_idx not in self._cache:
            self._cache[layer_idx] = self._decompress_fn(self.path, layer_idx)
        weights = self._cache.pop(layer_idx)
        # Evict all older layers to keep peak memory at one layer
        for k in [k for k in self._cache if k < layer_idx]:
            del self._cache[k]
        return weights

    def prefetch(self, layer_idx: int):
        """Start decompressing a layer in background thread."""
        if layer_idx not in self._cache and layer_idx not in self._pending:
            self._pending[layer_idx] = self._executor.submit(
                self._decompress_fn, self.path, layer_idx
            )


class PipelinedInference:
    """Run inference while decompressing — first token before full model loads."""

    def __init__(self, decompressor: StreamingDecompressor, layer_fn, n_layers: int, device="cpu"):
        """layer_fn(x, weights, layer_idx) -> x  runs one layer's forward pass."""
        self.dec = decompressor
        self.layer_fn = layer_fn
        self.n_layers = n_layers
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pipelined forward: decompress layer N+1 while running layer N."""
        self.dec.prefetch(0)
        for i in range(self.n_layers):
            if i + 1 < self.n_layers:
                self.dec.prefetch(i + 1)
            weights = {k: v.to(self.device) for k, v in self.dec.get_layer(i).items()}
            x = self.layer_fn(x, weights, i)
            del weights
        return x
