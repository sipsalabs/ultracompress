"""Wavelet Weight Compression — Haar multi-resolution decomposition."""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class WaveletCompressed:
    approx: torch.Tensor
    details: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # (H, V, D) per level
    orig_shape: Tuple[int, ...]
    n_levels: int

def _pad_even(x):
    h, w = x.shape[-2:]
    if h % 2: x = torch.nn.functional.pad(x, (0, 0, 0, 1))
    if w % 2: x = torch.nn.functional.pad(x, (0, 1, 0, 0))
    return x

def _haar_forward(x):
    x = _pad_even(x)
    lo_r = (x[..., 0::2, :] + x[..., 1::2, :]) / 2
    hi_r = (x[..., 0::2, :] - x[..., 1::2, :]) / 2
    ll = (lo_r[..., :, 0::2] + lo_r[..., :, 1::2]) / 2
    lh = (lo_r[..., :, 0::2] - lo_r[..., :, 1::2]) / 2
    hl = (hi_r[..., :, 0::2] + hi_r[..., :, 1::2]) / 2
    hh = (hi_r[..., :, 0::2] - hi_r[..., :, 1::2]) / 2
    return ll, lh, hl, hh

def _haar_inverse(ll, lh, hl, hh):
    lo_r = torch.zeros(*ll.shape[:-1], ll.shape[-2], ll.shape[-1]*2, device=ll.device)
    lo_r[..., :, 0::2] = ll + lh; lo_r[..., :, 1::2] = ll - lh
    hi_r = torch.zeros_like(lo_r)
    hi_r[..., :, 0::2] = hl + hh; hi_r[..., :, 1::2] = hl - hh
    out = torch.zeros(*ll.shape[:-2], ll.shape[-2]*2, lo_r.shape[-1], device=ll.device)
    out[..., 0::2, :] = lo_r + hi_r; out[..., 1::2, :] = lo_r - hi_r
    return out

def _threshold(t, thr):
    return t * (t.abs() > thr)

class WaveletCompressor:
    def compress(self, weight: torch.Tensor, n_levels: int = 3, threshold: float = 0.01) -> WaveletCompressed:
        orig_shape = weight.shape
        x = weight.view(-1, weight.shape[-1]) if weight.ndim > 2 else weight.clone()
        if x.ndim == 1: x = x.unsqueeze(0)
        details = []
        for _ in range(n_levels):
            if x.shape[-1] < 2 or x.shape[-2] < 2: break
            x, lh, hl, hh = _haar_forward(x)
            details.append((_threshold(hl, threshold), _threshold(lh, threshold), _threshold(hh, threshold)))
        return WaveletCompressed(approx=x, details=details, orig_shape=orig_shape, n_levels=len(details))

    def decompress(self, c: WaveletCompressed) -> torch.Tensor:
        x = c.approx
        for hl, lh, hh in reversed(c.details):
            x = _haar_inverse(x, lh, hl, hh)
        return x[..., :c.orig_shape[-2], :c.orig_shape[-1]].reshape(c.orig_shape)

class AdaptiveWavelet(WaveletCompressor):
    def compress_adaptive(self, weight: torch.Tensor, importance: float = 1.0, n_levels: int = 3,
                          base_threshold: float = 0.02) -> WaveletCompressed:
        threshold = base_threshold / max(importance, 1e-8)
        return self.compress(weight, n_levels=n_levels, threshold=threshold)
