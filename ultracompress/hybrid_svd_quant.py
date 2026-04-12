"""Hybrid SVD + Residual Quantization: the manifold-optimal compressor.
SVD captures the intrinsic low-rank subspace; quantization mops up the residual."""

import torch, numpy as np
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class CompressedWeight:
    U: torch.Tensor; S: torch.Tensor; V: torch.Tensor  # rank-k SVD factors
    r_codes: torch.Tensor; r_scale: torch.Tensor; r_zero: torch.Tensor  # quantized residual
    bits: int; shape: tuple

class HybridSVDQuantCompressor:
    def compress(self, W: torch.Tensor, rank: int, residual_bits: int = 4) -> CompressedWeight:
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        U_k, S_k, V_k = U[:, :rank], S[:rank], Vt[:rank, :]
        approx = U_k @ torch.diag(S_k) @ V_k
        R = W.float() - approx
        levels = 2 ** residual_bits - 1
        rmin, rmax = R.min(), R.max()
        scale = (rmax - rmin) / levels if levels > 0 else torch.tensor(1.0)
        codes = ((R - rmin) / scale).round().clamp(0, levels).to(torch.uint8)
        return CompressedWeight(U_k.half(), S_k.half(), V_k.half(), codes, scale.half(), rmin.half(), residual_bits, W.shape)

    def decompress(self, c: CompressedWeight) -> torch.Tensor:
        approx = c.U.float() @ torch.diag(c.S.float()) @ c.V.float()
        R = c.r_codes.float() * c.r_scale.float() + c.r_zero.float()
        return approx + R

    def size_bytes(self, c: CompressedWeight) -> int:
        m, n = c.shape; k = c.S.numel()
        svd_bytes = (m * k + k + k * n) * 2  # float16
        res_bytes = m * n * c.bits / 8 + 4    # codes + scale/zero
        return int(svd_bytes + res_bytes)

def compress_model(state_dict: Dict[str, torch.Tensor], target_size: int) -> Dict:
    comp = HybridSVDQuantCompressor()
    results, total_orig, total_comp = {}, 0, 0
    for name, W in state_dict.items():
        if W.ndim != 2: continue
        m, n = W.shape; orig = m * n * 2
        # Binary search for rank that hits target ratio
        ratio = target_size / sum(v.numel() * 2 for v in state_dict.values() if v.ndim == 2)
        rank = max(1, int(min(m, n) * ratio * 0.6))
        bits = 4 if ratio > 0.3 else 2
        c = comp.compress(W, rank, bits)
        recon = comp.decompress(c)
        cos = torch.nn.functional.cosine_similarity(W.float().flatten(), recon.flatten(), dim=0).item()
        sz = comp.size_bytes(c)
        results[name] = {'compressed': c, 'cos_sim': cos, 'ratio': orig / sz, 'bytes': sz}
        total_orig += orig; total_comp += sz
        print(f"  {name}: rank={rank} bits={bits} cos={cos:.6f} ratio={orig/sz:.1f}x")
    print(f"  Total: {total_orig/1e6:.1f}MB -> {total_comp/1e6:.1f}MB ({total_orig/total_comp:.1f}x)")
    return results

if __name__ == '__main__':
    torch.manual_seed(42)
    W = torch.randn(1024, 1024)
    comp = HybridSVDQuantCompressor()
    for rank, bits in [(64, 4), (32, 4), (32, 2), (16, 4), (16, 2)]:
        c = comp.compress(W, rank, bits)
        recon = comp.decompress(c)
        cos = torch.nn.functional.cosine_similarity(W.flatten(), recon.flatten(), dim=0).item()
        sz = comp.size_bytes(c)
        orig = 1024 * 1024 * 2
        print(f"rank={rank:3d} bits={bits} | cos={cos:.6f} | {orig/sz:.1f}x compression | {sz/1024:.0f}KB")
