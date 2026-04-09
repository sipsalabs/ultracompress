"""
UltraCompress Pipeline v5 — Product Quantization Revolution

Four compression paths compete for best quality-per-bit:
  0. Product Quantization (PQ) — THE flagship method. Splits weight groups into
     M sub-vectors, each quantized to K codebook entries. Effective codebook
     size K^M from M*K stored entries. Achieves sub-0.1 BPW.
  1. SVD + binarize + codebook — for steep-spectrum weights
  2. Standard Vector Quantization (RVQ) — sub-1 BPW
  3. Direct scalar quantization (INT2-8) — fast fallback

PQ configurations for target BPW levels:
  0.016 BPW: M=8 K=4 G=1024  (10T -> 20GB)
  0.076 BPW: M=8 K=4 G=1024  (10T -> 95GB, 235B -> 2.2GB)
  0.25  BPW: M=8 K=4 G=128   (235B -> 7.3GB)
  0.37  BPW: M=32 K=16 G=512 (235B -> 10.9GB)

Target: 235B -> 20GB (0.68 BPW), scaling to 10T -> 20GB (0.016 BPW)
"""

import torch
import time
import math
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from .profiler import profile_layer, profile_model, LayerProfile
from .factorize import factorize_weight, factorize_residual, reconstruct_from_factors
from .binarize import binarize_weight, BinarizedWeight
from .codebook import compress_binarized_factor, CodebookCompressed
from .quantize import quantize_absmax, smart_quantize, quantize_vector_codebook, QuantizedTensor, VectorQuantized
from .product_quantize import product_quantize, ProductQuantized
from .metrics import compute_quality, CompressionResult, ModelCompressionReport


@dataclass
class PipelineConfig:
    """Configuration for the compression pipeline."""
    # Quality target
    target_cosine_sim: float = 0.999
    max_retries: int = 3

    # Global target
    target_bpw: float = 0.5

    # Stage toggles
    enable_profiling: bool = True
    enable_factorization: bool = True
    enable_residual: bool = True
    enable_binarization: bool = True
    enable_codebook: bool = True
    enable_direct_quant: bool = True     # Hybrid: use direct quant for flat spectra
    enable_sparsity: bool = False

    # Spectrum threshold: if effective_rank_ratio > this, skip SVD
    svd_skip_threshold: float = 0.5

    # Factorization params
    energy_target: float = 0.999
    residual_levels: int = 3
    min_rank: int = 4
    max_rank: int = 1024

    # Binarization params
    group_size: int = 128
    use_rotation: bool = True
    use_sigma_delta: bool = True

    # Codebook params
    codebook_size: int = 4096

    # Direct quantization params
    quant_bits: int = 2
    quant_group_size: int = 128

    # Vector quantization params
    enable_vq: bool = True
    vq_codebook_size: int = 256
    vq_group_sizes: tuple = (16, 12, 8)  # Try largest group first (lowest BPW)
    vq_residual_levels: int = 1
    vq_n_iter: int = 15

    # Product quantization params (the flagship method)
    enable_pq: bool = True
    pq_configs: tuple = (
        # (M, K, G) tuples sorted by ascending BPW
        # BPW = M * log2(K) / G + scale_overhead
        (8, 4, 1024),    # ~0.08 BPW — extreme
        (8, 4, 512),     # ~0.09 BPW
        (8, 4, 256),     # ~0.14 BPW
        (16, 4, 256),    # ~0.20 BPW
        (8, 4, 128),     # ~0.26 BPW
        (8, 16, 128),    # ~0.40 BPW
        (16, 16, 256),   # ~0.36 BPW
        (8, 16, 64),     # ~0.76 BPW
        (4, 256, 32),    # ~1.6 BPW (high quality)
    )
    pq_n_iter: int = 20

    # Runtime
    device: str = "cuda"
    min_tensor_size: int = 1024


@dataclass
class CompressedLayer:
    name: str
    original_shape: tuple
    profile: Optional[LayerProfile] = None
    method: str = "none"  # "svd_binary", "vq", "intN", "skip"
    # SVD path
    codebook_U: Optional[CodebookCompressed] = None
    codebook_V: Optional[CodebookCompressed] = None
    binarized: Optional[BinarizedWeight] = None
    residual_codebooks: list = field(default_factory=list)
    # Direct quant path
    quantized: Optional[QuantizedTensor] = None
    # Vector quantization path
    vq: Optional[VectorQuantized] = None
    # Product quantization path
    pq: Optional[ProductQuantized] = None


def estimate_spectrum_flatness(weight: torch.Tensor, device: str = "cuda") -> float:
    """
    Quickly estimate how "flat" the singular value spectrum is.
    Returns ratio of rank needed for 90% energy to full rank.
    Low ratio = steep spectrum (SVD-friendly)
    High ratio = flat spectrum (use direct quant)
    """
    w = weight.float()
    if w.ndim < 2:
        return 0.0
    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)

    m, n = w.shape
    full_rank = min(m, n)

    # Quick estimate using a small number of singular values
    probe = min(64, full_rank)
    try:
        _, S, _ = torch.svd_lowrank(w.to(device), q=probe, niter=2)
        # Extrapolate: if top-64 SVs capture X% of energy,
        # estimate how much more rank we'd need
        if len(S) > 1:
            energy_captured = (S ** 2).sum()
            # Estimate total energy from Frobenius norm
            total_energy = torch.norm(w.to(device), p='fro') ** 2
            ratio_captured = (energy_captured / total_energy).item()

            if ratio_captured > 0.99:
                return probe / full_rank  # Steep — 64 SVs capture 99%+
            elif ratio_captured > 0.9:
                # Estimate: need ~probe/ratio for 99%
                estimated_rank_99 = int(probe / ratio_captured)
                return min(estimated_rank_99 / full_rank, 1.0)
            else:
                return 0.9  # Very flat — 64 SVs don't even get 90%
    except Exception:
        return 0.5

    return 0.5


def compress_weight(
    name: str,
    weight: torch.Tensor,
    config: PipelineConfig,
    profile: Optional[LayerProfile] = None,
) -> tuple:
    """Compress a single weight tensor using the optimal strategy.

    Routes between two paths based on spectrum flatness:
      - Steep spectrum (low effective rank) → SVD + binarize + codebook
      - Flat spectrum → Direct low-bit quantization (INT2-8)

    Both paths are quality-gated: if the result doesn't meet the target
    cosine similarity, we escalate (more bits or higher SVD rank).

    When both paths are enabled, we try SVD first for steep-spectrum
    matrices since it achieves lower BPW at the same quality. If SVD
    fails the quality gate, we fall back to direct quant.
    """
    original_shape = tuple(weight.shape)
    original_bytes = weight.numel() * 2
    device = config.device

    layer = CompressedLayer(name=name, original_shape=original_shape)

    # Skip tiny or 1D tensors
    if weight.numel() < config.min_tensor_size or weight.ndim < 2:
        layer.method = "skip"
        quality = {"mse": 0.0, "cosine_sim": 1.0, "relative_error": 0.0}
        result = CompressionResult(
            name=name, original_shape=original_shape,
            original_bytes=original_bytes,
            compressed_bytes=weight.numel() * (1 if weight.ndim < 2 else 2),
            **quality,
        )
        return result, layer

    w = weight.float().to(device)

    # Profile
    if config.enable_profiling and profile is None:
        profile = profile_layer(name, weight, config.target_bpw)
    layer.profile = profile

    # --- Path 0: Product Quantization (flagship — tried first) ---
    # PQ achieves sub-0.1 BPW with combinatorial codebook expressiveness.
    # Try configs in order of ascending BPW, stop when quality gate met.
    pq_result = None
    pq_layer = None
    if config.enable_pq and weight.numel() >= 256:
        import math as _math
        pq_configs_sorted = sorted(
            config.pq_configs,
            key=lambda c: c[0] * _math.log2(max(c[1], 2)) / c[2]
        )

        for M, K, G in pq_configs_sorted:
            if G % M != 0 or G // M < 2:
                continue
            if weight.numel() < G * 4:
                continue

            try:
                pq = product_quantize(
                    weight, n_subvectors=M, codebook_size=K,
                    group_size=G, n_iter=config.pq_n_iter,
                )
                compressed_bytes = pq.storage_bytes()
                reconstructed = pq.decompress().to(device)
                if reconstructed.shape != w.shape:
                    reconstructed = reconstructed.reshape(w.shape)
                quality = compute_quality(w, reconstructed)

                candidate_layer = CompressedLayer(
                    name=name, original_shape=original_shape, profile=profile,
                    method=f"pq_m{M}_k{K}_g{G}", pq=pq,
                )
                candidate_result = CompressionResult(
                    name=name, original_shape=original_shape,
                    original_bytes=original_bytes,
                    compressed_bytes=max(compressed_bytes, 1),
                    **quality,
                    stage_details={"method": "pq", "M": M, "K": K, "G": G},
                )

                if pq_result is None or quality["cosine_sim"] > pq_result.cosine_sim:
                    pq_result = candidate_result
                    pq_layer = candidate_layer

                if quality["cosine_sim"] >= config.target_cosine_sim:
                    return candidate_result, candidate_layer

            except Exception:
                continue

    # --- Decide strategy: SVD vs Direct Quant ---
    use_svd = False
    flatness = 1.0
    if config.enable_direct_quant and config.enable_factorization:
        flatness = estimate_spectrum_flatness(weight, device)
        use_svd = flatness < config.svd_skip_threshold
    elif config.enable_factorization:
        use_svd = True

    # --- Path 1: SVD + binarize + codebook (steep spectrum) ---
    svd_result = None
    svd_layer = None
    if use_svd:
        svd_layer = CompressedLayer(name=name, original_shape=original_shape, profile=profile)
        try:
            compressed_bytes, reconstructed = _compress_svd(w, svd_layer, config)
            if reconstructed.shape != w.shape:
                reconstructed = reconstructed.reshape(w.shape)
            quality = compute_quality(w, reconstructed)
            svd_layer.method = "svd_binary"
            svd_result = CompressionResult(
                name=name, original_shape=original_shape,
                original_bytes=original_bytes,
                compressed_bytes=max(compressed_bytes, 1),
                **quality,
                stage_details={"method": "svd_binary", "flatness": flatness},
            )
        except Exception:
            svd_result = None

    # If SVD met the quality gate, use it (it's typically lower BPW)
    if svd_result is not None and svd_result.cosine_sim >= config.target_cosine_sim:
        return svd_result, svd_layer

    # --- Path 1b: SVD + VQ fusion (the sub-0.1 BPW path) ---
    # Factorize W ≈ U@V, then VQ the smaller factors instead of the full matrix.
    # This is where extreme compression lives: rank 64 on a 4096x4096 matrix
    # means VQ operates on 64*(4096+4096) = 524K elements instead of 16M.
    svd_vq_result = None
    svd_vq_layer = None
    if config.enable_vq and config.enable_factorization and weight.ndim >= 2:
        svd_vq_layer = CompressedLayer(name=name, original_shape=original_shape, profile=profile)
        try:
            compressed_bytes, reconstructed = _compress_svd_vq(w, svd_vq_layer, config)
            if reconstructed.shape != w.shape:
                reconstructed = reconstructed.reshape(w.shape)
            quality = compute_quality(w, reconstructed)
            svd_vq_result = CompressionResult(
                name=name, original_shape=original_shape,
                original_bytes=original_bytes,
                compressed_bytes=max(compressed_bytes, 1),
                **quality,
                stage_details={"method": svd_vq_layer.method, "flatness": flatness},
            )
            if svd_vq_result.cosine_sim >= config.target_cosine_sim:
                return svd_vq_result, svd_vq_layer
        except Exception:
            svd_vq_result = None

    # --- Path 2: Direct Vector Quantization (sub-1 BPW) ---
    # Try VQ configs in order of ascending BPW. Use binary search over group
    # sizes: if G=16 (lowest BPW) meets the gate, skip larger configs.
    # If it doesn't, jump to the next size.
    vq_result = None
    vq_layer = None
    if config.enable_vq and weight.numel() >= config.vq_codebook_size:
        import math
        # Build configs sorted by BPW ascending
        vq_configs = []
        for gs in config.vq_group_sizes:
            for n_levels in range(1, config.vq_residual_levels + 1):
                bpw_est = math.log2(max(config.vq_codebook_size, 2)) / gs * n_levels
                vq_configs.append((bpw_est, gs, n_levels))
        vq_configs.sort()

        # Scale iterations: fewer for large tensors, more for small
        n_groups_est = weight.numel() / min(gs for _, gs, _ in vq_configs)
        vq_iter = config.vq_n_iter if n_groups_est < 50000 else max(6, config.vq_n_iter // 2)

        for bpw_est, gs, n_levels in vq_configs:
            try:
                # Quick probe: run VQ on a few complete rows to estimate quality.
                # If the probe can't get close to the gate, skip the full run.
                if weight.numel() > 100000 and weight.ndim >= 2:
                    # Take a subset of complete rows (preserves matrix structure)
                    n_probe_rows = max(4, min(weight.shape[0] // 4, 64))
                    probe_w = weight[:n_probe_rows]
                    try:
                        probe_vq = quantize_vector_codebook(
                            probe_w, codebook_size=config.vq_codebook_size,
                            group_size=gs, n_iter=max(3, vq_iter // 3),
                            n_residual_levels=n_levels,
                        )
                        probe_recon = probe_vq.decompress().to(device)
                        probe_w_dev = probe_w.float().to(device)
                        if probe_recon.shape != probe_w_dev.shape:
                            probe_recon = probe_recon.reshape(probe_w_dev.shape)
                        probe_q = compute_quality(probe_w_dev, probe_recon)
                        # If probe is far below gate, skip
                        if probe_q["cosine_sim"] < config.target_cosine_sim - 0.03:
                            continue
                    except Exception:
                        pass  # Probe failed, try full run anyway

                vq = quantize_vector_codebook(
                    weight, codebook_size=config.vq_codebook_size,
                    group_size=gs, n_iter=vq_iter,
                    n_residual_levels=n_levels,
                )
                compressed_bytes = vq.storage_bytes()
                reconstructed = vq.decompress().to(device)
                if reconstructed.shape != w.shape:
                    reconstructed = reconstructed.reshape(w.shape)
                quality = compute_quality(w, reconstructed)

                candidate = CompressedLayer(
                    name=name, original_shape=original_shape, profile=profile,
                    method=f"vq_k{config.vq_codebook_size}_g{gs}_r{n_levels}",
                    vq=vq,
                )
                candidate_result = CompressionResult(
                    name=name, original_shape=original_shape,
                    original_bytes=original_bytes,
                    compressed_bytes=max(compressed_bytes, 1),
                    **quality,
                    stage_details={
                        "method": "vq", "codebook_size": config.vq_codebook_size,
                        "group_size": gs, "residual_levels": n_levels,
                        "flatness": flatness,
                    },
                )

                if vq_result is None or quality["cosine_sim"] > vq_result.cosine_sim:
                    vq_result = candidate_result
                    vq_layer = candidate

                if quality["cosine_sim"] >= config.target_cosine_sim:
                    if (svd_result is not None
                            and svd_result.cosine_sim >= config.target_cosine_sim
                            and svd_result.compressed_bytes < compressed_bytes):
                        return svd_result, svd_layer
                    return candidate_result, candidate
            except Exception:
                continue

    if vq_result is not None and vq_result.cosine_sim >= config.target_cosine_sim:
        return vq_result, vq_layer

    # --- Path 3: Direct scalar quantization (fallback) ---
    # Collect all candidates so far
    candidates = []
    if pq_result is not None:
        candidates.append((pq_result, pq_layer))
    if svd_result is not None:
        candidates.append((svd_result, svd_layer))
    if svd_vq_result is not None:
        candidates.append((svd_vq_result, svd_vq_layer))
    if vq_result is not None:
        candidates.append((vq_result, vq_layer))

    if config.enable_direct_quant:
        bit_sequence = sorted(set([config.quant_bits, 3, 4, 5, 6, 8]))
        for bits in bit_sequence:
            quantized = quantize_absmax(
                weight, bits=bits, group_size=config.quant_group_size,
            )
            quant_layer = CompressedLayer(
                name=name, original_shape=original_shape, profile=profile,
                method=f"int{bits}", quantized=quantized,
            )
            compressed_bytes = quantized.storage_bytes()
            reconstructed = quantized.decompress().to(device)

            if reconstructed.shape != w.shape:
                reconstructed = reconstructed.reshape(w.shape)

            quality = compute_quality(w, reconstructed)
            quant_result = CompressionResult(
                name=name, original_shape=original_shape,
                original_bytes=original_bytes,
                compressed_bytes=max(compressed_bytes, 1),
                **quality,
                stage_details={"method": f"int{bits}", "bits": bits, "flatness": flatness},
            )
            candidates.append((quant_result, quant_layer))

            if quality["cosine_sim"] >= config.target_cosine_sim:
                # Gate met — pick the smallest candidate that also meets the gate
                gate_passing = [
                    (r, l) for r, l in candidates
                    if r.cosine_sim >= config.target_cosine_sim
                ]
                if gate_passing:
                    return min(gate_passing, key=lambda x: x[0].compressed_bytes)
                return quant_result, quant_layer

    # No path met the gate — return the one with best quality
    if candidates:
        return max(candidates, key=lambda x: x[0].cosine_sim)
    return best_result, best_layer


def _compress_svd(w: torch.Tensor, layer: CompressedLayer, config: PipelineConfig) -> tuple:
    """SVD + binarize + codebook path. Returns (compressed_bytes, reconstructed)."""
    compressed_bytes = 0
    original_shape = tuple(w.shape)

    if config.enable_residual and config.residual_levels > 1:
        residual_fact = factorize_residual(
            w, n_levels=config.residual_levels,
            min_rank=config.min_rank, max_rank=config.max_rank,
            device=config.device,
        )
        layer.residual_codebooks = []

        for level in residual_fact.levels:
            if config.enable_binarization:
                binarized = binarize_weight(
                    level.U, level.V, rank=level.rank,
                    original_shape=original_shape,
                    group_size=config.group_size,
                    use_rotation=config.use_rotation,
                    use_sigma_delta=config.use_sigma_delta,
                )
                if config.enable_codebook:
                    cb_U = compress_binarized_factor(binarized.U_bin, config.codebook_size)
                    cb_V = compress_binarized_factor(binarized.V_bin, config.codebook_size)
                    compressed_bytes += cb_U.storage_bytes() + cb_V.storage_bytes()
                    layer.residual_codebooks.append((cb_U, cb_V))
                else:
                    compressed_bytes += binarized.storage_bytes()
            else:
                compressed_bytes += (level.U.numel() + level.V.numel()) * 2

        # Reconstruct
        with torch.no_grad():
            reconstructed = torch.zeros_like(w)
            if layer.residual_codebooks:
                for cb_U, cb_V in layer.residual_codebooks:
                    reconstructed = reconstructed + cb_U.decompress() @ cb_V.decompress()
            else:
                reconstructed = residual_fact.reconstruct()
    else:
        factorized = factorize_weight(
            w, energy_target=config.energy_target,
            min_rank=config.min_rank, max_rank=config.max_rank,
            device=config.device,
        )
        if config.enable_binarization:
            binarized = binarize_weight(
                factorized.U, factorized.V, rank=factorized.rank,
                original_shape=original_shape,
                group_size=config.group_size,
                use_rotation=config.use_rotation,
                use_sigma_delta=config.use_sigma_delta,
            )
            layer.binarized = binarized
            if config.enable_codebook:
                cb_U = compress_binarized_factor(binarized.U_bin, config.codebook_size)
                cb_V = compress_binarized_factor(binarized.V_bin, config.codebook_size)
                layer.codebook_U = cb_U
                layer.codebook_V = cb_V
                compressed_bytes = cb_U.storage_bytes() + cb_V.storage_bytes()
            else:
                compressed_bytes = binarized.storage_bytes()
        else:
            compressed_bytes = (factorized.U.numel() + factorized.V.numel()) * 2

        with torch.no_grad():
            if layer.codebook_U is not None:
                reconstructed = layer.codebook_U.decompress() @ layer.codebook_V.decompress()
            elif layer.binarized is not None:
                reconstructed = layer.binarized.decompress().to(config.device)
            else:
                reconstructed = factorized.U @ factorized.V

    return compressed_bytes, reconstructed


def _compress_svd_vq(w: torch.Tensor, layer: CompressedLayer, config: PipelineConfig) -> tuple:
    """SVD + VQ fusion: factorize first, then vector-quantize the factors.

    This is the key insight behind AQLM/QuIP#:
      W ≈ U @ V   where U is (m, r) and V is (r, n)
      Instead of storing U and V in FP16, vector-quantize both.

    BPW calculation:
      U has m*r elements, V has r*n elements, total = r*(m+n)
      Original W has m*n elements
      If we VQ the factors at B bits per element:
        compressed_bpw = B * r * (m + n) / (m * n)
      For r << min(m,n), this is much less than B.

    Example: 4096x4096 matrix, rank 64, VQ at 2 BPW on factors:
      Factor elements: 64 * (4096 + 4096) = 524,288
      Original elements: 16,777,216
      Effective BPW: 2 * 524,288 / 16,777,216 = 0.063 BPW
    """
    original_shape = tuple(w.shape)

    # Step 1: SVD factorization (adaptive rank for energy target)
    factorized = factorize_weight(
        w, energy_target=config.energy_target,
        min_rank=config.min_rank, max_rank=config.max_rank,
        device=config.device,
    )

    # Step 2: VQ the factors
    # Try multiple VQ configs on the factors, pick the one that meets quality gate
    import math
    best_bytes = float('inf')
    best_recon = None
    best_vq_U = None
    best_vq_V = None
    best_label = None

    for gs in config.vq_group_sizes:
        for n_levels in range(1, config.vq_residual_levels + 1):
            try:
                vq_U = quantize_vector_codebook(
                    factorized.U, codebook_size=config.vq_codebook_size,
                    group_size=gs, n_iter=config.vq_n_iter,
                    n_residual_levels=n_levels,
                )
                vq_V = quantize_vector_codebook(
                    factorized.V, codebook_size=config.vq_codebook_size,
                    group_size=gs, n_iter=config.vq_n_iter,
                    n_residual_levels=n_levels,
                )

                compressed_bytes = vq_U.storage_bytes() + vq_V.storage_bytes()

                with torch.no_grad():
                    U_recon = vq_U.decompress().to(config.device)
                    V_recon = vq_V.decompress().to(config.device)
                    if U_recon.shape != factorized.U.shape:
                        U_recon = U_recon.reshape(factorized.U.shape)
                    if V_recon.shape != factorized.V.shape:
                        V_recon = V_recon.reshape(factorized.V.shape)
                    reconstructed = U_recon @ V_recon

                quality = compute_quality(w, reconstructed.reshape(w.shape))

                if quality["cosine_sim"] >= config.target_cosine_sim and compressed_bytes < best_bytes:
                    best_bytes = compressed_bytes
                    best_recon = reconstructed
                    best_vq_U = vq_U
                    best_vq_V = vq_V
                    best_label = f"svd_vq_r{factorized.rank}_g{gs}_r{n_levels}"
                    break  # This group size works, no need for more residual levels
                elif best_recon is None or compressed_bytes < best_bytes:
                    # Track best even if below gate (might be best we can do)
                    best_bytes = compressed_bytes
                    best_recon = reconstructed
                    best_vq_U = vq_U
                    best_vq_V = vq_V
                    best_label = f"svd_vq_r{factorized.rank}_g{gs}_r{n_levels}"
            except Exception:
                continue

        if best_recon is not None:
            q = compute_quality(w, best_recon.reshape(w.shape))
            if q["cosine_sim"] >= config.target_cosine_sim:
                break  # Found a passing config, stop searching

    if best_recon is None:
        raise RuntimeError("SVD+VQ failed to produce any result")

    layer.method = best_label
    return best_bytes, best_recon.reshape(original_shape)


def compress_model(
    named_weights: list,
    config: PipelineConfig = None,
) -> ModelCompressionReport:
    """Compress all weight tensors in a model."""
    if config is None:
        config = PipelineConfig()

    report = ModelCompressionReport()

    if config.enable_profiling:
        profiles = profile_model(named_weights, config.target_bpw)
        profile_map = {p.name: p for p in profiles}
    else:
        profile_map = {}

    print(f"\nCompressing {len(named_weights)} tensors")
    print(f"  Target BPW: {config.target_bpw}")
    print(f"  Quality gate: cosine_sim >= {config.target_cosine_sim}")
    print(f"  Paths: PQ={'ON' if config.enable_pq else 'OFF'}"
          f"  SVD={'ON' if config.enable_factorization else 'OFF'}"
          f"  VQ={'ON' if config.enable_vq else 'OFF'}"
          f"  Scalar={'ON' if config.enable_direct_quant else 'OFF'}")
    if config.enable_vq:
        print(f"  VQ: K={config.vq_codebook_size} G={config.vq_group_sizes} R={config.vq_residual_levels}")
    print()

    for name, weight in tqdm(named_weights, desc="Compressing"):
        profile = profile_map.get(name)
        result, layer = compress_weight(name, weight, config, profile)
        report.layers.append(result)

    return report
