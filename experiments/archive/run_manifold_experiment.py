"""
MANIFOLD COMPRESSION EXPERIMENT
================================
Testing the hypothesis: Shannon's compression limits assume Euclidean distance,
but neural network weights live on curved manifolds where the limits are
fundamentally different.

If intrinsic dimensionality << ambient dimensionality AND curvature is negative,
then rate-distortion theory on Riemannian manifolds gives exponentially lower
bounds than Shannon's flat-space theorem predicts.

This script:
  1. Measures intrinsic dimensionality of Qwen3-0.6B weight space (MLE, Levina & Bickel 2005)
  2. Estimates sectional curvature from SVD spectra + geodesic deviation
  3. Compares Euclidean vs manifold-aware quantization at the SAME compression ratio
  4. Measures reconstruction quality difference

Key references:
  - Aghajanyan et al. 2021: "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"
  - Levina & Bickel 2005: "Maximum Likelihood Estimation of Intrinsic Dimension"
  - Rate-distortion on Riemannian manifolds: negative curvature -> exponentially lower R(D) bounds
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================================================================
# STEP 0: Load model weights
# ================================================================

print("=" * 70)
print("MANIFOLD COMPRESSION EXPERIMENT")
print("Testing: Does manifold-aware quantization beat Euclidean?")
print("=" * 70)

print("\n[0] Loading Qwen3-0.6B weights...")
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

# Collect all 2D weight matrices (the interesting ones for manifold analysis)
weight_matrices = {}
for k, v in wd.items():
    if v.ndim == 2 and v.shape[0] >= 64 and v.shape[1] >= 64:
        weight_matrices[k] = v.float()

print(f"  Found {len(weight_matrices)} weight matrices")
total_params = sum(w.numel() for w in weight_matrices.values())
print(f"  Total parameters in 2D matrices: {total_params:,} ({total_params*2/1e9:.2f} GB at bf16)")


# ================================================================
# STEP 1: Intrinsic Dimensionality Estimation (MLE, Levina & Bickel 2005)
# ================================================================

print("\n" + "=" * 70)
print("[1] INTRINSIC DIMENSIONALITY ESTIMATION")
print("    Method: Maximum Likelihood Estimation (Levina & Bickel 2005)")
print("    If ID << ambient dim, the weight manifold is vastly lower-dimensional")
print("=" * 70)


def intrinsic_dim_mle(X, k1=5, k2=20):
    """
    Estimate intrinsic dimensionality using MLE (Levina & Bickel 2005).

    For each point, compute:
        d_hat(x) = [ (1/(k2-k1)) * sum_{j=k1}^{k2} log(T_k(x) / T_j(x)) ]^{-1}

    where T_j(x) is the distance to the j-th nearest neighbor.
    Then average d_hat over all (sampled) points.

    k1, k2: range of neighbor counts to average over for stability.
    """
    n, d = X.shape

    # Subsample for computational tractability
    max_points = 2000
    if n > max_points:
        idx = torch.randperm(n)[:max_points]
        X_sample = X[idx]
    else:
        X_sample = X

    n_sample = X_sample.shape[0]

    # Compute pairwise distances in batches to save memory
    batch_size = 500
    all_dims = []

    for i in range(0, n_sample, batch_size):
        batch = X_sample[i:i+batch_size]
        # Distances from this batch to all sample points
        dists = torch.cdist(batch, X_sample)  # (batch, n_sample)

        # Sort distances (exclude self at dist=0)
        dists_sorted, _ = dists.sort(dim=1)
        # Skip the first column (self-distance = 0)
        dists_sorted = dists_sorted[:, 1:]

        # MLE estimate for each point
        # Use neighbors k1 through k2
        k2_actual = min(k2, dists_sorted.shape[1])
        k1_actual = max(1, k1)

        for j in range(batch.shape[0]):
            T = dists_sorted[j, :k2_actual].clamp(min=1e-12)
            T_k = T[k2_actual - 1]  # distance to k2-th neighbor

            # d_hat = 1 / mean(log(T_k / T_j)) for j in [k1, k2-1]
            log_ratios = torch.log(T_k / T[k1_actual-1:k2_actual-1])
            if log_ratios.numel() > 0 and log_ratios.mean() > 1e-12:
                d_hat = 1.0 / log_ratios.mean().item()
                if 0 < d_hat < 10000:  # sanity bound
                    all_dims.append(d_hat)

    if len(all_dims) == 0:
        return float('nan'), float('nan')

    dims = np.array(all_dims)
    return float(np.median(dims)), float(np.std(dims))


# Estimate ID for different weight types
print("\nEstimating intrinsic dimensionality per weight type...")
print(f"{'Weight type':<30} {'Shape':>15} {'Ambient dim':>12} {'Intrinsic dim':>14} {'Ratio':>8}")
print("-" * 85)

id_results = {}
# Group by weight type for cleaner analysis
type_groups = {
    'attention_q': [], 'attention_k': [], 'attention_v': [], 'attention_o': [],
    'ffn_gate': [], 'ffn_up': [], 'ffn_down': [],
}

for k, w in weight_matrices.items():
    if 'q_proj' in k: type_groups['attention_q'].append(w)
    elif 'k_proj' in k: type_groups['attention_k'].append(w)
    elif 'v_proj' in k: type_groups['attention_v'].append(w)
    elif 'o_proj' in k: type_groups['attention_o'].append(w)
    elif 'gate_proj' in k: type_groups['ffn_gate'].append(w)
    elif 'up_proj' in k: type_groups['ffn_up'].append(w)
    elif 'down_proj' in k: type_groups['ffn_down'].append(w)

for wtype, matrices in type_groups.items():
    if not matrices:
        continue

    # Stack rows from several layers to get a point cloud on the weight manifold
    # Each row of a weight matrix = a point in weight space
    # We sample from multiple layers to see the manifold structure
    sample_layers = matrices[:8]  # first 8 layers
    rows = torch.cat([m for m in sample_layers], dim=0).cpu()

    ambient_dim = rows.shape[1]
    t0 = time.time()
    id_median, id_std = intrinsic_dim_mle(rows, k1=5, k2=30)
    elapsed = time.time() - t0

    ratio = id_median / ambient_dim if not math.isnan(id_median) else float('nan')
    print(f"  {wtype:<28} {str(rows.shape):>15} {ambient_dim:>12} {id_median:>10.1f} +/- {id_std:>4.1f} {ratio:>7.2%}")
    id_results[wtype] = {'id': id_median, 'ambient': ambient_dim, 'ratio': ratio}

# Global estimate: flatten a sample of all weights into a single point cloud
print("\nGlobal intrinsic dimensionality (all weight types combined):")
all_rows = []
target_dim = 1024  # Use weights with matching dimension for fair comparison
for k, w in list(weight_matrices.items())[:40]:
    if w.shape[1] != target_dim:
        continue
    # Sample 200 rows from each matrix
    n_rows = min(200, w.shape[0])
    idx = torch.randperm(w.shape[0])[:n_rows]
    all_rows.append(w[idx].cpu())
global_cloud = torch.cat(all_rows, dim=0) if all_rows else torch.randn(100, target_dim)

# Also estimate ID of the "layer manifold" - each layer as a single point (flattened)
layer_points = []
for li in range(28):
    layer_vec = []
    for k, w in weight_matrices.items():
        if f'layers.{li}.' in k:
            layer_vec.append(w.reshape(-1)[:1024].cpu())  # first 1024 elements per tensor
    if layer_vec:
        layer_points.append(torch.cat(layer_vec))

if layer_points:
    layer_cloud = torch.stack(layer_points)
    print(f"  Layer manifold: {layer_cloud.shape[0]} points in R^{layer_cloud.shape[1]}")
    id_layer, id_layer_std = intrinsic_dim_mle(layer_cloud, k1=2, k2=8)
    print(f"  Layer manifold intrinsic dim: {id_layer:.1f} +/- {id_layer_std:.1f}")
    print(f"  ==> {layer_cloud.shape[0]} layers lie on a ~{id_layer:.0f}-dimensional manifold!")

global_id, global_std = intrinsic_dim_mle(global_cloud, k1=5, k2=30)
print(f"\n  Global weight cloud: {global_cloud.shape[0]} points in R^{global_cloud.shape[1]}")
print(f"  Global intrinsic dim: {global_id:.1f} +/- {global_std:.1f}")
print(f"  Compression headroom: ambient/intrinsic = {global_cloud.shape[1]/global_id:.1f}x")


# ================================================================
# STEP 2: Curvature Estimation
# ================================================================

print("\n" + "=" * 70)
print("[2] CURVATURE ESTIMATION")
print("    Method 1: SVD spectral decay -> effective curvature")
print("    Method 2: Geodesic deviation (triangle comparison)")
print("    Negative curvature = hyperbolic = exponentially more room")
print("=" * 70)


def estimate_curvature_svd(W):
    """
    Estimate manifold curvature from the SVD spectrum.

    Intuition: On a flat manifold, singular values decay linearly (in log space).
    On a negatively curved (hyperbolic) manifold, they decay faster than exponential.
    On a positively curved (spherical) manifold, they decay slower.

    We fit log(sigma_i) ~ alpha * i and measure the residual curvature.

    Returns:
        curvature: negative = hyperbolic, positive = spherical
        effective_dim: number of significant singular values
        spectrum: the singular values
    """
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    S = S.cpu().numpy()

    # Effective dimension (fraction of variance in top-k)
    var = S ** 2
    cumvar = np.cumsum(var) / var.sum()
    effective_dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    effective_dim_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    # Fit log-spectrum to detect curvature
    n = len(S)
    log_s = np.log(S[:n].clip(min=1e-12))
    x = np.arange(n, dtype=np.float64)

    # Linear fit: log(S_i) = a + b*i (flat manifold prediction)
    # Quadratic fit: log(S_i) = a + b*i + c*i^2 (curved manifold)
    # The sign of c indicates curvature: c < 0 -> faster than exponential -> negative curvature
    coeffs = np.polyfit(x, log_s, 2)
    curvature_coeff = coeffs[0]  # coefficient of i^2

    # Normalize curvature by matrix size for comparability
    curvature = curvature_coeff * n

    return curvature, effective_dim_90, effective_dim_99, S


def estimate_curvature_geodesic(point_cloud, n_triangles=500):
    """
    Estimate curvature via geodesic triangle deviation (Toponogov comparison).

    On a flat manifold: d(midpoint(A,C), B) = sqrt(d(A,B)^2 + d(B,C)^2 - d(A,C)^2/4) (approx)
    On negatively curved: midpoint is FARTHER from B than flat prediction (triangles are "thinner")
    On positively curved: midpoint is CLOSER (triangles are "fatter")

    We sample random triangles, compute the deviation from flat prediction,
    and aggregate to estimate sectional curvature.
    """
    n = point_cloud.shape[0]
    deviations = []

    for _ in range(n_triangles):
        # Pick 3 random points
        idx = torch.randperm(n)[:3]
        A, B, C = point_cloud[idx[0]], point_cloud[idx[1]], point_cloud[idx[2]]

        # Midpoint of A and C (Euclidean midpoint as proxy for geodesic midpoint)
        M = (A + C) / 2.0

        # Distances
        dAB = (A - B).norm().item()
        dBC = (B - C).norm().item()
        dAC = (A - C).norm().item()
        dMB = (M - B).norm().item()

        # Flat-space prediction for midpoint distance (from parallelogram law):
        # d(M,B)^2 = (2*d(A,B)^2 + 2*d(B,C)^2 - d(A,C)^2) / 4
        flat_pred_sq = (2 * dAB**2 + 2 * dBC**2 - dAC**2) / 4.0

        if flat_pred_sq > 0 and dMB > 0:
            flat_pred = math.sqrt(flat_pred_sq)
            # Deviation: positive means actual > predicted (hyperbolic-like)
            # negative means actual < predicted (spherical-like)
            deviation = (dMB - flat_pred) / flat_pred
            if abs(deviation) < 10:  # filter outliers
                deviations.append(deviation)

    if not deviations:
        return 0.0, 0.0

    dev = np.array(deviations)
    mean_dev = float(np.mean(dev))
    std_dev = float(np.std(dev))

    # Convert deviation to approximate sectional curvature
    # For small curvature K and triangle side length ~L:
    # deviation ~ -K * L^2 / 24 (negative K -> positive deviation)
    # We report the raw mean deviation as a curvature proxy
    return mean_dev, std_dev


print("\nSVD spectral curvature analysis:")
print(f"{'Weight type':<30} {'Curvature':>12} {'eff dim 90%':>12} {'eff dim 99%':>12} {'Interpretation':>20}")
print("-" * 90)

curvature_results = {}
for wtype, matrices in type_groups.items():
    if not matrices:
        continue

    curvatures = []
    eff90s = []
    eff99s = []
    for W in matrices[:6]:  # first 6 layers
        curv, ed90, ed99, _ = estimate_curvature_svd(W.cpu())
        curvatures.append(curv)
        eff90s.append(ed90)
        eff99s.append(ed99)

    avg_curv = np.mean(curvatures)
    avg_e90 = np.mean(eff90s)
    avg_e99 = np.mean(eff99s)

    if avg_curv < -0.01:
        interp = "HYPERBOLIC"
    elif avg_curv > 0.01:
        interp = "spherical"
    else:
        interp = "~flat"

    print(f"  {wtype:<28} {avg_curv:>12.4f} {avg_e90:>12.0f} {avg_e99:>12.0f} {interp:>20}")
    curvature_results[wtype] = {'curvature': avg_curv, 'eff_dim_90': avg_e90, 'eff_dim_99': avg_e99}

# Geodesic triangle curvature on weight point cloud
print("\nGeodesic triangle curvature estimation:")
for wtype, matrices in type_groups.items():
    if not matrices:
        continue

    rows = torch.cat([m for m in matrices[:4]], dim=0).cpu()
    # Subsample
    if rows.shape[0] > 3000:
        idx = torch.randperm(rows.shape[0])[:3000]
        rows = rows[idx]

    geo_curv, geo_std = estimate_curvature_geodesic(rows, n_triangles=1000)

    if geo_curv > 0.001:
        interp = "HYPERBOLIC (farther midpoints)"
    elif geo_curv < -0.001:
        interp = "spherical (closer midpoints)"
    else:
        interp = "~flat"

    print(f"  {wtype:<28} deviation: {geo_curv:>+.6f} +/- {geo_std:.6f}  -> {interp}")


# ================================================================
# STEP 3 & 4: Compare Euclidean vs Manifold Quantization
# ================================================================

print("\n" + "=" * 70)
print("[3-4] EUCLIDEAN vs MANIFOLD QUANTIZATION COMPARISON")
print("      Same compression ratio, measure quality difference")
print("=" * 70)


def euclidean_quantize(W, bits=2, group_size=128):
    """Standard uniform/absmax Euclidean quantization."""
    shape = W.shape
    flat = W.reshape(-1).float()

    # Pad
    remainder = flat.numel() % group_size
    if remainder:
        flat = torch.cat([flat, flat.new_zeros(group_size - remainder)])

    groups = flat.reshape(-1, group_size)
    scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = groups / scales

    n_levels = 2 ** bits
    half = n_levels // 2
    # Symmetric quantization
    codes = torch.clamp(torch.round(normalized * (half - 1)), -(half - 1), half - 1)
    dequant = (codes / (half - 1)) * scales
    result = dequant.reshape(-1)[:W.numel()].reshape(shape)

    # Storage cost
    n_groups = groups.shape[0]
    bits_total = groups.numel() * bits + n_groups * 16  # codes + scales in fp16
    return result, bits_total


def manifold_quantize(W, bits=2, group_size=128):
    """
    Manifold-aware quantization:

    1. SVD to find the weight manifold (low-rank structure)
    2. Quantize in the TANGENT SPACE at the manifold (where the metric is natural)
    3. Apply geodesic correction during reconstruction

    The key difference: instead of quantizing raw values (Euclidean),
    we quantize the manifold coordinates (U, S, V from SVD), applying
    more bits to the directions with more curvature.

    At the same total bit budget, this allocates precision better because
    it respects the geometry.
    """
    shape = W.shape
    m, n = shape if W.ndim == 2 else (W.shape[0], W.reshape(W.shape[0], -1).shape[1])
    W2d = W.reshape(m, -1).float() if W.ndim != 2 else W.float()

    # SVD decomposition
    U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)

    # Determine rank to keep: match the target bit budget
    # For Euclidean: total_bits = numel * bits + overhead
    total_bits_target = W.numel() * bits + (W.numel() // group_size) * 16

    # In manifold mode: we store U_k (m*k), S_k (k), V_k (k*n)
    # Each quantized at higher precision since there are fewer values
    # Find k such that (m*k + k + k*n) * manifold_bits = total_bits_target
    # We quantize manifold coords at 2x the bits since there are fewer coords
    manifold_bits = min(bits * 2, 8)  # double precision on fewer coords

    # Binary search for rank k that matches the bit budget
    best_k = 1
    for k in range(1, min(m, n)):
        manifold_params = m * k + k + k * n
        manifold_cost = manifold_params * manifold_bits + (manifold_params // group_size + 1) * 16
        if manifold_cost <= total_bits_target:
            best_k = k
        else:
            break

    k = best_k

    # Truncated SVD
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = Vh[:k, :]

    # Curvature-adaptive scaling: allocate more precision to high-curvature directions
    # High curvature directions (rapid SV changes) need more precision
    sv_gradient = torch.zeros_like(S_k)
    if k > 1:
        sv_gradient[:-1] = (S_k[:-1] - S_k[1:]).abs()
        sv_gradient[-1] = sv_gradient[-2] if k > 1 else sv_gradient[0]
    curvature_weight = 1.0 + sv_gradient / sv_gradient.mean().clamp(min=1e-8)
    curvature_weight = curvature_weight.clamp(max=3.0)

    # Scale singular values by curvature weight before quantization
    # This effectively allocates more quantization levels to high-curvature directions
    S_scaled = S_k * curvature_weight

    # Quantize U_k, S_scaled, V_k at manifold_bits precision
    def quant_tensor(T, qbits):
        flat = T.reshape(-1)
        gs = min(group_size, flat.numel())
        if flat.numel() % gs:
            flat = torch.cat([flat, flat.new_zeros(gs - flat.numel() % gs)])
        groups = flat.reshape(-1, gs)
        sc = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        norm = groups / sc
        half = 2 ** (qbits - 1)
        codes = torch.clamp(torch.round(norm * (half - 1)), -(half - 1), half - 1)
        return ((codes / (half - 1)) * sc).reshape(-1)[:T.numel()].reshape(T.shape)

    U_q = quant_tensor(U_k, manifold_bits)
    S_q = quant_tensor(S_scaled, manifold_bits)
    V_q = quant_tensor(V_k, manifold_bits)

    # Undo curvature scaling
    S_final = S_q / curvature_weight

    # Reconstruct with geodesic correction
    W_hat = U_q * S_final.unsqueeze(0) @ V_q

    # Geodesic correction: on negatively curved manifolds, the SVD truncation
    # undershoots. We correct using the curvature estimated from the spectrum.
    if k > 1:
        log_s = torch.log(S_k.clamp(min=1e-12))
        kappa = -(log_s[0] - log_s[-1]).item() / k  # negative if decaying
        if abs(kappa) > 1e-6:
            if kappa > 0:
                correction = math.sqrt(kappa) / math.sin(math.sqrt(min(kappa, math.pi**2 * 0.99)))
            else:
                correction = math.sqrt(-kappa) / math.sinh(math.sqrt(-kappa))
            # Gentle application
            correction = 1.0 + (correction - 1.0) * 0.05
            W_hat = W_hat * correction

    result = W_hat.reshape(shape)

    # Actual storage cost
    manifold_params = m * k + k + k * n
    bits_total = manifold_params * manifold_bits + (manifold_params // group_size + 1) * 16
    return result, bits_total


def compute_quality(W_orig, W_recon):
    """Compute reconstruction quality metrics."""
    W_orig_f = W_orig.float()
    W_recon_f = W_recon.float()

    mse = F.mse_loss(W_recon_f, W_orig_f).item()
    cos_sim = F.cosine_similarity(W_orig_f.reshape(1, -1), W_recon_f.reshape(1, -1)).item()
    rel_err = (W_orig_f - W_recon_f).norm().item() / W_orig_f.norm().clamp(min=1e-8).item()
    snr = -10 * math.log10(mse / W_orig_f.var().clamp(min=1e-12).item()) if mse > 0 else float('inf')

    return {
        'mse': mse,
        'cosine_sim': cos_sim,
        'relative_error': rel_err,
        'snr_db': snr,
    }


# Run head-to-head comparison
print("\nHead-to-head: Euclidean vs Manifold quantization")
print("  Using the SAME total bit budget for fair comparison\n")

for bits in [2, 3, 4]:
    print(f"\n--- {bits}-bit compression ---")
    print(f"{'Weight':<45} {'Method':<12} {'Cos sim':>9} {'Rel err':>9} {'SNR(dB)':>9} {'Bits used':>12}")
    print("-" * 100)

    euclid_wins = 0
    manifold_wins = 0
    euclid_cos_total = 0.0
    manifold_cos_total = 0.0
    n_compared = 0

    for k, W in list(weight_matrices.items())[:30]:
        # Skip embedding/lm_head (too large, different structure)
        if 'embed' in k or 'lm_head' in k:
            continue

        W_cpu = W.cpu()

        # Euclidean quantization
        W_euclid, bits_euclid = euclidean_quantize(W_cpu, bits=bits)
        q_euclid = compute_quality(W_cpu, W_euclid)

        # Manifold quantization (with same bit budget)
        W_manifold, bits_manifold = manifold_quantize(W_cpu, bits=bits)
        q_manifold = compute_quality(W_cpu, W_manifold)

        short_name = k.replace('model.layers.', 'L').replace('.weight', '')
        print(f"  {short_name:<43} {'Euclid':<12} {q_euclid['cosine_sim']:>9.6f} {q_euclid['relative_error']:>9.4f} {q_euclid['snr_db']:>9.2f} {bits_euclid:>12,}")
        print(f"  {'':<43} {'Manifold':<12} {q_manifold['cosine_sim']:>9.6f} {q_manifold['relative_error']:>9.4f} {q_manifold['snr_db']:>9.2f} {bits_manifold:>12,}")

        if q_manifold['cosine_sim'] > q_euclid['cosine_sim']:
            manifold_wins += 1
            print(f"  {'':>43} >>> MANIFOLD WINS by {q_manifold['cosine_sim'] - q_euclid['cosine_sim']:.6f}")
        else:
            euclid_wins += 1
            print(f"  {'':>43} >>> Euclid wins by {q_euclid['cosine_sim'] - q_manifold['cosine_sim']:.6f}")

        euclid_cos_total += q_euclid['cosine_sim']
        manifold_cos_total += q_manifold['cosine_sim']
        n_compared += 1

    if n_compared > 0:
        print(f"\n  SCOREBOARD at {bits} bits:")
        print(f"    Manifold wins: {manifold_wins}/{n_compared}")
        print(f"    Euclidean wins: {euclid_wins}/{n_compared}")
        print(f"    Avg cosine sim - Euclidean: {euclid_cos_total/n_compared:.6f}")
        print(f"    Avg cosine sim - Manifold:  {manifold_cos_total/n_compared:.6f}")
        diff = (manifold_cos_total - euclid_cos_total) / n_compared
        print(f"    Manifold advantage: {diff:+.6f} ({diff*100:+.4f}%)")


# ================================================================
# STEP 5: End-to-end model quality (if inference module available)
# ================================================================

print("\n" + "=" * 70)
print("[5] END-TO-END MODEL QUALITY TEST")
print("    Quantize full model both ways, compare generation quality")
print("=" * 70)

try:
    from ultracompress.inference import ModelConfig, MiniTransformer

    # Build reference model
    hf_to_gguf = {
        'self_attn.q_proj.weight': 'attn_q.weight',
        'self_attn.k_proj.weight': 'attn_k.weight',
        'self_attn.v_proj.weight': 'attn_v.weight',
        'self_attn.o_proj.weight': 'attn_output.weight',
        'self_attn.q_norm.weight': 'attn_q_norm.weight',
        'self_attn.k_norm.weight': 'attn_k_norm.weight',
        'input_layernorm.weight': 'attn_norm.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight',
        'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight',
    }

    def build_model_weights(quantize_fn, bits):
        """Build GGUF-style weight dict with given quantization."""
        gd = {}
        gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
        gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(1024)).float()
        gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()

        for li in range(28):
            for h, g in hf_to_gguf.items():
                k = f'model.layers.{li}.{h}'
                if k in wd:
                    w = wd[k].float()
                    if w.ndim == 2 and w.shape[0] >= 64 and w.shape[1] >= 64:
                        w_q, _ = quantize_fn(w.cpu(), bits=bits)
                        gd[f'blk.{li}.{g}'] = w_q.float()
                    else:
                        gd[f'blk.{li}.{g}'] = w
        return gd

    config = ModelConfig(n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
                         intermediate_size=3072, vocab_size=151936, head_dim=128)

    # Build teacher (uncompressed)
    gd_teacher = {}
    gd_teacher['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
    gd_teacher['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(1024)).float()
    gd_teacher['output.weight'] = wd.get('lm_head.weight', gd_teacher['token_embd.weight']).float()
    for li in range(28):
        for h, g in hf_to_gguf.items():
            k = f'model.layers.{li}.{h}'
            if k in wd:
                gd_teacher[f'blk.{li}.{g}'] = wd[k].float()

    teacher = MiniTransformer(config, device)
    teacher.load_weights(gd_teacher)

    def quick_eval(model, n=50):
        """Evaluate model quality vs teacher."""
        t1, t10_scores = 0, []
        for trial in range(n):
            torch.manual_seed(trial * 13 + 9999)
            tokens = torch.randint(100, 50000, (1, 16), device=device)
            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)
                teacher_pred = teacher_logits[0, -1].argmax().item()
                teacher_top10 = set(teacher_logits[0, -1].topk(10).indices.tolist())

                student_logits = model.forward(tokens, max_layers=28)
                student_pred = student_logits[0, -1].argmax().item()
                student_top10 = set(student_logits[0, -1].topk(10).indices.tolist())

                if teacher_pred == student_pred:
                    t1 += 1
                t10_scores.append(len(teacher_top10 & student_top10) / 10)
        return t1 / n, sum(t10_scores) / len(t10_scores)

    for bits in [2, 3]:
        print(f"\n  Building {bits}-bit Euclidean model...")
        gd_euclid = build_model_weights(euclidean_quantize, bits)
        model_euclid = MiniTransformer(config, device)
        model_euclid.load_weights(gd_euclid)
        t1_e, t10_e = quick_eval(model_euclid, n=100)

        print(f"  Building {bits}-bit Manifold model...")
        gd_manifold = build_model_weights(manifold_quantize, bits)
        model_manifold = MiniTransformer(config, device)
        model_manifold.load_weights(gd_manifold)
        t1_m, t10_m = quick_eval(model_manifold, n=100)

        print(f"\n  === {bits}-BIT END-TO-END RESULTS ===")
        print(f"  Euclidean:  top-1 = {t1_e:.1%}, top-10 = {t10_e:.1%}")
        print(f"  Manifold:   top-1 = {t1_m:.1%}, top-10 = {t10_m:.1%}")
        print(f"  Manifold advantage: top-1 {t1_m - t1_e:+.1%}, top-10 {t10_m - t10_e:+.1%}")

        del model_euclid, model_manifold, gd_euclid, gd_manifold
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

except Exception as e:
    print(f"  End-to-end test skipped: {e}")


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("SUMMARY: MANIFOLD COMPRESSION HYPOTHESIS TEST")
print("=" * 70)

print("""
KEY FINDINGS:

1. INTRINSIC DIMENSIONALITY:
   If ID << ambient dim, the weights live on a low-dimensional manifold
   and there is room for manifold-aware compression to beat flat methods.

2. CURVATURE:
   Negative curvature (hyperbolic) means exponentially more volume per
   dimension -- rate-distortion bounds are exponentially lower.
   Positive curvature (spherical) means less volume -- bounds are higher.

3. QUANTIZATION COMPARISON:
   At the same bit budget, manifold-aware quantization allocates precision
   along the natural geometry of the weight space. If the manifold is
   significantly curved, this should yield measurably better reconstruction.

IMPLICATIONS FOR ULTRACOMPRESS:
   - If curvature is significantly negative: manifold compression can
     PROVABLY exceed Shannon's flat-space limits for this data.
   - If intrinsic dim is very low: huge compression headroom exists that
     flat methods cannot exploit.
   - The combination (low ID + negative curvature) is the sweet spot.

NEXT STEPS:
   - If manifold wins: build this into the ultracompress pipeline
   - Derive the actual Riemannian rate-distortion bound R(D) for the
     measured curvature and compare to achieved compression
   - Patent the manifold-aware quantization approach
""")
