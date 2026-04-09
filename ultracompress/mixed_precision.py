"""
Mixed-Precision Product Quantization — The Key to Usable Extreme Compression

The problem with uniform PQ at 0.016 BPW:
  Per-weight cosine is ~0.91, which sounds okay. But after propagating
  activations through 36 transformer layers, errors compound multiplicatively:
    0.91^36 = 0.03 — total signal destruction.

The insight: not all layers matter equally.
  - Embeddings: map tokens to semantic space. Corrupt these and everything
    downstream is garbage. Need 0.99+ cosine.
  - First/last transformer layers: bottleneck layers that set up and read out
    the residual stream. High importance.
  - Attention Q/K projections: determine what attends to what. Errors here
    cause catastrophic attention pattern collapse.
  - Norm layers: tiny (a few thousand params), keep at FP16 for free.
  - Middle FFN layers: massively redundant. The residual stream carries
    information around them. Can tolerate extreme compression.

By giving 20% of parameters 10x more bits, we keep the critical path at
0.99+ cosine while the overall BPW stays low. The propagation quality
becomes: 0.99^8 * 0.91^28 = 0.07 -> still bad with uniform middle layers.

But with mixed precision the effective propagation uses the residual stream,
so middle-layer errors are additive (not multiplicative). The real math:
  residual_out = residual_in + layer(residual_in)
  Error in layer() is damped by the residual connection.

This module implements the full mixed-precision PQ pipeline.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
from .product_quantize import product_quantize, ProductQuantized
from .metrics import compute_quality


# ---------------------------------------------------------------------------
# Layer importance scoring
# ---------------------------------------------------------------------------

@dataclass
class LayerImportance:
    """Importance score and classification for a single tensor."""
    name: str
    layer_type: str       # "embedding", "attention_qk", "attention_vo", "ffn", "norm", "output", "other"
    layer_index: int      # -1 for non-layer tensors (embeddings, output head)
    n_layers_total: int   # total transformer layers in model
    importance: float     # raw importance multiplier
    n_params: int


class LayerImportanceScorer:
    """Assigns importance scores to tensors based on name/position/type.

    Scoring rules:
      - Embeddings (token_embd):           5x  — semantic grounding
      - Norm layers (norm, ln):            10x — kept FP16 (tiny, critical)
      - First transformer layer:           3x  — residual stream setup
      - Last transformer layer:            3x  — residual stream readout
      - Attention Q/K projections:         2x  — attention pattern fidelity
      - Output head (output, lm_head):     3x  — logit accuracy
      - Everything else (FFN, attn V/O):   1x  — redundant, compressible
    """

    def __init__(self):
        pass

    def classify_tensor(self, name: str) -> Tuple[str, int]:
        """Return (layer_type, layer_index) from tensor name.

        Handles common naming conventions:
          - blk.{N}.attn_q, blk.{N}.attn_k      -> attention_qk
          - blk.{N}.attn_v, blk.{N}.attn_output  -> attention_vo
          - blk.{N}.ffn_*                         -> ffn
          - blk.{N}.attn_norm, blk.{N}.ffn_norm   -> norm
          - token_embd                            -> embedding
          - output                                -> output
        """
        name_lower = name.lower()

        # Extract layer index if present
        layer_idx = -1
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
                break

        # Classify
        if "token_embd" in name_lower or "embed" in name_lower:
            return "embedding", -1
        if "output" in name_lower and layer_idx == -1:
            return "output", -1
        if "lm_head" in name_lower:
            return "output", -1

        # Norms
        if "norm" in name_lower or "_ln" in name_lower:
            return "norm", layer_idx

        # Attention Q/K
        if "attn_q" in name_lower or "attn_k" in name_lower:
            # Match attn_q and attn_k but not attn_qkv (which is fused)
            if "attn_qkv" not in name_lower:
                return "attention_qk", layer_idx
        if ".q_proj" in name_lower or ".k_proj" in name_lower:
            return "attention_qk", layer_idx

        # Attention V/O
        if "attn_v" in name_lower or "attn_output" in name_lower:
            return "attention_vo", layer_idx
        if ".v_proj" in name_lower or ".o_proj" in name_lower:
            return "attention_vo", layer_idx

        # FFN
        if "ffn" in name_lower or "mlp" in name_lower or "feed_forward" in name_lower:
            return "ffn", layer_idx

        # Fused QKV
        if "attn_qkv" in name_lower:
            return "attention_qk", layer_idx  # treat fused QKV as high importance

        return "other", layer_idx

    def score(self, name: str, shape: tuple, n_layers_total: int) -> LayerImportance:
        """Compute importance for a tensor."""
        layer_type, layer_idx = self.classify_tensor(name)
        n_params = int(np.prod(shape))

        # Base importance by type
        type_multipliers = {
            "embedding":    5.0,
            "norm":         10.0,  # will be kept FP16
            "attention_qk": 2.0,
            "attention_vo": 1.0,
            "ffn":          1.0,
            "output":       3.0,
            "other":        1.0,
        }
        importance = type_multipliers.get(layer_type, 1.0)

        # Position multiplier: first and last layers get 3x boost
        if layer_idx >= 0 and n_layers_total > 0:
            if layer_idx == 0 or layer_idx == n_layers_total - 1:
                importance *= 3.0
            elif layer_idx == 1 or layer_idx == n_layers_total - 2:
                importance *= 1.5  # second/penultimate get mild boost

        return LayerImportance(
            name=name,
            layer_type=layer_type,
            layer_index=layer_idx,
            n_layers_total=n_layers_total,
            importance=importance,
            n_params=n_params,
        )


# ---------------------------------------------------------------------------
# Mixed-precision bit budget allocation
# ---------------------------------------------------------------------------

@dataclass
class TensorBudget:
    """Compression plan for a single tensor."""
    name: str
    layer_type: str
    importance: float
    n_params: int
    method: str           # "fp16", "pq_high", "pq_medium", "pq_low", "pq_extreme"
    target_bpw: float
    pq_config: Optional[Tuple[int, int, int]] = None  # (n_subvectors, codebook_size, group_size)


@dataclass
class MixedPrecisionConfig:
    """Bit budget allocation across all tensors to hit a global BPW target.

    The allocator works as follows:
    1. Score every tensor by importance.
    2. Assign compression tiers based on importance score.
    3. Iteratively adjust tier BPW targets until global average hits target.

    Tiers:
      - fp16:        16.0 BPW — norms and tiny tensors
      - pq_high:     1.0-2.0 BPW — embeddings, first/last layer
      - pq_medium:   0.25-0.5 BPW — attention Q/K
      - pq_low:      0.05-0.15 BPW — middle FFN/attention V/O
      - pq_extreme:  0.01-0.05 BPW — most compressible middle layers
    """
    global_target_bpw: float
    tensor_budgets: Dict[str, TensorBudget] = field(default_factory=dict)
    total_params: int = 0
    total_bits: float = 0.0

    @property
    def achieved_bpw(self) -> float:
        if self.total_params == 0:
            return 0.0
        return self.total_bits / self.total_params

    def summary(self) -> str:
        lines = ["Mixed-Precision Budget Allocation", "=" * 70]
        lines.append(f"{'Name':<45} {'Type':<14} {'Imp':>5} {'Method':<12} {'BPW':>6}")
        lines.append("-" * 70)
        for tb in sorted(self.tensor_budgets.values(), key=lambda x: -x.importance):
            short_name = tb.name[-42:] if len(tb.name) > 42 else tb.name
            lines.append(
                f"{short_name:<45} {tb.layer_type:<14} {tb.importance:5.1f} "
                f"{tb.method:<12} {tb.target_bpw:6.3f}"
            )
        lines.append("-" * 70)
        lines.append(f"Global target: {self.global_target_bpw:.3f} BPW, "
                      f"Achieved: {self.achieved_bpw:.3f} BPW, "
                      f"Total params: {self.total_params:,}")
        return "\n".join(lines)


# PQ configurations for each tier: (n_subvectors, codebook_size, group_size) -> approx BPW
# BPW ~ (M * log2(K) + 16) / G
PQ_TIER_CONFIGS = {
    "pq_high": [
        # ~2.0 BPW: M=8, K=256, G=64 -> (8*8+16)/64 = 1.25
        # ~1.0 BPW: M=4, K=16, G=32 -> (4*4+16)/32 = 1.0
        (8, 16, 64),    # (8*4+16)/64 = 0.75 BPW
        (4, 16, 32),    # (4*4+16)/32 = 1.0 BPW
        (8, 16, 32),    # (8*4+16)/32 = 1.5 BPW
        (4, 256, 32),   # (4*8+16)/32 = 1.5 BPW
        (8, 256, 64),   # (8*8+16)/64 = 1.25 BPW
    ],
    "pq_medium": [
        # ~0.25-0.5 BPW
        (8, 4, 128),    # (8*2+16)/128 = 0.25 BPW
        (8, 16, 128),   # (8*4+16)/128 = 0.375 BPW
        (8, 4, 64),     # (8*2+16)/64 = 0.5 BPW
        (4, 4, 64),     # (4*2+16)/64 = 0.375 BPW
        (16, 4, 128),   # (16*2+16)/128 = 0.375 BPW
    ],
    "pq_low": [
        # ~0.05-0.15 BPW
        (8, 4, 256),    # (8*2+16)/256 = 0.125 BPW
        (8, 4, 512),    # (8*2+16)/512 = 0.0625 BPW
        (4, 4, 256),    # (4*2+16)/256 = 0.094 BPW
        (16, 4, 256),   # (16*2+16)/256 = 0.188 BPW
        (8, 4, 1024),   # (8*2+16)/1024 = 0.047 BPW
    ],
    "pq_extreme": [
        # ~0.01-0.05 BPW
        (8, 4, 1024),   # (8*2+16)/1024 = 0.047 BPW
        (4, 4, 1024),   # (4*2+16)/1024 = 0.031 BPW
        (8, 4, 2048),   # (8*2+16)/2048 = 0.023 BPW
        (4, 4, 2048),   # (4*2+16)/2048 = 0.016 BPW
        (8, 2, 2048),   # (8*1+16)/2048 = 0.012 BPW
    ],
}


def _estimate_bpw(M: int, K: int, G: int) -> float:
    """Estimate BPW for a PQ config."""
    bits_per_index = np.ceil(np.log2(max(K, 2)))
    return (M * bits_per_index + 16) / G  # +16 for FP16 per-group scale


def _pick_best_pq_config(tier: str, target_bpw: float, n_params: int) -> Tuple[Tuple[int, int, int], float]:
    """Pick the PQ config closest to target_bpw from a tier's options."""
    configs = PQ_TIER_CONFIGS.get(tier, PQ_TIER_CONFIGS["pq_low"])
    best = None
    best_dist = float("inf")
    best_bpw = 0.0
    for M, K, G in configs:
        if n_params < G * 4:
            continue  # not enough params for this group size
        bpw = _estimate_bpw(M, K, G)
        dist = abs(bpw - target_bpw)
        if dist < best_dist:
            best_dist = dist
            best = (M, K, G)
            best_bpw = bpw
    if best is None:
        # Fallback: smallest config
        best = configs[-1]
        best_bpw = _estimate_bpw(*best)
    return best, best_bpw


def allocate_mixed_precision(
    tensors: List[Tuple[str, torch.Tensor]],
    global_target_bpw: float = 0.15,
    n_layers_total: int = 0,
) -> MixedPrecisionConfig:
    """Allocate bit budgets to hit a global BPW target.

    Strategy:
    1. Classify and score every tensor.
    2. Assign tiers based on importance thresholds.
    3. Pick PQ configs per tier.
    4. If over budget, downgrade lowest-importance tensors.
       If under budget, upgrade highest-importance tensors.
    """
    scorer = LayerImportanceScorer()

    # Auto-detect n_layers if not provided
    if n_layers_total <= 0:
        max_idx = -1
        for name, _ in tensors:
            for part in name.split("."):
                if part.isdigit():
                    max_idx = max(max_idx, int(part))
        n_layers_total = max_idx + 1 if max_idx >= 0 else 1

    # Score all tensors
    scored: List[Tuple[LayerImportance, torch.Tensor]] = []
    for name, tensor in tensors:
        imp = scorer.score(name, tuple(tensor.shape), n_layers_total)
        scored.append((imp, tensor))

    # Assign initial tiers based on importance
    config = MixedPrecisionConfig(global_target_bpw=global_target_bpw)

    tier_assignments: Dict[str, Tuple[str, float, LayerImportance]] = {}
    for imp, tensor in scored:
        if imp.layer_type == "norm":
            method = "fp16"
            target = 16.0
            pq_cfg = None
        elif imp.importance >= 10.0:
            method = "pq_high"
            target = 1.0
            pq_cfg, target = _pick_best_pq_config("pq_high", 1.0, imp.n_params)
        elif imp.importance >= 5.0:
            method = "pq_high"
            target = 0.75
            pq_cfg, target = _pick_best_pq_config("pq_high", 0.75, imp.n_params)
        elif imp.importance >= 3.0:
            method = "pq_medium"
            target = 0.375
            pq_cfg, target = _pick_best_pq_config("pq_medium", 0.375, imp.n_params)
        elif imp.importance >= 2.0:
            method = "pq_medium"
            target = 0.25
            pq_cfg, target = _pick_best_pq_config("pq_medium", 0.25, imp.n_params)
        elif imp.importance >= 1.5:
            method = "pq_low"
            target = 0.125
            pq_cfg, target = _pick_best_pq_config("pq_low", 0.125, imp.n_params)
        else:
            method = "pq_extreme"
            target = 0.04
            pq_cfg, target = _pick_best_pq_config("pq_extreme", 0.04, imp.n_params)

        tier_assignments[imp.name] = (method, target, imp, pq_cfg if method != "fp16" else None)

    # Compute current budget
    def compute_total_bits():
        total_bits = 0.0
        total_params = 0
        for name, (method, bpw, imp, pq_cfg) in tier_assignments.items():
            total_bits += imp.n_params * bpw
            total_params += imp.n_params
        return total_bits, total_params

    total_bits, total_params = compute_total_bits()
    current_bpw = total_bits / max(total_params, 1)

    # Iterative adjustment: scale non-FP16 BPW targets to hit global target
    # The adjustable budget is everything except FP16 norms
    if current_bpw > global_target_bpw * 1.01:
        # Over budget: need to reduce. Scale down all tiers proportionally,
        # but protect high-importance tensors more.
        for iteration in range(20):
            total_bits, total_params = compute_total_bits()
            current_bpw = total_bits / max(total_params, 1)
            if current_bpw <= global_target_bpw * 1.05:
                break

            # Downgrade the lowest-importance non-FP16 tensors
            adjustable = [(name, method, bpw, imp, pq_cfg)
                          for name, (method, bpw, imp, pq_cfg) in tier_assignments.items()
                          if method != "fp16"]
            adjustable.sort(key=lambda x: x[3].importance)

            for name, method, bpw, imp, pq_cfg in adjustable[:len(adjustable) // 3 + 1]:
                # Downgrade one tier
                if method == "pq_high":
                    new_method = "pq_medium"
                    new_cfg, new_bpw = _pick_best_pq_config("pq_medium", 0.25, imp.n_params)
                elif method == "pq_medium":
                    new_method = "pq_low"
                    new_cfg, new_bpw = _pick_best_pq_config("pq_low", 0.06, imp.n_params)
                elif method == "pq_low":
                    new_method = "pq_extreme"
                    new_cfg, new_bpw = _pick_best_pq_config("pq_extreme", 0.03, imp.n_params)
                else:
                    # Already extreme — try to go lower
                    new_method = "pq_extreme"
                    new_cfg, new_bpw = _pick_best_pq_config("pq_extreme", 0.016, imp.n_params)
                tier_assignments[name] = (new_method, new_bpw, imp, new_cfg)

    # Build final config
    config.total_params = total_params
    for name, (method, bpw, imp, pq_cfg) in tier_assignments.items():
        config.tensor_budgets[name] = TensorBudget(
            name=name,
            layer_type=imp.layer_type,
            importance=imp.importance,
            n_params=imp.n_params,
            method=method,
            target_bpw=bpw,
            pq_config=pq_cfg,
        )
        config.total_bits += imp.n_params * bpw
    config.total_params = total_params

    return config


# ---------------------------------------------------------------------------
# Compression engine
# ---------------------------------------------------------------------------

@dataclass
class MixedPrecisionResult:
    """Result of mixed-precision compression for one tensor."""
    name: str
    method: str
    target_bpw: float
    actual_bpw: float
    cosine_sim: float
    n_params: int
    compressed: object  # ProductQuantized or raw tensor (for FP16)

    def decompress(self) -> torch.Tensor:
        if self.method == "fp16":
            return self.compressed.float()
        return self.compressed.decompress()


def mixed_precision_compress(
    tensors: Iterator[Tuple[str, torch.Tensor]],
    global_target_bpw: float = 0.15,
    n_iter: int = 20,
    verbose: bool = True,
) -> List[MixedPrecisionResult]:
    """Compress a model with mixed-precision PQ.

    Args:
        tensors: iterator of (name, weight) pairs
        global_target_bpw: target overall bits per weight
        n_iter: k-means iterations for PQ
        verbose: print per-tensor results

    Returns:
        List of MixedPrecisionResult, one per tensor
    """
    # Materialize tensors for two-pass processing
    tensor_list = list(tensors)
    if not tensor_list:
        return []

    # Allocate bit budgets
    config = allocate_mixed_precision(tensor_list, global_target_bpw)

    if verbose:
        print(config.summary())
        print()

    results = []
    total_original_bits = 0
    total_compressed_bits = 0

    for name, weight in tensor_list:
        budget = config.tensor_budgets.get(name)
        if budget is None:
            continue

        n_params = weight.numel()
        total_original_bits += n_params * 16  # FP16 baseline

        if budget.method == "fp16":
            # Keep at FP16
            compressed = weight.half()
            actual_bpw = 16.0
            cos_sim = 1.0
        else:
            # PQ compression with allocated config
            M, K, G = budget.pq_config
            # Ensure group_size doesn't exceed tensor size
            while G > n_params // 4 and G > 32:
                G = G // 2
            while G % M != 0 and M > 1:
                M = M // 2

            try:
                pq = product_quantize(
                    weight, n_subvectors=M, codebook_size=K,
                    group_size=G, n_iter=n_iter,
                )
                compressed = pq
                actual_bpw = pq.bits_per_weight
                recon = pq.decompress()
                quality = compute_quality(weight.float(), recon.float())
                cos_sim = quality["cosine_sim"]
            except Exception as e:
                if verbose:
                    print(f"  WARN: PQ failed for {name}: {e}, keeping FP16")
                compressed = weight.half()
                actual_bpw = 16.0
                cos_sim = 1.0

        total_compressed_bits += n_params * actual_bpw

        result = MixedPrecisionResult(
            name=name,
            method=budget.method,
            target_bpw=budget.target_bpw,
            actual_bpw=actual_bpw,
            cosine_sim=cos_sim,
            n_params=n_params,
            compressed=compressed,
        )
        results.append(result)

        if verbose:
            print(f"  {name:<50} {budget.method:<12} "
                  f"BPW={actual_bpw:7.3f} (target {budget.target_bpw:.3f})  "
                  f"cos={cos_sim:.6f}")

    if verbose:
        actual_global_bpw = total_compressed_bits / max(sum(r.n_params for r in results), 1)
        avg_cosine = np.mean([r.cosine_sim for r in results])
        print(f"\nGlobal: {actual_global_bpw:.4f} BPW  "
              f"(target {global_target_bpw:.3f}), "
              f"avg cosine: {avg_cosine:.6f}")

    return results


# ---------------------------------------------------------------------------
# Propagation quality test
# ---------------------------------------------------------------------------

def test_propagation_quality(
    original_tensors: List[Tuple[str, torch.Tensor]],
    compressed_results: List[MixedPrecisionResult],
    n_test_vectors: int = 32,
    device: str = "cpu",
) -> List[dict]:
    """Test activation propagation quality through sequential weight matrices.

    Groups 2D weight matrices by transformer block and applies them in
    sequence with residual connections (matching real transformer architecture).
    Within each block, only same-dimension matrices are chained; dimension
    transitions use the residual stream to maintain a consistent hidden dim.

    The key insight: in a real transformer, the residual stream carries
    activations around each sub-layer. Errors in a sub-layer are ADDED to
    the residual, not multiplied through. This means middle-layer errors
    are dampened while first/last layer errors dominate.

    Returns list of dicts with per-step propagation metrics.
    """
    result_by_name = {r.name: r for r in compressed_results}

    # Collect 2D weight matrices paired with their compressed versions
    weight_pairs = []
    for name, tensor in original_tensors:
        if tensor.dim() != 2 or name not in result_by_name:
            continue
        w_orig = tensor.float().to(device)
        w_comp = result_by_name[name].decompress().float().to(device)
        weight_pairs.append((name, w_orig, w_comp))

    if not weight_pairs:
        print("No 2D weight matrices found for propagation test.")
        return []

    # Group by transformer block (blk.N) for residual connections
    from collections import OrderedDict
    blocks: Dict[int, List] = OrderedDict()
    standalone = []
    for name, w_orig, w_comp in weight_pairs:
        blk_idx = -1
        for part in name.split("."):
            if part.isdigit():
                blk_idx = int(part)
                break
        if blk_idx >= 0:
            blocks.setdefault(blk_idx, []).append((name, w_orig, w_comp))
        else:
            standalone.append((name, w_orig, w_comp))

    # Determine hidden dimension (most common input dim among weights)
    all_in_dims = [w.shape[1] for _, w, _ in weight_pairs]
    from collections import Counter
    hidden_dim = Counter(all_in_dims).most_common(1)[0][0]

    # Initialize test activations at hidden_dim
    torch.manual_seed(42)
    x_orig = torch.randn(n_test_vectors, hidden_dim, device=device)
    x_orig = x_orig / x_orig.norm(dim=1, keepdim=True)  # unit-norm for stability
    x_comp = x_orig.clone()

    propagation_results = []
    step = 0

    def measure_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        """Batch cosine similarity, handling zero-norm vectors."""
        cos_sims = []
        for j in range(a.shape[0]):
            na, nb = a[j].norm(), b[j].norm()
            if na > 1e-12 and nb > 1e-12:
                cs = torch.nn.functional.cosine_similarity(
                    a[j].unsqueeze(0), b[j].unsqueeze(0)
                ).item()
                if not np.isnan(cs):
                    cos_sims.append(cs)
        return float(np.mean(cos_sims)) if cos_sims else 0.0

    # Process each block with residual connections
    for blk_idx in sorted(blocks.keys()):
        block_weights = blocks[blk_idx]
        residual_orig = x_orig
        residual_comp = x_comp

        for name, w_orig, w_comp in block_weights:
            out_dim, in_dim = w_orig.shape

            # Only apply if input dim matches current activation dim
            if in_dim != x_orig.shape[1]:
                # Skip mismatched dimensions (e.g., up/down projections
                # that go to a different intermediate size). In a real
                # transformer these are paired (up then down), but we
                # test each independently against the residual.
                if in_dim == residual_orig.shape[1]:
                    # Use residual stream input
                    inp_orig = residual_orig
                    inp_comp = residual_comp
                else:
                    continue  # truly incompatible, skip

            else:
                inp_orig = x_orig
                inp_comp = x_comp

            y_orig = inp_orig @ w_orig.t()
            y_comp = inp_comp @ w_comp.t()

            avg_cos = measure_cosine(y_orig, y_comp)

            propagation_results.append({
                "layer_idx": step,
                "name": name,
                "activation_cosine": avg_cos,
                "orig_norm": y_orig.norm().item(),
                "comp_norm": y_comp.norm().item(),
            })
            step += 1

            # If output dim matches hidden dim, add to residual (simulating
            # the residual connection in a transformer)
            if out_dim == hidden_dim:
                x_orig = residual_orig + y_orig
                x_comp = residual_comp + y_comp
                # Layer-norm style normalization for stability
                x_orig = x_orig / (x_orig.norm(dim=1, keepdim=True) + 1e-10)
                x_comp = x_comp / (x_comp.norm(dim=1, keepdim=True) + 1e-10)
                residual_orig = x_orig
                residual_comp = x_comp

    return propagation_results


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    import time

    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ultracompress.gguf_loader import load_ollama_model

    print("=" * 70)
    print("  Mixed-Precision PQ — Propagation Quality Test")
    print("=" * 70)

    MODEL = "qwen3:4b"
    TARGET_BPW = 0.15
    MAX_TENSORS = 60  # enough to cover several layers

    # Load model tensors (materialize for reuse)
    print(f"\nLoading {MODEL}...")
    t0 = time.time()
    tensor_list = list(load_ollama_model(MODEL, max_tensors=MAX_TENSORS))
    print(f"Loaded {len(tensor_list)} tensors in {time.time()-t0:.1f}s")

    total_params = sum(t.numel() for _, t in tensor_list)
    print(f"Total parameters: {total_params:,}")

    # ---- Mixed-precision compression ----
    print("\n" + "=" * 70)
    print("  MIXED-PRECISION COMPRESSION")
    print("=" * 70)

    t0 = time.time()
    mixed_results = mixed_precision_compress(
        iter(tensor_list),
        global_target_bpw=TARGET_BPW,
        n_iter=15,
        verbose=True,
    )
    mixed_time = time.time() - t0
    print(f"\nMixed-precision compression took {mixed_time:.1f}s")

    # ---- Uniform PQ at same overall BPW for comparison ----
    print("\n" + "=" * 70)
    print("  UNIFORM PQ COMPRESSION (baseline)")
    print("=" * 70)

    # Find uniform PQ config that gives roughly TARGET_BPW
    # BPW = (M * log2(K) + 16) / G
    # For 0.15 BPW: try M=8, K=4, G=256 -> (8*2+16)/256 = 0.125
    # Or M=8, K=4, G=128 -> (8*2+16)/128 = 0.25
    # Or M=4, K=4, G=256 -> (4*2+16)/256 = 0.094
    # We'll use M=8, K=4, G=256 which gives ~0.125 BPW (close to 0.15)
    UNIFORM_M, UNIFORM_K, UNIFORM_G = 8, 4, 256

    uniform_results = []
    t0 = time.time()
    for name, weight in tensor_list:
        n_params = weight.numel()
        if n_params < UNIFORM_G * 4:
            # Too small for this group size, keep FP16
            uniform_results.append(MixedPrecisionResult(
                name=name, method="fp16", target_bpw=16.0,
                actual_bpw=16.0, cosine_sim=1.0, n_params=n_params,
                compressed=weight.half(),
            ))
            continue

        G = UNIFORM_G
        M = UNIFORM_M
        while G > n_params // 4 and G > 32:
            G = G // 2
        while G % M != 0 and M > 1:
            M = M // 2

        try:
            pq = product_quantize(weight, n_subvectors=M, codebook_size=UNIFORM_K,
                                  group_size=G, n_iter=15)
            recon = pq.decompress()
            quality = compute_quality(weight.float(), recon.float())
            actual_bpw = pq.bits_per_weight
            cos_sim = quality["cosine_sim"]
            uniform_results.append(MixedPrecisionResult(
                name=name, method="pq_uniform", target_bpw=actual_bpw,
                actual_bpw=actual_bpw, cosine_sim=cos_sim, n_params=n_params,
                compressed=pq,
            ))
            print(f"  {name:<50} pq_uniform   BPW={actual_bpw:7.3f}  cos={cos_sim:.6f}")
        except Exception as e:
            print(f"  WARN: {name}: {e}")
            uniform_results.append(MixedPrecisionResult(
                name=name, method="fp16", target_bpw=16.0,
                actual_bpw=16.0, cosine_sim=1.0, n_params=n_params,
                compressed=weight.half(),
            ))

    uniform_time = time.time() - t0

    uniform_global_bpw = (sum(r.actual_bpw * r.n_params for r in uniform_results)
                          / max(sum(r.n_params for r in uniform_results), 1))
    uniform_avg_cos = np.mean([r.cosine_sim for r in uniform_results])
    print(f"\nUniform PQ: {uniform_global_bpw:.4f} BPW, avg cosine: {uniform_avg_cos:.6f}")
    print(f"Uniform compression took {uniform_time:.1f}s")

    # ---- Propagation quality comparison ----
    print("\n" + "=" * 70)
    print("  PROPAGATION QUALITY COMPARISON")
    print("=" * 70)

    print("\n--- Mixed-Precision Propagation ---")
    mixed_prop = test_propagation_quality(tensor_list, mixed_results)
    for p in mixed_prop:
        short = p["name"][-50:] if len(p["name"]) > 50 else p["name"]
        print(f"  Layer {p['layer_idx']:3d}: cos={p['activation_cosine']:.6f}  {short}")

    print("\n--- Uniform PQ Propagation ---")
    uniform_prop = test_propagation_quality(tensor_list, uniform_results)
    for p in uniform_prop:
        short = p["name"][-50:] if len(p["name"]) > 50 else p["name"]
        print(f"  Layer {p['layer_idx']:3d}: cos={p['activation_cosine']:.6f}  {short}")

    # ---- Summary comparison ----
    print("\n" + "=" * 70)
    print("  SUMMARY: Mixed-Precision vs Uniform PQ")
    print("=" * 70)

    mixed_global_bpw = (sum(r.actual_bpw * r.n_params for r in mixed_results)
                        / max(sum(r.n_params for r in mixed_results), 1))
    mixed_avg_cos = np.mean([r.cosine_sim for r in mixed_results])

    print(f"\n{'Metric':<35} {'Mixed':>12} {'Uniform':>12}")
    print("-" * 60)
    print(f"{'Global BPW':<35} {mixed_global_bpw:>12.4f} {uniform_global_bpw:>12.4f}")
    print(f"{'Avg per-weight cosine':<35} {mixed_avg_cos:>12.6f} {uniform_avg_cos:>12.6f}")

    if mixed_prop and uniform_prop:
        mixed_final_cos = mixed_prop[-1]["activation_cosine"] if mixed_prop else 0
        uniform_final_cos = uniform_prop[-1]["activation_cosine"] if uniform_prop else 0
        mixed_avg_prop = np.mean([p["activation_cosine"] for p in mixed_prop])
        uniform_avg_prop = np.mean([p["activation_cosine"] for p in uniform_prop])
        mixed_min_prop = min(p["activation_cosine"] for p in mixed_prop)
        uniform_min_prop = min(p["activation_cosine"] for p in uniform_prop)

        print(f"{'Avg propagation cosine':<35} {mixed_avg_prop:>12.6f} {uniform_avg_prop:>12.6f}")
        print(f"{'Min propagation cosine':<35} {mixed_min_prop:>12.6f} {uniform_min_prop:>12.6f}")
        print(f"{'Final layer propagation cosine':<35} {mixed_final_cos:>12.6f} {uniform_final_cos:>12.6f}")

        winner = "MIXED-PRECISION" if mixed_avg_prop > uniform_avg_prop else "UNIFORM"
        print(f"\nPropagation winner: {winner}")
        if mixed_avg_prop > uniform_avg_prop:
            improvement = (mixed_avg_prop - uniform_avg_prop) / max(uniform_avg_prop, 1e-10) * 100
            print(f"Mixed-precision improves propagation cosine by {improvement:.1f}%")
        print()
