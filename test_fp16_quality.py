#!/usr/bin/env python3
"""
FP16 vs GGUF PQ Compression Quality Test

Tests Product Quantization on FP16 source weights (Qwen/Qwen3-8B) and
compares against known GGUF quality levels.

Critical question: Does PQ on FP16 weights achieve significantly higher
cosine similarity than on GGUF weights?
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
import gc

from ultracompress.product_quantize import product_quantize
from ultracompress.quantize import quantize_absmax
from ultracompress.metrics import compute_quality


# Snapshot path (already cached)
SNAPSHOT_DIR = r"C:\Users\sip\.cache\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\b968826d9c46dd6066d109eabc6255188de91218"

# PQ configurations to test
PQ_CONFIGS = [
    {"name": "PQ M=8 K=4 G=256",   "M": 8,  "K": 4,   "G": 256,  "target_bpw": 0.13},
    {"name": "PQ M=4 K=2 G=2048",   "M": 4,  "K": 2,   "G": 2048, "target_bpw": 0.016},
    {"name": "PQ M=16 K=16 G=256",  "M": 16, "K": 16,  "G": 256,  "target_bpw": 0.36},
    {"name": "PQ M=4 K=256 G=32",   "M": 4,  "K": 256, "G": 32,   "target_bpw": 1.6},
]

# Known GGUF quality baselines (from prior testing on qwen3:4b GGUF weights)
GGUF_BASELINES = {
    0.13:  {"weight_cos": 0.89, "activ_cos": 0.85},
    0.016: {"weight_cos": 0.72, "activ_cos": 0.60},
    0.36:  {"weight_cos": 0.91, "activ_cos": 0.88},
    1.6:   {"weight_cos": 0.95, "activ_cos": 0.93},
}


def load_tensors_streaming(snapshot_dir, max_tensors=15):
    """Load tensors one-at-a-time from safetensors shards to save memory."""
    from safetensors import safe_open
    from pathlib import Path

    shard_files = sorted(Path(snapshot_dir).glob("*.safetensors"))
    print(f"  Found {len(shard_files)} shard files")

    count = 0
    for shard_file in shard_files:
        with safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for name in f.keys():
                if count >= max_tensors:
                    return
                tensor = f.get_tensor(name)
                yield name, tensor.float()
                count += 1
                if count >= max_tensors:
                    return


def compute_activation_cosine(w_orig, w_recon):
    """Compute activation output cosine: y = X @ W^T."""
    with torch.no_grad():
        in_dim = w_orig.shape[1] if w_orig.ndim == 2 else w_orig.shape[-1]
        X = torch.randn(1, 32, in_dim)

        y_orig = torch.nn.functional.linear(X, w_orig)
        y_recon = torch.nn.functional.linear(X, w_recon)

        cos = torch.nn.functional.cosine_similarity(
            y_orig.reshape(1, -1), y_recon.reshape(1, -1)
        ).item()
    return cos


def test_pq_config(weight, cfg):
    """Test a single PQ config on a weight tensor."""
    M, K, G = cfg["M"], cfg["K"], cfg["G"]

    if weight.numel() < G * 4:
        return None

    try:
        pq = product_quantize(weight, n_subvectors=M, codebook_size=K,
                              group_size=G, n_iter=20)
        recon = pq.decompress()
        if recon.shape != weight.shape:
            recon = recon.reshape(weight.shape)

        quality = compute_quality(weight, recon)
        activ_cos = compute_activation_cosine(weight.float(), recon.float())
        actual_bpw = pq.bits_per_weight

        # Free memory
        del pq, recon
        gc.collect()

        return {
            "weight_cos": quality["cosine_sim"],
            "activ_cos": activ_cos,
            "rel_error": quality["relative_error"],
            "actual_bpw": actual_bpw,
        }
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def test_int4(weight):
    """INT4 baseline quantization."""
    try:
        qt = quantize_absmax(weight, bits=4, group_size=128)
        recon = qt.decompress()
        if recon.shape != weight.shape:
            recon = recon.reshape(weight.shape)

        quality = compute_quality(weight, recon)
        activ_cos = compute_activation_cosine(weight.float(), recon.float())
        actual_bpw = qt.bits_per_weight

        del qt, recon
        gc.collect()

        return {
            "weight_cos": quality["cosine_sim"],
            "activ_cos": activ_cos,
            "rel_error": quality["relative_error"],
            "actual_bpw": actual_bpw,
        }
    except Exception as e:
        print(f"    INT4 ERROR: {e}")
        return None


def main():
    print("=" * 80)
    print("  UltraCompress - FP16 vs GGUF PQ Quality Comparison")
    print("  Model: Qwen/Qwen3-8B (FP16 source weights)")
    print("=" * 80)

    print(f"\n  Device: cpu")
    print(f"  Loading tensors from {SNAPSHOT_DIR}")

    t0 = time.time()
    all_tensors = []
    for name, w in load_tensors_streaming(SNAPSHOT_DIR, max_tensors=15):
        all_tensors.append((name, w))
        print(f"    Loaded: {name} shape={tuple(w.shape)} numel={w.numel():,}")
        sys.stdout.flush()

    load_time = time.time() - t0
    print(f"  Loaded {len(all_tensors)} tensors in {load_time:.1f}s")

    # Filter: ndim >= 2, numel >= 10000
    eligible = [(name, w) for name, w in all_tensors if w.ndim >= 2 and w.numel() >= 10000]
    print(f"  Eligible tensors (ndim>=2, numel>=10000): {len(eligible)}")

    # Free non-eligible tensors
    del all_tensors
    gc.collect()

    # ================================================================
    # Per-config aggregated results
    # ================================================================
    config_results = {}
    for cfg in PQ_CONFIGS:
        config_results[cfg["name"]] = {"weight_cos": [], "activ_cos": [], "bpw": []}
    config_results["INT4 baseline"] = {"weight_cos": [], "activ_cos": [], "bpw": []}

    # ================================================================
    # Test each tensor
    # ================================================================
    print("\n" + "=" * 80)
    print("  Per-Tensor Results")
    print("=" * 80)

    for i, (name, weight) in enumerate(eligible):
        print(f"\n  [{i+1}/{len(eligible)}] {name} -- shape {tuple(weight.shape)}, {weight.numel():,} params")
        sys.stdout.flush()
        w = weight.float()

        # Test PQ configs
        for cfg in PQ_CONFIGS:
            t1 = time.time()
            result = test_pq_config(w, cfg)
            dt = time.time() - t1
            if result:
                config_results[cfg["name"]]["weight_cos"].append(result["weight_cos"])
                config_results[cfg["name"]]["activ_cos"].append(result["activ_cos"])
                config_results[cfg["name"]]["bpw"].append(result["actual_bpw"])
                print(f"    {cfg['name']:<25} wt_cos={result['weight_cos']:.6f}  act_cos={result['activ_cos']:.6f}  bpw={result['actual_bpw']:.4f}  ({dt:.1f}s)")
            else:
                print(f"    {cfg['name']:<25} SKIPPED (tensor too small or error)")
            sys.stdout.flush()

        # INT4 baseline
        t1 = time.time()
        int4_result = test_int4(w)
        dt = time.time() - t1
        if int4_result:
            config_results["INT4 baseline"]["weight_cos"].append(int4_result["weight_cos"])
            config_results["INT4 baseline"]["activ_cos"].append(int4_result["activ_cos"])
            config_results["INT4 baseline"]["bpw"].append(int4_result["actual_bpw"])
            print(f"    {'INT4 baseline':<25} wt_cos={int4_result['weight_cos']:.6f}  act_cos={int4_result['activ_cos']:.6f}  bpw={int4_result['actual_bpw']:.4f}  ({dt:.1f}s)")
        sys.stdout.flush()

    # ================================================================
    # Summary Table
    # ================================================================
    print("\n" + "=" * 80)
    print("  SUMMARY: FP16 Source Weight Quality (Qwen/Qwen3-8B)")
    print("=" * 80)
    print(f"\n  {'Config':<25} {'BPW':>8} {'Wt Cos':>10} {'Act Cos':>10} {'N':>4}")
    print("  " + "-" * 62)

    for cfg_name in [c["name"] for c in PQ_CONFIGS] + ["INT4 baseline"]:
        data = config_results[cfg_name]
        if data["weight_cos"]:
            avg_wcos = np.mean(data["weight_cos"])
            avg_acos = np.mean(data["activ_cos"])
            avg_bpw = np.mean(data["bpw"])
            n = len(data["weight_cos"])
            print(f"  {cfg_name:<25} {avg_bpw:>8.4f} {avg_wcos:>10.6f} {avg_acos:>10.6f} {n:>4}")

    # ================================================================
    # FP16 vs GGUF Comparison
    # ================================================================
    print("\n" + "=" * 80)
    print("  FP16 vs GGUF COMPARISON")
    print("  (GGUF baselines from prior testing on qwen3:4b Q4/Q5 weights)")
    print("=" * 80)
    print(f"\n  {'Config':<25} {'BPW':>6} {'FP16 Wt':>10} {'GGUF Wt':>10} {'Delta':>8} {'FP16 Act':>10} {'GGUF Act':>10} {'Delta':>8}")
    print("  " + "-" * 88)

    for cfg in PQ_CONFIGS:
        data = config_results[cfg["name"]]
        if not data["weight_cos"]:
            continue
        fp16_wcos = np.mean(data["weight_cos"])
        fp16_acos = np.mean(data["activ_cos"])
        target_bpw = cfg["target_bpw"]

        gguf = GGUF_BASELINES.get(target_bpw, {})
        gguf_wcos = gguf.get("weight_cos", float('nan'))
        gguf_acos = gguf.get("activ_cos", float('nan'))

        d_wcos = fp16_wcos - gguf_wcos
        d_acos = fp16_acos - gguf_acos

        print(f"  {cfg['name']:<25} {target_bpw:>6.3f} {fp16_wcos:>10.6f} {gguf_wcos:>10.4f} {d_wcos:>+8.4f} {fp16_acos:>10.6f} {gguf_acos:>10.4f} {d_acos:>+8.4f}")

    # ================================================================
    # Verdict
    # ================================================================
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)

    improvements_w = []
    improvements_a = []
    for cfg in PQ_CONFIGS:
        data = config_results[cfg["name"]]
        if not data["weight_cos"]:
            continue
        fp16_wcos = np.mean(data["weight_cos"])
        fp16_acos = np.mean(data["activ_cos"])
        target_bpw = cfg["target_bpw"]
        gguf = GGUF_BASELINES.get(target_bpw, {})
        if "weight_cos" in gguf:
            improvements_w.append(fp16_wcos - gguf["weight_cos"])
            improvements_a.append(fp16_acos - gguf["activ_cos"])

    if improvements_w:
        avg_imp_w = np.mean(improvements_w)
        avg_imp_a = np.mean(improvements_a)
        print(f"\n  Average weight cosine improvement (FP16 over GGUF): {avg_imp_w:+.4f}")
        print(f"  Average activation cosine improvement:               {avg_imp_a:+.4f}")

        if avg_imp_w > 0.02:
            print(f"\n  CONCLUSION: YES - FP16 source weights compress SIGNIFICANTLY better.")
            print(f"  PQ on FP16 gains ~{avg_imp_w:.2f} weight cosine over GGUF at same BPW.")
        elif avg_imp_w > 0.005:
            print(f"\n  CONCLUSION: MODERATE improvement from FP16 source weights.")
        else:
            print(f"\n  CONCLUSION: Minimal difference between FP16 and GGUF source weights.")

    # Min/max per config
    print(f"\n  Per-config detail:")
    for cfg in PQ_CONFIGS:
        data = config_results[cfg["name"]]
        if data["weight_cos"]:
            print(f"    {cfg['name']}: min_wt_cos={min(data['weight_cos']):.6f} max_wt_cos={max(data['weight_cos']):.6f} "
                  f"min_act_cos={min(data['activ_cos']):.6f} max_act_cos={max(data['activ_cos']):.6f}")

    print("\n" + "=" * 80)
    print("  Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
