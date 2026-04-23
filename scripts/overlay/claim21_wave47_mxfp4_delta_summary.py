"""claim21_wave47_mxfp4_delta_summary.py

Aggregates per-pair JSONs from claim21_wave47_mxfp4_delta.py into a
cohort table. Params-weighted rel-Frobenius at cohort level (same
caveat as wave 46: true cohort-level relerr requires per-tensor
||D||_F^2; per-pair scalar is a proxy).
"""
from __future__ import annotations

import json
from pathlib import Path


PAIRS = [
    ("olmo2_1b",
     "results/claim21_wave47_mxfp4_delta_olmo2_1b.json"),
    ("smollm2_1.7b",
     "results/claim21_wave47_mxfp4_delta_smollm2_1.7b.json"),
    ("qwen3_1.7b",
     "results/claim21_wave47_mxfp4_delta_qwen3_1.7b.json"),
]


def main() -> None:
    root = Path(".")
    per_model: dict = {}
    tot_params = 0
    tot_blocks = 0
    tot_bf16 = 0
    tot_br_bf16 = 0
    tot_br_i8 = 0
    tot_br_i4 = 0
    tot_overhead_i8 = 0
    tot_overhead_i4 = 0
    relerr8_list: list[tuple[int, float]] = []
    relerr4_list: list[tuple[int, float]] = []

    print()
    print("=" * 104)
    print("CLAIM 21 WAVE 47 MXFP4 BLOCK-FLOAT DELTA COHORT SUMMARY")
    print("=" * 104)
    hdr = (f"{'model':<16} {'params':>14} {'blocks':>12} {'br(bf16)':>14} "
           f"{'br(i8+bsc)':>14} {'br(i4+bsc)':>14} {'relerr8':>10} {'relerr4':>10}")
    print(hdr)
    print("-" * len(hdr))

    block_size = None
    for name, path in PAIRS:
        p = root / path
        if not p.exists():
            print(f"  MISSING: {path}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if block_size is None:
            block_size = d.get("block_size")
        nparams = d["n_params"]
        nblocks = d["n_blocks"]
        nb = d["raw_bytes"]["bf16"]
        br_bf16 = d["brotli_11_bytes"]["bf16"]
        br_i8   = d["brotli_11_bytes"]["int8+block_scales_fp16"]
        br_i4   = d["brotli_11_bytes"]["int4+block_scales_fp16"]
        re8 = d["rel_frobenius_reconstruction_error"]["int8"]
        re4 = d["rel_frobenius_reconstruction_error"]["int4"]
        ovh8 = d["raw_bytes"]["block_scales_overhead_int8"]
        ovh4 = d["raw_bytes"]["block_scales_overhead_int4"]

        per_model[name] = {
            "base_repo": d["base_repo"],
            "ft_repo":   d["ft_repo"],
            "n_params":  nparams,
            "n_blocks":  nblocks,
            "bf16_bytes": nb,
            "brotli11_bytes_bf16": br_bf16,
            "brotli11_bytes_int8": br_i8,
            "brotli11_bytes_int4": br_i4,
            "block_scales_overhead_int8": ovh8,
            "block_scales_overhead_int4": ovh4,
            "rel_frobenius_err_int8": re8,
            "rel_frobenius_err_int4": re4,
            "ratio_bf16_over_int8": br_bf16 / br_i8,
            "ratio_bf16_over_int4": br_bf16 / br_i4,
        }
        tot_params += nparams
        tot_blocks += nblocks
        tot_bf16 += nb
        tot_br_bf16 += br_bf16
        tot_br_i8 += br_i8
        tot_br_i4 += br_i4
        tot_overhead_i8 += ovh8
        tot_overhead_i4 += ovh4
        relerr8_list.append((nparams, re8))
        relerr4_list.append((nparams, re4))
        print(f"{name:<16} {nparams:>14,} {nblocks:>12,} {br_bf16:>14,} "
              f"{br_i8:>14,} {br_i4:>14,} {re8:>10.3e} {re4:>10.3e}")

    print("-" * len(hdr))
    if tot_br_i8 == 0:
        print("(no pairs parsed; nothing to aggregate)")
        return

    def weighted(lst: list[tuple[int, float]]) -> float:
        num = sum(w * v for w, v in lst)
        den = sum(w for w, _ in lst)
        return num / den if den else 0.0

    re8_w = weighted(relerr8_list)
    re4_w = weighted(relerr4_list)

    cohort = {
        "n_pairs": len(per_model),
        "block_size": block_size,
        "n_params_total":       tot_params,
        "n_blocks_total":       tot_blocks,
        "bf16_bytes_total":     tot_bf16,
        "brotli11_bf16_total":  tot_br_bf16,
        "brotli11_int8_total":  tot_br_i8,
        "brotli11_int4_total":  tot_br_i4,
        "block_scales_overhead_int8_total": tot_overhead_i8,
        "block_scales_overhead_int4_total": tot_overhead_i4,
        "ratio_bf16_over_int8": tot_br_bf16 / tot_br_i8,
        "ratio_bf16_over_int4": tot_br_bf16 / tot_br_i4,
        "weighted_rel_frob_err_int8": re8_w,
        "weighted_rel_frob_err_int4": re4_w,
        "int8_bpB_brotli11":   8.0 * tot_br_i8  / tot_bf16,
        "int4_bpB_brotli11":   8.0 * tot_br_i4  / tot_bf16,
        "bf16_delta_bpB":      8.0 * tot_br_bf16 / tot_bf16,
    }

    print()
    print(f"COHORT RATIOS (block={block_size}, vs brotli-11 bf16 delta baseline from wave 45):")
    print(f"  br(bf16 delta) / br(int8 block+sc) = "
          f"{cohort['ratio_bf16_over_int8']:.3f}x  "
          f"(params-weighted relerr {re8_w:.3e})")
    print(f"  br(bf16 delta) / br(int4 block+sc) = "
          f"{cohort['ratio_bf16_over_int4']:.3f}x  "
          f"(params-weighted relerr {re4_w:.3e})")
    print(f"  int8 @ brotli-11  bpB = {cohort['int8_bpB_brotli11']:.4f}")
    print(f"  int4 @ brotli-11  bpB = {cohort['int4_bpB_brotli11']:.4f}")
    print(f"  bf16 delta bpB        = {cohort['bf16_delta_bpB']:.4f}")

    out_path = root / "results" / "claim21_wave47_mxfp4_delta_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"per_model": per_model, "cohort": cohort}
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    import os as _os
    _os.replace(tmp, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
