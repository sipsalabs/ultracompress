"""claim21_wave46_quant_delta_summary.py

Aggregates per-pair JSONs from claim21_wave46_quant_delta.py into a
cohort table + headline ratios + rel-Frobenius reconstruction errors.
"""
from __future__ import annotations

import json
from pathlib import Path


PAIRS = [
    ("olmo2_1b",
     "results/claim21_wave46_quant_delta_olmo2_1b.json"),
    ("smollm2_1.7b",
     "results/claim21_wave46_quant_delta_smollm2_1.7b.json"),
    ("qwen3_1.7b",
     "results/claim21_wave46_quant_delta_qwen3_1.7b.json"),
]


def main() -> None:
    root = Path(".")
    per_model: dict = {}
    # Cohort accumulators
    tot_params = 0
    tot_bf16 = 0
    tot_br_bf16 = 0
    tot_br_i8 = 0
    tot_br_i4 = 0
    # For weighted aggregate rel-Frobenius we need ||D||_F^2 per tensor
    # but per-pair we only have the pair-level scalar. Best we can do
    # at cohort level is report min/max and params-weighted average.
    relerr8_list: list[tuple[int, float]] = []
    relerr4_list: list[tuple[int, float]] = []

    print()
    print("=" * 96)
    print("CLAIM 21 WAVE 46 QUANT-DELTA COHORT SUMMARY")
    print("=" * 96)
    hdr = (f"{'model':<16} {'params':>14} {'br(bf16)':>14} "
           f"{'br(i8+sc)':>14} {'br(i4+sc)':>14} {'relerr8':>10} "
           f"{'relerr4':>10}")
    print(hdr)
    print("-" * len(hdr))

    for name, path in PAIRS:
        p = root / path
        if not p.exists():
            print(f"  MISSING: {path}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        nparams = d["n_params"]
        nb = d["raw_bytes"]["bf16"]
        br_bf16 = d["brotli_11_bytes"]["bf16"]
        br_i8   = d["brotli_11_bytes"]["int8+scales_fp16"]
        br_i4   = d["brotli_11_bytes"]["int4+scales_fp16"]
        re8 = d["rel_frobenius_reconstruction_error"]["int8"]
        re4 = d["rel_frobenius_reconstruction_error"]["int4"]
        per_model[name] = {
            "base_repo": d["base_repo"],
            "ft_repo":   d["ft_repo"],
            "n_params":  nparams,
            "bf16_bytes": nb,
            "brotli11_bytes_bf16": br_bf16,
            "brotli11_bytes_int8": br_i8,
            "brotli11_bytes_int4": br_i4,
            "rel_frobenius_err_int8": re8,
            "rel_frobenius_err_int4": re4,
            "ratio_bf16_over_int8": br_bf16 / br_i8,
            "ratio_bf16_over_int4": br_bf16 / br_i4,
        }
        tot_params  += nparams
        tot_bf16    += nb
        tot_br_bf16 += br_bf16
        tot_br_i8   += br_i8
        tot_br_i4   += br_i4
        relerr8_list.append((nparams, re8))
        relerr4_list.append((nparams, re4))
        print(f"{name:<16} {nparams:>14,} {br_bf16:>14,} "
              f"{br_i8:>14,} {br_i4:>14,} {re8:>10.3e} {re4:>10.3e}")

    print("-" * len(hdr))
    if tot_br_i8 == 0:
        print("(no pairs parsed; nothing to aggregate)")
        return

    # Params-weighted mean of rel-Frobenius error
    def weighted(lst: list[tuple[int, float]]) -> float:
        num = sum(w * v for w, v in lst)
        den = sum(w for w, _ in lst)
        return num / den if den else 0.0

    re8_w = weighted(relerr8_list)
    re4_w = weighted(relerr4_list)

    cohort = {
        "n_pairs": len(per_model),
        "n_params_total":       tot_params,
        "bf16_bytes_total":     tot_bf16,
        "brotli11_bf16_total":  tot_br_bf16,
        "brotli11_int8_total":  tot_br_i8,
        "brotli11_int4_total":  tot_br_i4,
        "ratio_bf16_over_int8": tot_br_bf16 / tot_br_i8,
        "ratio_bf16_over_int4": tot_br_bf16 / tot_br_i4,
        "weighted_rel_frob_err_int8": re8_w,
        "weighted_rel_frob_err_int4": re4_w,
        "int8_bpB_brotli11":   8.0 * tot_br_i8  / tot_bf16,
        "int4_bpB_brotli11":   8.0 * tot_br_i4  / tot_bf16,
        "bf16_delta_bpB":      8.0 * tot_br_bf16 / tot_bf16,
    }

    print()
    print("COHORT RATIOS (vs brotli-11 bf16 delta baseline from wave 45):")
    print(f"  br(bf16 delta) / br(int8+sc) = {cohort['ratio_bf16_over_int8']:.3f}x  "
          f"(params-weighted relerr {re8_w:.3e})")
    print(f"  br(bf16 delta) / br(int4+sc) = {cohort['ratio_bf16_over_int4']:.3f}x  "
          f"(params-weighted relerr {re4_w:.3e})")
    print(f"  int8 @ brotli-11  bpB = {cohort['int8_bpB_brotli11']:.4f}")
    print(f"  int4 @ brotli-11  bpB = {cohort['int4_bpB_brotli11']:.4f}")

    out = {"per_model": per_model, "cohort": cohort}
    out_path = root / "results" / "claim21_wave46_quant_delta_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
