"""claim21_finetune_delta_summary.py -- wave 45 aggregator.

Aggregates per-pair JSONs from claim21_finetune_delta.py into a single
cohort table + headline ratios."""
from __future__ import annotations

import json
from pathlib import Path


PAIRS = [
    ("olmo2_1b",
     "results/claim21_finetune_delta_olmo2_1b.json"),
    ("smollm2_1.7b",
     "results/claim21_finetune_delta_smollm2_1.7b.json"),
]


def main():
    root = Path(".")
    per_model = {}
    tot_params = 0
    tot_bf16 = 0
    tot_br_base = 0
    tot_br_ft = 0
    tot_br_delta = 0

    print()
    print("="*92)
    print("CLAIM 21 WAVE 45 FINE-TUNE DELTA STORAGE SUMMARY")
    print("="*92)
    hdr = f"{'model':<16} {'params':>14} {'bf16 B':>14} {'br(base)':>14} {'br(ft)':>14} {'br(delta)':>14}"
    print(hdr)
    print("-"*len(hdr))

    for name, path in PAIRS:
        p = root / path
        if not p.exists():
            print(f"  MISSING: {path}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        nparams = d["n_params"]
        nb = d["bf16_bytes_per_stream"]
        br_base  = d["compressed_bytes"]["base"]["brotli_11"]["bytes"]
        br_ft    = d["compressed_bytes"]["ft"]["brotli_11"]["bytes"]
        br_delta = d["compressed_bytes"]["delta"]["brotli_11"]["bytes"]
        per_model[name] = {
            "base_repo": d["base_repo"],
            "ft_repo":   d["ft_repo"],
            "n_params":  nparams,
            "bf16_bytes_per_stream": nb,
            "brotli11_bytes_base":   br_base,
            "brotli11_bytes_ft":     br_ft,
            "brotli11_bytes_delta":  br_delta,
            "delta_vs_ft_ratio":     br_ft / br_delta,
            "delta_vs_base_ratio":   br_base / br_delta,
            "ft_bpB_brotli11":       8.0*br_ft/nb,
            "delta_bpB_brotli11":    8.0*br_delta/nb,
        }
        tot_params   += nparams
        tot_bf16     += nb
        tot_br_base  += br_base
        tot_br_ft    += br_ft
        tot_br_delta += br_delta
        print(f"{name:<16} {nparams:>14,} {nb:>14,} {br_base:>14,} {br_ft:>14,} {br_delta:>14,}")

    print("-"*len(hdr))
    print(f"{'COHORT':<16} {tot_params:>14,} {tot_bf16:>14,} {tot_br_base:>14,} {tot_br_ft:>14,} {tot_br_delta:>14,}")

    cohort = {
        "n_pairs": len(per_model),
        "n_params_total":       tot_params,
        "bf16_bytes_total":     tot_bf16,
        "brotli11_base_total":  tot_br_base,
        "brotli11_ft_total":    tot_br_ft,
        "brotli11_delta_total": tot_br_delta,
        "delta_vs_ft_ratio":    tot_br_ft    / tot_br_delta,
        "delta_vs_base_ratio":  tot_br_base  / tot_br_delta,
        "ft_vs_bf16_ratio":     tot_bf16     / tot_br_ft,
        "delta_vs_bf16_ratio":  tot_bf16     / tot_br_delta,
        "ft_bpB_brotli11":      8.0*tot_br_ft    / tot_bf16,
        "delta_bpB_brotli11":   8.0*tot_br_delta / tot_bf16,
    }

    print()
    print("COHORT RATIOS:")
    print(f"  brotli11(ft)    / brotli11(delta) = {cohort['delta_vs_ft_ratio']:.3f}x  "
          f"(how much smaller the delta is than the full fine-tune)")
    print(f"  brotli11(base)  / brotli11(delta) = {cohort['delta_vs_base_ratio']:.3f}x")
    print(f"  bf16(raw)       / brotli11(ft)    = {cohort['ft_vs_bf16_ratio']:.3f}x")
    print(f"  bf16(raw)       / brotli11(delta) = {cohort['delta_vs_bf16_ratio']:.3f}x")
    print(f"  ft    @ brotli11  bpB = {cohort['ft_bpB_brotli11']:.4f}")
    print(f"  delta @ brotli11  bpB = {cohort['delta_bpB_brotli11']:.4f}")

    out = {
        "claim": 21,
        "wave": 45,
        "experiment": "finetune_delta_storage_summary",
        "per_model": per_model,
        "cohort": cohort,
    }
    out_path = root / "results" / "claim21_finetune_delta_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
