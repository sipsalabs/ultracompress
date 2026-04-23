"""claim21_headline_summary.py -- wave 44 cohort aggregator.

Aggregates the per-model headline JSON outputs into a single cohort
end-to-end ratio table: the canonical industry-quotable result for
Claim 21.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]


def main():
    per_model = {}
    for m in MODELS:
        per_model[m] = json.loads(
            (RES / f"claim21_headline_{m}_rho0.01.json").read_text())

    bf16 = sum(per_model[m]["bf16_bytes"] for m in MODELS)
    nparams = sum(per_model[m]["n_params"] for m in MODELS)
    gz = sum(per_model[m]["baselines_on_raw_bf16"]["gzip_9"]["bytes"]
             for m in MODELS)
    zd = sum(per_model[m]["baselines_on_raw_bf16"]["zstd_19"]["bytes"]
             for m in MODELS)
    br = sum(per_model[m]["baselines_on_raw_bf16"]["brotli_11"]["bytes"]
             for m in MODELS)
    c21 = sum(per_model[m]["claim21_pipeline"]["total_compressed_bytes"]
              for m in MODELS)

    cohort = {
        "bf16_bytes_total":       bf16,
        "gzip9_bytes_total":      gz,
        "zstd19_bytes_total":     zd,
        "brotli11_bytes_total":   br,
        "claim21_bytes_total":    c21,
        "n_params_total":         nparams,
        "headline_ratios_vs_bf16": {
            "gzip_9":    bf16 / gz,
            "zstd_19":   bf16 / zd,
            "brotli_11": bf16 / br,
            "claim21":   bf16 / c21,
        },
        "claim21_vs_baseline_ratios": {
            "vs_gzip9":    gz / c21,
            "vs_zstd19":   zd / c21,
            "vs_brotli11": br / c21,
        },
        "bytes_per_param": {
            "bf16_original": bf16 / nparams,
            "gzip_9":        gz / nparams,
            "zstd_19":       zd / nparams,
            "brotli_11":     br / nparams,
            "claim21":       c21 / nparams,
        },
    }

    out = {
        "claim": 21,
        "wave": 44,
        "experiment": "headline_end_to_end_ratio_summary",
        "rho": 0.010,
        "models": MODELS,
        "per_model": {m: {
            "n_params":     per_model[m]["n_params"],
            "bf16_bytes":   per_model[m]["bf16_bytes"],
            "ratios":       per_model[m]["headline_ratios"],
            "bytes_per_param": per_model[m]["bytes_per_param"],
        } for m in MODELS},
        "cohort": cohort,
    }

    print("=" * 78)
    print("Wave 44: end-to-end ratio headline (cohort aggregate)")
    print("=" * 78)
    print(f"  models:    {MODELS}")
    print(f"  params:    {nparams:,}")
    print(f"  bf16:      {bf16:,} B ({bf16/1e9:.3f} GB)")
    print()
    print(f"  {'method':<12} {'bytes':>18} {'bpP':>8} {'ratio':>10}")
    for name, b in [("bf16", bf16), ("gzip-9", gz), ("zstd-19", zd),
                    ("brotli-11", br), ("Claim-21", c21)]:
        r = bf16 / b if b else float('nan')
        print(f"  {name:<12} {b:>18,} {b/nparams:>8.4f} {r:>9.3f}x")
    print()
    print("  Claim-21 vs raw-bf16 baselines:")
    print(f"    vs gzip-9   : {gz/c21:.3f}x smaller")
    print(f"    vs zstd-19  : {zd/c21:.3f}x smaller")
    print(f"    vs brotli-11: {br/c21:.3f}x smaller")

    out_path = RES / "claim21_headline_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
