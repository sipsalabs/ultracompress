"""claim21_fp8_decomp_summary.py -- wave 43 cohort aggregator.

Aggregates the per-model fp8_decomp JSON outputs into a cohort table.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]


def main():
    per_model = {}
    decomps = None
    for m in MODELS:
        per_model[m] = json.loads(
            (RES / f"claim21_fp8_decomp_{m}_rho0.01.json").read_text())
        if decomps is None:
            decomps = [r["decomp"] for r in per_model[m]["results"]]

    cohort = {}
    cohort_n = sum(per_model[m]["fp8_bytes"] for m in MODELS)
    for d in decomps:
        total_b = 0
        all_ok = True
        for m in MODELS:
            r = next(x for x in per_model[m]["results"] if x["decomp"] == d)
            total_b += r["total_brotli11_bytes"]
            all_ok &= r["roundtrip_ok"]
        cohort[d] = {
            "total_brotli11_bytes": total_b,
            "bpB": 8.0 * total_b / cohort_n,
            "all_roundtrips_ok": all_ok,
        }

    raw_bpB = cohort["raw"]["bpB"]
    for d, v in cohort.items():
        v["delta_vs_raw_bpB"] = v["bpB"] - raw_bpB

    best_d = min(cohort.keys(), key=lambda k: cohort[k]["bpB"])
    out = {
        "claim": 21,
        "wave": 43,
        "experiment": "fp8_decomp_summary",
        "rho": 0.010,
        "models": MODELS,
        "cohort_fp8_bytes": cohort_n,
        "cohort_by_decomp": cohort,
        "best_decomp": best_d,
        "best_cohort_bpB": cohort[best_d]["bpB"],
        "best_gain_vs_raw_bpB": cohort[best_d]["bpB"] - raw_bpB,
        "verdict": ("RAW WINS -- no preconditioning beats brotli-11 on raw bytes"
                    if best_d == "raw" else f"PRECONDITION WINS -- {best_d}"),
    }

    print("=" * 76)
    print("Wave 43: fp8 byte decomposition + brotli-11 (cohort aggregate)")
    print("=" * 76)
    print(f"cohort fp8 bytes: {cohort_n:,}")
    print()
    print(f"{'decomp':<14}{'brotli11 bytes':>18}{'bpB':>12}{'delta vs raw':>16}{'roundtrip':>12}")
    for d in decomps:
        v = cohort[d]
        flag = "OK" if v["all_roundtrips_ok"] else "*** FAIL ***"
        print(f"{d:<14}{v['total_brotli11_bytes']:>18,}{v['bpB']:>12.5f}"
              f"{v['delta_vs_raw_bpB']:>+16.5f}{flag:>12}")
    print()
    print(f"VERDICT: {out['verdict']}")
    print(f"  best decomp: {best_d}  cohort bpB = {cohort[best_d]['bpB']:.5f}")
    print(f"  gain vs raw: {out['best_gain_vs_raw_bpB']:+.5f} bpB")
    print("=" * 76)

    out_json = RES / "claim21_fp8_decomp_summary.json"
    out_txt = RES / "claim21_fp8_decomp_summary.txt"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    txt_lines = [
        "Claim 21 wave 43: fp8 byte decomposition + brotli-11 (cohort aggregate)",
        "=" * 76,
        f"cohort fp8 bytes: {cohort_n:,}",
        "",
        f"{'decomp':<14}{'brotli11 bytes':>18}{'bpB':>12}{'delta vs raw':>16}{'roundtrip':>12}",
    ]
    for d in decomps:
        v = cohort[d]
        flag = "OK" if v["all_roundtrips_ok"] else "*** FAIL ***"
        txt_lines.append(
            f"{d:<14}{v['total_brotli11_bytes']:>18,}{v['bpB']:>12.5f}"
            f"{v['delta_vs_raw_bpB']:>+16.5f}{flag:>12}"
        )
    txt_lines += [
        "",
        f"VERDICT: {out['verdict']}",
        f"  best decomp: {best_d}  cohort bpB = {cohort[best_d]['bpB']:.5f}",
        f"  gain vs raw: {out['best_gain_vs_raw_bpB']:+.5f} bpB",
    ]
    out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(f"[wrote] {out_json}")
    print(f"[wrote] {out_txt}")


if __name__ == "__main__":
    main()
