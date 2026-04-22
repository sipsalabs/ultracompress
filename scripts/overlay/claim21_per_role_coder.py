"""Claim-21 wave 29: per-role fp8 compressibility dispersion.

Wave 28 concluded that the only remaining path to sub-brotli-11 is
a stronger fp8-specific coder. Wave 29 tests one concrete hypothesis:
do the 7 roles (q/k/v/o/gate/up/down projections) have sufficiently
different fp8 byte distributions that a role-adaptive coder (brotli
or arithmetic trained per role) would beat a single cohort coder?

The test is operational. For each (model, role) from the existing
claim21_per_role_<model>_rho0.01.json artifacts, we already have the
per-role brotli-11 bytes on fp8. Compute:
  - per-role fp8 bpB = fp8_bytes_brotli * 8 / fp8_raw_bytes
  - weighted mean across roles (= aggregate brotli if codec is
    order-of-operations-invariant, which brotli is not because each
    call re-primes the dictionary)
  - aggregate single-coder brotli bpB on fp8 from wave 15

The delta
  weighted_mean_per_role - aggregate_single
is the bpB SAVED (or LOST) by running brotli independently per role
vs running it on one concatenated fp8 stream. Dictionary priming loss
vs distribution-specialization gain.

Pure aggregator over wave-20 per-role JSONs + wave-15 codec_sweep.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
MODELS = ["tinyllama", "smollm2_1.7b", "olmo2_1b", "qwen3_1.7b"]
RHO_TAG = "0.01"


def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    per_model = []
    for m in MODELS:
        pr = load(RES / f"claim21_per_role_{m}_rho{RHO_TAG}.json")
        cs = load(RES / f"claim21_codec_sweep_{m}_rho{RHO_TAG}.json")
        agg_fp8_bpB = float(
            cs["codec_sweep"]["fp8"]["codecs"]["brotli-11"]["bits_per_byte"])
        agg_fp8_raw = int(cs["codec_sweep"]["fp8"]["raw_bytes"])
        roles = pr["roles"]

        per_role_rows = []
        total_raw = 0
        total_brotli_bytes = 0
        for role, rd in roles.items():
            raw = int(rd["raw_bytes"]["fp8"])
            br_b = int(rd["codecs"]["brotli-11"]["fp8_bytes"])
            bpB = br_b * 8.0 / raw
            per_role_rows.append(dict(
                role=role, n_linears=int(rd["n_linears"]),
                raw=raw, brotli_bytes=br_b, bpB=bpB,
                frac_of_fp8=raw/agg_fp8_raw,
            ))
            total_raw += raw
            total_brotli_bytes += br_b
        # weighted mean of per-role bpB (by fp8 bytes)
        per_role_rows.sort(key=lambda r: r["bpB"])
        weighted_mean = total_brotli_bytes * 8.0 / total_raw
        bpB_min = per_role_rows[0]["bpB"]
        bpB_max = per_role_rows[-1]["bpB"]
        per_model.append(dict(
            model=m,
            agg_brotli_bpB=agg_fp8_bpB,
            per_role_weighted_mean_bpB=weighted_mean,
            per_role_min_bpB=bpB_min,
            per_role_max_bpB=bpB_max,
            spread_bpB=bpB_max - bpB_min,
            gain_vs_aggregate_bpB=agg_fp8_bpB - weighted_mean,
            roles=per_role_rows,
        ))

    lines = []
    lines.append("Claim-21 wave 29: per-role fp8 brotli-11 compressibility")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Per-role brotli-11 independently primes the dictionary per")
    lines.append("role vs a single-stream brotli on the concatenated fp8. The")
    lines.append("comparison tells us whether distribution-specialization by")
    lines.append("role beats a single coder after accounting for 7x dictionary")
    lines.append("priming overhead.")
    lines.append("")
    for pm in per_model:
        lines.append(f"--- {pm['model']}  rho=0.010 ---")
        lines.append(f"  {'role':<12}{'n_lin':>6}{'raw bytes':>12}"
                     f"{'br bpB':>10}{'frac of fp8':>14}")
        for r in pm["roles"]:
            lines.append(
                f"  {r['role']:<12}{r['n_linears']:>6}"
                f"{r['raw']:>12,}{r['bpB']:>10.4f}"
                f"{r['frac_of_fp8']*100:>13.2f}%"
            )
        lines.append(
            f"  aggregate single-coder brotli-11 fp8 bpB: "
            f"{pm['agg_brotli_bpB']:.4f}")
        lines.append(
            f"  per-role weighted mean bpB:                "
            f"{pm['per_role_weighted_mean_bpB']:.4f}")
        lines.append(
            f"  spread (max-min) across roles:             "
            f"{pm['spread_bpB']:.4f} bpB")
        lines.append(
            f"  gain (positive = per-role beats aggregate): "
            f"{pm['gain_vs_aggregate_bpB']:+.4f} bpB")
        lines.append("")

    # Cohort (byte-weighted)
    tot_raw = sum(sum(r["raw"] for r in pm["roles"]) for pm in per_model)
    tot_br  = sum(sum(r["brotli_bytes"] for r in pm["roles"]) for pm in per_model)
    tot_agg_raw = tot_raw
    tot_agg_br  = sum(int(pm["agg_brotli_bpB"]*sum(r["raw"] for r in pm["roles"])/8)
                      for pm in per_model)
    lines.append("=" * 100)
    lines.append("COHORT (byte-weighted across 4 models):")
    wm = tot_br*8.0/tot_raw
    ag = sum(pm["agg_brotli_bpB"]*sum(r["raw"] for r in pm["roles"])
             for pm in per_model) / tot_raw
    lines.append(f"  aggregate single-coder brotli-11 fp8 bpB: {ag:.4f}")
    lines.append(f"  per-role weighted mean bpB:                {wm:.4f}")
    lines.append(f"  gain (positive = per-role beats aggregate): {ag-wm:+.4f} bpB")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 100)
    lines.append(
        "Gain > 0 => per-role coder beats aggregate even with 7x priming")
    lines.append(
        "  overhead, meaning role-specific fp8 distributions differ enough")
    lines.append(
        "  to more than pay the dictionary-resetup cost.")
    lines.append(
        "Gain ~ 0 => role-adaptive coder is a wash at brotli level.")
    lines.append(
        "Gain < 0 => priming overhead dominates and a single coder wins.")
    lines.append(
        "Spread is the WITHIN-role-class bpB range: a true oracle role-")
    lines.append(
        "  adaptive coder could reach per-role-min on each role (lower")
    lines.append(
        "  bound); the weighted mean reported here uses the shipped brotli.")
    lines.append("")

    txt = "\n".join(lines) + "\n"
    (RES/"claim21_per_role_coder.txt").write_text(txt, encoding="utf-8")
    out = dict(
        claim=21, experiment="per_role_fp8_coder",
        models=MODELS, rho=0.010,
        per_model=per_model,
        cohort=dict(
            agg_bpB=ag,
            per_role_weighted_mean_bpB=wm,
            gain_bpB=ag - wm,
        ),
    )
    (RES/"claim21_per_role_coder.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(txt)
    print("[wrote] results/claim21_per_role_coder.txt")
    print("[wrote] results/claim21_per_role_coder.json")


if __name__ == "__main__":
    main()
