"""claim21_varint_emitter_summary.py -- aggregate wave-24 results.

Compares bit-level encodings vs observed idx_delta compression and
Shannon floor across the cohort. Emits
results/claim21_varint_emitter.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"

MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]


def main():
    rows = []
    for m in MODELS:
        p = RES / f"claim21_varint_emitter_{m}_rho0.01.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(d)

    lines = []
    lines.append("Claim-21 varint/rice idx_delta emitter measurement")
    lines.append("=" * 86)
    lines.append("")
    lines.append("Wave 23 predicted that a simple bit-level variable-length")
    lines.append("integer encoder would close most of the 1.55 bpB gap between")
    lines.append("observed brotli-11 idx_delta rate (4.34 bpB) and the Shannon")
    lines.append("order-0 floor (2.78 bpB). This wave measures the gap directly.")
    lines.append("")
    lines.append("All sizes reported in 'bpB(4-byte ref)' = total bits divided")
    lines.append("by the int32 raw size (so 8.0 = no compression; 2.78 = floor).")
    lines.append("")

    tot_n = 0
    tot = {"raw": 0, "leb": 0, "gamma": 0, "rice_med": 0,
           "rice_best": 0, "H": 0.0}

    for d in rows:
        n = d["n_deltas"]
        raw = d["raw_int32_bits"]
        leb = d["leb128_bits"]
        gm = d["gamma_bits"]
        rmed = d["rice_median_bits"]
        rbest = d["rice_best_bits"]
        H = d["shannon_H_total_bits"]
        lines.append(f"--- {d['model']}  n_deltas={n:,} ---")
        lines.append(f"  delta_stats: min={d['delta_stats']['min']}  "
                     f"max={d['delta_stats']['max']}  "
                     f"median={d['delta_stats']['median']}  "
                     f"p99={d['delta_stats']['p99']:.1f}")
        lines.append(
            f"  {'int32 LE (raw)':<26}  {raw/n:>8.3f} bpd   {raw/n/4:>6.3f} bpB"
        )
        lines.append(
            f"  {'LEB128 varint':<26}  {leb/n:>8.3f} bpd   {leb/n/4:>6.3f} bpB"
        )
        lines.append(
            f"  {'Elias gamma':<26}  {gm/n:>8.3f} bpd   {gm/n/4:>6.3f} bpB"
        )
        lines.append(
            f"  {'Rice k=' + str(d['rice_median_k']) + ' (med)':<26}  "
            f"{rmed/n:>8.3f} bpd   {rmed/n/4:>6.3f} bpB"
        )
        lines.append(
            f"  {'Rice k=' + str(d['rice_best_k']) + ' (best)':<26}  "
            f"{rbest/n:>8.3f} bpd   {rbest/n/4:>6.3f} bpB"
        )
        lines.append(
            f"  {'Shannon H (floor)':<26}  {H/n:>8.3f} bpd   {H/n/4:>6.3f} bpB"
        )
        lines.append("")
        tot_n += n
        tot["raw"] += raw
        tot["leb"] += leb
        tot["gamma"] += gm
        tot["rice_med"] += rmed
        tot["rice_best"] += rbest
        tot["H"] += H

    lines.append("=" * 86)
    lines.append(f"COHORT  (n_deltas total={tot_n:,})")
    lines.append("-" * 86)
    lines.append("  scheme                      bpd       bpB(4-byte ref)   savings vs raw")
    names = [
        ("int32 LE (raw)", tot["raw"]),
        ("LEB128 varint", tot["leb"]),
        ("Elias gamma", tot["gamma"]),
        ("Rice best (search)", tot["rice_best"]),
        ("Shannon H (floor)", tot["H"]),
    ]
    raw_total = tot["raw"]
    for name, tb in names:
        bpd = tb / tot_n
        bpB = bpd / 4.0
        sav = 100.0 * (1.0 - tb / raw_total)
        lines.append(f"  {name:<26}  {bpd:>8.3f}  {bpB:>12.3f}     {sav:>9.3f} %")
    lines.append("")
    lines.append("Comparison points (for context):")
    lines.append(
        "  brotli-11 observed cohort rate  (wave 23)  =  4.339 bpB   "
        "= 17.356 bpd"
    )
    lines.append(
        "  Shannon order-0 floor            (wave 19)  =  2.784 bpB   "
        "= 11.138 bpd"
    )
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 86)
    lines.append(
        "- If LEB128 / Rice-best bpB is below 4.339 bpB, a bit-level coder"
    )
    lines.append(
        "  SHIPS savings beyond brotli-11 on idx_delta. The margin is the"
    )
    lines.append(
        "  direct empirical validation of the wave-23 prediction."
    )
    lines.append(
        "- If Rice-best is near the Shannon floor, the distribution is well"
    )
    lines.append(
        "  modelled by a geometric distribution, confirming wave 21's"
    )
    lines.append(
        "  picture of small-positive-integer deltas."
    )
    lines.append("")

    (RES / "claim21_varint_emitter.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"[wrote] results/claim21_varint_emitter.txt")


if __name__ == "__main__":
    main()
