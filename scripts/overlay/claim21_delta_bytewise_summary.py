"""claim21_delta_bytewise_summary.py -- cohort per-byte idx_delta diag."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"
OUT = RES / "claim21_delta_bytewise.txt"

PAT = re.compile(r"claim21_delta_bytewise_(?P<model>.+?)_rho(?P<rho>[\d.]+)\.json$")


def main():
    files = sorted(RES.glob("claim21_delta_bytewise_*_rho*.json"))
    if not files:
        print("no files"); return
    lines: list[str] = []
    lines.append("Claim-21 idx_delta per-byte-position diagnostic")
    lines.append("=" * 82)
    lines.append("Reshape idx_delta as (N, 4) (little-endian int32 lanes) and")
    lines.append("compute Shannon H of each byte position independently.")
    lines.append("")
    lines.append("If the 4 byte lanes were mutually independent, any coder's")
    lines.append("achievable rate is bounded below by sum(H_pos_i) / 4 bits/byte.")
    lines.append("The gap between that bound and the as-emitted full-stream H")
    lines.append("measures cross-lane dependence (= joint savings beyond per-lane).")
    lines.append("")
    # cohort: weighted H per position
    cohort_n = {p: 0 for p in range(4)}
    cohort_nz = {p: 0.0 for p in range(4)}
    cohort_Hsum = 0.0
    cohort_bytes = 0
    cohort_H_full = 0.0
    for f in files:
        m = PAT.search(f.name)
        if not m: continue
        d = json.loads(f.read_text())
        lines.append(f"--- {d['model']}  rho={d['rho']} ---")
        lines.append(f"  n_deltas={d['n_deltas']:,}  bytes={d['idx_delta_bytes']:,}  "
                     f"full-stream H={d['full_stream_H_bpB']:.4f} bpB "
                     f"(floor {d['full_stream_floor_pct']:5.2f}%)")
        lines.append(f"  deltas: min={d['delta_stats']['min']}  "
                     f"max={d['delta_stats']['max']}  "
                     f"mean={d['delta_stats']['mean']:.2f}  "
                     f"median={d['delta_stats']['median']:.1f}  "
                     f"p99={d['delta_stats']['p99']:.1f}")
        lines.append("  byte_pos    H (bpB)  floor %%  zero %%   comment")
        zero_fracs = []
        for pb in d["per_byte_position"]:
            pos = pb["byte_position"]
            cohort_n[pos]  += d["n_deltas"]
            cohort_nz[pos] += d["n_deltas"] * pb["nonzero_fraction"]
            comment = ("LSB: primary entropy" if pos == 0
                       else "MSB-adjacent" if pos == 1
                       else "MSB lane (near-zero)")
            lines.append(f"     {pos}     {pb['shannon_H_bpB']:>7.4f}  "
                         f"{pb['order0_savings_floor_pct']:6.2f}%  "
                         f"{pb['zero_fraction']*100:6.2f}%  {comment}")
            zero_fracs.append(pb['zero_fraction'])
        cohort_Hsum += d["n_deltas"] * d["sum_per_byte_H_bits_per_int32"]
        cohort_bytes += d["idx_delta_bytes"]
        cohort_H_full += d["idx_delta_bytes"] * d["full_stream_H_bpB"]
        bound = d["sum_per_byte_H_bits_per_int32"]
        lines.append(f"  sum per-byte H = {bound:.3f} bits/int32  "
                     f"(= {bound/8:.3f} bytes/int32; floor {100*(32-bound)/32:.2f}%)")
        lines.append("")
    # cohort
    total_n = sum(cohort_n.values()) / 4  # each pos counted separately, same n per pos
    mean_Hsum = cohort_Hsum / total_n if total_n else 0.0
    mean_H_full = cohort_H_full / cohort_bytes if cohort_bytes else 0.0
    lines.append("=" * 82)
    lines.append(f"COHORT  (n={len(files)} runs, delta-weighted)")
    lines.append("-" * 82)
    for pos in range(4):
        if cohort_n[pos] == 0: continue
        nz = cohort_nz[pos] / cohort_n[pos]
        lines.append(f"  byte pos {pos}: nonzero_frac = {nz*100:6.2f}%  "
                     f"zero_frac = {(1-nz)*100:6.2f}%")
    lines.append(f"  full-stream H (byte-weighted) = {mean_H_full:.4f} bpB  "
                 f"(floor {100*(8-mean_H_full)/8:.2f}%)")
    lines.append(f"  sum per-byte H (delta-weighted) = {mean_Hsum:.3f} bits/int32")
    lines.append(f"  bytes/int32 if bytes independent = {mean_Hsum/8:.3f}")
    lines.append(f"  savings floor (32 - Hsum) / 32   = {100*(32-mean_Hsum)/32:.2f}%")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 82)
    lines.append("- MSB lanes (bytes 2, 3) have near-100%% zeros: deltas are small")
    lines.append("  positive integers (sorted-row indices advance slowly; since")
    lines.append("  restored rows are sparse, gaps between adjacent indices are")
    lines.append("  a few hundred at most). This is why byte permutation kills")
    lines.append("  savings sharply (wave 17: lzma-6 idx_delta -14.0pp): the")
    lines.append("  position-of-zero-byte information is what lzma exploits.")
    lines.append("- LSB lane (byte 0) carries most of the entropy (H near 6-7 bpB)")
    lines.append("  because the low byte of a small integer is essentially random.")
    lines.append("- The gap between sum-per-byte savings floor and the full-stream")
    lines.append("  order-0 floor measures cross-lane dependence: if Hsum/4 (in bpB)")
    lines.append("  equals full_H, byte lanes are marginally independent; any")
    lines.append("  additional savings from positional structure come from here.")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"[wrote] {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
