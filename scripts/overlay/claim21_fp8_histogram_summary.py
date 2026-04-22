"""claim21_fp8_histogram_summary.py -- cohort order-0 summary."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"
OUT = RES / "claim21_fp8_histogram.txt"

PAT = re.compile(r"claim21_fp8_histogram_(?P<model>.+?)_rho(?P<rho>[\d.]+)\.json$")


def main():
    files = sorted(RES.glob("claim21_fp8_histogram_*_rho*.json"))
    if not files:
        print("no files"); return
    lines: list[str] = []
    lines.append("Claim-21 order-0 byte-histogram diagnostic")
    lines.append("=" * 82)
    lines.append("For each stream, measure how far the byte distribution deviates")
    lines.append("from uniform. These deviations are the fundamental source of")
    lines.append("order-0 compressibility (wave 17 established that fp8 savings")
    lines.append("are ~89% order-0). Here we quantify 'how compressible must it")
    lines.append("be in principle', as an information-theoretic lower bound on")
    lines.append("bits/byte no coder can beat without exploiting context.")
    lines.append("")
    lines.append("Columns:")
    lines.append("  H (bits/B)  = Shannon entropy of byte histogram (uniform = 8.0000)")
    lines.append("  floor %     = order-0 savings floor = 100 * (8 - H) / 8")
    lines.append("                (= lower bound on savings achievable by any coder)")
    lines.append("  TV          = total-variation distance from uniform (0..1)")
    lines.append("  chi2/n      = chi-square / n_bytes (deviation, scale-free)")
    lines.append("  max/mean    = most-frequent-byte count / expected count at uniform")
    lines.append("")

    # Cohort aggregate (weighted by byte count)
    cohort: dict[str, dict[str, float]] = {}

    for f in files:
        m = PAT.search(f.name)
        if not m: continue
        d = json.loads(f.read_text())
        lines.append(f"--- {d['model']}  rho={d['rho']} ---")
        hdr = "  stream       n_bytes    H (bpB)  floor%     TV   chi2/n  max/mean  min/mean"
        lines.append(hdr)
        for sname in ("fp8", "idx_delta", "scale"):
            s = d["streams"][sname]
            lines.append(
                f"  {sname:10}  {s['n_bytes']:>10,}  {s['shannon_bits_per_byte']:7.4f}"
                f"  {s['order0_savings_floor_pct']:6.3f}%"
                f"  {s['tv_distance_from_uniform']:.4f}"
                f"  {s['chi2_vs_uniform']/s['n_bytes']:.4f}"
                f"  {s['max_count_over_mean']:6.2f}"
                f"  {s['min_count_over_mean']:6.3f}"
            )
            # cohort accumulation: H weighted by byte count; TV,max likewise
            c = cohort.setdefault(sname, {"n": 0.0, "H_num": 0.0, "tv_num": 0.0,
                                          "max": 0.0, "min": 1e9})
            c["n"]     += s["n_bytes"]
            c["H_num"] += s["n_bytes"] * s["shannon_bits_per_byte"]
            c["tv_num"]+= s["n_bytes"] * s["tv_distance_from_uniform"]
            c["max"]    = max(c["max"], s["max_count_over_mean"])
            c["min"]    = min(c["min"], s["min_count_over_mean"])
        lines.append("")

    lines.append("=" * 82)
    lines.append(f"COHORT  (n={len(files)} runs, byte-weighted)")
    lines.append("-" * 82)
    lines.append("  stream      bytes         H (bpB)   floor %    TV      max/mean  min/mean")
    for sname in ("fp8", "idx_delta", "scale"):
        c = cohort[sname]
        H = c["H_num"] / c["n"]; tv = c["tv_num"] / c["n"]
        floor = 100.0 * (8.0 - H) / 8.0
        lines.append(
            f"  {sname:10}  {int(c['n']):>12,}  {H:7.4f}  {floor:6.3f}%"
            f"  {tv:.4f}  {c['max']:7.2f}  {c['min']:6.3f}"
        )
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 82)
    lines.append("- fp8 H < 8 bpB directly: its byte histogram is measurably")
    lines.append("  sub-uniform. The order-0 savings floor is the minimum")
    lines.append("  compression ratio any coder (even one that ignores context)")
    lines.append("  could achieve. This is the fundamental cause of the")
    lines.append("  ~16 pp of fp8 savings that survive byte permutation (wave 17).")
    lines.append("- Hadamard rotation + FP8 quantization produces a byte")
    lines.append("  distribution that is provably, quantitatively non-uniform;")
    lines.append("  the sub-uniformity is a property of the encoding, not of")
    lines.append("  any particular model.")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"[wrote] {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
