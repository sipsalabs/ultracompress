"""claim21_stream_independence_summary.py -- aggregate concat-vs-split.

Reads claim21_stream_independence_<model>_rho<rho>.json files and
reports the cohort-mean gap between single-buffer concat compression
and per-stream split compression.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"
OUT = RES / "claim21_stream_independence.txt"

PAT = re.compile(r"claim21_stream_independence_(?P<model>.+?)_rho(?P<rho>[\d.]+)\.json$")


def main():
    files = sorted(RES.glob("claim21_stream_independence_*_rho*.json"))
    runs = []
    for f in files:
        m = PAT.search(f.name)
        if not m:
            continue
        runs.append((m.group("model"), m.group("rho"), json.loads(f.read_text())))
    if not runs:
        print("no files"); return

    codecs = sorted({c for (_, _, d) in runs for c in d["by_codec"].keys()})

    lines: list[str] = []
    lines.append("Claim-21 3-stream decomposition: concat-vs-split empirical test")
    lines.append("=" * 82)
    lines.append("SPLIT  = Claim 21's emission: sum_s | codec(stream_s) |")
    lines.append("CONCAT = alternative coder:   | codec(fp8 || idx_delta || scale) |")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  concat-split > 0  => 3-stream split is STRICTLY BETTER for this coder")
    lines.append("                       (streams have different byte distributions;")
    lines.append("                        concatenation dilutes each distribution's")
    lines.append("                        memoryless entropy by Jensen's inequality).")
    lines.append("  concat-split < 0  => a single coder captures cross-stream structure")
    lines.append("                       the 3-stream split was leaving on the table.")
    lines.append("")

    for model, rho, d in runs:
        raw = d["raw_sizes"]
        lines.append(f"--- {model}  rho={rho}  n_linears={d['n_linears']} ---")
        lines.append(
            f"  raw: fp8={raw['fp8']:,}  idx={raw['idx_delta']:,}  "
            f"scale={raw['scale']:,}  concat={raw['concat']:,}"
        )
        lines.append(f"  {'codec':<12} {'split':>12} {'concat':>12} "
                     f"{'split%':>8} {'concat%':>8} {'cs-gap%':>9}")
        lines.append("  " + "-" * 68)
        for c in codecs:
            if c not in d["by_codec"]:
                continue
            bc = d["by_codec"][c]
            lines.append(
                f"  {c:<12} {bc['split_bytes']['total']:>12,} "
                f"{bc['concat_bytes']:>12,} "
                f"{bc['split_vs_raw_pct']:>7.3f}% "
                f"{bc['concat_vs_raw_pct']:>7.3f}% "
                f"{bc['concat_vs_split_pct']:>+8.3f}%"
            )
        lines.append("")

    # Cohort: raw-byte-weighted mean of (concat-split) gap, per codec.
    lines.append("=" * 82)
    lines.append(f"COHORT  (n={len(runs)} runs, size-weighted by raw bytes)")
    lines.append("-" * 82)
    lines.append(f"  {'codec':<12} {'split_tot':>14} {'concat_tot':>14} "
                 f"{'split%':>8} {'concat%':>8} {'cs-gap%':>9}")
    lines.append("  " + "-" * 72)
    for c in codecs:
        raw_tot = sum(d["raw_sizes"]["concat"] for (_, _, d) in runs
                      if c in d["by_codec"])
        split_tot = sum(d["by_codec"][c]["split_bytes"]["total"] for (_, _, d) in runs
                        if c in d["by_codec"])
        concat_tot = sum(d["by_codec"][c]["concat_bytes"] for (_, _, d) in runs
                         if c in d["by_codec"])
        split_pct = 100.0 * (raw_tot - split_tot) / raw_tot
        concat_pct = 100.0 * (raw_tot - concat_tot) / raw_tot
        cs_gap = 100.0 * (concat_tot - split_tot) / split_tot
        lines.append(
            f"  {c:<12} {split_tot:>14,} {concat_tot:>14,} "
            f"{split_pct:>7.3f}% {concat_pct:>7.3f}% {cs_gap:>+8.3f}%"
        )

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"[wrote] {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
