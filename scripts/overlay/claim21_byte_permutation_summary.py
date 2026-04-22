"""claim21_byte_permutation_summary.py -- aggregate permutation test.

Reports per-codec per-stream "context gap" = orig savings pp - permuted
savings pp. Large gap => coder extracts higher-order context. Small
gap => coder is near-order-0 on this stream.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"
OUT = RES / "claim21_byte_permutation.txt"

PAT = re.compile(r"claim21_byte_permutation_(?P<model>.+?)_rho(?P<rho>[\d.]+)\.json$")
STREAMS = ("fp8", "idx_delta", "scale")


def main():
    files = sorted(RES.glob("claim21_byte_permutation_*_rho*.json"))
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
    lines.append("Claim-21 byte-permutation test: order-0 vs higher-order context")
    lines.append("=" * 82)
    lines.append("orig   = codec savings on Claim 21 emission (sorted ordering).")
    lines.append("perm   = codec savings on the SAME bytes after uniform byte-permutation")
    lines.append("         (preserves order-0 histogram; destroys all local structure).")
    lines.append("gap pp = orig pp - perm pp  (how many pp of savings came from context).")
    lines.append("")

    for model, rho, d in runs:
        lines.append(f"--- {model}  rho={rho} ---")
        raw = d["raw_sizes"]
        lines.append(f"  raw: fp8={raw['fp8']:,}  idx_delta={raw['idx_delta']:,}  "
                     f"scale={raw['scale']:,}")
        for c in codecs:
            if c not in d["by_codec"]:
                continue
            bc = d["by_codec"][c]
            lines.append(f"  {c}")
            lines.append(f"    {'stream':<10} {'orig%':>8} {'perm%':>8} {'gap_pp':>9}")
            for s in STREAMS:
                ps = bc[s]
                lines.append(
                    f"    {s:<10} "
                    f"{ps['orig_pct']:>7.3f}% "
                    f"{ps['perm_pct']:>7.3f}% "
                    f"{ps['context_gap_pp']:>+8.3f}"
                )
        lines.append("")

    # Cohort: size-weighted mean per (codec, stream).
    lines.append("=" * 82)
    lines.append(f"COHORT  (n={len(runs)} runs, size-weighted over raw bytes per stream)")
    lines.append("-" * 82)
    lines.append(f"  {'codec':<12} {'stream':<11} {'orig%':>8} {'perm%':>8} {'gap_pp':>9}")
    lines.append("  " + "-" * 55)
    for c in codecs:
        for s in STREAMS:
            raw_tot = sum(d["raw_sizes"][s] for (_, _, d) in runs if c in d["by_codec"])
            orig_tot = sum(d["by_codec"][c][s]["orig_bytes"] for (_, _, d) in runs
                           if c in d["by_codec"])
            perm_tot = sum(d["by_codec"][c][s]["perm_bytes"] for (_, _, d) in runs
                           if c in d["by_codec"])
            o_pct = 100.0 * (raw_tot - orig_tot) / raw_tot
            p_pct = 100.0 * (raw_tot - perm_tot) / raw_tot
            gap = o_pct - p_pct
            lines.append(
                f"  {c:<12} {s:<11} "
                f"{o_pct:>7.3f}% "
                f"{p_pct:>7.3f}% "
                f"{gap:>+8.3f}"
            )
        lines.append("")

    lines.append("=" * 82)
    lines.append("INTERPRETATION")
    lines.append("--------------")
    lines.append("- gap_pp near 0 => the coder is already operating at the order-0 Shannon")
    lines.append("  limit of that stream. Byte permutation cannot hurt a coder that")
    lines.append("  only uses marginal byte statistics.")
    lines.append("- gap_pp > 0   => the coder is finding higher-order context (e.g., run")
    lines.append("  length, repeated subsequences, delta patterns). Byte permutation")
    lines.append("  destroys this structure while preserving the histogram.")
    lines.append("- Histogram equality between orig and perm is asserted at encode time")
    lines.append("  (np.bincount match) so the order-0 distribution is provably identical.")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"[wrote] {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
