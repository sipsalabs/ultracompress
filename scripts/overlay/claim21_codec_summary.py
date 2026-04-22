"""claim21_codec_summary.py -- aggregate cross-codec overlay sweeps.

Scans results/claim21_codec_sweep_*.json and reports, for each
(model, rho), the percent overlay-bit reduction achieved by each
codec. Writes results/claim21_codec_summary.{json,txt}.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = REPO / "results"

PAT = re.compile(r"^claim21_codec_sweep_(?P<model>.+?)_rho(?P<rho>0\.[0-9]+)\.json$")
CODECS = ["zstd-3", "zstd-9", "zstd-15", "zstd-22", "zlib-9", "bz2-9", "lzma-6", "brotli-11", "lz4-hc"]


def main() -> None:
    rows = []
    for p in sorted(RESULTS.glob("claim21_codec_sweep_*.json")):
        m = PAT.match(p.name)
        if not m:
            continue
        d = json.loads(p.read_text())
        cs = d.get("codec_sweep")
        if not cs:
            continue

        raw_bytes = sum(cs[s]["raw_bytes"] for s in ("fp8", "idx_delta", "scale"))
        shannon_bytes = sum(cs[s]["shannon_bytes"] for s in ("fp8", "idx_delta", "scale"))
        raw_bits = raw_bytes * 8

        row = {
            "model": m.group("model"),
            "rho": float(m.group("rho")),
            "raw_bytes": raw_bytes,
            "shannon_bytes": shannon_bytes,
            "codecs": {},
        }
        for c in CODECS:
            # skip codecs not present in this file (older sweeps had 7 codecs)
            if c not in cs["fp8"]["codecs"]:
                continue
            total = sum(cs[s]["codecs"][c]["bytes"] for s in ("fp8", "idx_delta", "scale"))
            saved_pct = 100 * (raw_bytes - total) / raw_bytes
            gap = 100 * (total * 8 - shannon_bytes * 8) / max(raw_bits - shannon_bytes * 8, 1)
            row["codecs"][c] = {
                "bytes": total,
                "saved_pct": saved_pct,
                "gap_to_shannon_pct": gap,
            }
        rows.append(row)

    rows.sort(key=lambda r: (r["rho"], r["model"]))

    out_json = RESULTS / "claim21_codec_summary.json"
    out_json.write_text(json.dumps(rows, indent=2))

    # text table
    header_codecs = " ".join(f"{c:>9}" for c in CODECS)
    hdr = f"{'model':<14} {'rho':>6} {'raw MB':>8} {header_codecs}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        savings = " ".join(
            (f"{r['codecs'][c]['saved_pct']:7.2f}% " if c in r['codecs'] else f"{'-':>9}")
            for c in CODECS
        )
        lines.append(f"{r['model']:<14} {r['rho']:>6.3f} {r['raw_bytes']/1e6:>8.2f} {savings}")

    # cohort means
    from collections import defaultdict
    by_rho = defaultdict(list)
    for r in rows:
        by_rho[r["rho"]].append(r)
    lines.append("")
    lines.append("Cohort means (savings %):")
    lines.append(hdr)
    for rho, rs in sorted(by_rho.items()):
        means = []
        for c in CODECS:
            vals = [r['codecs'][c]['saved_pct'] for r in rs if c in r['codecs']]
            if vals:
                means.append(f"{sum(vals)/len(vals):7.2f}% ")
            else:
                means.append(f"{'-':>9}")
        lines.append(f"{'MEAN':<14} {rho:>6.3f} {sum(r['raw_bytes'] for r in rs)/len(rs)/1e6:>8.2f} {' '.join(means)}")

    out_txt = RESULTS / "claim21_codec_summary.txt"
    text = "\n".join(lines)
    out_txt.write_text(text)
    print(text)
    print()
    print(f"[wrote] {out_json}")
    print(f"[wrote] {out_txt}")


if __name__ == "__main__":
    main()
