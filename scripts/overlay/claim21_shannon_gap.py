"""Claim 21 per-stream Shannon-gap analysis.

For each (model, rho, stream) in the existing claim21_codec_sweep_*.json
files, compute:
  H (Shannon, marginal byte-entropy, bits/byte)
  best LZ-family coder bits/byte (min over zstd-3/9/15/22, zlib-9, lzma-6)
  gap_pct = 100 * (best - H) / H

Positive gap = coder is above Shannon floor (slack vs marginal entropy).
Negative gap = coder is SUB-Shannon on marginal byte entropy — i.e. the
coder is exploiting multi-byte Markov context that marginal byte-entropy
cannot see. This is the patent's "sub-Shannon evidence" for multi-byte
structure in the overlay payload.

Emits:
  results/claim21_shannon_gap.json
  results/claim21_shannon_gap.txt
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PATTERN = re.compile(r"^claim21_codec_sweep_(?P<model>.+?)_rho(?P<rho>0\.[0-9]+)\.json$")

LZ_CODERS = ["zstd-3", "zstd-9", "zstd-15", "zstd-22", "zlib-9", "lzma-6"]
STREAMS = ["fp8", "idx_delta", "scale"]


def main() -> None:
    rows: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("claim21_codec_sweep_*.json")):
        m = PATTERN.match(path.name)
        if not m:
            continue
        model = m.group("model")
        rho = float(m.group("rho"))
        with path.open("r") as f:
            data = json.load(f)
        cs = data.get("codec_sweep")
        if not cs:
            continue
        row = {"model": model, "rho": rho, "streams": {}}
        for stream in STREAMS:
            s = cs[stream]
            H = s["shannon_bits_per_byte"]
            coders = s["codecs"]
            best_name = None
            best_bpb = None
            for name in LZ_CODERS:
                if name not in coders:
                    continue
                bpb = coders[name]["bits_per_byte"]
                if best_bpb is None or bpb < best_bpb:
                    best_bpb = bpb
                    best_name = name
            gap_pct = 100.0 * (best_bpb - H) / H if H > 0 else float("nan")
            row["streams"][stream] = {
                "shannon_bpb": H,
                "best_lz_bpb": best_bpb,
                "best_lz_name": best_name,
                "gap_pct": gap_pct,
                "lzma6_bpb": coders["lzma-6"]["bits_per_byte"],
                "lzma6_gap_pct": 100.0 * (coders["lzma-6"]["bits_per_byte"] - H) / H if H > 0 else float("nan"),
            }
        rows.append(row)

    rows.sort(key=lambda r: (r["rho"], r["model"]))

    out_json = RESULTS_DIR / "claim21_shannon_gap.json"
    with out_json.open("w") as f:
        json.dump({"rows": rows}, f, indent=2)

    lines: list[str] = []
    lines.append("Per-stream Shannon-gap analysis (gap_pct = 100*(best_LZ_bpb - H)/H; negative = sub-Shannon)")
    lines.append("")
    lines.append(f"{'model':<14} {'rho':>6}  {'fp8 H':>7} {'fp8 LZ*':>8} {'fp8 gap':>8}  {'idx H':>7} {'idx LZ*':>8} {'idx gap':>8}  {'scl H':>7} {'scl LZ*':>8} {'scl gap':>8}")
    lines.append("-" * 140)
    for r in rows:
        parts = [f"{r['model']:<14}", f"{r['rho']:>6.3f}"]
        for stream in STREAMS:
            s = r["streams"][stream]
            parts.append(f" {s['shannon_bpb']:>7.3f} {s['best_lz_bpb']:>8.3f} {s['gap_pct']:>+7.2f}%")
        lines.append("".join(parts))

    # Per-rho cohort means
    lines.append("")
    lines.append("Cohort means per rho (n=6):")
    lines.append(f"{'MEAN':<14} {'rho':>6}  {'fp8 H':>7} {'fp8 LZ*':>8} {'fp8 gap':>8}  {'idx H':>7} {'idx LZ*':>8} {'idx gap':>8}  {'scl H':>7} {'scl LZ*':>8} {'scl gap':>8}")
    rhos = sorted({r["rho"] for r in rows})
    for rho in rhos:
        group = [r for r in rows if r["rho"] == rho]
        if not group:
            continue
        parts = ["MEAN          ", f"{rho:>6.3f}"]
        for stream in STREAMS:
            H = sum(r["streams"][stream]["shannon_bpb"] for r in group) / len(group)
            bpb = sum(r["streams"][stream]["best_lz_bpb"] for r in group) / len(group)
            gap = sum(r["streams"][stream]["gap_pct"] for r in group) / len(group)
            parts.append(f" {H:>7.3f} {bpb:>8.3f} {gap:>+7.2f}%")
        lines.append("".join(parts))

    # Summary counts
    lines.append("")
    sub_shannon_counts = {s: sum(1 for r in rows if r["streams"][s]["gap_pct"] < 0) for s in STREAMS}
    lines.append(f"Sub-Shannon rows (best LZ < H) out of {len(rows)}: "
                 f"fp8={sub_shannon_counts['fp8']}  idx_delta={sub_shannon_counts['idx_delta']}  scale={sub_shannon_counts['scale']}")

    out_txt = RESULTS_DIR / "claim21_shannon_gap.txt"
    out_txt.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[wrote] {out_json}")
    print(f"[wrote] {out_txt}")


if __name__ == "__main__":
    main()
