"""Aggregate per-model claim21_entropy_*.json outputs into a single table.

Emits:
  results/claim21_summary.json  (list[dict])
  results/claim21_summary.txt   (human-readable table)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = REPO / "results"

# Canonical sweep outputs only. Require a decimal point in rho to reject legacy
# "rho003" (=0.03) style filenames that were produced before the sweep existed.
PAT = re.compile(r"^claim21_entropy_(?P<model>.+?)_rho(?P<rho>0\.[0-9]+)\.json$")


def main() -> None:
    rows: list[dict] = []
    for p in sorted(RESULTS.glob("claim21_entropy_*.json")):
        m = PAT.match(p.name)
        if not m:
            continue
        j = json.loads(p.read_text())
        pb = j["payload_bytes"]
        H = j["shannon_entropy_bits_per_byte"]
        # Shannon lower bound on each stream: H (bits/byte) * raw_bytes / 8 = entropy-optimal bytes.
        shannon_bytes = (
            H["fp8"]   * pb["fp8_raw"]       / 8.0
            + H["idx"] * pb["idx_delta_raw"] / 8.0
            + H["scale"] * pb["scale_raw"]   / 8.0
        )
        zstd_bytes = pb["fp8_zstd22"] + pb["idx_delta_zstd"] + pb["scale_zstd"]
        # Overhead: row-index field (32 bits/row).
        n_rows = j["n_restored_rows"]
        idx_bits = 32 * n_rows
        shannon_bits = shannon_bytes * 8 + idx_bits
        zstd_bits    = zstd_bytes    * 8 + idx_bits
        raw_bits     = (pb["fp8_raw"] + pb["idx_delta_raw"] + pb["scale_raw"]) * 8 + idx_bits
        n_total_params = j["n_total_params"]
        shannon_overlay_bpw = shannon_bits / n_total_params
        zstd_overlay_bpw    = zstd_bits    / n_total_params
        # Gap: how much above the Shannon floor does zstd land, as % of raw->Shannon savings.
        raw_overlay_bpw = raw_bits / n_total_params
        if raw_overlay_bpw > shannon_overlay_bpw:
            gap_pct = 100.0 * (zstd_overlay_bpw - shannon_overlay_bpw) / (raw_overlay_bpw - shannon_overlay_bpw)
        else:
            gap_pct = 0.0
        rows.append({
            "model": m.group("model"),
            "rho": float(m.group("rho")),
            "restored_rows": n_rows,
            "total_rows": j["n_total_rows"],
            "base_bpw": round(j["base_bpw"], 4),
            "old_overlay_bpw": round(j["old_overlay_bpw"], 4),
            "new_overlay_bpw": round(j["new_overlay_bpw"], 4),
            "shannon_overlay_bpw": round(shannon_overlay_bpw, 4),
            "old_total_bpw": round(j["old_effective_bpw"], 4),
            "new_total_bpw": round(j["new_effective_bpw"], 4),
            "shannon_total_bpw": round(j["base_bpw"] + shannon_overlay_bpw, 4),
            "saved_pct_of_overlay_bits": round(j["saved_pct_of_overlay_bits"], 2),
            "zstd_gap_to_shannon_pct": round(gap_pct, 2),
            "fp8_entropy_bits_per_byte": round(H["fp8"], 3),
        })

    rows.sort(key=lambda r: (r["model"], r["rho"]))
    out_json = RESULTS / "claim21_summary.json"
    out_json.write_text(json.dumps(rows, indent=2))

    # Text table
    header = (
        f"{'model':<16} {'rho':>7} {'rows/total':>17} "
        f"{'old bpw':>9} {'zstd bpw':>9} {'Shannon':>9} {'saved %':>8} {'gap->H %':>9}"
    )
    lines = [
        "Claim 21 - Entropy coding of the overlay payload",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for r in rows:
        lines.append(
            f"{r['model']:<16} {r['rho']:>7.3f} "
            f"{r['restored_rows']:>8d}/{r['total_rows']:<8d} "
            f"{r['old_total_bpw']:>9.4f} {r['new_total_bpw']:>9.4f} "
            f"{r['shannon_total_bpw']:>9.4f} "
            f"{r['saved_pct_of_overlay_bits']:>7.2f}% "
            f"{r['zstd_gap_to_shannon_pct']:>8.2f}%"
        )
    txt = "\n".join(lines) + "\n"
    (RESULTS / "claim21_summary.txt").write_text(txt)
    print(txt)
    print(f"[summary] wrote {out_json} and claim21_summary.txt ({len(rows)} rows)")


if __name__ == "__main__":
    main()
