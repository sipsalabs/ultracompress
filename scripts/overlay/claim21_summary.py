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
        rows.append({
            "model": m.group("model"),
            "rho": float(m.group("rho")),
            "restored_rows": j["n_restored_rows"],
            "total_rows": j["n_total_rows"],
            "base_bpw": round(j["base_bpw"], 4),
            "old_overlay_bpw": round(j["old_overlay_bpw"], 4),
            "new_overlay_bpw": round(j["new_overlay_bpw"], 4),
            "old_total_bpw": round(j["old_effective_bpw"], 4),
            "new_total_bpw": round(j["new_effective_bpw"], 4),
            "saved_pct_of_overlay_bits": round(j["saved_pct_of_overlay_bits"], 2),
            "fp8_entropy_bits_per_byte": round(j["shannon_entropy_bits_per_byte"]["fp8"], 3),
        })

    rows.sort(key=lambda r: (r["model"], r["rho"]))
    out_json = RESULTS / "claim21_summary.json"
    out_json.write_text(json.dumps(rows, indent=2))

    # Text table
    header = (
        f"{'model':<16} {'rho':>7} {'rows/total':>17} "
        f"{'old bpw':>9} {'new bpw':>9} {'saved %':>8} {'H fp8':>6}"
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
            f"{r['saved_pct_of_overlay_bits']:>7.2f}% {r['fp8_entropy_bits_per_byte']:>6.3f}"
        )
    txt = "\n".join(lines) + "\n"
    (RESULTS / "claim21_summary.txt").write_text(txt)
    print(txt)
    print(f"[summary] wrote {out_json} and claim21_summary.txt ({len(rows)} rows)")


if __name__ == "__main__":
    main()
