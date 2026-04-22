"""claim21_per_role_summary.py -- per-role cohort aggregate."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"
OUT = RES / "claim21_per_role.txt"

PAT = re.compile(r"claim21_per_role_(?P<model>.+?)_rho(?P<rho>[\d.]+)\.json$")
REPORT_CODECS = ("zstd-9", "lzma-6", "brotli-11")
ROLE_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")


def main():
    files = sorted(RES.glob("claim21_per_role_*_rho*.json"))
    if not files:
        print("no files"); return
    lines: list[str] = []
    lines.append("Claim-21 per-role savings breakdown")
    lines.append("=" * 82)
    lines.append("Compress the Claim 21 payload separately for each linear role.")
    lines.append("Tests whether brotli-11 ~18% savings is uniform across roles")
    lines.append("or concentrated in attention vs MLP.")
    lines.append("")

    # cohort: byte-weighted savings per (role, codec)
    cohort_raw = {r: 0 for r in ROLE_ORDER}
    cohort_cmp = {r: {c: 0 for c in REPORT_CODECS} for r in ROLE_ORDER}

    for f in files:
        m = PAT.search(f.name)
        if not m: continue
        d = json.loads(f.read_text())
        lines.append(f"--- {d['model']}  rho={d['rho']} ---")
        hdr = "  role            n_lin   raw_B  "
        for c in REPORT_CODECS: hdr += f"  {c:>11}"
        lines.append(hdr)
        for role in ROLE_ORDER:
            if role not in d["roles"]: continue
            r = d["roles"][role]
            raw = r["raw_bytes"]["total"]
            row = f"  {role:<13}  {r['n_linears']:>5}  {raw:>8,}"
            for c in REPORT_CODECS:
                if c in r["codecs"]:
                    row += f"  {r['codecs'][c]['saved_pct']:>10.3f}%"
                    cohort_raw[role] += raw
                    cohort_cmp[role][c] += r["codecs"][c]["total_bytes"]
                else:
                    row += f"  {'n/a':>11}"
            lines.append(row)
        lines.append("")

    # dedupe raw: we added raw once per codec, so actual raw = cohort_raw/3 (3 codecs)
    # simpler: recompute raw separately
    cohort_raw2 = {r: 0 for r in ROLE_ORDER}
    for f in files:
        d = json.loads(f.read_text())
        for role in ROLE_ORDER:
            if role in d["roles"]:
                cohort_raw2[role] += d["roles"][role]["raw_bytes"]["total"]

    lines.append("=" * 82)
    lines.append(f"COHORT  (n={len(files)} runs, byte-weighted)")
    lines.append("-" * 82)
    hdr = "  role            raw_B      "
    for c in REPORT_CODECS: hdr += f"  {c:>11}"
    lines.append(hdr)
    for role in ROLE_ORDER:
        if cohort_raw2[role] == 0: continue
        row = f"  {role:<13}  {cohort_raw2[role]:>12,}"
        for c in REPORT_CODECS:
            saved = 100.0 * (cohort_raw2[role] - cohort_cmp[role][c]) / cohort_raw2[role]
            row += f"  {saved:>10.3f}%"
        lines.append(row)
    lines.append("")
    # per-codec spread across roles
    lines.append("SPREAD across roles per codec (cohort):")
    for c in REPORT_CODECS:
        saved_per_role = []
        for role in ROLE_ORDER:
            if cohort_raw2[role] == 0: continue
            saved_per_role.append(100.0 * (cohort_raw2[role] - cohort_cmp[role][c])
                                   / cohort_raw2[role])
        lo, hi = min(saved_per_role), max(saved_per_role)
        mean = sum(saved_per_role) / len(saved_per_role)
        lines.append(f"  {c:<12}  min={lo:6.3f}%  max={hi:6.3f}%  "
                     f"range={hi-lo:5.3f}pp  mean={mean:6.3f}%")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 82)
    lines.append("- A small range across roles means the Claim 21 savings mechanism")
    lines.append("  is uniformly applicable to every transformer linear, not a")
    lines.append("  role-specific trick (e.g. not unique to attention or to MLP).")
    lines.append("- A large range would indicate the effect is concentrated in a")
    lines.append("  subset of roles; the split should then be reported by role.")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"[wrote] {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
