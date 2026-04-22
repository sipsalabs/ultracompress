"""claim21_block_shuffle_summary.py -- context-scale report."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"
OUT = RES / "claim21_block_shuffle.txt"

PAT = re.compile(r"claim21_block_shuffle_(?P<model>.+?)_rho(?P<rho>[\d.]+)\.json$")
CODECS_REPORT = ("zstd-9", "lzma-6", "brotli-11")


def main():
    files = sorted(RES.glob("claim21_block_shuffle_*_rho*.json"))
    if not files:
        print("no files"); return
    lines: list[str] = []
    lines.append("Claim-21 block-shuffle: characteristic context scale per stream")
    lines.append("=" * 82)
    lines.append("Shuffle each stream in blocks of size B, then compress.")
    lines.append("B=1   : full byte permutation (order-0 only).")
    lines.append("B=full: as-emitted.")
    lines.append("The block size at which savings saturate to within codec noise")
    lines.append("of as-emitted is the characteristic context scale for that coder/stream.")
    lines.append("")

    for f in files:
        m = PAT.search(f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        lines.append(f"--- {d['model']}  rho={d['rho']} ---")
        raw = d["raw_sizes"]
        lines.append(f"  raw: fp8={raw['fp8']:,}  idx_delta={raw['idx_delta']:,}  "
                     f"scale={raw['scale']:,}")
        for stream in ("fp8", "idx_delta", "scale"):
            bs = d["by_stream"][stream]
            lines.append(f"  stream = {stream}")
            block_keys = [k for k in bs.keys() if k != "full"]
            # sort by int
            block_keys.sort(key=lambda x: int(x))
            full = bs["full"]["orig_encoded"]
            # header
            hdr = "    " + "block".rjust(7)
            for c in CODECS_REPORT:
                hdr += f"  {c:>11}"
            lines.append(hdr)
            for b in block_keys:
                row = "    " + f"{b:>7}"
                for c in CODECS_REPORT:
                    row += f"  {bs[b][c]['pct']:>10.3f}%"
                lines.append(row)
            row = "    " + f"{'full':>7}"
            for c in CODECS_REPORT:
                row += f"  {full[c]['pct']:>10.3f}%"
            lines.append(row)
            # characteristic scale = smallest B where pct within 0.2pp of full
            lines.append("    characteristic scale B* (pct within 0.2pp of full):")
            for c in CODECS_REPORT:
                target = full[c]["pct"]
                b_star = None
                for b in block_keys:
                    if target - bs[b][c]["pct"] <= 0.2:
                        b_star = b; break
                lines.append(f"      {c:<12} B* = {b_star}")
            lines.append("")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"[wrote] {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
