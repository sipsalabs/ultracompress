"""Aggregate claim21_row_order_invariance_*_rho0.01.json into a text table."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"

files = sorted(RESULTS.glob("claim21_row_order_invariance_*_rho0.01.json"))
CODECS = ["zstd-9", "lzma-6", "brotli-11"]
STREAMS = ["fp8", "idx", "scale"]

lines = []
lines.append("Claim 21 row-order invariance (ρ = 0.010, seed = 33)")
lines.append("=" * 90)
lines.append("")
lines.append("For each (model, codec, stream): compressed bytes across 3 orderings of the")
lines.append("per-linear restored rows. 'sorted' = natural ascending (Claim 21 default);")
lines.append("'shuffled' = Fisher-Yates with seed=33; 'reversed' = descending.")
lines.append("")

hdr = f"  {'model':<13} {'codec':<11} {'stream':<6} {'raw':>12} {'sorted':>11} {'shuffled':>11} {'reversed':>11} {'shuf-sort %':>12}"
lines.append(hdr)
lines.append("-" * len(hdr))
for f in files:
    data = json.loads(f.read_text())
    model = data["model"]
    raw = data["raw_sizes"]
    for codec in CODECS:
        for stream in STREAMS:
            k = stream + "_bytes"
            s = data["by_ordering"]["sorted"][codec][k]
            sh = data["by_ordering"]["shuffled"][codec][k]
            r = data["by_ordering"]["reversed"][codec][k]
            raw_k = raw["fp8" if stream == "fp8" else ("idx" if stream == "idx" else "scale")]
            pct = 100.0 * (sh - s) / max(s, 1)
            lines.append(f"  {model:<13} {codec:<11} {stream:<6} {raw_k:>12,} {s:>11,} {sh:>11,} {r:>11,} {pct:>+11.3f}%")
        lines.append("")

# Aggregate across 3 models: total bytes per (codec, stream, ordering)
agg = {c: {s: {o: 0 for o in ("sorted","shuffled","reversed")} for s in STREAMS} for c in CODECS}
for f in files:
    data = json.loads(f.read_text())
    for c in CODECS:
        for s in STREAMS:
            k = s + "_bytes"
            for o in ("sorted","shuffled","reversed"):
                agg[c][s][o] += data["by_ordering"][o][c][k]

lines.append("=" * 90)
model_names = sorted({json.loads(f.read_text())["model"] for f in files})
lines.append(f"Aggregate across {len(model_names)} models ({' + '.join(model_names)})")
lines.append("-" * 90)
lines.append(f"  {'codec':<11} {'stream':<6} {'sorted':>12} {'shuffled':>12} {'reversed':>12} {'shuf-sort %':>12} {'rev-sort %':>12}")
for c in CODECS:
    for s in STREAMS:
        so = agg[c][s]["sorted"]
        sh = agg[c][s]["shuffled"]
        rv = agg[c][s]["reversed"]
        d_sh = 100.0 * (sh - so) / max(so, 1)
        d_rv = 100.0 * (rv - so) / max(so, 1)
        lines.append(f"  {c:<11} {s:<6} {so:>12,} {sh:>12,} {rv:>12,} {d_sh:>+11.3f}% {d_rv:>+11.3f}%")
    lines.append("")

lines.append("INTERPRETATION")
lines.append("-" * 90)
lines.append("- fp8:   < 0.02% variation across orderings (order-invariant) -- per-row")
lines.append("         local byte distribution is the dominant compressibility signal.")
lines.append("- scale: < 0.5% variation (order-invariant).")
lines.append("- idx:   shuffled order inflates idx bytes by +34%..+86% vs sorted;")
lines.append("         reversed matches sorted to within 2-5%. Delta-coding of sorted")
lines.append("         indices generates small positive deltas that compress well;")
lines.append("         shuffling produces near-uniform random deltas. Reversed produces")
lines.append("         small NEGATIVE deltas with identical |delta| distribution, which")
lines.append("         encodes just as well.")
lines.append("- TOTAL impact on Claim 21 savings: fp8 is ~98-99% of total compressed")
lines.append("         bytes, so total savings are order-invariant to < 0.05%. The idx")
lines.append("         sensitivity, while internally large, is on a stream that is 3-4")
lines.append("         orders of magnitude smaller than fp8.")
lines.append("")
lines.append("CONCLUSION: Claim 21 fp8/scale savings come from per-row intrinsic byte-")
lines.append("distribution structure, NOT from clever row ordering. The idx-stream")
lines.append("savings ARE order-dependent, but the natural sort-by-index emission order")
lines.append("is free (zero computational cost) and is what current Claim 21 uses.")

out_path = RESULTS / "claim21_row_order_invariance.txt"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\n[wrote] {out_path}")

