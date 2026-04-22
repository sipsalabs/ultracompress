"""Aggregate claim21_real_payload_roundtrip_*.json -> summary text.

Produces a compact per-model × per-codec × per-stream PASS/FAIL table
plus a cohort-wide pass count and aggregate compressed-bytes totals.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"

files = sorted(RESULTS.glob("claim21_real_payload_roundtrip_*_rho*.json"))
if not files:
    raise SystemExit("no roundtrip JSONs found")

CODECS = ["zstd-3","zstd-9","zstd-15","zstd-22","zlib-9","bz2-9","lzma-6","brotli-11","lz4-hc"]
STREAMS = ["fp8","idx_delta","scale"]

lines = []
lines.append("Claim 21 EMPIRICAL lossless-roundtrip verification on REAL payload bytes")
lines.append("=" * 96)
lines.append("")
lines.append("Each row: SHA-256(dec(enc(stream))) == SHA-256(stream) on the ACTUAL")
lines.append("Claim-21 overlay payload (not random bytes). 9 codecs x 3 streams per model.")
lines.append("")

total_pass = 0
total_total = 0
agg = {c: {s: {"raw": 0, "comp": 0} for s in STREAMS} for c in CODECS}

for f in files:
    data = json.loads(f.read_text())
    m = data["model"]; rho = data["rho"]
    raw = data["raw_bytes"]
    n_pass = data["pass"]; n_total = data["total"]
    total_pass += n_pass; total_total += n_total
    lines.append(f"--- {m}  rho={rho}  (raw: fp8={raw['fp8']:,}  idx_delta={raw['idx_delta']:,}  scale={raw['scale']:,}) ---")
    lines.append(f"  {'codec':<11} {'fp8 comp':>12} {'idx comp':>10} {'scale comp':>11} {'fp8 ok':>7} {'idx ok':>7} {'scale ok':>9}")
    for c in CODECS:
        if c not in data["codecs"]:
            continue
        cd = data["codecs"][c]
        fp8_c = cd["fp8"]["compressed_bytes"]; fp8_ok = cd["fp8"]["sha256_match"]
        idx_c = cd["idx_delta"]["compressed_bytes"]; idx_ok = cd["idx_delta"]["sha256_match"]
        scl_c = cd["scale"]["compressed_bytes"]; scl_ok = cd["scale"]["sha256_match"]
        agg[c]["fp8"]["raw"] += raw["fp8"];       agg[c]["fp8"]["comp"] += fp8_c
        agg[c]["idx_delta"]["raw"] += raw["idx_delta"]; agg[c]["idx_delta"]["comp"] += idx_c
        agg[c]["scale"]["raw"] += raw["scale"];   agg[c]["scale"]["comp"] += scl_c
        lines.append(f"  {c:<11} {fp8_c:>12,} {idx_c:>10,} {scl_c:>11,} {str(fp8_ok):>7} {str(idx_ok):>7} {str(scl_ok):>9}")
    lines.append(f"  [summary] {n_pass}/{n_total} PASS")
    lines.append("")

lines.append("=" * 96)
lines.append(f"Aggregate across {len(files)} model(s): {total_pass}/{total_total} SHA-256 roundtrips PASSED  ({100.0*total_pass/max(total_total,1):.4f}%)")
lines.append("-" * 96)
lines.append(f"  {'codec':<11} {'fp8 raw':>14} {'fp8 comp':>14} {'fp8 save%':>10}  {'idx raw':>10} {'idx comp':>10} {'idx save%':>10}  {'scl raw':>9} {'scl comp':>9} {'scl save%':>10}")
for c in CODECS:
    f = agg[c]["fp8"]; i = agg[c]["idx_delta"]; s = agg[c]["scale"]
    if f["raw"] == 0:
        continue
    fp8_sv = 100.0*(1.0 - f["comp"]/f["raw"])
    idx_sv = 100.0*(1.0 - i["comp"]/i["raw"])
    scl_sv = 100.0*(1.0 - s["comp"]/s["raw"])
    lines.append(f"  {c:<11} {f['raw']:>14,} {f['comp']:>14,} {fp8_sv:>9.2f}%  {i['raw']:>10,} {i['comp']:>10,} {idx_sv:>9.2f}%  {s['raw']:>9,} {s['comp']:>9,} {scl_sv:>9.2f}%")

lines.append("")
lines.append("INTERPRETATION")
lines.append("-" * 96)
lines.append("- Every codec is a published lossless standard (zstd RFC 8478, zlib RFC")
lines.append("  1950/51, bzip2 spec, LZMA/xz, brotli RFC 7932, LZ4 frame). The SHA-256")
lines.append("  match on the REAL Claim-21 payload bytes (as opposed to the earlier")
lines.append("  random-buffer verification) is the strongest empirical evidence of")
lines.append("  losslessness: bit-exact recovery on the distribution the claim produces.")
lines.append("- Savings here use the real payload byte distribution, so aggregate")
lines.append("  percentages reflect the actual Claim-21 deployment shape (not random).")

out_path = RESULTS / "claim21_real_payload_roundtrip.txt"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[wrote] {out_path}")
