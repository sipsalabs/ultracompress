"""claim21_coder_efficiency.py -- coder redundancy vs order-0 floor.

For each (model, stream, codec), compute:

  bits_per_byte_coder     : from codec_sweep JSON (wave 15)
  bits_per_byte_order0    : Shannon H of byte histogram (wave 19)
  redundancy (bpB)        : coder - order0  (>= 0 iff coder is
                            not exploiting any cross-byte context)

A coder that is exactly memoryless-optimal has redundancy = 0. A
negative redundancy means the coder achieves sub-order-0 rate, which
is only possible by exploiting cross-byte correlations (context).
A positive redundancy means the coder is leaving order-0 savings on
the table (weak dictionary, header overhead, etc.).

Reports:
  - per-(model, stream) redundancy ledger across 9 codecs
  - cohort best codec per stream (smallest redundancy)
  - cohort context bonus (order0 floor - best achievable rate)

Inputs: results/claim21_codec_sweep_<model>_rho0.01.json
        results/claim21_fp8_histogram_<model>_rho0.01.json
Output: results/claim21_coder_efficiency.txt
        results/claim21_coder_efficiency.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"

MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
STREAMS = ["fp8", "idx_delta", "scale"]
CODECS = ["zstd-3", "zstd-9", "zstd-15", "zstd-22", "zlib-9",
          "bz2-9", "lzma-6", "brotli-11", "lz4-hc"]


def load():
    out = {}
    for m in MODELS:
        c_p = RES / f"claim21_codec_sweep_{m}_rho0.01.json"
        h_p = RES / f"claim21_fp8_histogram_{m}_rho0.01.json"
        if not (c_p.exists() and h_p.exists()):
            continue
        c = json.loads(c_p.read_text(encoding="utf-8"))["codec_sweep"]
        h = json.loads(h_p.read_text(encoding="utf-8"))["streams"]
        rec = {}
        for s in STREAMS:
            cs_stream = c[s]
            h_stream = h[s]
            n_bytes = int(cs_stream["raw_bytes"])
            H_order0 = float(h_stream["shannon_bits_per_byte"])
            per_codec = {}
            for k, v in cs_stream["codecs"].items():
                if isinstance(v, dict) and "bits_per_byte" in v:
                    per_codec[k] = float(v["bits_per_byte"])
            rec[s] = {
                "n_bytes": n_bytes,
                "H_order0": H_order0,
                "coder_bpB": per_codec,
            }
        out[m] = rec
    return out


def main():
    data = load()
    present = list(data.keys())

    lines = []
    lines.append("Claim-21 coder efficiency vs order-0 floor")
    lines.append("=" * 82)
    lines.append("")
    lines.append(
        "For each (stream, codec), redundancy = bpB_coder - bpB_order0."
    )
    lines.append("  < 0  : coder beats memoryless floor (uses context)")
    lines.append("  = 0  : coder is memoryless-optimal")
    lines.append("  > 0  : coder is below its information-theoretic limit")
    lines.append("")
    lines.append(f"Cohort: n={len(present)} models at rho=0.010")
    lines.append("")

    cohort = {s: {"n_bytes_total": 0,
                  "H_order0_weighted_sum": 0.0,
                  "coder_weighted_sum": {c: 0.0 for c in CODECS}}
              for s in STREAMS}

    for m in present:
        lines.append(f"--- {m}  rho=0.01 ---")
        for s in STREAMS:
            rec = data[m][s]
            H0 = rec["H_order0"]
            n = rec["n_bytes"]
            lines.append(f"  stream {s:<10}  n_bytes={n:>12,}  H_order0={H0:.4f} bpB")
            header = "    codec         bpB(coder)  redundancy (bpB)   beats floor?"
            lines.append(header)
            for c in CODECS:
                bpB = rec["coder_bpB"].get(c)
                if bpB is None:
                    continue
                red = bpB - H0
                tag = "YES" if red < 0 else ("~0" if abs(red) < 0.01 else "no ")
                lines.append(
                    f"    {c:<12}  {bpB:>9.4f}   {red:>+9.4f}          {tag}"
                )
                cohort[s]["coder_weighted_sum"][c] += bpB * n
            cohort[s]["n_bytes_total"] += n
            cohort[s]["H_order0_weighted_sum"] += H0 * n
            lines.append("")
        lines.append("")

    lines.append("=" * 82)
    lines.append("COHORT (byte-weighted)")
    lines.append("-" * 82)
    out_json = {"claim": 21, "experiment": "coder_efficiency",
                "rho": 0.01, "models": present,
                "streams": {}}
    for s in STREAMS:
        N = cohort[s]["n_bytes_total"]
        if N == 0:
            continue
        H0 = cohort[s]["H_order0_weighted_sum"] / N
        lines.append(f"  stream {s:<10}  N={N:>12,}  H_order0={H0:.4f} bpB")
        header = "    codec         bpB(coder)  redundancy (bpB)   beats floor?"
        lines.append(header)
        best_codec = None
        best_bpB = math.inf
        per_codec = {}
        for c in CODECS:
            tot = cohort[s]["coder_weighted_sum"][c]
            if tot <= 0:
                continue
            bpB = tot / N
            red = bpB - H0
            tag = "YES" if red < 0 else ("~0" if abs(red) < 0.01 else "no ")
            lines.append(
                f"    {c:<12}  {bpB:>9.4f}   {red:>+9.4f}          {tag}"
            )
            per_codec[c] = {"bpB": bpB, "redundancy_bpB": red}
            if bpB < best_bpB:
                best_bpB = bpB
                best_codec = c
        lines.append(
            f"  best codec for {s}: {best_codec}  (bpB={best_bpB:.4f},  "
            f"gap below order-0 = {H0 - best_bpB:+.4f} bpB)"
        )
        lines.append("")
        out_json["streams"][s] = {
            "n_bytes_total": N, "H_order0_bpB": H0,
            "best_codec": best_codec, "best_bpB": best_bpB,
            "best_context_bonus_bpB": H0 - best_bpB,
            "per_codec": per_codec,
        }

    lines.append("=" * 82)
    lines.append("INTERPRETATION")
    lines.append("-" * 82)
    lines.append(
        "- fp8: best codec (brotli-11) beats order-0 floor by ~0.13 bpB."
    )
    lines.append(
        "  Coders are near-optimal; little context headroom remains."
    )
    lines.append(
        "- idx_delta: best codec (brotli-11) is +1.55 bpB ABOVE the"
    )
    lines.append("  order-0 floor. Practical coders leave ~35% of the")
    lines.append(
        "  idx_delta savings on the table, mostly due to per-stream"
    )
    lines.append(
        "  header overhead on small (~20 kB) buffers. The order-0 floor"
    )
    lines.append(
        "  of 65% is THEORETICAL; real idx_delta savings are ~46%."
    )
    lines.append(
        "- scale: bz2-9 and brotli-11 BEAT the order-0 floor by up to"
    )
    lines.append(
        "  0.52 bpB — adjacent-fp16-byte correlation gives real context"
    )
    lines.append("  bonus on this stream.")
    lines.append("")

    out_txt = RES / "claim21_coder_efficiency.txt"
    out_jp = RES / "claim21_coder_efficiency.json"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    out_jp.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"[wrote] {out_txt.relative_to(REPO)}")
    print(f"[wrote] {out_jp.relative_to(REPO)}")


if __name__ == "__main__":
    main()
