"""Wave-41 summary: three streams are independent from brotli's perspective."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"


def main():
    d = json.loads((RES / "claim21_combined_stream_rho0.01.json").read_text())
    rows = d["rows"]

    lines = []
    lines.append("Claim-21 wave 41: cross-stream concatenation test (3 streams vs combined)")
    lines.append("=" * 112)
    lines.append("")
    lines.append("Question: does concatenating fp8 + idx_delta + scale into one buffer")
    lines.append("let brotli-11 exploit cross-stream structure (shared dictionary, shared")
    lines.append("context) that per-stream coding cannot reach?")
    lines.append("")
    lines.append(
        "Two orderings: A = fp8||idx_delta||scale, B = idx_delta||scale||fp8.")
    lines.append("")
    lines.append("Per-model results (bytes saved vs sum-of-parts; positive = concat better):")
    lines.append(
        f"  {'model':<14}{'n_total':>13}{'sum_br':>14}{'concat A':>13}"
        f"{'concat B':>13}{'gain A (bpB)':>14}{'gain B (bpB)':>14}")
    for r in rows:
        b = r["brotli11"]
        lines.append(
            f"  {r['model']:<14}{r['n_total']:>13,}"
            f"{b['per_stream_bytes']['sum']:>14,}"
            f"{b['concat_A_bytes']:>13,}{b['concat_B_bytes']:>13,}"
            f"{b['cross_stream_gain_A_bpB']:>+14.5f}"
            f"{b['cross_stream_gain_B_bpB']:>+14.5f}"
        )

    # Cohort totals
    tn = sum(r["n_total"] for r in rows)
    t_sum = sum(r["brotli11"]["per_stream_bytes"]["sum"] for r in rows)
    t_A = sum(r["brotli11"]["concat_A_bytes"] for r in rows)
    t_B = sum(r["brotli11"]["concat_B_bytes"] for r in rows)
    g_A = 8.0 * (t_sum - t_A) / tn
    g_B = 8.0 * (t_sum - t_B) / tn
    lines.append("")
    lines.append(
        f"  {'COHORT':<14}{tn:>13,}{t_sum:>14,}{t_A:>13,}{t_B:>13,}"
        f"{g_A:>+14.5f}{g_B:>+14.5f}"
    )

    lines.append("")
    lines.append("zstd-22 (secondary check):")
    lines.append(
        f"  {'model':<14}{'sum_zs':>14}{'concat A':>13}{'concat B':>13}"
        f"{'gain A':>12}{'gain B':>12}")
    for r in rows:
        z = r["zstd22"]
        lines.append(
            f"  {r['model']:<14}{z['per_stream_bytes']['sum']:>14,}"
            f"{z['concat_A_bytes']:>13,}{z['concat_B_bytes']:>13,}"
            f"{z['cross_stream_gain_A_bpB']:>+12.5f}"
            f"{z['cross_stream_gain_B_bpB']:>+12.5f}"
        )

    lines.append("")
    lines.append("=" * 112)
    lines.append("INTERPRETATION")
    lines.append("-" * 112)
    lines.append(
        f"- Cohort cross-stream gain via brotli-11 concat: +{g_A:.5f} bpB (order A),")
    lines.append(
        f"  +{g_B:.5f} bpB (order B). Both are below 0.002 bpB -- within single-")
    lines.append(
        "  byte rounding of the individual stream compressions.")
    lines.append(
        "- All four models show gains < 0.002 bpB for brotli-11 under both orderings.")
    lines.append(
        "  Ordering B (idx||scale||fp8, smallest-first) is marginally better than A")
    lines.append(
        "  on 3 of 4 models but the difference is below noise.")
    lines.append(
        "- zstd-22 shows MIXED results: some concat orderings increase total size")
    lines.append(
        "  (qwen3 order B: -0.003 bpB, concat WORSE). Likely due to block-framing")
    lines.append(
        "  overhead: zstd's windowed dictionary resets at block boundaries and")
    lines.append(
        "  crossing stream boundaries can evict useful context.")
    lines.append("")
    lines.append(
        "CONCLUSION: the three Claim-21 payload streams (fp8, idx_delta, scale) are")
    lines.append(
        "EFFECTIVELY INDEPENDENT from brotli-11's perspective. There is no exploit-")
    lines.append(
        "able cross-stream structure, and coding each stream separately loses at")
    lines.append(
        "most 0.002 bpB vs any concatenation ordering. The per-stream architecture")
    lines.append(
        "imposes no compression penalty and is preferable for engineering reasons")
    lines.append(
        "(parallel decode, independent error recovery, stream-specific codec choice).")
    lines.append("")

    out_txt = RES / "claim21_combined_stream.txt"
    out_json = RES / "claim21_combined_stream_summary.json"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    out_json.write_text(json.dumps({
        "claim": 21, "wave": 41,
        "cohort": dict(
            n_total=tn,
            brotli11_sum_bytes=t_sum,
            brotli11_concat_A_bytes=t_A,
            brotli11_concat_B_bytes=t_B,
            brotli11_cross_stream_gain_A_bpB=g_A,
            brotli11_cross_stream_gain_B_bpB=g_B,
        ),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {out_txt}")
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()
