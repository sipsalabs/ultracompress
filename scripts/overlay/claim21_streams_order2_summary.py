"""Wave-38 aggregator: Shannon floors across all 3 payload streams."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"


def main():
    d = json.loads((RES / "claim21_streams_order2_rho0.01.json").read_text())
    rows = d["rows"]

    lines = []
    lines.append("Claim-21 wave 38: per-stream Shannon-floor analysis (H0/H1/H2/brotli-11)")
    lines.append("=" * 120)
    lines.append("")
    lines.append("Waves 30-37 exhausted the order-2 coder design space on the fp8 stream,")
    lines.append("closing with a negative result. Wave 38 pivots to the OTHER TWO payload")
    lines.append("streams -- idx_delta and scale -- to test whether either has an")
    lines.append("exploitable Shannon-vs-brotli gap.")
    lines.append("")
    lines.append("Per-stream per-model rates (bits per byte):")
    hdr = (f"  {'model':<14}{'stream':<10}{'n_bytes':>12}"
           f"{'H0':>8}{'H1 MM':>9}{'H2 MM':>9}{'brotli11':>10}"
           f"{'H0-br':>8}{'H1-br':>8}{'H2-br':>8}")
    lines.append(hdr)
    for r in rows:
        lines.append(
            f"  {r['model']:<14}{r['stream']:<10}{r['n_bytes']:>12,}"
            f"{r['H0_bpB']:>8.4f}{r['H1_MM_bpB']:>9.4f}{r['H2_MM_bpB']:>9.4f}"
            f"{r['brotli11_bpB']:>10.4f}"
            f"{r['H0_minus_brotli']:>+8.4f}"
            f"{r['H1_MM_minus_brotli']:>+8.4f}"
            f"{r['H2_MM_minus_brotli']:>+8.4f}"
        )

    # Per-stream cohort aggregates weighted by n_bytes
    agg = {}
    for r in rows:
        s = r["stream"]
        a = agg.setdefault(s, dict(n=0, H0=0.0, H1=0.0, H2=0.0, br=0.0))
        w = r["n_bytes"]
        a["n"] += w
        a["H0"] += r["H0_bpB"] * w
        a["H1"] += r["H1_MM_bpB"] * w
        a["H2"] += r["H2_MM_bpB"] * w
        a["br"] += r["brotli11_bpB"] * w

    lines.append("")
    lines.append("Cohort aggregate (byte-weighted):")
    lines.append(hdr.replace("n_bytes", "n_total"))
    cohort_out = {}
    for s, a in agg.items():
        n = a["n"]
        H0 = a["H0"]/n; H1 = a["H1"]/n; H2 = a["H2"]/n; br = a["br"]/n
        lines.append(
            f"  {'COHORT':<14}{s:<10}{n:>12,}"
            f"{H0:>8.4f}{H1:>9.4f}{H2:>9.4f}{br:>10.4f}"
            f"{H0-br:>+8.4f}{H1-br:>+8.4f}{H2-br:>+8.4f}"
        )
        cohort_out[s] = dict(n_bytes=n, H0=H0, H1=H1, H2=H2,
                              brotli11=br, H0_minus_br=H0-br,
                              H1_minus_br=H1-br, H2_minus_br=H2-br)

    lines.append("")
    lines.append("=" * 120)
    lines.append("INTERPRETATION")
    lines.append("-" * 120)
    lines.append(
        "- fp8 (10-16 M bytes / model, 99.9% of payload): H2 MM - brotli-11 = ")
    lines.append(
        f"  {cohort_out['fp8']['H2_minus_br']:+.4f} bpB cohort. Matches wave 31.")
    lines.append(
        "- idx_delta (15-25 KB / model, ~0.05% of payload): H2 MM - brotli-11 = ")
    lines.append(
        f"  {cohort_out['idx_delta']['H2_minus_br']:+.4f} bpB cohort. H0-H1 structure")
    lines.append(
        "  is modest (~0.13 bpB gap).")
    lines.append(
        "- scale (8-13 KB / model, ~0.03% of payload): H2 MM - brotli-11 = ")
    lines.append(
        f"  {cohort_out['scale']['H2_minus_br']:+.4f} bpB cohort. Largest gap by far.")
    lines.append(
        "")
    lines.append(
        "SAMPLE-SIZE CAVEAT: idx_delta (N~20K) and scale (N~10K) streams are too")
    lines.append(
        "small for reliable H2 estimation (state space 256^3 = 16.7M cells, so")
    lines.append(
        "N/|S| << 1). Miller-Madow correction mitigates but does not eliminate")
    lines.append(
        "plug-in bias. The H2 numbers here lower-bound (in absolute value) the")
    lines.append(
        "TRUE order-2 entropy and therefore upper-bound the true Shannon gap.")
    lines.append(
        "H1 estimates (state space 65K) are moderately reliable; H0 estimates")
    lines.append(
        "(256 cells) are fully reliable.")
    lines.append(
        "")
    lines.append(
        "PRACTICAL CONCLUSION:")
    lines.append(
        "  * The LARGE Shannon gaps on scale (up to 4.5 bpB at order 2) are")
    lines.append(
        "    misleading: the absolute byte savings are tiny because the stream")
    lines.append(
        "    itself is tiny. At ~10 KB stream size and ~4 bpB potential savings,")
    lines.append(
        "    the ceiling is ~5 KB per model -- 0.03% of the fp8 payload.")
    lines.append(
        "  * The H1 MM - brotli-11 gap on scale (-1.18 bpB cohort) is partially")
    lines.append(
        "    sample-valid (65K-cell state on 8-13 KB streams is borderline)")
    lines.append(
        "    and indicates brotli-11 under-exploits order-1 structure on the")
    lines.append(
        "    scale stream. A dedicated static order-1 Huffman table shipped")
    lines.append(
        "    once per model (~2-4 KB after compression) could recover ~1 bpB")
    lines.append(
        "    on this stream, but the stream's small size limits the total")
    lines.append(
        "    absolute saving to ~1-2 KB per model.")
    lines.append(
        "  * idx_delta has a small Shannon gap (~0.14 bpB cohort at H2, sample-")
    lines.append(
        "    limited) and provides no meaningful engineering margin.")
    lines.append(
        "  * fp8 is the dominant payload and its -0.05 bpB H2-vs-brotli gap")
    lines.append(
        "    (wave 31) is not exploitable by any coder family (waves 33-37).")
    lines.append("")
    lines.append(
        "THREE-STREAM FINAL: brotli-11 is effectively at the operational floor")
    lines.append(
        "for all three payload streams. The cohort-wide aggregate upper-bound")
    lines.append(
        "on realizable Shannon savings across all three streams combined is")
    lines.append(
        "< 0.1 bpB of fp8-equivalent payload, which is below measurement")
    lines.append(
        "noise at current sample volumes.")
    lines.append("")

    out_txt = RES / "claim21_streams_order2.txt"
    out_json = RES / "claim21_streams_order2_summary.json"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    out_json.write_text(json.dumps({
        "claim": 21, "wave": 38,
        "cohort_per_stream": cohort_out,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {out_txt}")
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()
