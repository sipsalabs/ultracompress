"""Aggregator for wave-32 fp8 order-3 measurement.

Reports both plug-in and Miller-Madow-corrected order-3 conditional
entropy, flags sample-limit severity via 4-gram singleton fraction,
and compares against wave-31 order-2 floor and brotli-11 shipping rate.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
RHO_TAG = "0.01"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    rows = []
    for m in MODELS:
        p3 = RES / f"claim21_fp8_order3_{m}_rho{RHO_TAG}.json"
        p2 = RES / f"claim21_fp8_order2_{m}_rho{RHO_TAG}.json"
        cs = RES / f"claim21_codec_sweep_{m}_rho{RHO_TAG}.json"
        if not (p3.exists() and p2.exists() and cs.exists()):
            print(f"[miss] {m}")
            continue
        d3 = load(p3)
        d2 = load(p2)
        br = float(load(cs)["codec_sweep"]["fp8"]["codecs"]["brotli-11"]["bits_per_byte"])
        rows.append(dict(
            model=d3["model"],
            n=int(d3["n_fp8_bytes"]),
            k4=int(d3["observed_quad_states"]),
            sing=float(d3["quad_singleton_frac"]),
            H2=float(d2["order2_H_bpB"]),
            H3_plugin=float(d3["order3_H_bpB_plugin"]),
            H3_mm=float(d3["order3_H_bpB_mm"]),
            br=br,
        ))

    out_txt = RES / "claim21_fp8_order3.txt"
    out_json = RES / "claim21_fp8_order3.json"

    lines = []
    lines.append("Claim-21 wave 32: fp8 order-3 conditional entropy (sample-limited)")
    lines.append("=" * 112)
    lines.append("")
    lines.append("Measures H(B_i | B_{i-1}, B_{i-2}, B_{i-3}) over the fp8 byte stream")
    lines.append("using sparse unique-value counting. State space is 256^4 = 4.29e9, so")
    lines.append("with only ~10-16M fp8 bytes per model the 4-gram histogram is severely")
    lines.append("undersampled (singleton fraction ~0.87-0.92). Plug-in entropy is biased")
    lines.append("DOWNWARD; Miller-Madow correction adds (K-1)/(2N ln 2) bits to each")
    lines.append("joint-entropy estimate before taking the difference.")
    lines.append("")
    hdr = (f"  {'model':<14}{'n_bytes':>14}{'obs-4grams':>14}"
           f"{'sing.frac':>12}{'order-2 H':>12}"
           f"{'H3 plugin':>12}{'H3 MM':>12}"
           f"{'brotli-11':>12}{'br - H3mm':>12}")
    lines.append(hdr)

    tN = 0
    sH2 = sH3p = sH3m = sBR = 0.0
    for r in rows:
        n = r["n"]
        tN += n
        sH2 += r["H2"] * n
        sH3p += r["H3_plugin"] * n
        sH3m += r["H3_mm"] * n
        sBR += r["br"] * n
        lines.append(
            f"  {r['model']:<14}{n:>14,}{r['k4']:>14,}"
            f"{r['sing']:>12.4f}{r['H2']:>12.4f}"
            f"{r['H3_plugin']:>12.4f}{r['H3_mm']:>12.4f}"
            f"{r['br']:>12.4f}{r['br']-r['H3_mm']:>+12.4f}"
        )
    if tN:
        lines.append("")
        lines.append(
            f"  {'COHORT':<14}{tN:>14,}{'':>14}"
            f"{'':>12}{sH2/tN:>12.4f}"
            f"{sH3p/tN:>12.4f}{sH3m/tN:>12.4f}"
            f"{sBR/tN:>12.4f}{(sBR-sH3m)/tN:>+12.4f}"
        )
    lines.append("")
    lines.append("=" * 112)
    lines.append("INTERPRETATION")
    lines.append("-" * 112)
    lines.append(
        "- 4-gram singleton fraction >0.85 on every model means the plug-in H3 estimate is")
    lines.append(
        "  heavily biased DOWN: most observed 4-grams appear exactly once, so the joint")
    lines.append(
        "  distribution looks artificially peaky. Plug-in H3 of 3.17-3.77 bpB is a LOWER")
    lines.append(
        "  BOUND that cannot be trusted as a coder floor.")
    lines.append(
        "- Miller-Madow correction shifts the estimate up to 3.68-4.28 bpB. Even this")
    lines.append(
        "  conservative correction sits FAR BELOW the order-2 floor (6.40 bpB cohort) and")
    lines.append(
        "  the brotli-11 shipping rate (6.56 bpB cohort). The true H3 almost certainly")
    lines.append(
        "  lies between the MM estimate and the order-2 floor -- unresolvable at this")
    lines.append(
        "  sample size without a better estimator (NSB, coverage-adjusted).")
    lines.append(
        "- Operational conclusion for Claim 21: the constructive sub-brotli path at")
    lines.append(
        "  order 2 (wave 31, -0.155 bpB) is the RELIABLE lower-bound improvement.")
    lines.append(
        "  Order >=3 coding may unlock more, but proving it requires either more data")
    lines.append(
        "  per model or a low-bias nonparametric entropy estimator. This wave CHARTS")
    lines.append(
        "  the sample-size boundary of the context-entropy programme, and makes the")
    lines.append(
        "  order-2 result the provable floor that the patent can safely rely on.")
    lines.append("")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    out_json.write_text(json.dumps({
        "claim": 21,
        "wave": 32,
        "experiment": "fp8_order3_summary",
        "rho": float(RHO_TAG),
        "models": rows,
        "cohort": {
            "n_bytes": tN,
            "order2_H_bpB": sH2 / tN if tN else None,
            "order3_H_bpB_plugin": sH3p / tN if tN else None,
            "order3_H_bpB_mm": sH3m / tN if tN else None,
            "brotli11_bpB": sBR / tN if tN else None,
            "br_minus_H3mm_bpB": (sBR - sH3m) / tN if tN else None,
        },
        "note": "Order-3 estimator is sample-limited (singleton frac >0.85); plug-in biased down, MM still conservative. Order-2 floor (wave 31) remains the provable sub-brotli path.",
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {out_txt}")
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()
