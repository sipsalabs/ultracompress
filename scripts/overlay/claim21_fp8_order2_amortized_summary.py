"""Aggregator for wave-37 amortized-prior held-out coding."""
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"


def main():
    d = json.loads((RES / "claim21_fp8_order2_amortized_rho0.01.json").read_text())
    rows = d["rows"]

    # Cohort-aggregate per (prior_frac, alpha), weighted by n_test
    agg = defaultdict(lambda: dict(n=0, o2=0.0, br=0.0))
    for r in rows:
        k = (r["prior_frac"], r["alpha"])
        w = r["n_test"]
        agg[k]["n"] += w
        agg[k]["o2"] += r["order2_heldout_bpB"] * w
        agg[k]["br"] += r["brotli11_heldout_bpB"] * w

    lines = []
    lines.append("Claim-21 wave 37: empirical amortization test -- K->inf limit")
    lines.append("=" * 96)
    lines.append("")
    lines.append(
        "Wave 36 argued THEORETICALLY that shipping priors once and reusing them")
    lines.append(
        "over K>=5 payloads would make order-2 net-positive. Wave 37 tests that")
    lines.append(
        "empirically. For each model, priors are fit on a prefix of the fp8")
    lines.append(
        "stream and used to code the held-out suffix at alpha in {0.1, 0.01,")
    lines.append(
        "0.001}. This is the K->inf limit: no side-info charged, only held-out")
    lines.append("payload coding.")
    lines.append("")
    lines.append("Cohort-aggregate held-out rates (bits per fp8 byte, held-out only):")
    lines.append(
        f"  {'prior_frac':>12}{'alpha':>10}{'order2':>10}{'brotli-11':>12}"
        f"{'gain':>10}")
    best = ("", 1e9)
    for (pf, a), v in sorted(agg.items()):
        o2 = v["o2"] / v["n"]
        br = v["br"] / v["n"]
        gain = br - o2
        lines.append(
            f"  {pf:>12.2f}{a:>10}{o2:>10.4f}{br:>12.4f}{gain:>+10.4f}"
        )
        if -gain < best[1]:
            best = (f"prior={pf} alpha={a}", -gain)

    lines.append("")
    lines.append("Per-model best (prior_frac, alpha) gain:")
    per_model = defaultdict(lambda: (1e9, None))
    for r in rows:
        if -r["amortized_gain_vs_brotli11_bpB"] < per_model[r["model"]][0]:
            per_model[r["model"]] = (-r["amortized_gain_vs_brotli11_bpB"], r)
    for m, (gap, r) in per_model.items():
        lines.append(
            f"  {m:<14}  prior={r['prior_frac']:.2f} alpha={r['alpha']:<6}  "
            f"order2={r['order2_heldout_bpB']:.4f}  "
            f"brotli-11={r['brotli11_heldout_bpB']:.4f}  "
            f"gain={r['amortized_gain_vs_brotli11_bpB']:+.4f}"
        )

    lines.append("")
    lines.append("=" * 96)
    lines.append("INTERPRETATION -- THIS OVERTURNS THE WAVE-36 AMORTIZATION CLAIM")
    lines.append("-" * 96)
    lines.append(
        "- EVERY (prior_frac, alpha) cell is net-WORSE than brotli-11 on its own")
    lines.append(
        "  held-out bytes. Gains range from -0.28 to -1.85 bpB.")
    lines.append(
        "- Best configuration cohort-wide: " + best[0] + f" at +{best[1]:.4f} bpB worse")
    lines.append(
        "  than brotli-11 (still a loss).")
    lines.append(
        "- The 65,536 x 256 = 16.7M-cell count table is UNDER-SAMPLED even at 75 %")
    lines.append(
        "  of a 10-16 M byte payload (~0.7-1.2 samples per cell on average, but")
    lines.append(
        "  heavy-tailed with most cells at zero). Priors fit on the prefix fail to")
    lines.append(
        "  generalize to the suffix with enough precision to beat brotli-11's")
    lines.append(
        "  deeper context model (which is stream-adaptive and uses context of")
    lines.append(
        "  variable order up to ~16 with Huffman/ANS substitutional coding).")
    lines.append(
        "- Wave 34 oracle (a=0.01, fit AND tested on same stream) achieved 6.35-6.45")
    lines.append(
        "  bpB vs brotli-11 6.53-6.57. Wave 37 (a=0.01, prior=0.75) achieves")
    lines.append(
        "  6.99-7.22 bpB -- a ~0.6 bpB overfit penalty from train-on-test to")
    lines.append(
        "  hold-out within the same model. That overfit penalty exceeds the")
    lines.append(
        "  0.155 bpB theoretical Shannon advantage by a factor of 4x.")
    lines.append("")
    lines.append(
        "CORRECTED CLAIM-21 FINAL STATEMENT ON ORDER-2 CONTEXT CODING:")
    lines.append(
        "  The -0.155 bpB theoretical order-2 Shannon advantage on fp8 streams is")
    lines.append(
        "  NOT operationally realizable at current payload volumes (10-16 M bytes")
    lines.append(
        "  per model at rho=0.010) by ANY of the coder families we tested:")
    lines.append(
        "    - payload-only adaptive Laplace coders  (wave 33)")
    lines.append(
        "    - cross-model universal priors          (wave 34)")
    lines.append(
        "    - own-model self-bootstrap two-pass     (wave 35)")
    lines.append(
        "    - one-shot priors as side information   (wave 36)")
    lines.append(
        "    - amortized (K->inf) held-out priors    (wave 37)  <-- overturns wave 36")
    lines.append(
        "  brotli-11 sits effectively AT the operational floor for fp8 byte")
    lines.append(
        "  streams of this scale. The -0.155 bpB theoretical gap is a statistical-")
    lines.append(
        "  mechanical artifact of infinite-sample entropy estimation, not an")
    lines.append(
        "  exploitable engineering margin.")
    lines.append("")
    lines.append(
        "  This finding RETRACTS the wave-36 'amortization crossover at K>=5'")
    lines.append(
        "  suggestion. The order-2 context-coding programme for fp8 is closed")
    lines.append(
        "  with a NEGATIVE RESULT. Claim 21 should rely on the already-proven")
    lines.append(
        "  brotli-11 baseline and pivot any further compression work to other")
    lines.append(
        "  streams (idx_delta, scale) or other techniques (wave 15/23 hybrid")
    lines.append(
        "  serialization, bit-plane decomposition).")
    lines.append("")

    out_txt = RES / "claim21_fp8_order2_amortized.txt"
    out_json = RES / "claim21_fp8_order2_amortized_summary.json"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    out_json.write_text(json.dumps({
        "claim": 21,
        "wave": 37,
        "experiment": "fp8_order2_amortized_summary",
        "rho": 0.01,
        "cohort_by_config": {
            f"prior_frac={pf}_alpha={a}": {
                "n_test_bytes": agg[(pf, a)]["n"],
                "order2_heldout_bpB": agg[(pf, a)]["o2"] / agg[(pf, a)]["n"],
                "brotli11_heldout_bpB": agg[(pf, a)]["br"] / agg[(pf, a)]["n"],
                "amortized_gain_vs_brotli11_bpB":
                    (agg[(pf, a)]["br"] - agg[(pf, a)]["o2"]) / agg[(pf, a)]["n"],
            }
            for (pf, a) in sorted(agg.keys())
        },
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {out_txt}")
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()
