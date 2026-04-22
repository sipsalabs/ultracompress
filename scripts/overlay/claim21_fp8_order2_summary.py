"""Aggregator for wave-31 fp8 order-2."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
RHO_TAG = "0.01"


def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    rows = []
    for m in MODELS:
        d1 = load(RES / f"claim21_fp8_order1_{m}_rho{RHO_TAG}.json")
        d2 = load(RES / f"claim21_fp8_order2_{m}_rho{RHO_TAG}.json")
        cs = load(RES / f"claim21_codec_sweep_{m}_rho{RHO_TAG}.json")
        br = float(cs["codec_sweep"]["fp8"]["codecs"]["brotli-11"]["bits_per_byte"])
        rows.append(dict(
            model=m, n=d2["n_fp8_bytes"],
            H0=d1["order0_H_bpB"], H1=d1["order1_H_bpB"],
            H2=d2["order2_H_bpB"], br=br,
        ))

    lines = []
    lines.append("Claim-21 wave 31: fp8 order-2 conditional entropy")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Extends wave 30 one context position deeper:")
    lines.append("  H(B_i | B_{i-1}, B_{i-2}) via the empirical 256^3")
    lines.append("joint histogram. Tells us how much of brotli-11's")
    lines.append("sub-order-1 advantage is attributable to order-2.")
    lines.append("")
    lines.append(f"  {'model':<14}{'n_bytes':>14}"
                 f"{'order-0':>10}{'order-1':>10}{'order-2':>10}"
                 f"{'brotli-11':>12}{'br - H2':>10}")
    tN=0; sH0=sH1=sH2=sBR=0.0
    for r in rows:
        n = r["n"]; tN += n
        sH0 += r["H0"]*n; sH1 += r["H1"]*n; sH2 += r["H2"]*n; sBR += r["br"]*n
        lines.append(
            f"  {r['model']:<14}{n:>14,}"
            f"{r['H0']:>10.4f}{r['H1']:>10.4f}{r['H2']:>10.4f}"
            f"{r['br']:>12.4f}{r['br']-r['H2']:>+10.4f}"
        )
    lines.append("")
    lines.append(
        f"  {'COHORT':<14}{tN:>14,}"
        f"{sH0/tN:>10.4f}{sH1/tN:>10.4f}{sH2/tN:>10.4f}"
        f"{sBR/tN:>12.4f}{(sBR-sH2)/tN:>+10.4f}"
    )
    lines.append("")
    lines.append("Context gains per model:")
    lines.append(f"  {'model':<14}{'H0 -> H1':>12}{'H1 -> H2':>12}{'H0 -> H2':>12}")
    sG1=sG2=0.0
    for r in rows:
        n = r["n"]
        g1 = r["H0"]-r["H1"]; g2 = r["H1"]-r["H2"]
        sG1 += g1*n; sG2 += g2*n
        lines.append(f"  {r['model']:<14}{g1:>+12.4f}{g2:>+12.4f}{g1+g2:>+12.4f}")
    lines.append(
        f"  {'COHORT':<14}{sG1/tN:>+12.4f}{sG2/tN:>+12.4f}{(sG1+sG2)/tN:>+12.4f}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("INTERPRETATION")
    lines.append("-" * 100)
    lines.append("- order-2 H is the tight lower bound for any coder that")
    lines.append("  conditions on 2 previous bytes (65536 contexts x 256")
    lines.append("  symbols = 16.8M states).")
    lines.append("- 'br - H2' is how far brotli-11 sits above that floor.")
    lines.append("  Still negative => brotli-11 is using order >= 3 context.")
    lines.append("  Near-zero or positive => brotli-11 is essentially")
    lines.append("  capturing all context-2 and no more.")
    lines.append("- H1 -> H2 marginal gain quantifies how much additional")
    lines.append("  savings a context-2 coder gets over a context-1 coder.")
    lines.append("")

    txt = "\n".join(lines) + "\n"
    (RES/"claim21_fp8_order2.txt").write_text(txt, encoding="utf-8")
    out = dict(
        claim=21, experiment="fp8_order2", models=MODELS, rho=0.010,
        per_model=rows,
        cohort=dict(
            n=tN, H0=sH0/tN, H1=sH1/tN, H2=sH2/tN, brotli_11=sBR/tN,
            br_minus_H2=(sBR-sH2)/tN,
            gain_0_to_1=sG1/tN, gain_1_to_2=sG2/tN, gain_0_to_2=(sG1+sG2)/tN,
        ),
    )
    (RES/"claim21_fp8_order2.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(txt)
    print("[wrote] results/claim21_fp8_order2.txt")
    print("[wrote] results/claim21_fp8_order2.json")


if __name__ == "__main__":
    main()
