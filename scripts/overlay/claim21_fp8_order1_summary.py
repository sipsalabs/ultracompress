"""Aggregator for wave-30 fp8 order-1 measurement."""
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
        p = RES / f"claim21_fp8_order1_{m}_rho{RHO_TAG}.json"
        if not p.exists():
            print(f"[miss] {p}"); continue
        d = load(p)
        # pull brotli-11 fp8 bpB from wave-15 codec_sweep
        cs = load(RES / f"claim21_codec_sweep_{m}_rho{RHO_TAG}.json")
        br = float(cs["codec_sweep"]["fp8"]["codecs"]["brotli-11"]["bits_per_byte"])
        # pull all codec bpBs on fp8
        codecs = {k: float(v["bits_per_byte"])
                  for k, v in cs["codec_sweep"]["fp8"]["codecs"].items()}
        rows.append(dict(
            model=d["model"], n_bytes=int(d["n_fp8_bytes"]),
            H0=d["order0_H_bpB"], H1=d["order1_H_bpB"],
            gain=d["context_gain_bpB"], br=br, codecs=codecs,
        ))

    lines = []
    lines.append("Claim-21 wave 30: fp8 order-1 conditional entropy")
    lines.append("=" * 98)
    lines.append("")
    lines.append("Measures H(B_i | B_{i-1}) over the full fp8 byte stream of")
    lines.append("each model at rho=0.010, the information-theoretic lower")
    lines.append("bound for any byte-context-1 entropy coder. Comparison:")
    lines.append("  - order-0 H     : wave 19 baseline (memoryless floor)")
    lines.append("  - order-1 H     : this wave (tight lower bound at order 1)")
    lines.append("  - brotli-11 bpB : shipping coder (wave 15/23)")
    lines.append("")
    hdr = (f"  {'model':<14}{'n_bytes':>14}"
           f"{'order-0 H':>12}{'order-1 H':>12}"
           f"{'gain':>10}{'brotli-11':>12}"
           f"{'br - H1':>10}")
    lines.append(hdr)
    tN = 0; sH0 = 0.0; sH1 = 0.0; sBR = 0.0
    for r in rows:
        n = r["n_bytes"]; tN += n
        sH0 += r["H0"] * n; sH1 += r["H1"] * n; sBR += r["br"] * n
        lines.append(
            f"  {r['model']:<14}{n:>14,}"
            f"{r['H0']:>12.4f}{r['H1']:>12.4f}"
            f"{r['gain']:>+10.4f}{r['br']:>12.4f}"
            f"{r['br']-r['H1']:>+10.4f}"
        )
    if tN:
        lines.append("")
        lines.append(
            f"  {'COHORT':<14}{tN:>14,}"
            f"{sH0/tN:>12.4f}{sH1/tN:>12.4f}"
            f"{(sH0-sH1)/tN:>+10.4f}{sBR/tN:>12.4f}"
            f"{(sBR-sH1)/tN:>+10.4f}"
        )
    lines.append("")
    lines.append("=" * 98)
    lines.append("INTERPRETATION")
    lines.append("-" * 98)
    lines.append(
        "- 'gain' = H0 - H1 = how many bits B_{i-1} tells you about B_i.")
    lines.append(
        "  Wave 23 showed brotli-11 beats order-0 by 0.13 bpB on fp8, so")
    lines.append(
        "  brotli-11 already captures AT LEAST 0.13 bpB of context. The")
    lines.append(
        "  gain value here is the TIGHT upper bound on byte-context-1")
    lines.append(
        "  savings: any order-1 coder (including brotli-11) cannot beat")
    lines.append(
        "  order-0 by more than this.")
    lines.append(
        "- 'br - H1' = how far above the order-1 floor brotli-11 sits.")
    lines.append(
        "  Negative => brotli-11 is using longer-range context (order >=")
    lines.append(
        "  2) to beat the pure order-1 coder. Positive => there is")
    lines.append(
        "  residual context-1 headroom brotli-11 hasn't captured.")
    lines.append("")

    txt = "\n".join(lines) + "\n"
    (RES/"claim21_fp8_order1.txt").write_text(txt, encoding="utf-8")
    out = dict(
        claim=21, experiment="fp8_order1", models=MODELS, rho=0.010,
        per_model=rows,
        cohort=dict(
            n_bytes=tN,
            H0=sH0/tN if tN else 0.0,
            H1=sH1/tN if tN else 0.0,
            gain=(sH0-sH1)/tN if tN else 0.0,
            brotli_11=sBR/tN if tN else 0.0,
            brotli_minus_H1=(sBR-sH1)/tN if tN else 0.0,
        ) if tN else {},
    )
    (RES/"claim21_fp8_order1.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(txt)
    print("[wrote] results/claim21_fp8_order1.txt")
    print("[wrote] results/claim21_fp8_order1.json")


if __name__ == "__main__":
    main()
