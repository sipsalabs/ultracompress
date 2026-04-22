"""Aggregator for wave 26 per-model scale pair-decomp JSONs."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]


def main():
    rows = []
    for m in MODELS:
        p = RES / f"claim21_scale_pair_{m}_rho0.01.json"
        if not p.exists():
            print(f"[miss] {p}")
            continue
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    lines = []
    lines.append("Claim-21 wave 26: scale stream fp16 pair decomposition")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Wave 21 showed idx_delta's int32 layout is 2/4 bytes")
    lines.append("structurally zero. This wave does the equivalent measurement")
    lines.append("for the fp16 scale stream: per-byte H, joint H over the")
    lines.append("65536-bin pair distribution, and mutual information between")
    lines.append("the two bytes of each fp16. Also reports the Shannon H of the")
    lines.append("sign (1-bit), exponent (5-bit), and mantissa-hi/lo (2+8-bit)")
    lines.append("fields, which is an information-theoretic tight lower bound")
    lines.append("on any independent-field coder.")
    lines.append("")
    fmt = (f"  {'model':<14}{'byte0 H':>10}{'byte1 H':>10}"
           f"{'sum':>10}{'joint':>10}{'I(B0;B1)':>12}"
           f"{'field sum':>12}{'field bpB':>11}")
    lines.append(fmt)
    for r in rows:
        lines.append(
            f"  {r['model']:<14}"
            f"{r['byte0_H_bpB']:>10.4f}"
            f"{r['byte1_H_bpB']:>10.4f}"
            f"{r['byte_sum_bits_per_scale']:>10.4f}"
            f"{r['joint_H_bits_per_scale']:>10.4f}"
            f"{r['mutual_information_bits_per_scale']:>12.4f}"
            f"{r['field_sum_bits_per_scale']:>12.4f}"
            f"{r['field_sum_bpB']:>11.4f}"
        )
    # cohort averages (simple mean across models)
    def mean(key):
        return sum(r[key] for r in rows) / max(1, len(rows))
    lines.append("")
    lines.append(f"  Cohort (unweighted mean over {len(rows)} models):")
    lines.append(
        f"  {'mean':<14}"
        f"{mean('byte0_H_bpB'):>10.4f}"
        f"{mean('byte1_H_bpB'):>10.4f}"
        f"{mean('byte_sum_bits_per_scale'):>10.4f}"
        f"{mean('joint_H_bits_per_scale'):>10.4f}"
        f"{mean('mutual_information_bits_per_scale'):>12.4f}"
        f"{mean('field_sum_bits_per_scale'):>12.4f}"
        f"{mean('field_sum_bpB'):>11.4f}"
    )
    lines.append("")
    lines.append(
        "For context: wave 19 reported scale stream Shannon H (byte-level")
    lines.append(
        "mixed across both byte positions) cohort-aggregate ~5.39 bpB. The")
    lines.append(
        "per-byte-position H here is the actually-realisable floor for any")
    lines.append(
        "positionally-aware coder; the joint H is the tightest achievable")
    lines.append(
        "lower bound for any coder that sees both bytes of each fp16 as one")
    lines.append(
        "unit; the field-decomposed H is the lower bound for any coder that")
    lines.append(
        "splits fp16 into its IEEE 754 sign/exp/mantissa fields.")
    lines.append("")
    lines.append("=" * 90)
    txt = "\n".join(lines) + "\n"
    (RES / "claim21_scale_pair_decomp.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("[wrote] results/claim21_scale_pair_decomp.txt")


if __name__ == "__main__":
    main()
