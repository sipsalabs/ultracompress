"""Render the PATENT_CLAIMS wave-46 subsection from the summary JSON."""
from __future__ import annotations

import json
from pathlib import Path


SUMMARY = Path("results/claim21_wave46_quant_delta_summary.json")


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_ratio(x: float) -> str:
    return f"{x:.3f}x"


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY} -- run the summary script first")
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    c = summ["cohort"]
    per = summ["per_model"]

    print("### Wave 46 — lossy delta quantization (int8 / int4) + brotli-11")
    print()
    print("**Hypothesis.** Wave 45 showed the bf16 delta compresses "
          "~3x better losslessly than the full fine-tune. Deltas have "
          "much smaller magnitude than the fine-tune weights themselves, "
          "so per-tensor symmetric absmax quantization to int8 or even "
          "int4 should induce only small reconstruction error while "
          "multiplying the compression ratio further.")
    print()
    print("**Method (per pair).** Same key set as wave 45 (shape-matched "
          "2-D `.weight` tensors). For each tensor: compute `delta = ft - base` "
          "in fp32, per-tensor symmetric absmax scale, round to "
          "int8 / int4 (signed), pack (int4 nibble-packed), concatenate "
          "quantized bytes + per-tensor scales (fp16), brotli-11 the "
          "resulting stream. Report rel-Frobenius "
          "`||dequant - true_delta||_F / ||true_delta||_F` per pair, "
          "params-weighted mean for cohort. No retraining.")
    print()
    print(f"**Cohort ({c['n_pairs']} pairs, {fmt_int(c['n_params_total'])} "
          f"shared params).**")
    print()
    print("| stream | brotli-11 bytes | ratio vs bf16-delta | bpB | "
          "rel-Frobenius err |")
    print("|---|---:|---:|---:|---:|")
    print(f"| bf16 delta (wave-45 baseline) | "
          f"{fmt_int(c['brotli11_bf16_total'])} | 1.000x | "
          f"{c['bf16_delta_bpB']:.4f} | 0 (lossless) |")
    print(f"| int8 + fp16 scales | "
          f"{fmt_int(c['brotli11_int8_total'])} | "
          f"{fmt_ratio(c['ratio_bf16_over_int8'])} | "
          f"{c['int8_bpB_brotli11']:.4f} | "
          f"{c['weighted_rel_frob_err_int8']:.3e} |")
    print(f"| int4 + fp16 scales | "
          f"{fmt_int(c['brotli11_int4_total'])} | "
          f"{fmt_ratio(c['ratio_bf16_over_int4'])} | "
          f"{c['int4_bpB_brotli11']:.4f} | "
          f"{c['weighted_rel_frob_err_int4']:.3e} |")
    print()
    print("**Per pair.**")
    print()
    print("| pair | params | br(bf16) | br(int8+sc) | br(int4+sc) | "
          "bf16/int8 | bf16/int4 | relerr int8 | relerr int4 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in per.items():
        print(f"| {name} | {fmt_int(d['n_params'])} | "
              f"{fmt_int(d['brotli11_bytes_bf16'])} | "
              f"{fmt_int(d['brotli11_bytes_int8'])} | "
              f"{fmt_int(d['brotli11_bytes_int4'])} | "
              f"{fmt_ratio(d['ratio_bf16_over_int8'])} | "
              f"{fmt_ratio(d['ratio_bf16_over_int4'])} | "
              f"{d['rel_frobenius_err_int8']:.3e} | "
              f"{d['rel_frobenius_err_int4']:.3e} |")
    print()
    print("**Stacked implication (waves 44 + 45 + 46).** The enterprise "
          "fleet story multiplies: base stored once via wave-44's "
          "lossy+lossless scheme, every additional fine-tune stored as "
          "its brotli-11(int8 or int4 delta + fp16 scales). Per-tune "
          "marginal storage is the delta column above, which is "
          f"{fmt_ratio(c['ratio_bf16_over_int4'])} smaller than wave-45's "
          "bf16-delta baseline, which was already ~3x smaller than a "
          "standalone fine-tune under brotli-11.")
    print()
    print("**Artifacts.**")
    print()
    for name, path in [
        ("olmo",  "results/claim21_wave46_quant_delta_olmo2_1b.json"),
        ("smol",  "results/claim21_wave46_quant_delta_smollm2_1.7b.json"),
        ("qwen3", "results/claim21_wave46_quant_delta_qwen3_1.7b.json"),
    ]:
        if Path(path).exists():
            print(f"- [{path}]({path})")
    print("- [results/claim21_wave46_quant_delta_summary.json]"
          "(results/claim21_wave46_quant_delta_summary.json)")


if __name__ == "__main__":
    main()
