"""Render the PATENT_CLAIMS wave-47 subsection from the summary JSON."""
from __future__ import annotations

import json
from pathlib import Path


SUMMARY = Path("results/claim21_wave47_mxfp4_delta_summary.json")


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
    block = c.get("block_size", 32)

    print("### Wave 47 — MXFP4 block-float delta quantization (block=32) + brotli-11")
    print()
    print("**Hypothesis.** Wave 46's per-tensor absmax scheme is "
          "dominated by outliers: a handful of large-magnitude delta "
          "entries per tensor inflate the scale and waste int4 dynamic "
          "range on the dense small-magnitude majority. Grouping the "
          "delta into contiguous blocks of "
          f"{block} elements along the flattened axis and storing one "
          "fp16 scale per block (the OCP MX Block-Float pattern) should "
          "sharply reduce the rel-Frobenius reconstruction error at "
          "fixed bit width. The scale overhead is fixed at 2 B per "
          f"{block} weights ({2.0*8/block:.2f} bpB).")
    print()
    print("**Method (per pair).** Same key set as waves 45 + 46. For "
          "each tensor: flatten the fp32 delta, pad to a multiple of "
          f"block={block}, reshape to `(B, block)`, compute per-row "
          "absmax → fp16 scale, round to int8 / int4 (signed), pack "
          "(int4 nibble-packed), concatenate quantized bytes + per-block "
          "fp16 scale array, brotli-11 the resulting stream. Report "
          "rel-Frobenius error with fp16-round-tripped scales (decoder-"
          "faithful). No retraining.")
    print()
    print(f"**Cohort ({c['n_pairs']} pairs, {fmt_int(c['n_params_total'])} "
          f"shared params, {fmt_int(c['n_blocks_total'])} blocks @ "
          f"block={block}).**")
    print()
    print("| stream | brotli-11 bytes | ratio vs bf16-delta | bpB | "
          "rel-Frobenius err |")
    print("|---|---:|---:|---:|---:|")
    print(f"| bf16 delta (wave-45 baseline) | "
          f"{fmt_int(c['brotli11_bf16_total'])} | 1.000x | "
          f"{c['bf16_delta_bpB']:.4f} | 0 (lossless) |")
    print(f"| int8 + per-block fp16 scales | "
          f"{fmt_int(c['brotli11_int8_total'])} | "
          f"{fmt_ratio(c['ratio_bf16_over_int8'])} | "
          f"{c['int8_bpB_brotli11']:.4f} | "
          f"{c['weighted_rel_frob_err_int8']:.3e} |")
    print(f"| int4 + per-block fp16 scales | "
          f"{fmt_int(c['brotli11_int4_total'])} | "
          f"{fmt_ratio(c['ratio_bf16_over_int4'])} | "
          f"{c['int4_bpB_brotli11']:.4f} | "
          f"{c['weighted_rel_frob_err_int4']:.3e} |")
    print()
    print("**Per pair.**")
    print()
    print("| pair | params | blocks | br(bf16) | br(int8 block) | "
          "br(int4 block) | bf16/int8 | bf16/int4 | relerr int8 | "
          "relerr int4 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in per.items():
        print(f"| {name} | {fmt_int(d['n_params'])} | "
              f"{fmt_int(d['n_blocks'])} | "
              f"{fmt_int(d['brotli11_bytes_bf16'])} | "
              f"{fmt_int(d['brotli11_bytes_int8'])} | "
              f"{fmt_int(d['brotli11_bytes_int4'])} | "
              f"{fmt_ratio(d['ratio_bf16_over_int8'])} | "
              f"{fmt_ratio(d['ratio_bf16_over_int4'])} | "
              f"{d['rel_frobenius_err_int8']:.3e} | "
              f"{d['rel_frobenius_err_int4']:.3e} |")
    print()
    print("**Stacked implication (waves 44 + 45 + 46 + 47).** The fleet "
          "story now composes four substrates: (i) base stored once via "
          "wave-44's lossy+lossless scheme; (ii) each fine-tune stored "
          "as a delta against that base under brotli-11 (wave 45); "
          "(iii) delta further quantized to per-tensor int4 (wave 46); "
          "(iv) delta quantized to per-block int4 with fp16 block scales "
          "(this wave) to recover reconstruction accuracy the per-tensor "
          "scheme sacrificed. Per-fine-tune marginal storage is the "
          f"int4-block column above, {fmt_ratio(c['ratio_bf16_over_int4'])} "
          "smaller than the wave-45 bf16-delta baseline at "
          f"{c['weighted_rel_frob_err_int4']:.2e} rel-Frobenius "
          "reconstruction error.")
    print()
    print("**Artifacts.**")
    print()
    for _, path in [
        ("olmo",  "results/claim21_wave47_mxfp4_delta_olmo2_1b.json"),
        ("smol",  "results/claim21_wave47_mxfp4_delta_smollm2_1.7b.json"),
        ("qwen3", "results/claim21_wave47_mxfp4_delta_qwen3_1.7b.json"),
    ]:
        if Path(path).exists():
            print(f"- [{path}]({path})")
    print("- [results/claim21_wave47_mxfp4_delta_summary.json]"
          "(results/claim21_wave47_mxfp4_delta_summary.json)")


if __name__ == "__main__":
    main()
