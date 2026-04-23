"""Render the PATENT_CLAIMS wave-45 subsection from the summary JSON.

Usage:
  python scripts/overlay/claim21_finetune_delta_patent_block.py \
      > /tmp/wave45_block.md

Reads results/claim21_finetune_delta_summary.json and prints a
markdown block that can be pasted into PATENT_CLAIMS.md ahead of the
existing `### Measured throughput Pareto ...` header.  No hype --
every number is pulled directly from the summary JSON.  Honest lossless
framing: brotli-11 is bijective, no quantization is applied on top of
the bf16 cast that the repos already ship with.
"""
from __future__ import annotations

import json
from pathlib import Path


SUMMARY = Path("results/claim21_finetune_delta_summary.json")


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_bpB(x: float) -> str:
    return f"{x:.4f}"


def fmt_ratio(x: float) -> str:
    return f"{x:.3f}x"


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY} -- run the summary script first")
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    c = summ["cohort"]
    per = summ["per_model"]

    print("### Wave 45 — fine-tune delta storage (lossless, brotli-11)")
    print()
    print("**Hypothesis.** For a fleet of N fine-tunes of a shared base, "
          "store the base once plus N deltas (ft - base, bf16). The "
          "question: does brotli-11 compress the delta *substantially* "
          "better than the fine-tune itself?")
    print()
    print("**Method (per pair).** Load HF safetensors for base and "
          "fine-tune, intersect shape-matched 2-D `.weight` tensors "
          "(layers, embeddings, lm_head), concatenate each stream in "
          "bf16 row-major bytes, compute brotli-11 of base, ft, and "
          "(ft - base cast to bf16). Hashes recorded for reproducibility. "
          "No lossy quantization beyond the bf16 that upstream already "
          "ships.")
    print()
    print(f"**Cohort ({c['n_pairs']} pairs, {fmt_int(c['n_params_total'])} "
          f"shared params).**")
    print()
    print("| stream | bytes | bpB | ratio vs bf16 |")
    print("|---|---:|---:|---:|")
    print(f"| raw bf16 | {fmt_int(c['bf16_bytes_total'])} | 16.0000 | "
          f"1.000x |")
    print(f"| brotli-11(base) | {fmt_int(c['brotli11_base_total'])} | "
          f"{fmt_bpB(8.0*c['brotli11_base_total']/c['bf16_bytes_total'])} | "
          f"{fmt_ratio(c['bf16_bytes_total']/c['brotli11_base_total'])} |")
    print(f"| brotli-11(ft)   | {fmt_int(c['brotli11_ft_total'])} | "
          f"{fmt_bpB(c['ft_bpB_brotli11'])} | "
          f"{fmt_ratio(c['ft_vs_bf16_ratio'])} |")
    print(f"| brotli-11(delta)| {fmt_int(c['brotli11_delta_total'])} | "
          f"{fmt_bpB(c['delta_bpB_brotli11'])} | "
          f"{fmt_ratio(c['delta_vs_bf16_ratio'])} |")
    print()
    print("**Headline ratios (cohort aggregate).**")
    print()
    print(f"- brotli-11(ft) / brotli-11(delta) = "
          f"**{fmt_ratio(c['delta_vs_ft_ratio'])}** "
          "— fraction of a full fine-tune needed to store just the delta.")
    print(f"- brotli-11(base) / brotli-11(delta) = "
          f"{fmt_ratio(c['delta_vs_base_ratio'])}.")
    print(f"- raw bf16 / brotli-11(delta) = "
          f"{fmt_ratio(c['delta_vs_bf16_ratio'])}.")
    print()

    print("**Per pair.**")
    print()
    print("| pair | base | ft | params | br(base) | br(ft) | br(delta) | "
          "δ/ft | δ/base | ft bpB | δ bpB |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in per.items():
        print(f"| {name} | {d['base_repo']} | {d['ft_repo']} | "
              f"{fmt_int(d['n_params'])} | "
              f"{fmt_int(d['brotli11_bytes_base'])} | "
              f"{fmt_int(d['brotli11_bytes_ft'])} | "
              f"{fmt_int(d['brotli11_bytes_delta'])} | "
              f"{fmt_ratio(d['delta_vs_ft_ratio'])} | "
              f"{fmt_ratio(d['delta_vs_base_ratio'])} | "
              f"{fmt_bpB(d['ft_bpB_brotli11'])} | "
              f"{fmt_bpB(d['delta_bpB_brotli11'])} |")
    print()

    print("**Enterprise fleet implication.** For N fine-tunes of a "
          "single base, total storage under this scheme is "
          "`brotli11(base) + N * brotli11(delta)` instead of "
          "`N * brotli11(ft)`. The marginal cost of the N-th fine-tune "
          f"is {fmt_ratio(c['delta_vs_ft_ratio'])} of a standalone ft "
          "under brotli-11 — and this is purely the lossless-coder "
          "effect, before any Claim-21 overlay / lossy base storage is "
          "applied. Stacking wave-44's end-to-end scheme on top of this "
          "delta substrate multiplies the savings by N.")
    print()
    print("**Artifacts.**")
    print()
    for name, path in [("olmo",  "results/claim21_finetune_delta_olmo2_1b.json"),
                        ("smol",  "results/claim21_finetune_delta_smollm2_1.7b.json"),
                        ("qwen3", "results/claim21_finetune_delta_qwen3_1.7b.json")]:
        p = Path(path)
        if p.exists():
            print(f"- [{path}]({path})")
    print(f"- [{SUMMARY}]({SUMMARY})")
    print()
    print("_Full byte-exact reproduction: "
          "`python scripts/overlay/claim21_finetune_delta.py "
          "--base <repo> --ft <repo> --out <json> --skip-gzip-zstd`._")


if __name__ == "__main__":
    main()
