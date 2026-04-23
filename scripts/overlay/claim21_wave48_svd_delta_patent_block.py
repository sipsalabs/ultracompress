"""Render the PATENT_CLAIMS wave-48 subsection from the summary JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


SUMMARY = Path("results/claim21_wave48_svd_delta_summary.json")


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
    frac = c["rank_frac"]

    print(f"### Wave 48 — low-rank SVD delta (rank_frac={frac}) + brotli-11")
    print()
    print("**Hypothesis.** Post-hoc full fine-tune deltas have the same "
          "approximately-low-rank structure as LoRA-style adapters. "
          "A truncated SVD at rank "
          f"`r = round({frac} · min(m, n))` per 2-D weight should "
          "replace an `m × n` delta with `U (m × r)` and `Vᵀ (r × n)` "
          f"— asymptotically a `{100*frac:.1f}%` parameter fraction — "
          "at small rel-Frobenius reconstruction error.")
    print()
    print("**Method (per pair).** Same key set as waves 45–47. For each "
          "tensor: compute `D = ft_bf16 - base_bf16` in fp32, run "
          "randomized truncated SVD on GPU via `torch.svd_lowrank` at "
          f"rank `r = max(1, round({frac} · min(m, n)))`, absorb "
          "singular values into `Uhat` (so decoder is just `Uhat @ Vᵀ`), "
          "serialize both factors as bf16 and brotli-11. Report "
          "rel-Frobenius error accumulated across all tensors. No "
          "retraining.")
    print()
    print(f"**Cohort ({c['n_pairs']} pairs, {fmt_int(c['n_params_total'])} "
          f"delta params → "
          f"{fmt_int(c['n_params_lowrank_total'])} lowrank params "
          f"= {100*c['lowrank_param_fraction']:.2f}% raw reduction).**")
    print()
    print("| stream | brotli-11 bytes | ratio vs bf16-delta | "
          "rel-Frobenius err |")
    print("|---|---:|---:|---:|")
    print(f"| bf16 delta (wave-45 baseline) | "
          f"{fmt_int(c['brotli11_bf16_delta_total'])} | 1.000x | "
          f"0 (lossless) |")
    print(f"| low-rank SVD bf16 (U, Vᵀ) | "
          f"{fmt_int(c['brotli11_lowrank_total'])} | "
          f"{fmt_ratio(c['ratio_bf16_over_lowrank'])} | "
          f"{c['weighted_rel_frob_err']:.3e} |")
    print()
    print("**Per pair.**")
    print()
    print("| pair | delta params | lowrank params | lr frac | "
          "br(bf16 Δ) | br(lowrank) | ratio | relerr |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in per.items():
        print(f"| {name} | {fmt_int(d['n_params'])} | "
              f"{fmt_int(d['n_params_lowrank'])} | "
              f"{100*d['lowrank_param_fraction']:.2f}% | "
              f"{fmt_int(d['brotli11_bytes_bf16_delta'])} | "
              f"{fmt_int(d['brotli11_bytes_lowrank'])} | "
              f"{fmt_ratio(d['ratio_bf16_over_lowrank'])} | "
              f"{d['rel_frobenius_err']:.3e} |")
    print()
    print("**Stacked implication (waves 44 + 45 + 46 + 47 + 48).** The "
          "fleet substrate now has two orthogonal axes for per-fine-tune "
          "marginal compression: element-wise low-bit block-float (wave "
          "47, no structural assumption on the delta) and rank-truncated "
          "SVD (this wave, exploits the empirical low-rank structure). "
          "A decoder can choose whichever beats the other per-tensor, or "
          "stack: block-float quantize the SVD factors themselves. Per-"
          "tune marginal storage observed here is "
          f"{fmt_ratio(c['ratio_bf16_over_lowrank'])} smaller than wave-"
          "45's bf16-delta baseline at "
          f"{c['weighted_rel_frob_err']:.2e} rel-Frobenius error.")
    print()
    print("**Artifacts.**")
    print()
    for _, path in [
        ("olmo",  "results/claim21_wave48_svd_delta_olmo2_1b.json"),
        ("smol",  "results/claim21_wave48_svd_delta_smollm2_1.7b.json"),
        ("qwen3", "results/claim21_wave48_svd_delta_qwen3_1.7b.json"),
    ]:
        if Path(path).exists():
            print(f"- [{path}]({path})")
    print("- [results/claim21_wave48_svd_delta_summary.json]"
          "(results/claim21_wave48_svd_delta_summary.json)")


if __name__ == "__main__":
    main()
