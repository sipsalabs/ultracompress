# results/

All measurement JSONs referenced by [../RESULTS.md](../RESULTS.md) and [../PATENT_CLAIMS.md](../PATENT_CLAIMS.md).

## Claim 20 (row-overlay vs external quantizers, n=500, 6-model cohort)

- `h2h_n500_full.json` — merged cohort (48 rows = 6 models × 8 methods).
- `h2h_n500_small.json` — cuda:0 partition (32 rows: 1B–2B models).
- `h2h_n500_large.json` — cuda:1 partition (16 rows: 7B–8B models).

## Claim 19 (vs bnb nf4)

- `head_to_head_results_6model_n80.json` — primary 6-model n=80 run.
- `head_to_head_results_qwen3_8_cuda1.json` — Qwen3-8B confirmation.
- `head_to_head_results_pair.json` — OLMo + TinyLlama n=120 validation.

## Claim 17/18A/18D (row-overlay sweeps)

- `lambada_overlay_results.json` — Claim 17 fp16 sparse row-overlay.
- `lambada_overlay_fp8_results.json` — Claim 18A fp8 row-overlay.
- `lambada_overlay_mixed_results.json` — Claim 18D mixed-precision overlay.
- `lambada_overlay_adaptive_results.json` — Claim 18B (disclaimed negative).
- `lambada_overlay_int4_results.json` — Claim 18C (disclaimed, int4 insufficient).

## Claim 16 (packed v17)

- `pack_summary.json` — per-model packing stats.
- `verify_all_results.json` — round-trip integrity verification.
- `lambada_all_results.json`, `lambada_hifi_results.json` — held-out LAMBADA.
- `wikitext_results.json` — WikiText-103 test split (disjoint eval).

## FRR track

- `hires_results_hq5.json` — HQ5 flagship eval (n=1000).
- `combined_stack_results_hq5.json` — FRR body + ASVD head stack.
- `stress_synthetic_results.json` — determinism / synthetic stress harness.
- `v17hi_fit_summary.json` — v17 hi-fidelity fit summary.
