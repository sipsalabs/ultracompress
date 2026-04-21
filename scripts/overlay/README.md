# scripts/overlay/ — Row-overlay weight compression (Claims 17-20)

Drivers for the sub-3-bpw row-overlay stack and the external head-to-head harness.

## Flagship drivers

- **`benchmark_head_to_head.py`** — unified harness. Runs our overlays head-to-head vs `bitsandbytes` (nf4/int8) and `hqq` (2/3/4-bit, g16/g64) on LAMBADA, emits resumable JSON. Claim 19 (bnb) and Claim 20 (HQQ) were produced by this script.
- **`_analyze_claim20.py`** — merges `results/h2h_n500_small.json` + `results/h2h_n500_large.json` and emits the Claim 20 summary tables.

## Overlay drivers (Claims 17, 18A, 18D)

- **`lambada_overlay.py`** — Claim 17: sparse fp16 row-overlay atop the v17 packed base.
- **`lambada_overlay_fp8.py`** — Claim 18A: fp8 E4M3 row-overlay.
- **`lambada_overlay_mixed.py`** — Claim 18D: mixed-precision (fp16 top-K₁ + fp8 next-K₂).

## v17 pack + fit + verify (Claim 16)

- **`fit_v17_hifi.py`**, **`fit_v17_8b.py`** — weight-row fitter.
- **`compress_v17.py`**, **`pack_v17.py`**, **`pack_all_v17.py`** — packer.
- **`verify_all_v17.py`** — round-trip integrity check.

## Evaluation (LAMBADA / WikiText / stress)

- **`lambada_all.py`**, **`lambada_hifi.py`** — held-out LAMBADA drivers.
- **`wikitext_eval.py`** — disjoint WikiText-103 test split eval.
- **`eval_claim16_topk.py`**, **`eval_topk_8b.py`**, **`eval_v17_*.py`**, **`eval_sweep_ppl.py`** — per-claim top-k and PPL drivers.
- **`stress_synthetic.py`**, **`determinism_check.py`** — regression harnesses.
- **`demo_claim16.py`** — interactive demo.

## Data prep

- **`tokenize_lambada.py`**, **`tokenize_wikitext.py`** — dataset tokenization.
- **`cache_activations.py`**, **`cache_activations_io.py`**, **`cache_teacher_8b.py`** — teacher activation caches.
- **`aggregate_results.py`** — result aggregation.

## Running from the repo root

All scripts are driven from the repo root:

```
python scripts/overlay/benchmark_head_to_head.py --methods our_fp8_2p79,bnb_nf4,hqq_4bit_g64 --n 500
python scripts/overlay/lambada_overlay_mixed.py --rho_hi 0.001 --rho_lo 0.003
python scripts/overlay/_analyze_claim20.py
```
