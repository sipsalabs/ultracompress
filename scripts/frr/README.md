# scripts/frr/ — Fractal Residual Recursion training + eval (Claims 1-16)

Drivers for the FRR architectural-compression track: training the shared recursive block, launching dual-GPU runs, evaluation harnesses, and ablations.

## Flagship training drivers

- **`run_hq4_ceiling_break.py`** — flagship HQ4 trainer (entropy-weighted 5-loss, latent decay).
- **`run_hq3_breakthrough.py`** — HQ3 trainer (confidence-weighted CE + margin).
- **`run_frr_generic.py`** — architecture-agnostic trainer with `--teacher_cache` flag.
- **`run_frr_hq8.py`** — HQ8 experimental trainer.
- **`run_baseline_distill.py`** — matched-param standard-KD baseline.
- **`run_deq_frr.py`**, **`run_spectral_latent.py`**, **`run_1.7b_tinyfrr_hq2.py`** — ablations.

## Windows detached launchers

Use `subprocess.Popen` with `DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB | CREATE_NEW_PROCESS_GROUP` so training survives terminal closure.

- **`launch_hq4_detached.py`** / **`launch_hq5_detached.py`** / **`launch_hq6_detached.py`** / **`launch_hq7_longhorizon.py`**
- **`launch_hires_eval_hq5.py`**, **`launch_monday_eval.py`** — eval launchers.
- **`chain_combined_after_hires.py`**, **`chain_hq6_to_hq7.py`** — training-to-eval chains.

## Evaluation

- **`hires_eval.py`** — held-out eval with bootstrap CIs (Claim set).
- **`combined_stack_eval.py`** — FRR body + ASVD head stack eval.
- **`scale_eval.py`** — cross-scale eval (0.6B + 1.7B).
- **`smoke_any_model.py`** — minimal smoke test.

## Sweeps and analysis

- **`alpha_sweep.py`**, **`beta_sweep.py`**, **`per_role_alpha_sweep.py`** — hyperparameter sweeps.
- **`make_pareto_chart.py`**, **`plot_results.py`** — figure generators.

## Running from the repo root

```
python scripts/frr/run_hq4_ceiling_break.py --h 256 --steps 80000 --tag my_run
python scripts/frr/launch_hq4_detached.py
python scripts/frr/hires_eval.py --tags hq5_h256 hq5_h128 --n 1000
```
