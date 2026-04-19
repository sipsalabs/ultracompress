# What Works — UltraCompress distillation recipe (as of HQ5)

## Architecture

- **FRR body**: one shared Qwen-style transformer block `B`, applied in a fractal schedule `FractalModel(n_scales=4, iters_per_scale=7)` — that is, 4 scales × 7 iterations = 28 effective applications, matching the teacher's 28-block depth with a single set of weights.
- **Projections**: `proj_in: 2048 → h` before the body, `proj_out: h → 2048` after, where `h ∈ {128, 256, 384}`.
- **Head**: teacher's `final_norm` + `lm_head` frozen and re-used unchanged.
- Only `proj_in`, body block `B`, and `proj_out` are trainable: **0.64 M (h=128) / 1.51 M (h=256) / 2.61 M (h=384)**.

## Training objective (HQ5)

```
hard_weight  = (1 + H(teacher_logits)) ^ entropy_power     # h256: 1.5, h128: 1.0
fkl          = hard_weight · KL(T || S)
rkl          = 0.3 · KL(S || T)
latent       = latent_w(step) · MSE(S_hidden, T_hidden)    # 1.0 → 0.1 (h256) / 0.3 (h128)
ce           = 0.5 · ce_ramp(step) · CE(teacher_argmax)    # 0.5 → 1.0
margin       = 0.3 · ce_ramp · hard_weight · margin_loss
total_loss   = fkl + rkl + latent + ce + margin
```

Key insight: the `hard_weight` **upweights** high-entropy (uncertain) teacher tokens rather than down-weighting them, which is the opposite of standard confidence weighting. This is what breaks the 54% T1 plateau.

## Schedules

- `latent_w`: 1.0 → `latent_w_final` linearly over steps 20K → 50K.
- `ce_ramp`: 0.5 → 1.0 linearly over steps 16K → 48K.
- `T` (KD temperature): 2.0 → 1.0 linearly until step 64K.
- LR: 2e-4 → 1e-5 cosine, 500-step warmup.

## Hyperparameters (fixed)

- `SEQ_LEN=128`, `BATCH=4`, `ACCUM=2`, `STEPS=80 000`.
- Data: `fineweb_edu_500M_tokens.pt` (500 M tokens).
- Optimizer: AdamW, `weight_decay=0.01`.
- Precision: bf16 autocast, fp32 params.

## Empirically verified

1. **Inverted entropy weighting scales.** ENT_POW 0 → 1.0 → 1.5 produced monotone gains in quality and peak T1. Extrapolation to 2.0 is HQ6's test.
2. **Per-width latent floor matters.** h128 benefits from a higher latent floor (0.3) for stability; h256 prefers 0.1 for frontier performance.
3. **Warm-starting cascades.** HQ5 warm-started from HQ4, which warm-started from HQ3. Each generation's best.pt is a strictly better starting point than random init for the next objective.
4. **Detached launcher is reliable.** Both HQ4 and HQ5 completed unattended over ~6 h per run. No supervision required.

## What did not work

- **Standard confidence weighting** (ENT_POW = −1 or similar): plateaus at 54% T1 (HQ3 ceiling).
- **Dropping latent loss entirely after warmup**: degrades ppl-ratio without improving T1/T10.
- **Larger batches at shorter SEQ_LEN**: quality tracks effective-tokens-seen, not batch size.

## Files

- Training: [run_hq4_ceiling_break.py](../run_hq4_ceiling_break.py) (shared HQ4/5/6).
- Launchers: [launch_hq4_detached.py](../launch_hq4_detached.py), [launch_hq5_detached.py](../launch_hq5_detached.py), [launch_hq6_detached.py](../launch_hq6_detached.py).
- Data prep: `prepare_fineweb_edu_500M.py`.
- Eval: inline `eval_loop()` inside the training script (100 seqs, matches hires protocol structure).
