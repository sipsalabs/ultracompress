# V3 Cure Direction — Rank-Redistribution at Constant Total Budget

**Author:** Sipsa Labs research thread (overnight 2026-05-08 → 2026-05-09)
**Status:** Recommendation, not yet executed. PRIMARY = Option A (rank-redistribution). Secondary candidates = B / C / D below.
**Source of truth for v1 / v2 results:** `docs/PPL_EVAL_qwen3-1.7b-base-uniform-n50_2026_05_08.json`, `docs/PPL_EVAL_qwen3-1.7b-base-adaptive-bpw-v1_2026_05_08.json`, `docs/PPL_EVAL_qwen3-1.7b-base-v2-adaptive-train-steps_2026_05_08.json`, and the per-layer train_loss trace in `scripts/overlay/_e2e_qwen3_1_7b_base_v2_train_steps.log`.

---

## 1. Context — what we have measured

Apples-to-apples eval on Qwen3-1.7B-Base, n=50 prompts, seq=1024, baseline PPL = 12.0813:

| Run | Config | PPL ratio | Δ vs uniform | Notes |
|-----|--------|-----------|--------------|-------|
| Uniform | bpw=5, V18-C rank=32, train_steps=200 | **1.004876** | — | Reference |
| v1 adaptive bpw | k_proj@6bpw, others@5bpw, V18-C rank=32, train_steps=200 | 1.005097 | **+0.000220** | REFUTED (v1 slightly worse, within σ) |
| v2 adaptive train_steps | bpw=5, V18-C rank=32, train_steps=200→1000 ramp by depth | **1.004515** | -0.000361 | Marginal win, ~1.2σ |

Older record runs (NOT comparable — different baseline window / n_eval=30):
- `docs/PPL_EVAL_qwen3-1_7b-base_2026_05_08.json` — original record 1.0040 (n=30, baseline 12.7683)
- `docs/PPL_EVAL_qwen3-1_7b-base-v2-r64s400_2026_05_08.json` — earlier rank=64/steps=400 attempt 1.0042

There is a confounding earlier v2 run, `docs/PPL_EVAL_qwen3-1.7b-base-v18c-adaptive-train-steps_2026_05_08.json`, with the SAME nominal config as v2_adaptive_train_steps but ratio 1.005051. That is a 0.0005 spread between identical-config runs, larger than the σ≈0.0003 we have been quoting. **This must be resolved with a seed-sweep variance bound (seeds {1, 2, 3}) before any V2 win is declared real.**

## 2. Diagnostic — why v1 failed and v2 only barely won

The streaming compression runner reports per-layer `train_loss_final` (the converged MSE between V18-C-corrected layer output and bf16 teacher hidden state). On the v2_train_steps log (28 layers, Qwen3-1.7B-Base):

| Layer range | train_loss_final | Interpretation |
|-------------|------------------|----------------|
| 0 – 9 | < 0.002 | Rank-32 is OVERPROVISIONED. 200 steps saturate. Compute budget wasted. |
| 10 – 15 | 0.003 – 0.009 | Easy. Rank starting to bind. |
| 16 – 22 | 0.02 – 0.29 | Transition. Rank starting to bind harder than steps. |
| 23 – 27 | 0.35 – 0.81 | UNDERTRAINED. Loss does NOT converge even with 800–1000 ramped steps. Rank-32 cannot represent the deep-layer residual subspace. |

This is a 4000× train_loss range across depth at uniform config. The v1 hypothesis (per-Linear bpw) was attacking the WRONG knob: the residual at the per-Linear input is fine; the V18-C correction subspace is the bottleneck. The v2 hypothesis (more steps) is attacking a knob that helps mid-range layers (16-22) but cannot fix layers 23-27, which are RANK-bound, not STEPS-bound. v2's marginal +0.00037 improvement is consistent with that diagnosis: the mid-layers improve, the deep ones do not, and the per-layer PPL contribution averages.

## 3. Primary recommendation — Option A: rank-redistribution at constant total budget

### Hypothesis
The V18-C correction subspace at rank=32 saturates layers 0-15 and starves layers 23-27. Total parameter budget = 28 × 32 = 896 rank-units is correct; the per-layer allocation is wrong. Equalizing `train_loss_final` across depth (instead of equalizing rank) will close the gap.

### Mechanism
Linear ramp `rank_layer_i = round(16 + 32 * i / (n_layers - 1))` on Qwen3-1.7B-Base:
- Layer 0 → rank=16 (half of uniform — overprovisioned anyway)
- Layer 14 → rank=32 (uniform midpoint)
- Layer 27 → rank=48 (1.5× uniform — closer to the residual subspace dimension)
- Sum = 28 × 32 = 896 rank-units (UNCHANGED total budget)

Implementation: add `UC_RANK_REDISTRIBUTE=1` env flag to `streaming_compression_runner.py` (analogous to the existing `UC_ADAPTIVE_TRAIN_STEPS` gate at line 746). ~30 LOC. Per-layer rank goes into UC v3 layer metadata (already present), so no format change.

### Expected PPL signal direction
PPL ratio drops to **1.0030 – 1.0035** on the apples-to-apples eval. That is Δ ≥ -0.0010 vs uniform, ≥ 3σ above noise floor, i.e. the FIRST sub-1.004× signal at this `{bpw=5, GSQ k=32, V18-C, fixed total budget}` configuration.

If V3 lands in that band, it becomes the new headline number for Qwen3-1.7B-Base and the empirical anchor for the patent CIP re-anchoring (see §6).

### GPU cost estimate
SAME wallclock as uniform (~5 hr on cuda:1 for compression + 11 min for n=50 PPL eval). Justification: rank=16 layers train ~2× faster than rank=32, rank=48 layers train ~1.5× slower; weighted average ≈ 1×. Memory peak unchanged (the V18-C state for the loaded layer is rebuilt each layer; only that layer's V/U/α is on GPU). Output dir: `scripts/overlay/_e2e_qwen3_1_7b_base_v3_rank_redistribute/`. Result JSON: `docs/PPL_EVAL_qwen3-1.7b-base-v3-rank-redistribute_2026_05_09.json`.

### Risk assessment
- **Low.** Constant total memory budget; UC v3 format already supports per-layer rank; reuses the v2_train_steps autopipe pattern.
- **Decision criteria:**
  - PPL ratio ≤ 1.0035 → STRONG WIN, publishable, becomes new headline.
  - PPL ratio ≤ 1.0040 → WIN, file as research result; consider combining with Option B.
  - PPL ratio in [1.0040, 1.0048] → NEUTRAL; rank-redistribution and rank-uniform are equivalent under V18-C; refute the rank-bound diagnosis.
  - PPL ratio > 1.0048 → REFUTED. Tells us the deep-layer high train_loss is NOT a rank issue (probably an objective-function issue — see Option C).

## 4. Alternative — Option B: deep-only train_steps ramp

### Hypothesis
v2's linear ramp wasted compute on layers 0-15 (already saturated). Restrict the ramp to layers ≥ 16.

### Mechanism
`if layer_idx >= 16: train_steps = base * (1 + 4 * (layer_idx - 16) / 11)` else 200. Total compute ~1.5× uniform instead of v2's ~3×.

### Expected signal direction
Reproduces v2's ratio (1.004515 ± noise) at ~60% the wallclock. Could be slightly better if the deep-only ramp lets layers 23-27 get more steps within the same budget cap.

### GPU cost
~3 hr compression on cuda:1 (1.7× faster than v2).

### Why NOT primary
This is a v2 efficiency tweak, not a new mechanism. The deepest layers are STILL rank-bound; more steps just keeps grinding loss-plateau territory. **Run AFTER V3** to bound how much of v2's win was "ramp" vs "deep-only ramp".

## 5. Alternative — Option C: layer-grouped joint training (4-layer blocks)

### Hypothesis
The streaming compression treats each layer independently against frozen teacher hidden state. Joint training of 4 consecutive layers would let V/U/α absorb cross-layer error correlation, which the per-layer objective cannot see.

### Mechanism
Load 4 layers into memory simultaneously (peak VRAM goes from 2.78GB → ~10GB, well under 32GB on a 5090). Train V/U/α for all 4 layers jointly against the END-of-block teacher hidden state instead of the per-layer teacher. Backprop runs through all 4 layers.

### Expected signal direction
PPL ratio could drop to 1.0020-1.0035 IF cross-layer error compensation is the missing ingredient. Could also be a no-op if per-layer hidden-state distillation already captures most of it. **Sign uncertain.**

### GPU cost
~8-10 hr compression (per-block instead of per-layer; each block is 4× bigger and trains slower because backprop runs through 4 layers). Memory peak ~10GB.

### Why NOT primary
- High implementation cost (the streaming runner is currently per-layer iteration; moving to per-block requires teacher cache restructuring).
- Risk of "no signal" outcome that doesn't isolate cause.
- **Defer until V3 lands** so we know whether the residual gap is rank-bound (V3 fixes it) or objective-function-bound (V3 doesn't fix it — then Option C becomes the next move).

## 6. Alternative — Option D: block-classified bpw allocation (the "v1 done correctly" path)

### Hypothesis
The bottleneck isn't k_proj specifically — it's the LINEAR CLASS that dominates the residual on each layer. Linear classes group by activation distribution: attention projections (q/k/v/o) vs MLP gate/up vs MLP down. Each class has different quant friendliness, but ALSO different per-LAYER quant friendliness. Allocate bpw at the (class × layer) cross-product, not just per-class.

### Mechanism
Two-pass:
1. Profile `quant_rel_l2[class, layer]` matrix on the bf16 model with a 5-bpw uniform pre-pack. (~1 hr on cuda:1.)
2. Solve a knapsack: total bpw budget = 5 bpw × 7 Linears × 28 layers = 980 bit-units. Allocate +1bpw to top-k highest-error (class, layer) cells. Subject to: average bpw per layer in [4.5, 5.5] (so runtime VRAM is uniform per layer).

### Expected signal direction
Could push PPL ratio to 1.0025 IF the (class × layer) error variance is genuinely heavy-tailed. Could also be no-op if V18-C absorbs it (which is what killed v1).

### GPU cost
~5 hr compression + 1 hr profiling. Same as V3. 

### Why NOT primary
This is "v1 done correctly", but the v1 result tells us V18-C absorbs per-Linear quant residual at this rank. **Don't do (class × layer) bpw until V3 changes the rank ceiling.** Then re-test under the new rank profile — under V3, the deep-layer V18-C is no longer absorbing all per-Linear residual, so (class × layer) bpw might actually move PPL.

## 7. Recommended execution order

1. **V3 rank-redistribution** (this week) — PRIMARY. Single env flag, ~30 LOC, ~5h on cuda:1. Decision criterion: ratio ≤ 1.0040 = WIN; ≤ 1.0035 = STRONG WIN (publishable). Output JSON: `docs/PPL_EVAL_qwen3-1.7b-base-v3-rank-redistribute_2026_05_09.json`.

2. **Seed-sweep variance bound** (parallel, can co-exist on cuda:0) — re-run uniform + v2_train_steps with seeds {1, 2, 3}. If σ > 0.0005, retract v2 marginal-win claim until properly powered. Output: `docs/SEED_SWEEP_VARIANCE_BOUND_2026_05_09.md`.

3. **V3 + Option B combined** (next week, conditional on V3 winning) — add `UC_DEEP_ONLY_STEPS_RAMP=1` on top of V3's rank ramp. Test whether deep-only ramp adds anything once rank is right.

4. **Defer Options C and D** until V3 either lands a real signal or refutes the rank-bound diagnosis. Their value is only legible relative to V3's outcome.

## 8. Patent implication

The v1 patent CIP draft (commit 2fb18f8: "ip: continuation-in-part draft for per-Linear adaptive bpw, 12 method-form claims, $65 micro-entity") is anchored to a refuted result. **Do NOT file CIP until V3 lands.** The legitimate patent angle is now "per-layer rank schedule for streaming activation-distillation correction matrices", not "per-Linear adaptive bpw". Re-anchor the CIP on V3's empirical evidence + the train_loss-by-depth diagnostic in §2 before filing.

The Track A supplement (drafted 2026-04-25, due 2026-05-09 for $65 micro-entity, see `provisional_track_a_supplement.pdf`) is the prioritized pre-funding patent expense and is unaffected by this research arc.

## 9. What this document IS NOT

This is a recommendation, not a result. V3 has not been executed. PPL signals quoted in §3 are predictions. They will be replaced with measured values once V3 runs and a JSON eval lands in `docs/`. If V3 refutes the rank-bound diagnosis, this document gets a §10 "REFUTED" addendum, not a quiet revision.
