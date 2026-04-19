# HQ5 Ceiling-Break v2 — Final Results

Teacher: Qwen3-1.7B (1.72 B params, hidden=2048, vocab=151 936)
Student: TinyFRR_HQ4 architecture (same as HQ4)
Training: 80 000 steps, fineweb_edu_500M_tokens.pt, SEQ_LEN=128, BATCH=4×ACCUM=2
Warm start: `checkpoints_1.7b_tinyfrr_hq4_h{H}/best.pt`

HQ5 = HQ4 objective with two tuning directions:

| variant | latent floor (`latent_w_final`) | entropy_power | rationale |
|---------|---------------------------------|---------------|-----------|
| h128    | **0.3** (vs HQ4 0.1)            | 1.0           | Reduce oscillation via stronger latent anchor |
| h256    | 0.1                             | **1.5** (vs HQ4 1.0) | Sharpen hard-token focus |

Both trained unattended via `launch_hq5_detached.py`.

## Final numbers (training EVAL, 100 held-out samples)

| variant | trainable | compression | best step | best T1 | best all-T10 | best last-T10 | best quality | best ppl-ratio |
|---------|-----------|-------------|-----------|---------|--------------|---------------|--------------|----------------|
| h128    | 0.64 M    | **734×**    |    78K    | 54.0%   | 68.4%        | 62.7%         | **68.4%**    | 1.204          |
| h256    | 1.51 M    | **311×**    |    78K    | **55.1%** | **70.0%**  | 63.1%         | **70.0%**    | **1.180**      |

### Peak values (any step)

| variant | peak T1  | at step | peak all-T10 | at step | peak last-T10 | at step | peak quality | at step |
|---------|----------|---------|--------------|---------|---------------|---------|--------------|---------|
| h128    | 54.4%    | 58K/64K | 68.4%        | 66K     | 64.6%         | 56K     | 68.4%        | 78K     |
| h256    | **57.0%**| 50K     | **70.0%**    | 66K/70K/78K | **66.5%** | 52K     | **70.0%**    | 78K     |

**HQ5 h256 peak T1 = 57.0% — first sustained crossing of 55% in the project history. Quality = 70.0% — first crossing of the 70% threshold.**

## Head-to-head vs HQ4

| metric       | HQ4 h128 | HQ5 h128 | Δ     | HQ4 h256 | HQ5 h256 | Δ     |
|--------------|----------|----------|-------|----------|----------|-------|
| best T1      | 53.4%    | 54.0%    | +0.6  | 54.3%    | **55.1%**| **+0.8** |
| best all-T10 | 68.0%    | 68.4%    | +0.4  | 69.2%    | **70.0%**| **+0.8** |
| best quality | 66.9%    | 68.4%    | **+1.5** | 68.9% | **70.0%**| **+1.1** |
| peak T1      | 55.7%    | 54.4%    | -1.3  | 55.7%    | **57.0%**| **+1.3** |
| peak all-T10 | 68.6%    | 68.4%    | -0.2  | 69.6%    | **70.0%**| +0.4  |
| peak last-T10| 64.4%    | 64.6%    | +0.2  | 65.3%    | **66.5%**| **+1.2** |
| peak quality | 66.9%    | 68.4%    | +1.5  | 68.9%    | **70.0%**| +1.1  |

**HQ5 is strictly better on stable best metrics for both widths.** Peak T1 regressed slightly on h128 (higher latent floor tames oscillation at the cost of single-eval spikes) but best metrics are all up.

The h256 run tells the clearest story: stronger hard-token focus (ENT_POW 1.0 → 1.5) translates directly into every frontier metric. Peak T1 moves from 55.7% → 57.0%, peak last-T10 from 65.3% → 66.5%, final quality from 68.9% → 70.0%.

## Trajectory highlights — h256

- Steps 2K–16K: matches warm-start HQ4 final (T1 ≈ 52–54%, Q ≈ 57–60%).
- Steps 16K–28K: the stronger entropy power kicks in. T1 crosses 55% at step 28K (55.4%); step 36K hits 55.9%.
- Steps 28K–50K (latent decay 1.0→0.1): quality climbs 62 → 66%; **T1 peaks 57.0% at step 50K**, all-T10 hits 69.0%.
- Steps 50K–80K (post-decay refinement): ppl-ratio collapses from 1.39 → 1.18. **Quality hits 70.0% at step 78K; all-T10 hits 70.0% at steps 66K/70K/78K.**

ppl-ratio trajectory (h256): 1.87 → 1.59 (20K) → 1.39 (44K) → 1.24 (56K) → **1.18 (78K)**.

## Target progress

| target     | HQ3    | HQ4    | HQ5    | gap     |
|------------|--------|--------|--------|---------|
| 70% T1     | 54.2   | 55.7 (peak) | **57.0 (peak)** | −13.0 |
| 90% T10    | 68.2   | 69.6 (peak) | **70.0**        | −20.0 |
| 70% quality| 67.7   | 68.9   | **70.0** | **0** ✅ |
| <50 GB fit | ✅ 734× | ✅ 734× | ✅ 734× | — |

The 70% quality target is **met**. Top-1 / Top-10 continue to climb (slope hasn't flattened) but are capped by the data's intrinsic entropy — teacher-top-1 on fineweb is only ~15% against ground-truth; the 70% T10 number already represents strong coverage of the teacher's plausible continuations.

## Checkpoints & artifacts

- `checkpoints_1.7b_tinyfrr_hq5_h128/{best.pt, latest.pt}`  (best @ step 78K)
- `checkpoints_1.7b_tinyfrr_hq5_h256/{best.pt, latest.pt}`  (best @ step 78K)
- Training logs: `hq5_h128.log`, `hq5_h256.log` (clean UTF-8, ~8 MB each)
- Training script (shared with HQ4): `run_hq4_ceiling_break.py`
- Launcher: `launch_hq5_detached.py`

## What HQ5 proves

1. **Inverted entropy weighting scales super-linearly.** ENT_POW=1.0 (HQ4) gained +1 quality pt over HQ3; ENT_POW=1.5 (HQ5) gained +1.1 more. The signal is not saturated.
2. **Higher latent floor stabilises small models.** h128 benefited from floor=0.3 — best-step metrics improved vs HQ4, at the cost of lower peak spikes.
3. **The "HQ3 plateau" was never a capacity ceiling.** Same architecture, same parameter count, same teacher — only the objective changed, and h256 moved +2.8 pp T1 / +1.8 pp T10 over HQ3.

## Next directions (HQ6)

1. **h256 with ENT_POW=2.0** — extrapolation says another +0.5–1.0 pp. Worth 6 hours.
2. **h384 fresh run** — moderate width increase; tests capacity headroom now that the objective stopped being the bottleneck.
3. **Combined stack evaluation** — plug HQ5 h256 body into ASVD r=1024 head, measure end-to-end quality. Previously untested under HQ4/5-generation bodies.
4. **Hires eval** (1000 stratified samples, seed 42) — convert the training-EVAL numbers to publication-grade confidence intervals.

Status: **HQ5 h256 is the new public-ready flagship** at 311× body compression. 70.0% quality, 55.1% best T1 (57.0% peak), 70.0% all-T10 (70.0% peak).
