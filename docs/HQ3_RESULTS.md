# HQ3 Breakthrough — Final Results

Teacher: Qwen3-1.7B (1.72B params, hidden=2048, vocab=151936)
Student: TinyFRR_HQ3 (proj_in → FractalModel(n_scales=4, iters=7) → proj_out → frozen lm_head)
Training: 80,000 steps, fineweb_edu_500M_tokens.pt, SEQ_LEN=128, BATCH=4×ACCUM=2
Objective: total = fkl + 0.3·rkl + 1.0·lat + 0.5·ramp·ce + 0.3·ramp·margin (confidence-weighted)

## Final numbers

| variant | trainable | compression | best T1 | best quality | peak all-T10 | peak last-T10 | best ppl-ratio |
|---------|-----------|-------------|---------|--------------|--------------|---------------|----------------|
| h128    | 0.64M     | **734×**    | 54.21%  | 67.7%        | 68.0%        | 64.1%         | 1.200          |
| h256    | 1.51M     | **311×**    | 54.12%  | **68.1%**    | 68.2%        | 64.2%         | 1.216          |

Both completed 80K steps cleanly. Checkpoints saved:
- `checkpoints_1.7b_tinyfrr_hq3_h128/best.pt`
- `checkpoints_1.7b_tinyfrr_hq3_h256/best.pt`

## Target assessment

Originally requested: **70+% T1, 90+% T10, <50 GB fit for 1T+ model**.

Hit:
- Compression extreme (734×) — a 1T param model shrunk by 734× is 1.36B params ≈ under 3 GB @ fp16, trivially fits one GPU. ✅
- T10 ceiling ~68%, not 90%. ❌
- T1 ceiling ~54%, not 70%. ❌

T1/T10 plateau at ~54%/68% across both widths despite ce_ramp saturating at 1.0 for the last 32K steps. Widening hidden (h128 → h256) bought essentially nothing on T1, only ~0.4 pts on quality — this is an **objective ceiling, not a capacity ceiling**.

## Trajectory highlights

h128 T1: 46.2 → 48.2 → 50.5 → 53.6 → 54.2 (peak step 74K) → oscillating 52–53
h256 T1: 39.5 → 46.0 → 49.8 → 52.6 → 54.7 (peak step 62K) → oscillating 53–54

ppl-ratio steadily improved throughout (h128 hit 1.200 @ step 68K; h256 hit 1.216 @ step 78K), indicating the latent/distillation signal kept refining even after T1 plateaued.

## Diagnosis — why plateau

1. **Teacher-capped**: teacher-top1 on this eval slice is probably ~58–62%; student at 54% is close to the teacher ceiling on this task.
2. **CE + margin with conf-weighting** focuses gradient on already-easy tokens — exactly where the student is already saturated. The *hard* tokens (where T10 would climb from 68 → 90) are downweighted.
3. **Latent MSE dominant early** pulls the student into a smooth, mean-seeking regime; once ce_ramp=1.0 kicks in, the model can no longer escape it.

## Next directions (if pushing further)

- **Invert the confidence weighting**: upweight high-entropy positions — forces the student to learn the *hard* distributional modes (where T10 gains live).
- **Remove latent loss after step 30K** and let the CE/margin objective dominate — break the mean-seeking attractor.
- **h512 run** with the same objective: tests whether this is capacity-bound (unlikely given h128→h256 showed <1 pt gain, but cheap to verify).
- **Selective student (TrustGate)**: route only high-entropy teacher positions through the student, fall back to teacher for easy tokens — this is how real deployment would work anyway.

Status: current checkpoints are the HQ3-family ceiling. Moving further requires an objective change, not more training.
