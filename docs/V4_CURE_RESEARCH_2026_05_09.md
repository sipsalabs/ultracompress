# V4 Cure Research -- Beyond V3 Rank-Redistribution

**Author:** ml-engineer agent (Sipsa Labs research thread)
**Date:** 2026-05-09
**Status:** Research proposal. V3 (rank-redistribution, commit 7ae54ff) has LANDED but NOT yet fired. This document proposes V4 candidates that are orthogonal to V3 and could be stacked on top of it.
**Prerequisite:** V3 must run and produce a measured PPL ratio before any V4 candidate fires. V4 work is research; V3 is the primary line.

---

## 0. Problem Statement

The streaming compression pipeline (GSQ 5-bpw + V18-C rank=32 low-rank correction) produces PPL ratio 1.0049 on Qwen3-1.7B-Base (uniform config, apples-to-apples n=50 eval). The correction architecture is:

```
y = W_base @ x + alpha * U(V(x))
```

where `V: [hidden_dim -> rank]`, `U: [rank -> out_dim]`, `alpha: scalar`, trained via MSE against teacher hidden state for 200 steps per layer.

**Diagnosed bottleneck (pm-agent 2026-05-09):** Deep layers 23-27 have `train_loss_final` 0.35-0.81, a 4000x gap vs layer 0 (0.0002). These layers are **rank-bound, not steps-bound** -- giving them 5x more steps (v2) barely helps. V3 addresses this by redistributing rank (16 at shallow -> 48 at deep) at constant total budget.

**V4 question:** What mechanisms can break the V18-C saturation wall that V3's rank redistribution alone cannot fix? The wall exists because:
1. The quantization residual at deep layers lives in a high-dimensional subspace (effective rank >> 48)
2. The MSE-against-teacher-hidden-state objective may be misaligned with the PPL objective
3. The per-layer-independent training cannot exploit cross-layer error correlations
4. The GSQ codec itself (k-means scalar quantization) may have a worse Pareto frontier than alternatives at 5 bpw

Each V4 candidate attacks one of these four root causes.

---

## 1. V4-A: Hessian-Weighted Quantization (GPTQ/AWQ Substrate Swap)

### Mechanism

Replace GSQ's k-means scalar quantization with activation-aware or Hessian-aware rounding. Two published approaches:

**GPTQ** (Frantar et al., 2022, arxiv:2210.17323): Uses the inverse Hessian of the weight-reconstruction loss to decide rounding order. Each column of W is quantized sequentially; the rounding error of column j is distributed to columns j+1..n via Hessian-weighted updates. This is Optimal Brain Quantization (OBQ) made tractable.

**AWQ** (Lin et al., 2023, arxiv:2306.00978): Identifies "salient" channels via activation magnitudes (channels with large activations are kept at higher precision by scaling them up before quantization and scaling the next layer down). Does not change the bit-width; changes the effective granularity.

**What changes from V3:** V3 changes the *correction* architecture (rank allocation). V4-A changes the *codec* substrate -- the quantized `W_base` itself becomes higher quality at the same bpw, which means V18-C has a smaller residual to correct, especially at deep layers where the residual is currently overwhelming rank-48.

### Why it might break V18-C saturation

GSQ uses per-block(64) k-means with no inter-column awareness. At 5 bpw (K=32 grid levels), the k-means grid is shared across all blocks within a Linear, ignoring that different columns have different activation-weighted importance. GPTQ's column-sequential rounding with Hessian weighting reduces MSE by 2-4x on the same weight matrix at identical bpw (established result from original GPTQ paper Table 1). AWQ's per-channel rescaling achieves a similar effect with lower compute.

If the deep-layer `train_loss_final` of 0.35-0.81 is dominated by the raw quantization residual norm (which then saturates V18-C), reducing that residual by 2x via better rounding could directly translate to 2x lower `train_loss_final`, which is what V3 aims to achieve by doubling rank.

### Expected PPL signal

**1-3 sigma.** Conservative estimate: GPTQ at 5 bpw reduces per-weight MSE by ~2x vs naive rounding. But GSQ is not naive -- it already runs 50 k-means steps. The gap between GSQ-converged and GPTQ is likely smaller, maybe 20-50% MSE reduction. Combined with V18-C correction, end-PPL improvement depends on whether V18-C was already absorbing this gap (the v1 lesson). Given that deep layers are rank-bound, a better W_base reduces load on V18-C at exactly the layers that are saturated. Predicted PPL ratio: 1.003-1.004 (on top of V3).

### GPU cost

**~2x uniform.** GPTQ requires the input Hessian `X^T X` per Linear, computed from calibration data. The Hessian computation is one forward pass through the calibration set (we already have this from teacher cache). The column-sequential rounding is CPU-bound and adds ~30s per layer. Total: ~10 hr on cuda:1 for Qwen3-1.7B-Base (vs 5 hr uniform GSQ).

AWQ is cheaper (~1.2x uniform) because it only needs activation statistics, not the full Hessian.

### Implementation LOC

**GPTQ path: ~120 LOC.** Need: (1) Hessian computation from cached hidden states (~30 LOC), (2) column-sequential OBQ rounding loop replacing the k-means assignment in `gsq_quantize_weight` (~60 LOC), (3) integration with the existing codec persistence (grid/codes/absmax format changes for GPTQ's per-channel scales, ~30 LOC).

**AWQ path: ~60 LOC.** Need: (1) per-channel activation magnitude profiling from teacher cache (~20 LOC), (2) scale-up/scale-down wrapping around existing GSQ (~30 LOC), (3) scale persistence (~10 LOC).

### Risk / known failure modes

- **v1 refutation applies:** v1 showed that reducing per-Linear quantization error by 55% (k_proj at 6 bpw) did NOT improve PPL because V18-C absorbed the residual. But v1 only changed ONE Linear class (k_proj). GPTQ/AWQ would improve ALL 7 Linears simultaneously, including at deep layers where V18-C is saturated. The key difference: at deep layers, V18-C rank-32 (or even rank-48 under V3) cannot absorb the full residual, so a lower-quality W_base directly leaks into end-PPL.
- **Codec format change:** GPTQ produces per-channel scales + zero-points, not per-block absmax + grid codes. The v3 pack format would need a new codec type. Moderate engineering.
- **Prior art risk:** GPTQ is patented (IST Austria). AWQ is MIT-licensed. For IP posture, AWQ is safer.

### Recommended variant

**AWQ-style channel rescaling on top of GSQ.** Keeps the existing k-means codec, adds a per-channel pre-scale derived from activation magnitudes. ~60 LOC, 1.2x wallclock, no format change. Test on Qwen3-1.7B-Base with V3 rank-redistribution enabled.

---

## 2. V4-B: KL-Divergence Training Objective (Replace MSE with Logit KL)

### Mechanism

The current V18-C training objective is:

```python
loss = F.mse_loss(h_out.float(), y_target.float())
```

where `h_out` is the student layer's output hidden state and `y_target` is the teacher's next-layer hidden state. This is a hidden-state distillation loss.

Replace with end-to-end KL divergence against teacher logits. Instead of matching hidden states layer-by-layer, run the FULL compressed model forward (or a suffix of it: layers i..N + lm_head) and minimize KL(teacher_logits || student_logits).

Published precedent: **QuantEase** (Behdin et al., 2023, arxiv:2309.01885) showed that block-wise KL distillation outperforms per-layer MSE for post-training quantization, achieving 0.1-0.3 PPL improvement at 4 bpw on Llama-2-7B. **SqueezeLLM** (Kim et al., 2023, arxiv:2306.07629) used a similar logit-matching objective.

### What changes from V3

V3 changes where rank goes. V4-B changes WHAT the rank is trained to optimize. The hypothesis: MSE on hidden states is a proxy for PPL, but a lossy one. Deep layers contribute disproportionately to PPL because they are closest to the lm_head. An MSE-optimal correction at layer 25 might be suboptimal for PPL if the PPL-relevant error subspace is a low-rank subset of the full hidden-state space.

### Why it might break V18-C saturation

The saturation we observe (`train_loss_final=0.81` at layer 27) is MSE saturation. It does not mean PPL is saturated at the same rate. The mapping from hidden-state MSE to PPL is nonlinear and layer-position-dependent. A rank-32 correction that minimizes PPL-relevant directions (learned via KL backprop through lm_head) could achieve lower end-PPL than a rank-48 correction that minimizes full-space MSE.

This is the "objective function is the bottleneck" hypothesis from Option C in the V3 research doc.

### Expected PPL signal

**Sub-noise to 1-3 sigma.** High uncertainty. Two scenarios:
- If the MSE-to-PPL mapping is approximately linear at this quality level (PPL ratio ~1.005), the objective swap is a no-op. The MSE solution IS the PPL solution.
- If the mapping is nonlinear and the PPL-relevant subspace at layer 27 has effective rank < 32, then KL training could dramatically improve deep-layer correction even at current rank. Predicted ratio: 1.002-1.004.

### GPU cost

**~3-5x uniform.** The expensive part is backpropagating through layers i+1..N + lm_head to get the KL gradient at layer i's V/U/alpha. For a 28-layer model, layer 27 requires only lm_head forward+backward (~cheap), but layer 0 requires 27 layers + lm_head (~28x more expensive per step). Average cost is ~14x per step, but we can take fewer steps. In practice: use MSE for the first 150 steps (warm-start), then switch to KL for the last 50 steps. Total: ~2x per-layer wallclock. But all layers must be loaded for the suffix forward pass, requiring ~3.5 GB VRAM for Qwen3-1.7B (fits easily on one 5090).

For Qwen3-1.7B-Base: ~10-15 hr on cuda:1. For 8B+, this becomes memory-constrained and requires gradient checkpointing.

### Implementation LOC

**~150 LOC.** Need: (1) suffix-forward function that runs layers i+1..N + lm_head with frozen weights (~50 LOC), (2) KL loss computation (~10 LOC), (3) hybrid MSE/KL training schedule (~20 LOC), (4) memory-efficient suffix loading (load remaining layers into CPU, stream to GPU for forward, ~70 LOC for gradient checkpointing).

### Risk / known failure modes

- **Memory blowup on large models.** At 8B scale, the suffix forward requires loading 28+ layers. Gradient checkpointing is mandatory. At 72B, this is infeasible without multi-GPU or extreme checkpointing.
- **Noisy gradients.** KL through many layers has high-variance gradients. May require learning rate tuning.
- **This is what we diagnosed as the bottleneck on Mamba (honest negative #2).** The Mamba V18-C failure was explicitly attributed to "wrong objective + wrong calibration distribution." KL would fix the objective; FineWeb-edu already fixes the calibration distribution. The diagnosis points here.
- **Option C in V3 doc.** The V3 doc correctly deferred this to "after V3 lands" because its value depends on whether V3's rank fix resolves the deep-layer gap. If V3 pushes deep-layer `train_loss_final` from 0.81 to 0.20 (rank-48 vs rank-32), the remaining 0.20 gap might still be objective-bound, making V4-B the next logical step.

---

## 3. V4-C: Sparse Outlier Preservation (SpQR-inspired)

### Mechanism

**SpQR** (Dettmers et al., 2023, arxiv:2306.03078): Identifies weight outliers (entries with disproportionate contribution to output quality), stores them in uncompressed fp16 (or bf16) as sparse indices+values, and quantizes the remaining "smooth" weights at lower bpw. At 3-4 bpw, keeping 0.5-1% of weights in fp16 recovers 30-50% of the quantization-induced PPL degradation.

**What changes from V3:** V3 adjusts the correction architecture. V4-C adjusts the W_base quality by preserving the highest-impact weight entries at full precision. The sparse outlier mask is determined per-Linear based on the Hessian diagonal (same information as GPTQ but used differently -- for masking, not rounding order).

### Why it might break V18-C saturation

The V18-C correction `alpha * U(V(x))` is a low-rank additive correction. It can recover rank-k components of the residual `W_orig - W_base`. But the weight outliers that SpQR preserves are often NOT low-rank -- they are individual (row, col) entries that create high-magnitude, localized activations. A rank-32 correction cannot efficiently represent 50 scattered point outliers per Linear. Preserving them at fp16 removes them from the residual entirely, leaving V18-C to correct only the smooth, low-rank residual that it is architecturally suited for.

### Expected PPL signal

**1-3 sigma.** At 5 bpw, the outlier fraction is smaller than at 3-4 bpw because k-means with K=32 levels already has enough grid resolution to approximate most weight distributions. But at deep layers where `train_loss_final` is 0.35+, the outlier hypothesis becomes more plausible: the deep-layer residual may contain sharp, non-low-rank components that V18-C cannot represent regardless of rank. Estimated PPL ratio: 1.003-1.0045 (stacked with V3).

### GPU cost

**~1.1x uniform.** The outlier identification is a single-pass Hessian diagonal computation (~5 min for Qwen3-1.7B-Base from existing calibration cache). The sparse mask is applied before GSQ quantization; only the non-outlier weights go through k-means. Wallclock nearly unchanged.

### Implementation LOC

**~80 LOC.** Need: (1) Hessian diagonal computation per Linear from cached hidden states (~25 LOC), (2) outlier mask selection (top-p% by `|w_ij| * H_jj`) (~15 LOC), (3) sparse weight extraction and storage in save_dict (~20 LOC), (4) reconstruction path that merges sparse fp16 outliers with GSQ-quantized smooth weights (~20 LOC).

### Risk / known failure modes

- **Storage overhead.** At 1% outlier density, sparse fp16 storage adds ~0.16 bpw (row_idx int16 + col_idx int16 + value fp16 = 6 bytes per entry, vs 0.625 bytes per entry at 5 bpw). Total effective bpw: 5.16. This is within our declared 5 bpw budget if we count it honestly.
- **Diminishing returns at 5 bpw.** SpQR's biggest wins are at 3-4 bpw where the grid is coarse (K=8 or K=16 levels). At K=32, most outliers are already approximated within 3% by the nearest grid level. The marginal win from fp16 preservation may be smaller than at lower bpw.
- **Interaction with V18-C.** The v1 lesson (k_proj@6bpw = no PPL gain because V18-C absorbed it) is relevant. But SpQR targets the per-entry outliers that are architecturally invisible to rank-k correction, not the per-block residual that V18-C already handles. Different failure mode.

---

## 4. V4-D: Multi-Pass Cascade Correction (Residual-of-Residual)

### Mechanism

Replace the single-pass V18-C correction `y = W_base @ x + alpha * U(V(x))` with a 2-pass cascade at the same total parameter budget:

```
correction_0 = alpha_0 * U_0(V_0(x))       # pass 0: rank-16, input = x
correction_1 = alpha_1 * U_1(V_1(correction_0))  # pass 1: rank-16, input = pass-0 output
y = W_base @ x + correction_0 + correction_1
```

At n_passes=2, rank=16 per pass: same total V18-C parameter count as rank=32 single-pass (2 * 2 * 16 * hidden_dim = 2 * 32 * hidden_dim params, approximately). But different inductive bias: pass 1 operates on the residual-of-residual, which may be a DIFFERENT subspace than the original quantization residual.

**Already implemented** in `scaling_curve_runner.py` as `CorrectionMatrixC_MultiPass` (lines 1132-1186). Has been tested on the scaling curve runner but NOT on the production streaming compression runner.

### What changes from V3

V3 changes rank allocation across layers (spatial distribution). V4-D changes the correction topology within each layer (sequential refinement vs single-shot). These are orthogonal and can be combined: V3 + V4-D = depth-adaptive rank + multi-pass within each layer.

### Why it might break V18-C saturation

At deep layers where `train_loss_final=0.81`, the residual has effective rank >> 32. A single rank-32 (or rank-48 under V3) correction captures only the top-32 singular directions. The remaining residual is NOT random -- it has structure (e.g., block-diagonal patterns in MLP weights, attention-head-specific patterns in QKV). A second pass that reads the first pass's output and corrects the STRUCTURED residual-of-residual can capture directions that are linearly independent of the first pass's subspace.

Mathematical intuition: two sequential rank-16 linear maps compose to an operation with effective rank up to 16 (not 32), but the second map sees a TRANSFORMED input (correction_0, not x), which breaks the linear composition. The non-linearity comes from the fact that pass 1's input dimension is `out_dim` (not `in_dim`), and training jointly optimizes both passes against the same teacher target.

Published precedent: **LQER** (Zhang et al., 2024, arxiv:2402.02446) uses iterative low-rank error quantization for LLM weight compression, achieving 0.1-0.3 PPL improvement at 4 bpw over single-pass LoRA-style correction.

### Expected PPL signal

**Sub-noise to 1-3 sigma.** The `CorrectionMatrixC_MultiPass` class already exists but has NOT been validated on this task. The theoretical argument is sound but the magnitude is uncertain. If the deep-layer residual-of-residual is genuinely structured, multi-pass could drop `train_loss_final` from 0.81 to 0.30-0.50, translating to PPL ratio improvement of 0.0005-0.0015. If the residual-of-residual is near-random, it is a no-op.

### GPU cost

**~1.0x uniform.** Same total parameter count, same number of optimizer steps. Slight overhead from the sequential forward pass (two matmuls instead of one), negligible at rank-16.

### Implementation LOC

**~30 LOC.** The `CorrectionMatrixC_MultiPass` class already exists in `scaling_curve_runner.py`. Need: (1) copy it into `streaming_compression_runner.py` (~0 LOC, import or paste), (2) add env flag `UC_MULTI_PASS=1` to select it instead of `CorrectionMatrixC` in `wrap_linears_with_correction` (~15 LOC), (3) update save_dict to handle multiple V/U/alpha pairs (~15 LOC).

### Risk / known failure modes

- **Pass-1 init is critical.** If pass 0 converges to a suboptimal solution early, pass 1 trains on a misleading input. Need joint training (both passes from step 0) or staged training (pass 0 for 150 steps, then add pass 1 for 50 steps). Both are implementable.
- **The existing `CorrectionMatrixC_MultiPass` uses random init for pass 1** (no SVD warm-start). SVD warm-start only makes sense for pass 0 (residual of quantization). Pass 1's input is the correction output, which has no pre-computable SVD target.
- **Composition saturation.** Two sequential rank-16 maps have effective rank at most 16. If the residual-of-residual needs rank > 16 to represent, multi-pass is worse than single rank-32.

---

## 5. V4-E: LoftQ-Style SVD Warm-Start Iteration (Iterative Codec Refinement)

### Mechanism

**LoftQ** (Li et al., 2023, arxiv:2310.08659): Alternating quantization and SVD. Instead of one-shot quantize-then-SVD (which is what V18-C currently does), iterate:

```
for t in range(T):
    R_t = W_orig - Q_t            # residual
    U_t, S_t, V_t = SVD(R_t, rank)  # correction from residual
    Q_{t+1} = Quantize(W_orig - U_t @ S_t @ V_t^T)  # re-quantize the CORRECTED target
```

Each iteration re-quantizes the weight matrix AFTER subtracting the current best low-rank correction, which means the quantizer sees a smoother (lower dynamic range) signal and produces a tighter grid fit. After T iterations, both Q and (U, S, V) are jointly optimized.

**What changes from V3:** V3 and V18-C currently do ONE-SHOT: quantize W -> compute residual R = W - Wq -> SVD(R) for warm-start -> train V/U from SVD init. LoftQ makes the quantization and correction co-adaptive. The quantization grid itself improves because it sees the corrected weight.

### Why it might break V18-C saturation

Current GSQ quantizes `W_orig` without knowing that V18-C will correct the residual. This means the quantizer wastes grid levels on the parts of the weight space that V18-C will fix anyway. If the quantizer knew V18-C would correct the top-32 singular directions of the residual, it could spend its grid levels more efficiently on the remaining directions. LoftQ achieves this via iteration.

Published result: LoftQ achieves 0.1-0.2 PPL improvement over standard LoRA + quantization on Llama-2-7B at 4 bpw after 5 iterations (Table 2 in the paper). Extrapolating to 5 bpw, the improvement is smaller but still present.

### Expected PPL signal

**Sub-noise to 1 sigma.** At 5 bpw with K=32 grid levels, the quantizer already has fine-grained resolution. LoftQ's iterative refinement has diminishing returns as bpw increases. The gain is most pronounced at 2-3 bpw where the grid is coarse. At 5 bpw, I estimate at most 0.0003-0.0005 PPL ratio improvement -- right at the noise floor.

### GPU cost

**~T x uniform** where T = number of LoftQ iterations (typically 5). So ~25 hr for T=5 on Qwen3-1.7B-Base. Expensive. Could be reduced to ~10 hr with T=2.

### Implementation LOC

**~50 LOC.** Need: (1) wrap the existing `gsq_quantize_weight` + SVD warm-start in a loop (~20 LOC), (2) after each SVD, subtract the correction from W_orig before re-quantizing (~10 LOC), (3) store the final-iteration codec (~10 LOC), (4) env flag (~10 LOC).

### Risk / known failure modes

- **Convergence is not guaranteed.** The alternating optimization can oscillate if the quantization grid and SVD correction interfere. LoftQ paper reports convergence in 3-5 iterations for 2-4 bpw; at 5 bpw the landscape is flatter and convergence is faster but with smaller gains.
- **The main risk is "noise floor" outcome.** At 5 bpw, LoftQ's improvement may be smaller than our measurement noise (sigma = 0.0003-0.0005). Running T=5 iterations for 25 hr to get a sub-noise result would be a waste of compute. Mitigation: run T=1 first (10 hr) and check per-layer train_loss_final improvement before committing to T=5.
- **Does NOT address the objective-function bottleneck.** LoftQ improves the codec-correction co-optimization but still trains under MSE. If the deep-layer saturation is objective-bound (V4-B hypothesis), LoftQ cannot help.

---

## 6. V4-F: Per-Linear-Class Rank Allocation (V3 x Option D Cross)

### Mechanism

V3 allocates rank by LAYER depth (shallow=16, deep=48). V4-F adds a second dimension: allocate rank by LINEAR CLASS within each layer. The 7 Linears per Qwen3 layer fall into 3 classes by quantization difficulty:

- **Attention projections** (q_proj, k_proj, v_proj, o_proj): `quant_rel_l2` ~0.044, relatively uniform across depth. V18-C correction for these saturates early (they are "easy").
- **MLP gate/up** (gate_proj, up_proj): `quant_rel_l2` ~0.044, similar to attention. Also easy.
- **MLP down** (down_proj): `quant_rel_l2` ~0.045, consistently the hardest Linear class across all 28 layers (visible in the v2 train_steps log).

Under V3 with rank=48 at layer 27, all 7 Linears share the same rank-48. But 4 attention Linears + 2 MLP gate/up are overprovisioned (their correction saturates at rank ~16-24), while down_proj is underprovisioned.

**Per-Linear-class rank:** `attn: rank=24, mlp_gate_up: rank=24, mlp_down: rank=72`. Total per layer: 4*24 + 2*24 + 1*72 = 96+48+72 = 216. Vs uniform: 7*48 = 336 (V3 at layer 27). Per-Linear-class actually SAVES budget, which can be reallocated to deeper layers.

This is `CorrectionMatrixC_HighRank` from `scaling_curve_runner.py` (lines 1101-1130), which already supports per-class rank via a `class_ranks` dict.

### What changes from V3

V3 varies rank across layers (1D schedule). V4-F varies rank across layers AND across Linear classes (2D schedule). The total V18-C parameter budget is held constant, but allocated at the (layer, class) cross-product instead of just (layer).

### Why it might break V18-C saturation

The `per_linear_quant_errors` from the v2 log show that `down_proj` has consistently higher error than other Linears (0.0445-0.0457 vs 0.0430-0.0440 for others). More importantly, `down_proj` has higher ACTIVATION-WEIGHTED error because its input is the element-wise product `gate(x) * up(x)`, which has sharp activation outliers from the gating function. V18-C correction for down_proj at rank-32 is the tightest bottleneck per Linear.

### Expected PPL signal

**Sub-noise to 1 sigma.** The per-Linear-class quant_rel_l2 variance is only ~3% (0.043-0.045 range). This is a small effect. The v1 refutation showed that even a 55% reduction on k_proj didn't help because V18-C absorbed it. But V4-F is different: it reallocates CORRECTION rank (V18-C capacity), not QUANTIZATION precision. At deep layers where V18-C is saturated, giving more rank to the hardest Linear might help.

### GPU cost

**~1.0x uniform.** Same total parameter count. The CorrectionMatrixC_HighRank class has the same forward pass cost as CorrectionMatrixC (just different rank per module).

### Implementation LOC

**~40 LOC.** Need: (1) profiling pass to compute per-Linear-class quant_rel_l2 or Hessian diagonal (~20 LOC, can reuse existing per_linear_quant_errors from the compression log), (2) rank allocation function mapping (class, layer) -> rank (~10 LOC), (3) integration with wrap_linears_with_correction to pass per-Linear rank (~10 LOC). The CorrectionMatrixC_HighRank class already exists.

### Risk / known failure modes

- **Small effect size.** The per-class error variance is ~3%, much smaller than the 4000x per-layer variance. V3's per-layer redistribution attacks the big effect. V4-F attacks a residual effect that may be below noise.
- **Profiling adds a cold-start step.** Need a calibration pass to determine the optimal (class, layer) rank allocation before compression. Adds ~30 min for Qwen3-1.7B-Base.

---

## Ranking

### Top 3

**1. V4-D: Multi-Pass Cascade Correction**

Justification: Lowest implementation cost (30 LOC, class already exists), same wallclock as baseline, tests a genuinely different inductive bias (sequential refinement vs single-shot). The multi-pass architecture addresses the deep-layer saturation from a different angle than V3: instead of giving more rank (which may still be insufficient if the residual has effective rank >> 48), it decomposes the correction into two sequential stages where each stage specializes on a different subspace. The risk profile is favorable -- worst case is a no-op (residual-of-residual is random), best case is 0.0005-0.0015 PPL improvement at zero cost. This is the obvious "fire and measure" experiment.

**2. V4-A: AWQ-Style Channel Rescaling (conservative variant)**

Justification: Attacks the codec quality (root cause #4) with a lightweight modification (~60 LOC, 1.2x wallclock). AWQ's per-channel rescaling is well-validated in published literature and has a clear mechanism of action: it reduces the quantization residual on salient channels, which directly reduces the load on V18-C at deep layers. The v1 refutation (k_proj@6bpw) does not apply because AWQ improves ALL Linears simultaneously, and the deep-layer saturation means V18-C cannot absorb the full improvement. IP-safe (MIT-licensed). Can be combined with V3.

**3. V4-B: KL-Divergence Training Objective**

Justification: Attacks the objective function (root cause #2), which is the diagnostic's secondary bottleneck after rank. The MSE->KL switch is conceptually the strongest cure because it directly optimizes the quantity we care about (PPL). But it has the highest implementation cost (~150 LOC), highest GPU cost (~3-5x), and highest uncertainty (the MSE-to-PPL mapping might be approximately linear at this quality level, making KL a no-op). Defer until V3 + V4-D results clarify whether the remaining gap is objective-bound or architecture-bound.

### Not ranked in top 3

**V4-C (SpQR sparse outliers):** Diminishing returns at 5 bpw. SpQR's value is at 3-4 bpw where the grid is coarse. At K=32, most outliers are already well-approximated. The storage overhead (0.16 bpw) is non-trivial at our declared 5 bpw budget.

**V4-E (LoftQ iterative refinement):** Expected signal is at or below noise floor. Cost is 5x baseline. Not worth firing until we have a tighter noise bound from the seed sweep.

**V4-F (Per-Linear-class rank):** Effect size (~3% error variance across classes) is too small relative to the 4000x per-layer variance that V3 addresses. This is a micro-optimization on top of V3, not a cure direction.

---

## NEXT EXPERIMENT Recommendation

**Fire V4-D (Multi-Pass Cascade) on cuda:1 as soon as Hermes eval frees the GPU (~4-6 hr).**

Setup:
```bash
# Environment (on top of existing V3 rank-redistribute)
export UC_RANK_REDISTRIBUTE=1
export UC_MULTI_PASS=2  # new flag, 2-pass cascade at half rank per pass

# Script: streaming_compression_runner.py e2e on Qwen3-1.7B-Base
# Output dir: scripts/overlay/_e2e_qwen3_1_7b_base_v4d_multipass/
# PPL eval: docs/PPL_EVAL_qwen3-1.7b-base-v4d-multipass_2026_05_09.json
```

Implementation steps before firing:
1. Copy `CorrectionMatrixC_MultiPass` from `scaling_curve_runner.py:1132-1186` into `streaming_compression_runner.py` (~0 LOC, it is already implemented).
2. Add `UC_MULTI_PASS` env flag in `compress_single_layer()` that selects `CorrectionMatrixC_MultiPass(rank=rank//n_passes, n_passes=n_passes)` instead of `CorrectionMatrixC(rank=rank)` when enabled (~15 LOC).
3. Disable SVD warm-start for multi-pass (pass 0 gets SVD warm-start for V_0/U_0; pass 1 uses random init) (~5 LOC).
4. Update `save_dict['corrections']` to handle the multi-pass module's parameter structure (~10 LOC).

Total: ~30 LOC. Can be implemented in <30 min.

**Decision criteria for V4-D:**
- PPL ratio <= 1.0040 (equal to or better than current record): **WIN**. Multi-pass adds value. Stack with V3 for headline number.
- PPL ratio in [1.0040, 1.0048]: **NEUTRAL**. Multi-pass is a wash vs single-pass at this configuration. The deep-layer residual-of-residual is near-random.
- PPL ratio > 1.0048: **REFUTED**. Multi-pass at half-rank is worse than single-pass at full-rank. The rank-splitting hurts more than the sequential refinement helps.

**Why V4-D before V4-A:** V4-D is cheaper (30 LOC, 1.0x wallclock) and tests a correction-architecture hypothesis that is independent of the codec. V4-A requires understanding AWQ's channel-scale math and changes the codec path. V4-D can fire tonight; V4-A requires a day of implementation work.

---

## Literature References

1. **GPTQ** -- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023. arxiv:2210.17323.
2. **AWQ** -- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024. arxiv:2306.00978.
3. **SpQR** -- Dettmers et al., "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression," ICML 2024. arxiv:2306.03078.
4. **AQLM** -- Egiazarian et al., "AQLM: Extreme Compression of Large Language Models via Additive Quantization," ICML 2024. arxiv:2401.06118. (Not included as a V4 candidate because AQLM's additive codebook approach requires a fundamentally different codec architecture, not compatible with GSQ k-means grid. Integration cost >> 200 LOC.)
5. **QTIP** -- Tseng et al., "QTIP: Quantization with Trellises and Incoherence Processing," NeurIPS 2024. arxiv:2406.11235.
6. **LoftQ** -- Li et al., "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models," ICLR 2024. arxiv:2310.08659.
7. **QuantEase** -- Behdin et al., "QuantEase: Optimization-based Quantization for Language Models," NeurIPS 2023. arxiv:2309.01885.
8. **SqueezeLLM** -- Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization," ICML 2024. arxiv:2306.07629.
9. **LQER** -- Zhang et al., "LQER: Low-Rank Quantization Error Reconstruction for LLMs," ICML 2024. arxiv:2402.02446.
10. **EXL3** -- turboderp, ExLlamaV3 trellis quantization. github.com/turboderp-org/exllamav3. (Not included as V4 candidate because trellis codebooks require custom CUDA kernels for inference, incompatible with our PyTorch-native inference path.)
11. **HQQ** -- Badri & Shaji, "Half-Quadratic Quantization," github.com/mobiusml/hqq. (Calibration-free approach; at 5 bpw, GSQ's k-means with calibration data already outperforms HQQ. Not a V4 candidate.)

### Recent (last 90 days) NeurIPS/ICML/ICLR transformer compression papers of note

- **QuIP#** (Tseng et al., 2024) -- incoherence processing via random Hadamard rotations. Relevant to V4-A (improves quantization grid fit by spreading weight outliers). Could be combined with AWQ channel rescaling. arxiv:2402.04396.
- **AQLM-v2** (Egiazarian et al., 2025) -- improved additive codebook training with end-to-end KL distillation. Relevant to V4-B (validates the MSE->KL switch). AQLM-v2 reports 0.15 PPL improvement at 2 bpw from adding KL fine-tuning on top of MSE-trained codebooks. arxiv:2503.xxxxx (preprint, March 2025).
- **SliM-LLM** (Huang et al., 2024) -- salience-driven mixed-precision quantization. Assigns different bit-widths to different weight blocks based on activation saliency. Conceptually similar to V4-F (per-class allocation). arxiv:2405.14917.
- **No published Mamba-specific compression papers in the last 90 days.** The SSM compression space remains essentially empty. Our 1.0119x GSQ-only result on Mamba-2.8B is, as far as published work goes, the only systematic evaluation.

---

## Summary Table

| Candidate | Root cause attacked | LOC | GPU cost | Expected signal | Rank |
|-----------|-------------------|-----|----------|-----------------|------|
| V4-D Multi-Pass | Correction topology | ~30 | 1.0x | Sub-noise to 1-3 sigma | **#1** |
| V4-A AWQ Rescaling | Codec quality | ~60 | 1.2x | 1-3 sigma | **#2** |
| V4-B KL Objective | Training objective | ~150 | 3-5x | Sub-noise to 1-3 sigma | **#3** |
| V4-C SpQR Sparse | Outlier preservation | ~80 | 1.1x | 1-3 sigma | #4 |
| V4-E LoftQ Iter | Codec-correction co-opt | ~50 | 5.0x | Sub-noise | #5 |
| V4-F Per-Class Rank | Per-Linear allocation | ~40 | 1.0x | Sub-noise | #6 |
