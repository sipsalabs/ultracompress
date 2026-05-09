# V4-D Multi-Pass Cascade Correction -- Research Log

**Date:** 2026-05-09
**Operator:** ml-engineer agent (Sipsa Labs)
**GPU:** cuda:0 (RTX 5090 #0, idle)
**Model:** Qwen3-1.7B-Base (28 layers)
**Status:** IN PROGRESS

---

## Hypothesis

The single-pass V18-C correction `y = W_base @ x + alpha * U(V(x))` at rank=32 saturates at deep layers (23-27) with `train_loss_final` 0.35-0.81. A 2-pass cascade at rank=16 per pass (same total param count) has a different inductive bias: pass 1 corrects the STRUCTURED residual-of-residual left by pass 0. If the deep-layer residual is not random noise but has block-diagonal or attention-head-specific patterns, the cascade can capture directions that are linearly independent of pass 0's subspace.

**Target:** PPL ratio <= 1.00400x (improvement on the 1.00488x uniform baseline at n=30/seq=1024).

## Mechanism

```
correction_0 = alpha_0 * U_0(V_0(x))        # pass 0: rank-16, input = x (hd-dim)
correction_1 = alpha_1 * U_1(V_1(correction_0))  # pass 1: rank-16, input = pass-0 output (nr-dim)
y = W_base @ x + correction_0 + correction_1
```

Key architectural detail: pass 1's V_1 takes `correction_0` as input (dimension = nr = out_dim), NOT x (dimension = hd = in_dim). This breaks linear composition: two sequential rank-16 maps with DIFFERENT input spaces cannot be trivially collapsed into a single rank-16 map.

Pass 0: SVD warm-start from quantization residual (same as standard V18-C).
Pass 1: Random init (no pre-computable SVD target for residual-of-residual).

## Parameter Budget

At rank=32 single-pass per Linear:
- V: [hd, 32], U: [32, nr], alpha: 1 scalar
- Total per Linear: 32*(hd+nr) + 1

At n_passes=2, rank_per_pass=16:
- Pass 0: V_0: [hd, 16], U_0: [16, nr], alpha_0: 1
- Pass 1: V_1: [nr, 16], U_1: [16, nr], alpha_1: 1
- Total per Linear: 16*(hd+nr) + 16*(nr+nr) + 2

For Qwen3-1.7B self_attn (hd=2048, nr varies by proj):
- q_proj/o_proj: nr=2048. Single: 32*4096+1=131073. Multi: 16*4096+16*4096+2=131074. ~EQUAL.
- k_proj/v_proj: nr=1024. Single: 32*3072+1=98305. Multi: 16*3072+16*2048+2=81922. Multi is CHEAPER (16% less).
- MLP (nr=hd=2048): Single: 32*4096+1=131073. Multi: 16*4096+16*4096+2=131074. ~EQUAL.

TOTAL across all 7 Linears/layer: multi-pass is approximately equal to or slightly cheaper than single-pass. The effective bpw overhead is the same. No extra params to count.

## Experiment Config

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | Qwen/Qwen3-1.7B-Base | Validated config |
| GSQ bpw | 5 | Validated config |
| GSQ block_size | 64 | Validated config |
| V18-C rank (base) | 32 | Validated config |
| V4-D n_passes | 2 | This experiment |
| V4-D rank_per_pass | 16 | = rank / n_passes |
| Train steps | 200 | Validated config |
| Train LR | 1e-3 | Validated config |
| Train BS | 8 | Validated config |
| n_calib | 100 | Default |
| n_eval | 30 | Validated config |
| seq_len | 1024 | Validated config |
| seed | 42 | Validated config |
| Data | FineWeb-edu held-out tail | Validated config |
| Device | cuda:0 | GPU 0 idle |
| Env flags | UC_V4D_MULTIPASS=2 | This experiment |

## Implementation

Wired `CorrectionMatrixC_MultiPass` (ported from `scaling_curve_runner.py:1132-1186`) into `streaming_compression_runner.py` behind `UC_V4D_MULTIPASS=<n_passes>` env flag. Changes:

1. Added `CorrectionMatrixC_MultiPass` class after `CorrectionMatrixC` (with `init_from_svd` for pass 0 only).
2. Added V4-D env flag parsing in `compress_single_layer()` after rank-redistribute block.
3. Modified `wrap_linears_with_correction()` to select `CorrectionMatrixC_MultiPass` when flag >= 2.
4. Updated freeze logic to handle `nn.ModuleList` of Vs/Us and `nn.ParameterList` of alphas.
5. Updated save_dict corrections persistence for multi-pass (V_weights list, U_weights list, alphas list).
6. Updated eval reconstruction in `streaming_eval_ppl()` to detect and reconstruct multi-pass modules.
7. Added V4D reporting in results JSON and print banner.

Total: ~75 LOC (slightly more than the estimated ~30 because of eval reconstruction).

## Decision Criteria

- PPL ratio <= 1.00400x: **WIN**. Multi-pass adds value. Stack with V3 for headline number.
- PPL ratio in [1.00400, 1.00488]: **NEUTRAL**. Multi-pass is a wash vs single-pass at this configuration.
- PPL ratio > 1.00488: **REFUTED**. Multi-pass at half-rank is worse than single-pass at full-rank.

## Run Command

```bash
UC_V4D_MULTIPASS=2 python scripts/overlay/streaming_compression_runner.py \
  --model qwen3-1.7b-base --n_eval 30 --seq_len 1024 \
  --out_json docs/PPL_EVAL_qwen3-1.7b-base-v4d-multipass_2026_05_09.json
```

## Measurement

Run completed 2026-05-09T11:16:15 MDT. JSON: `docs/PPL_EVAL_qwen3-1.7b-base-v4d-multipass_2026_05_09.json`.

- **Baseline PPL:** 12.7683 (bf16, n=30, seq=1024, seed=42)
- **Compressed PPL:** 13.6395
- **PPL Ratio:** **1.0682x** (6.82% degradation)
- **Total wallclock:** 852.6s (445s compress + 400s eval + 7s baseline)
- **Peak VRAM:** 3.46 GB (compress), 2.29 GB (eval)

### Per-layer train_loss_final comparison

| Layer | V4-D MultiPass (rank=16x2) | Uniform (rank=32x1, from V3 doc) | Ratio |
|-------|---------------------------|----------------------------------|-------|
| 0 | 0.001150 | ~0.0002 | ~5.8x worse |
| 5 | 0.004220 | ~0.002 | ~2.1x worse |
| 10 | 0.014645 | ~0.009 | ~1.6x worse |
| 15 | 0.058191 | ~0.02 | ~2.9x worse |
| 20 | 0.998714 | ~0.29 | ~3.4x worse |
| 23 | 2.112574 | ~0.35 | **6.0x worse** |
| 25 | 3.664543 | ~0.60 | **6.1x worse** |
| 27 | 5.195891 | ~0.81 | **6.4x worse** |

The V4-D cascade is UNIFORMLY worse at EVERY layer, and the gap WIDENS at deep layers (6x at layers 23-27 vs 2x at layers 5-10). The multi-pass architecture does not help at any depth.

## Conclusion

**REFUTED.** V4-D Multi-Pass Cascade Correction at n_passes=2 / rank_per_pass=16 is catastrophically worse than single-pass rank=32: PPL ratio 1.0682x vs 1.00488x (13.7x larger degradation).

### Root cause analysis

The hypothesis that the residual-of-residual is STRUCTURED and capturable by a second pass is wrong at this configuration. Three contributing failure mechanisms:

1. **Rank halving is the dominant effect.** Each pass sees rank=16 instead of rank=32. At shallow layers where rank-32 is overprovisioned, rank-16 is still adequate but less flexible (train_loss 2-6x worse). At deep layers where rank-32 is already insufficient, rank-16 is catastrophically underpowered (6x worse loss, reaching 5.2 at layer 27 vs 0.81 uniform).

2. **Pass 1 sees a useless input.** Pass 1's input is `correction_0 = alpha_0 * U_0(V_0(x))`, which is a rank-16 projection of x. This is an extremely information-poor signal -- it has discarded all but 16 directions of the original hidden-state. Pass 1 cannot recover information that pass 0 already discarded. The "residual-of-residual" hypothesis assumed pass 1 would correct structured errors, but pass 1 does not SEE the original residual -- it only sees pass 0's output, which is a lossy compression of the residual.

3. **Pass 1 random init + alpha=0 start means it contributes nothing useful.** Pass 1 starts from random init with alpha=0. In 200 training steps, it must simultaneously learn (a) what the residual-of-pass-0 looks like, and (b) how to correct it. With no SVD warm-start (there is no pre-computable target for pass 1), the optimizer is lost in a flat landscape.

### What this rules out

- Multi-pass cascade at half-rank is NOT a viable cure for V18-C saturation.
- The "sequential refinement at constant param budget" inductive bias does NOT help when the budget per pass drops below the effective rank of the residual.

### What this does NOT rule out

- Multi-pass at FULL rank per pass (rank=32 x 2 = double the total params) might work, but that is a different experiment: it would be comparing 2x parameter budget, not a topology change at constant budget. Not useful for the compression ratio story.
- V3 rank-redistribution (varying rank by depth, single-pass) remains the primary cure direction. This result reinforces that rank is the binding constraint: halving rank causes 6x worse loss even with a second correction pass.

### Disposition

V4-D is CLOSED. Do not re-run. Move to V3 (rank-redistribution) as the next experiment on the PPL floor.

