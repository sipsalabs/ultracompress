---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
  - ultracompress
  - frr
  - model-compression
  - fractal-residual-recursion
  - distillation
pipeline_tag: text-generation
---

# UltraCompress — Qwen3-1.7B FRR (HQ5)

**Extreme architectural compression** of [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) via Fractal Residual Recursion (FRR) with the HQ5 entropy-aware distillation objective (entropy_power=1.5 for h256, latent-floor=0.3 for h128).

## Model Details

| Property                  | HQ5 h256        | HQ5 h128        |
|---------------------------|-----------------|-----------------|
| Base model                | Qwen/Qwen3-1.7B | Qwen/Qwen3-1.7B |
| Body compression          | **311×**        | **734×**        |
| Trainable parameters      | 1.51 M          | 0.64 M          |
| Teacher-top-1 agreement   | **55.1%** (peak 57.0%) | 54.0% (peak 54.4%) |
| Teacher-all-top-10        | **70.0%** (peak 70.0%) | 68.4% (peak 68.4%) |
| Teacher-last-top-10       | 63.1% (peak 66.5%) | 62.7% (peak 64.6%) |
| Perplexity ratio (vs T.)  | **1.18×**       | 1.20×           |
| Quality score             | **70.0%**       | 68.4%           |
| Training steps            | 80 000          | 80 000          |
| Training data             | FineWeb-Edu (500M tokens) | FineWeb-Edu (500M tokens) |
| Eval protocol             | 100 held-out seqs, SEQ_LEN=128 | same |

`Quality = 100 − (PPL_student − PPL_teacher) / PPL_teacher × 100`, floored at 0.

## Architecture

FRR replaces Qwen3-1.7B's 28 transformer layers with a single shared `FractalBlock` applied over 4 scales × 7 iterations. Each scale has learned `γ, β` modulation vectors (~8 KB) that induce layer-specific behavior from a single block.

```
teacher_embed(frozen) → proj_in(2048 → h) → FractalModel(h, 4, 7)
                    → proj_out(h → 2048) → teacher_norm(frozen) → teacher_lm_head(frozen)
```

Only `proj_in`, `FractalModel` (shared block + 4 scales of γ/β + 28 iter-scales), and `proj_out` are trainable. Teacher embedding, final norm, and lm_head are reused frozen.

## Training Objective (HQ5)

```
hard_weight  = (1 + H(teacher_logits)) ^ entropy_power          # h256: 1.5, h128: 1.0
fkl          = hard_weight · KL(T || S)        # entropy-weighted forward KL
rkl          = 0.3 · KL(S || T)                # reverse KL for coverage
latent       = latent_w(step) · MSE(S_hidden, T_hidden)   # 1.0 → 0.1 (h256) / 0.3 (h128) across 20K→50K
ce           = 0.5 · ce_ramp(step) · CE(teacher_argmax)   # 0.5 → 1.0 across 16K→48K
margin       = 0.3 · ce_ramp · hard_weight · margin_loss   # top-1 margin, same schedule
total_loss   = fkl + rkl + latent + ce + margin
```

Temperature schedule: `T = 2.0 → 1.0` across steps 0 → 64K.
Optimizer: AdamW, LR 2×10⁻⁴ → 1×10⁻⁵ cosine, 500-step warmup, weight decay 0.01.
Batch: 4 × accumulation 2, sequence length 128, bf16 mixed precision.

## Intended Use

- Fast single-GPU distillation of large language models (architectural compression).
- Research into weight-sharing transformer architectures.
- As a compact body to pair with the separately-trained ASVD lm_head (see [`finetune_asvd_head.py`](../finetune_asvd_head.py)).

## Out-of-Scope Use

- Direct production inference — these checkpoints were optimized against teacher agreement metrics, not downstream benchmarks. Evaluate on your target task before deployment.
- Safety / alignment — inherits any limitations of Qwen3-1.7B plus distillation-induced drift.

## Limitations

- 100-sample training eval has ±1–2 pp noise. Hires eval (1000 stratified samples) recommended for final numbers.
- Trained on FineWeb-Edu only; domain shift (code, dialogue, math) is untested.
- FRR body evaluated with the full 311 M-parameter frozen lm_head. Combined with ASVD lm_head, end-to-end compression rises but quality must be re-measured jointly.

## Usage

```python
import torch
from run_hq4_ceiling_break import TinyFRR_HQ4

ckpt  = torch.load('checkpoints_1.7b_tinyfrr_hq4_h256/best.pt', map_location='cuda')
# (loader wires teacher embed/norm/lm_head — see run_hq4_ceiling_break.py)
```

See the top-level README and `run_hq4_ceiling_break.py` for the full loader.

## Compression Stack

| Stack                           | Expected quality         | Effective compression |
|---------------------------------|--------------------------|-----------------------|
| FRR body (HQ5 h256) only        | 70.0% quality, 70.0% T10 | 311× (body)           |
| FRR body + ASVD r=1024 head     | projected ≥ 90% T10 head × 70% T10 body ≈ **64% joint T10** | **~2.3× end-to-end** |
| FRR + Q2 weight quantization    | −1 to −2 pp quality      | + 16×                 |
| FRR + Q2 + entropy coding       | same                     | + additional 6×       |

## Files

| File                                         | Contents                                        |
|----------------------------------------------|-------------------------------------------------|
| `checkpoints_1.7b_tinyfrr_hq5_h256/best.pt`  | Best HQ5 h256 checkpoint (step 78 000)          |
| `checkpoints_1.7b_tinyfrr_hq5_h128/best.pt`  | Best HQ5 h128 checkpoint (step 78 000)          |
| `hq5_h{128,256}.log`                         | Full 80K-step training logs (EVAL every 2K)     |
| [`HQ5_RESULTS.md`](../HQ5_RESULTS.md)        | Detailed results + trajectory + diagnosis       |
| [`HQ4_RESULTS.md`](../HQ4_RESULTS.md)        | Previous-generation results                     |
| [`HQ3_RESULTS.md`](../HQ3_RESULTS.md)        | Baseline results                                |

## Citation

```bibtex
@misc{ultracompress2026,
  title  = {Fractal Residual Recursion: Extreme Transformer Compression
            via Shared Recursive Blocks},
  author = {Mounir},
  year   = {2026},
  url    = {https://github.com/mounnar/ultracompress}
}
```

## License

Apache 2.0 (matches base model).
