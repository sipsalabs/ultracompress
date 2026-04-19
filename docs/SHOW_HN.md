# Show HN: UltraCompress — 311× compression of Qwen3-1.7B at 70% teacher quality

**GitHub:** https://github.com/mounnar/ultracompress

TL;DR: A single shared transformer block, applied recursively in a fractal schedule, replaces all 28 blocks of Qwen3-1.7B. End result: **1.51 M trainable parameters** (a **311× compression of the transformer body**) at **70.0% teacher-agreement quality** and **55.1% top-1 / 70.0% top-10** on a held-out FineWeb-Edu set.

## What's new vs standard distillation

Standard KD objectives concentrate gradient on tokens where the teacher is confident. This caps student quality at a fixed plateau (~54% T1 in our setup) because the student saturates on the easy tokens and never learns the hard ones.

We **invert** the weighting:

```
hard_weight = (1 + H(teacher_logits)) ^ entropy_power
```

High-entropy (uncertain) teacher tokens get *more* gradient, not less. This single change pushed our T1 from 54.2% (HQ3 baseline) → 55.1% (HQ5 best, 57.0% peak) and quality from 67.7% → 70.0%. ENT_POW sweeps are monotone up to 1.5; HQ6 tests 2.0.

## Architecture

- One shared Qwen-style transformer block `B`.
- Applied in a fractal schedule: 4 scales × 7 iterations = 28 effective applications.
- Linear projections `2048 → h → 2048` bracket the fractal body (`h ∈ {128, 256}`).
- Teacher's `final_norm` + `lm_head` are frozen and re-used.

## Results (Qwen3-1.7B, 80K-step distillation, 500 M-token FineWeb-Edu)

| Variant      | Trainable | Body compression | Best T1 | Best T10 | Quality |
|--------------|-----------|------------------|---------|----------|---------|
| HQ5 h256     | 1.51 M    | **311×**         | **55.1%** (peak 57.0%) | **70.0%** | **70.0%** |
| HQ5 h128     | 0.64 M    | **734×**         | 54.0%   | 68.4%    | 68.4%   |

## Reproducibility

All code, checkpoints, and logs are in the repo. Two RTX 5090s, ~6 hours for a full 80K-step run. Detached launcher survives terminal / editor close — [launch_hq5_detached.py](https://github.com/mounnar/ultracompress/blob/master/launch_hq5_detached.py).

## What's next

HQ6 (entropy_power 2.0, h384 capacity test) is training now. Combined with ASVD head factorization, Q2 weights, and entropy coding, we project the full stack fits under 12 GB for 100T-equivalent models.

Feedback welcome — the inverted-entropy idea feels general and probably applies to any distillation setup that's plateau-ed on easy tokens.
