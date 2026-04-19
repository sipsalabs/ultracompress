# Twitter / X Thread

**1/** We just crossed 70% teacher-quality on Qwen3-1.7B distillation with **1.51 M trainable parameters**.

That's a **311× compression of the transformer body** — one shared block, applied recursively, replaces all 28 layers.

Top-1 agreement 55.1% (peak 57.0%). Top-10 70.0%.

**2/** The trick: **invert** the entropy weighting.

Standard distillation upweights tokens where the teacher is confident. Plateaus at ~54% T1 because the student saturates on easy tokens.

We upweight *hard* tokens:
`hard_weight = (1 + H(teacher)) ^ entropy_power`

**3/** Monotone gains as we crank `entropy_power`:

- HQ3 (power 0, confidence-weighted): 54.1% T1 / 68.1% Q
- HQ4 (power 1.0, inverted): 54.3% T1 / 68.9% Q
- HQ5 (power 1.5): **55.1% T1 / 70.0% Q**
- HQ6 (power 2.0): training now

**4/** Architecture is a single Qwen-style block `B`, applied in a fractal schedule (4 scales × 7 iterations = 28 effective layers). Linear projections 2048→h→2048 bracket it. h=256. Teacher's final_norm + lm_head frozen.

**5/** 80 000 steps, 500 M tokens of FineWeb-Edu, bf16 on dual RTX 5090. ~6 h per run. Detached launcher survives VS Code closing.

All code, checkpoints, logs on GitHub: https://github.com/mounnar/ultracompress

**6/** End-to-end roadmap:
- FRR body: 311× (shipped)
- + ASVD r=1024 head: ~2.3× end-to-end (next)
- + Q2 weights: +16×
- + entropy coding: +~6×

Projection for 100T-equivalent model: ~12 GB VRAM.

**7/** The inverted-entropy idea is probably general — any distillation setup that's plateau-ed on easy tokens likely has room to improve with this objective. Interested in hearing if it transfers.
