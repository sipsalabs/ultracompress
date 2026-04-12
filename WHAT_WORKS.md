# What Actually Works (and What Doesn't)

*Honest assessment after 2 days, 124 commits, 81 modules, 42K lines of code.*

## WORKS (proven with real numbers)

### 1. FRR Single Shared Block
One transformer block, applied 28 times with per-scale gamma/beta modulation.
- **0.6B: 65% T10 at 60x** (100K steps)
- **1.7B: 66% T10 at 48x** (50K steps, record)
- Why: CKA functional similarity >0.9 between layers. Shared-weight transformers are Turing complete.
- Key: more training steps = more quality. 10K->100K = 54%->65%.

### 2. FRR + Q2 Pipeline (End-to-End)
Hadamard rotation -> SVD -> Q2 quantization -> residual correction -> entropy coding.
- **959x compression at -1.5% quality drop** (proven E2E)
- Entropy coding gives 6x free on Q2 weights.

### 3. PHM (Hypercomplex Multiplication)
Replace linear layers with 4-component hypercomplex multiplication.
- **53% T10 at 239x compression** — 4x fewer params, -2% vs baseline.

### 4. Inference Speed (L2 Cache)
FRR block (14.7 MB) fits in GPU L2 cache. Teacher (3 GB) doesn't.
- **3.1-3.4x faster inference** across all sequence lengths. Proven on RTX 5090.

### 5. Scaling
Bigger models compress better with FRR.
- 0.6B -> 1.7B: +3-5% T10 at same training steps.
- Projected: 8B -> 70%+ T10.

## DOESN'T WORK (honest failures)

### 1. Error-Only Compression
Predict layer N+1 from layer N, store only errors.
- **-40% accuracy** on real weights. Layers are statistically independent (cosine ~0.000).

### 2. Multi-Block FRR
2-3 specialized shared blocks instead of 1.
- **Same quality (57% T10), 11x more params.** Single block is already optimal.

### 3. Controller Hypernetwork
Input-dependent modulation (Ouroboros V2 style).
- **NaN at step 3K.** Unstable. Matches Ouroboros V2 paper finding.

### 4. Quant-Aware Training
Train FRR with Q2 quantization penalty.
- **6% T10 after Q2** (vs 53% with normal training + pipeline Q2). Backfires.

### 5. Real Text Training (on random-token eval)
FineWeb-Edu training.
- **41% T10** (vs 56% with random tokens). Eval metric mismatch, not a real failure.

### 6. Intermediate Hidden State Matching
Match per-layer hidden states during distillation.
- **No T10 improvement** with random tokens. +4% T1 only.

### 7. 4D Cross-Depth Attention (Sip's idea)
- Collapsed at step 4K -> NaN. Same instability pattern as controller.
- **Pattern: any dynamic computation in the shared block is unstable.**
- Needs: warmup from static, gradient scaling, or late-phase-only activation.

### 8. Sparse30, NeuroFractal, Seed Architecture
- All dead (0-4% T10). Too aggressive or unstable.

## UNTESTED BUT PROMISING

1. **MoL (Mixture of LoRAs)** — token-conditional routing, different from controller
2. **Born-again distillation** — +2-4% per generation, literature-backed
3. **8B scaling** — cached, script ready, needs both GPUs
4. **Optimized training** — 2x batch, 1.5x LR, should converge faster
5. **Speculative decoding** — FRR as draft model, 2x speedup, zero quality loss
6. **Prelude/Coda** — keep first/last layers untied, share middle only

## THE FORMULA

Quality = f(model_size, training_steps)

Nothing else matters as much. Architecture enhancements give marginal gains (+1-7%).
Just train longer on a bigger teacher.
