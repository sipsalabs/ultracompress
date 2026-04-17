# Day 2 Summary — April 12, 2026

**103 commits. 82 modules. 42K lines. Records broken.**

## Headlines

1. **1.7B at 67% T10 at 48x compression** — ALL-TIME RECORD
2. **3.1-3.4x faster inference** — L2 cache proven on RTX 5090
3. **959x E2E compression** — only 1.5% quality drop
4. **HellaSwag 26.5%** at 60x — only 2.5% drop from teacher (FIRST REAL BENCHMARK)
5. **Scaling confirmed** — bigger models compress better
6. **Speculative decoding** — reframes product as inference accelerator (2x speedup)
7. **8B cached and ready** — next scaling milestone

## All-Time Records

| Record | Value | Model | Steps |
|--------|-------|-------|-------|
| Best T10 | **67%** | 1.7B | 100K |
| Best 0.6B T10 | 65% | 0.6B | 100K |
| Best T1 | 48% | 0.6B | 100K |
| Best efficiency | 53% T10 at 239x | PHM | 10K |
| Best E2E | 53% T10 at 959x | FRR+Q2 | 10K |
| Fastest | 3.4x speedup | FRR | — |
| Best HellaSwag | 26.5% at 60x | FRR | 100K |

## What Works (proven)
- FRR single shared block (60x, 65%)
- FRR + bigger models (48x, 67%)
- FRR + Q2 pipeline (959x, 53%)
- PHM hypercomplex (239x, 53%)
- More training steps (keeps improving to 100K+)
- L2 cache speed advantage (3x faster)
- Static gamma/beta modulation

## What Doesn't Work (honest)
- Error-only (layers independent)
- Multi-block (same quality, more params)
- Controller hypernetwork (NaN)
- 4D cross-depth attention (NaN)
- Quant-aware training (backfires)
- Real text eval on random-token metrics (mismatch)
- Sparse30, NeuroFractal, Seed (dead)

## Key Insight
**Quality = f(model_size, training_steps)**
Nothing else matters as much. Architecture tricks are marginal.
Dynamic computation in the shared block is unstable.
Static modulation + more training = the formula.

## What's Running
- MoL test on GPU 0 (finishing ~4:50 PM)
- 8B auto-launches on both GPUs after MoL

## What's Ready for Tonight
`python run_tonight.py` on GPU 0:
1. Dual-objective test (Sip's T10/T1 split idea)
2. Top-1 focused loss test
3. Optimized training (2x batch, 1.5x LR)
4. Born-again distillation (3 generations)

## Monday Plan
1. File provisional patent ($80, 25 claims)
2. Make GitHub repo public
3. Submit arxiv paper
4. Post Show HN + Twitter thread
5. Upload compressed models to HuggingFace
