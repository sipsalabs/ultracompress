# UltraCompress - Session Status & Recovery Notes

**Last updated:** 2026-04-09 (session 2, post-crash)
**Version:** v8 (Weight Genome)
**Location:** C:\Users\sip\Desktop\Projects\ultracompress

---

## Project Goal
Extreme LLM weight compression. Flagship the industry.
- 235B params -> 20GB (0.68 BPW)
- 10T params -> 20GB (0.016 BPW)
- 1000T params -> 20GB (0.00016 BPW)

## CRITICAL REALITY CHECK (2026-04-10)
The 0.9999 output cosine metric was WRONG — it measured against random calibration
activations that don't match real inference. Actual text generation shows:
- 3.0 BPW, 0.955 weight cosine: 40% top-10 agreement, top-1 MISS
- 1.5 BPW, 0.953 weight cosine: 0% agreement, broken
- Sub-1 BPW: completely unusable for text generation

The compound pipeline framework is VALID but needs fundamentally different optimization:
1. Real text calibration TESTED — helps marginally (+0.006 layer cosine, +0.02 logit cosine)
2. Hessian-weighted PQ TESTED — helps marginally (10% top-10 vs 0%)
3. Output-aware gradient refinement HURTS — overfits to calibration, drops weight cosine
4. Need 0.99+ weight cosine per layer for actual text quality
5. GPTQ-style ROW-BY-ROW optimization is the industry path — we need the PQ equivalent

**Honest BPW quality ladder (Qwen3-0.6B, 4 layers, plain PQ):**
| BPW | Wt Cos | L0 cos | Logit cos | Top-10 | Usable? |
|-----|--------|--------|-----------|--------|---------|
| 3.0 | 0.955 | 0.90 | 0.89 | 40% | Barely |
| 2.6 | 0.953 | 0.89 | 0.92 | 0-10% | No |
| 1.6 | 0.834 | 0.61 | 0.70 | 0% | No |
| 1.0 | 0.753 | 0.47 | 0.05 | 0% | No |
| 0.8 | 0.633 | 0.34 | -0.06 | 0% | No |

## What Exists (Complete)

### Core Modules (ultracompress/)
| File | Purpose | Status |
|------|---------|--------|
| `product_quantize.py` | PQ with per-subvector k-means. Flagship method. | DONE |
| `ultra_pq.py` | Residual PQ, gradient codebook refinement, entropy estimation, global codebooks | DONE |
| `calibrated_pq.py` | Calibration-aware PQ (importance-weighted k-means) | DONE but NOT integrated |
| `pipeline.py` | 4-path compression pipeline (PQ > SVD > VQ > scalar) | DONE |
| `inference.py` | MiniTransformer for end-to-end output quality testing | DONE |
| `gguf_loader.py` | Load GGUF models from Ollama | DONE |
| `safetensors_loader.py` | Load HuggingFace safetensors (FP16) | DONE |
| `metrics.py` | Cosine sim, MSE, relative error | DONE |
| `quantize.py` | INT2-8 scalar quantization, VQ | DONE |
| `factorize.py` | SVD factorization | DONE |
| `binarize.py` | Binary weight compression | DONE |
| `codebook.py` | Codebook compression for binarized factors | DONE |
| `spectral.py` | DCT spectral compression | DONE |
| `crosslayer.py` | Cross-layer weight sharing | DONE |
| `sparsify.py` | Weight sparsification | DONE |
| `mixed_precision.py` | Mixed precision allocation | DONE |
| `profiler.py` | Layer profiling | DONE |

### Runner Scripts
| File | Purpose |
|------|---------|
| `run_ultra.py` | Full model compression benchmark (presets: extreme/balanced/quality/10t_quality) |
| `run_inference_compare.py` | End-to-end inference comparison (the real quality test) |
| `run_compress.py` | Pipeline-based compression |
| `run_quality_test.py` | Quality testing |
| `run_all_tests.py` | Run all tests |
| `test_fp16_quality.py` | FP16 source weight quality test |

## Key Results (CORRECTED 2026-04-09)

### IMPORTANT: Previous results were WRONG
The 0.91+ weight cosine claims from 2026-04-08 were incorrect. Re-testing confirms:
- PQ at K=4, G=256 (SVS=32) gives **0.22 cosine** (essentially random)
- The sub-vector dimension was too high for the codebook size
- SVS (sub-vector size) = G/M is the dominant quality factor

### Corrected Weight-Level Quality (GGUF source)
| Config | SVS | BPW | Weight Cosine | Notes |
|--------|-----|-----|---------------|-------|
| M=16 K=16 G=64 | 4 | 1.26 | 0.81 | Best SVS=4 |
| M=8 K=256 G=64 | 8 | 1.35 | 0.83 | Best SVS=8 |
| M=32 K=16 G=128 | 4 | 1.14 | 0.81 | Good |
| M=8 K=4 G=256 | 32 | 0.13 | 0.22 | BROKEN - SVS too high |
| M=4 K=2 G=2048 | 512 | 0.016 | ~0.10 | BROKEN - SVS way too high |

### Output-Aware Refinement Results (FP16 source, Qwen3-0.6B)
**THIS IS THE BREAKTHROUGH:**

At 0.5 BPW (M=8 K=16 G=64, SVS=8) over 4 layers:
| Layer | k-means | REFINED | Delta |
|-------|---------|---------|-------|
| 0 | 0.63 | 0.66 | +0.04 |
| 1 | 0.59 | 0.72 | +0.11 |
| 2 | 0.55 | 0.83 | +0.23 |
| 3 | 0.52 | **0.93** | +0.31 |

At 0.32 BPW (M=8 K=8 G=128, SVS=16) over 8 layers:
| Layer | k-means | REFINED | Delta |
|-------|---------|---------|-------|
| 0 | 0.39 | 0.47 | +0.08 |
| 1 | 0.27 | 0.79 | +0.40 |
| 2 | 0.22 | 0.98 | +0.59 |
| 3 | 0.22 | **1.00** | +0.58 |
| 4-7 | 0.20 | **1.00** | +0.63 |

**Error REVERSAL confirmed!** Refined weights learn to compensate for upstream errors.
Later layers get BETTER, not worse.

**BUT**: Only 2/7 weights per layer get refined (others have dimension mismatch).
Average output cosine is 0.54 because 5 weights stay at ~0.39.

### The New Core Problem
Not error compounding — it's **coverage**. Output-aware refinement solves compounding
completely (1.0000 cosine!) but only when applied. Need to refine ALL weights,
not just those where activation dims happen to match.

### COMPOUND PIPELINE RESULTS (v8, 2026-04-09 session 2)
`run_compound.py` — Full transformer activation tracking, ALL 7 weights refined.

| BPW | Compression | L0 avg | L1 avg | L3+ avg | 1T size | Notes |
|-----|-------------|--------|--------|---------|---------|-------|
| 0.76 | 21x | 0.9265 | 0.9970 | 0.9999 | ~95 GB | 0.5 BPW target |
| 0.32 | 50x | 0.8865 | 0.9978 | 1.0000 | 40 GB | 0.19 BPW target |
| 0.08 | 208x | 0.8681 | 0.9986 | 1.0000 | 9.6 GB | 0.05 BPW target |
| 0.06 | 265x | 0.8644 | 0.9977 | 1.0000 | **7.6 GB** | 0.016 BPW target |

**Key findings:**
- Weight cosine is IRRELEVANT (0.05 at 0.06 BPW) — output cosine is what matters
- Error REVERSAL confirmed: layers 3+ hit 1.0000 output cosine at ALL BPW levels
- Layer 0 is the bottleneck: v_proj gets 0.38-0.52 (first layer, no upstream context)
- All other weights: 0.99+ from layer 1 onwards
- **1T model fits in 7.6 GB at 0.06 BPW**

### FULL DEPTH VALIDATION (28 layers, 0.32 BPW):
Pattern holds PERFECTLY across entire model. 0.9958 avg, 0.9999 from layer 3 to 27.
No degradation at depth. Error reversal is STABLE.

### Remaining challenges:
1. Layer 0 v_proj is weak (0.52) — needs special handling (higher BPW or mixed precision)
2. 10T still 400 GB at 0.32 BPW — need cross-layer SVD or Weight Genome for 10T+ scale
3. Need to validate with actual text generation (not just cosine similarity)
4. Need to test on larger models (Qwen3-8B needs more RAM)

## What Needs to Happen Next

### Priority 1: BUILD THE COMPOUND PIPELINE (Phase 2)
Create `compound_pipeline.py` that:
1. Tracks activations through FULL transformer layer (attention + FFN + residual)
2. Applies output-aware PQ refinement to ALL 7 weight types per layer
3. Uses cross-layer SVD to reduce residual magnitudes before PQ
4. Allocates BPW budgets via mixed precision scoring
5. Reports per-layer output cosine for ALL weights

The output-aware technique WORKS (1.0000 cosine!) — we just need full coverage.

### Priority 2: FP16 Source Weights (CRITICAL)
FP16 weights give much cleaner PQ results. GGUF re-quantization is fighting uphill.
- Qwen3-0.6B cached at HuggingFace (1.5GB FP16)
- Use `safetensors_loader.py` for all testing
- Qwen3-8B cached but needs 32GB RAM for float32 loading

### Priority 3: The Weight Genome (Phase 3)
Hierarchical generative compression:
- SIREN generator + per-layer modulation + PQ residuals
- Differentiable PQ for joint training
- Target: 0.001 BPW with scaling to 1000T

### Weight Genome Status (Phase 3)
- Framework built: weight_genome.py, differentiable_pq.py, run_genome.py
- Direct SIREN prediction DOES NOT WORK (cosine=0.01, MSE stuck at 1.0)
- Weight matrices lack spatial smoothness for INR-style prediction
- Need different approach: predict codebook structure, not raw values
- Research direction, not production-ready

### SCALING ANALYSIS — 10T FITS IN 20GB

The "0.06 BPW" on small models is inflated by codebook overhead.
At scale (large tensors), codebook overhead -> 0:
- 1024x1024 tensor: 0.094 BPW (66% codebook overhead)
- 8192x8192 tensor: 0.032 BPW (3% overhead)
- 32768x8192 tensor: 0.031 BPW (<1% overhead)

**10T+ models have massive tensors. Theoretical BPW at scale:**
| Config | BPW | 10T | 100T | 1000T |
|--------|-----|-----|------|-------|
| M=8 K=4 G=2048 | 0.016 | **19.5 GB** | 195 GB | 1953 GB |
| M=8 K=2 G=2048 | 0.012 | **14.6 GB** | 147 GB | 1465 GB |
| M=4 K=2 G=4096 | 0.005 | **6.1 GB** | 61 GB | 610 GB |

**All these configs give 1.0000 output cosine from layer 3+ on Qwen3-0.6B.**

Cross-layer SVD was tested but weak on 0.6B (only 40% reduction at k=10).
Larger models likely have much more cross-layer redundancy — this remains
the path to 1000T in 20GB if SVD gives 10x+ on large models.

### Cross-layer SVD analysis (Qwen3-0.6B)
- k=1: 5% variance, 0.22 cosine (barely helps)
- k=10: 41-49% variance, 0.51 cosine, 0.62 residual ratio
- Small model = low cross-layer redundancy
- Need to test on 70B+ for meaningful cross-layer gains

### Deprioritized
- Calibrated PQ: tested, only +0.02 improvement over standard PQ
- Global codebooks: minor savings vs output-aware breakthrough
- Genome INR approach: doesn't converge, needs fundamental rethink

## Architecture Notes
- Default test model: `qwen3:4b` via Ollama
- Supports any GGUF model through `gguf_loader.py`
- MiniTransformer in inference.py does real forward passes (attention + FFN + RoPE)
- All PQ configs use format: (M=subvectors, K=codebook_size, G=group_size)
- BPW formula: M * log2(K) / G + overhead

## How to Run
```bash
# Full benchmark
python run_ultra.py --mode balanced

# Inference quality test (THE important one)
python run_inference_compare.py --mode balanced --max-layers 4

# Different configs
python run_ultra.py --mode extreme      # 0.016 BPW
python run_ultra.py --mode quality      # RPQ multi-level
python run_ultra.py --mode 10t_quality  # 10T target
```

## Dependencies
- torch, numpy, tqdm, gguf (pip install gguf)
- Ollama running locally with model pulled (e.g., `ollama pull qwen3:4b`)
