# UltraCompress - Session Status & Recovery Notes

**Last updated:** 2026-04-09
**Version:** v5
**Location:** C:\Users\sip\Desktop\Projects\ultracompress

---

## Project Goal
Extreme LLM weight compression. Flagship the industry.
- 235B params -> 20GB (0.68 BPW)
- 10T params -> 20GB (0.016 BPW)

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

## Key Results (from last session 2026-04-08)

### Weight-Level Quality
| Config | BPW | Weight Cosine | Method |
|--------|-----|---------------|--------|
| M=4 K=2 G=2048 | 0.016 | 0.895 | Binary PQ (extreme) |
| M=8 K=4 G=256 | 0.13 | 0.915 | Standard PQ (balanced) |
| M=16 K=16 G=256 | 0.36 | ~0.94 | High-K PQ |
| INT4 | 4.25 | 0.999+ | Baseline |

### Inference-Level Quality (the real test)
| Config | Weight Cos | After 36 Layers | Assessment |
|--------|-----------|-----------------|------------|
| 0.016 BPW | 0.89 | ~0.82 | Degrades significantly |
| 0.13 BPW | 0.90 | ~0.53 | SEVERE compounding |
| INT4 (4.25 BPW) | 0.992 | 0.992 | Near-perfect |

### The Core Problem
Errors compound through layers. Even 10% weight error per layer becomes 50%+ output error after 36 layers. Need **0.99+ weight cosine per layer** for acceptable inference quality.

## What Needs to Happen Next

### Priority 1: Integrate Calibrated PQ into Pipeline
`calibrated_pq.py` exists but is NOT wired into `run_ultra.py` or `run_inference_compare.py`.
- Calibrated PQ weights k-means by activation importance (like GPTQ/AWQ)
- Theory: 80% of activations near zero with ReLU/GELU, so errors in unused dimensions don't matter
- Could turn 0.90 weight cosine into 0.99+ OUTPUT cosine
- **This is the most promising next step**

### Priority 2: Gradient Codebook Refinement at Scale  
`refine_codebooks_gradient()` in ultra_pq.py works but needs integration into main runners.
- Refines codebook entries via Adam after k-means
- Pushes quality beyond k-means ceiling

### Priority 3: FP16 Source Weights
Current pipeline loads from GGUF (already quantized weights). GGUF weights have a noise floor.
- `safetensors_loader.py` exists for loading FP16 from HuggingFace
- FP16 source = higher ceiling for PQ quality
- `test_fp16_quality.py` exists but needs full integration

### Priority 4: Global Codebooks + Entropy Coding
- `GlobalCodebookManager` in ultra_pq.py is done
- `estimate_entropy_bpw()` is done
- Need to integrate into full-model compression to show true BPW savings

### Stretch Goals
- Layer-adaptive PQ configs (more bits for early/late layers, fewer for middle)
- Beam search over PQ configs per tensor
- Online codebook adaptation during inference

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
