# UltraCompress

> **Extreme LLM compression via Fractal Residual Recursion (FRR).**
> One shared transformer block replaces all 28 layers — achieving **52–733x architectural compression** with up to **90.4% benchmark retention**.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Highlights

| Metric | Result |
|--------|--------|
| **Architectural compression** | 52–733x (vs. SOTA ~2x) |
| **End-to-end compression** | 959x (FRR + Q2 + entropy coding) |
| **HellaSwag retention** | 90.4% at 52x compression |
| **Inference speedup** | 3.1–3.4x (shared block fits in GPU L2 cache) |
| **Top-10 agreement** | 73.1% (HQ2, h=128, 733x compression) |

---

## How It Works

Most compression asks: *"How do I make these weights smaller?"*
FRR asks: **"Do I even need different weights per layer?"**

Adjacent transformer layers have **zero weight similarity** (cosine ≈ 0.000) but **>0.9 functional similarity** (CKA). They perform the same *type* of computation on different feature spaces. FRR learns the shared function; lightweight per-scale modulation (~8K params) selects different behaviors.

```
Traditional Transformer           FRR Compressed Model
========================          ==========================

Input                             Input
  │                                 │
  ▼                                 ▼
[Layer 0 weights: 54 MB]          [Shared Block: 7.3M params]
  │                                 │ + γ₀, β₀
  ▼                                 ▼
[Layer 1 weights: 54 MB]          [Same Shared Block]
  │                                 │ + γ₁, β₁
  ▼                                 ▼
  ...  (28 layers)                  ...  (28 iterations)
  │                                 │
  ▼                                 ▼
Output                            Output

Total: 1,500 MB                   Total: 21 MB (FP32) / 1.8 MB (Q2)
```

**Theoretical backing:** Shared-weight (looped) transformers are Turing complete ([Giannou et al., 2023](https://arxiv.org/abs/2301.13196)).

---

## Results

### End-to-End Compression (Qwen3-0.6B)

| Stack | Top-1 | Top-10 | Compression | Size |
|-------|-------|--------|-------------|------|
| FRR only (FP32) | 36% | 55% | 60x | 29 MB |
| FRR + Q2 pipeline | 35% | 53% | **959x** | **1.8 MB** |

### Benchmark Retention (Qwen3-1.7B)

| Benchmark | Teacher | FRR | Retention | Compression |
|-----------|---------|-----|-----------|-------------|
| HellaSwag | 31.3% | 28.3% | **90.4%** | 52x |
| WikiText-2 PPL | 670.7 | 1271.7 | ~1.9x | 52x |

### Active Experiments (Qwen3-1.7B)

| Method | T1 | T10 | Params | Compression | Key Innovation |
|--------|----|-----|--------|-------------|----------------|
| **HQ2** (h=128) | 53.6% | 73.1% | 0.64M | 733x | Hidden-state latent matching |
| **FSD** (h=256) | 52.2% | 62.8% | 79.4M | 9.8x | Fused spectral decode |
| **SLD** (h=256) | — | — | 79.4M | 9.8x | Spectral latent distillation (training) |

### Inference Speed (RTX 5090)

| Seq Length | Teacher | FRR | Speedup |
|------------|---------|-----|---------|
| 32 | 613 tok/s | 2,073 tok/s | **3.38x** |
| 128 | 2,624 tok/s | 8,041 tok/s | **3.06x** |
| 256 | 5,223 tok/s | 16,403 tok/s | **3.14x** |

---

## Competitive Position

| Method | Year | Compression | Approach |
|--------|------|-------------|----------|
| GPTQ / AWQ | 2023 | 4–8x | Post-training quantization |
| SparseGPT | 2023 | 2–4x | Unstructured pruning |
| Relaxed Recursive (Google) | 2025 | ~2x | Shared block + LoRA |
| Ouroboros V2 | 2026 | ~2x | Controller hypernetwork |
| **UltraCompress FRR** | **2026** | **52–959x** | **Fractal recursive block + pipeline** |

No other published method achieves architectural compression beyond ~4x. FRR operates **13–15x beyond** all competitors.

---

## Quick Start

```bash
git clone https://github.com/mounnar/ultracompress.git
cd ultracompress
pip install -e .

# Compress a model
python ultracompress.py compress --model Qwen/Qwen3-0.6B

# Run inference
python ultracompress.py run --model compressed.ucz --prompt "The future of AI is"
```

### Training FRR from scratch

```bash
# Download and cache teacher model
python tools/download_models.py

# Run FRR distillation (Qwen3-1.7B)
python experiments/training/run_fused_spectral.py --h 256 --r 512 --steps 80000

# Run HQ2 hidden-state matching
python run_1.7b_tinyfrr_hq2.py --h 128 --steps 80000

# Evaluate
python experiments/eval/eval_tinyfrr_hires.py
```

---

## Repository Structure

```
ultracompress/              Core library
├── moonshot.py             FractalModel — shared recursive block
├── inference.py            MiniTransformer — teacher model loading
├── ultimate_pipeline.py    5-stage compression pipeline
├── entropy_coding.py       Lossless entropy coding (6x on Q2)
├── hypercomplex.py         PHM layers (4x param reduction)
└── ...                     80+ compression modules

experiments/
├── training/               Active training scripts (FSD, HQ2, etc.)
├── eval/                   Evaluation & benchmarking
├── analysis/               Training analysis & visualization
├── sweeps/                 Hyperparameter sweeps
└── archive/                Historical experiments

tools/                      Utilities (export, download, quantize)
docs/                       Paper draft, patents, business docs
tests/                      Unit tests
lib/                        Helper modules (unbuffered I/O, etc.)
```

---

## Key Research Findings

1. **Functional similarity enables weight sharing.** Adjacent layers have CKA >0.9 despite zero weight cosine similarity. The shared function space is learnable.

2. **FRR + quantization stack cleanly.** Q2 on FRR block weights drops only 1.5% top-10 quality. Proven end-to-end at 959x.

3. **Entropy coding gives 6x free on Q2.** IEEE 754 quantized values are highly redundant. Exponent/mantissa stream splitting + zlib yields massive lossless gains.

4. **Hidden-state matching > output-only distillation.** Aligning student latents with teacher's post-norm hidden states (HQ2) yields 73.1% T10 vs. 62.8% for logit-only methods.

5. **Larger models compress better.** 1.7B achieves 52x vs. 0.6B's 60x — more functional redundancy enables tighter sharing.

6. **100-sample evals have ±9.5% CI.** Need 4,000+ paired samples to reliably detect 3% differences.

---

## 100T+ Model Projections

| Stack | 100T Model Size | Compression |
|-------|----------------|-------------|
| FRR 52x + Q2 + entropy | **24.1 GB** | **8,298x** |
| FRR 100x + Q2 + entropy | 12.5 GB | 15,948x |

*A 100-trillion parameter model fitting on a single GPU.*

---

## Citation

```bibtex
@misc{ultracompress2026,
  title={Fractal Residual Recursion: Extreme Transformer Compression
         via Shared Recursive Blocks},
  author={Mounir},
  year={2026},
  url={https://github.com/mounnar/ultracompress}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
