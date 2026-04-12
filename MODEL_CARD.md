---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
- ultracompress
- frr
- model-compression
- fractal-residual-recursion
pipeline_tag: text-generation
---

# mounnar/qwen3-1.7b-frr-48x

**48x compressed** version of [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) using Fractal Residual Recursion (FRR).

## Model Details

| Property | Value |
|----------|-------|
| Base model | Qwen/Qwen3-1.7B |
| Compression | **48x** |
| Parameters | 29,380,636 (vs 751,632,384 original) |
| Top-10 agreement | 66% |
| Top-1 agreement | 37% |
| Training steps | 50,000 |
| Method | Fractal Residual Recursion (FRR) |

## How It Works

FRR replaces all 28 transformer layers with a single shared block applied recursively 28 times. Per-scale affine modulation vectors (~8K parameters) enable layer-specific behavior. The shared block stays in GPU L2 cache, making inference potentially **faster** than the original model.

## Usage

```python
from ultracompress.hf_model import FRRForCausalLM

model = FRRForCausalLM.from_frr("frr_model.pt", "Qwen/Qwen3-1.7B")
output = model.generate(input_ids, max_new_tokens=100)
```

## Compression Stack

This model uses FRR only (architectural compression). For maximum compression, apply our quantization pipeline on top:

| Stack | Quality | Compression |
|-------|---------|-------------|
| FRR only | 66% T10 | 48x |
| FRR + Q4 | ~64% T10 | ~384x |
| FRR + Q2 + entropy | ~63% T10 | ~768x |

## Citation

```bibtex
@misc{ultracompress2026,
  title={Fractal Residual Recursion: Extreme Transformer Compression via Shared Recursive Blocks},
  author={Mounir},
  year={2026},
  url={https://github.com/mounnar/ultracompress}
}
```

## License

Apache 2.0 (same as base model).
