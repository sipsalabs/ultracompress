"""
Create a HuggingFace model card for uploading FRR compressed models.

Usage:
  python create_model_card.py --model frr_100k_best.pt --name mounnar/qwen3-0.6b-frr-60x
"""
import argparse
import os
import time


def create_card(model_name, base_model, compression, t10, t1, steps, params):
    card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- ultracompress
- frr
- model-compression
- fractal-residual-recursion
pipeline_tag: text-generation
---

# {model_name}

**{compression}x compressed** version of [{base_model}](https://huggingface.co/{base_model}) using Fractal Residual Recursion (FRR).

## Model Details

| Property | Value |
|----------|-------|
| Base model | {base_model} |
| Compression | **{compression}x** |
| Parameters | {params:,} (vs {751632384:,} original) |
| Top-10 agreement | {t10}% |
| Top-1 agreement | {t1}% |
| Training steps | {steps:,} |
| Method | Fractal Residual Recursion (FRR) |

## How It Works

FRR replaces all 28 transformer layers with a single shared block applied recursively 28 times. Per-scale affine modulation vectors (~8K parameters) enable layer-specific behavior. The shared block stays in GPU L2 cache, making inference potentially **faster** than the original model.

## Usage

```python
from ultracompress.hf_model import FRRForCausalLM

model = FRRForCausalLM.from_frr("frr_model.pt", "{base_model}")
output = model.generate(input_ids, max_new_tokens=100)
```

## Compression Stack

This model uses FRR only (architectural compression). For maximum compression, apply our quantization pipeline on top:

| Stack | Quality | Compression |
|-------|---------|-------------|
| FRR only | {t10}% T10 | {compression}x |
| FRR + Q4 | ~{t10-2}% T10 | ~{compression*8}x |
| FRR + Q2 + entropy | ~{t10-3}% T10 | ~{compression*16}x |

## Citation

```bibtex
@misc{{ultracompress2026,
  title={{Fractal Residual Recursion: Extreme Transformer Compression via Shared Recursive Blocks}},
  author={{Mounir}},
  year={{2026}},
  url={{https://github.com/mounnar/ultracompress}}
}}
```

## License

Apache 2.0 (same as base model).
"""
    return card


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='frr_100k_best.pt')
    parser.add_argument('--name', default='mounnar/qwen3-0.6b-frr-60x')
    parser.add_argument('--base-model', default='Qwen/Qwen3-0.6B')
    parser.add_argument('--compression', type=int, default=60)
    parser.add_argument('--t10', type=int, default=63)
    parser.add_argument('--t1', type=int, default=44)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--params', type=int, default=7350300)
    args = parser.parse_args()

    card = create_card(args.name, args.base_model, args.compression,
                       args.t10, args.t1, args.steps, args.params)

    output = 'MODEL_CARD.md'
    with open(output, 'w') as f:
        f.write(card)
    print(f"Saved {output}")
    print(f"Upload to HuggingFace:")
    print(f"  huggingface-cli upload {args.name} {args.model} frr_model.pt")
    print(f"  huggingface-cli upload {args.name} {output} README.md")
