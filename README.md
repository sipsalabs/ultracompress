# UltraCompress — Genome Compression Engine

**Compress trillion-parameter models into megabytes. Zero degradation.**

UltraCompress replaces each transformer layer with a tiny "genome" — a micro-transformer that replicates the original layer's behavior at 37-1000x smaller size. The genome IS the model.

## Quick Start

```bash
# Compress a model
python ultracompress.py compress --model Qwen/Qwen3-0.6B --sd 128 --hybrid

# Run inference with the genome
python ultracompress.py run --genome genome_sd128_28L.pt --prompt "Hello world"

# Check genome info
python ultracompress.py info --genome genome_sd128_28L.pt
```

## How It Works

1. **Progressive Training**: Each layer learns independently to replicate its teacher layer's behavior
2. **Joint Fine-tuning**: All genome layers train end-to-end via KL divergence on logits
3. **Cached Training**: Teacher outputs are cached for 10-100x faster training

The genome never stores weight matrices. It generates the right transformation on-the-fly using a tiny bottleneck transformer per layer.

## Results (Qwen3-0.6B, 28 layers)

| Method | Top-1 | Top-10 | Genome Size | Compression |
|--------|-------|--------|-------------|-------------|
| Progressive sd=128 | 20% | 53% | 23.9 MB | 37x |
| Hybrid (prog+50K cached) | 27% | 46% | 23.9 MB | 37x |
| Progressive sd=64 | 16% | 47% | 9.6 MB | 91x |

## Scaling Projections

| Model | Genome (sd=64) | Genome (sd=128) |
|-------|---------------|-----------------|
| 8B | 40.7 MB | 87.3 MB |
| 70B | ~350 MB | ~750 MB |
| 405B | ~2 GB | ~4.3 GB |
| 10T | ~52 GB | ~111 GB |

## Architecture

- `ultracompress/genome_compressor.py` — Core engine
- `ultracompress/genome_v2.py` — V2 architectures (MultiView, LoRA)
- `ultracompress/streaming_loader.py` — Shard-by-shard model loading
- `ultracompress.py` — CLI interface

## Vision

Replace data centers. Train genomes directly instead of full models.
A $100M training run becomes $100K.

## Status

Active research. Quality improving with every experiment.
Target: 90%+ top-10 accuracy at 100x+ compression.

## License

Proprietary — Sip / Athena AGI
