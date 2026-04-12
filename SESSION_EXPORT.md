# UltraCompress — Full Session Export (2026-04-11)

## What This Project Is
**UltraCompress** — Compressing 10T-1000T parameter LLMs to fit in <20GB of memory with near-zero quality degradation.
**Location:** `C:\Users\sip\Desktop\Projects\ultracompress`
**For:** Athena AGI (Sip's project) + product to sell to companies
**Philosophy:** Think Tesla/Einstein — paradigm shift, not incremental. Invent new methods.

## Hardware
- **GPU 0:** RTX 5090 32GB (daily use — Discord, Chrome)
- **GPU 1:** RTX 5090 32GB (Athena/Ollama — PyTorch sees this as `cuda:0`, `device_count()=1`)
- **CPU:** Ryzen 9 9950X3D (16c/32t), **RAM:** 64GB DDR5 @4800MHz, **Storage:** 3TB NVMe
- Hardware upgrades NOT affordable (5090 at $3900-5000, DDR5 128GB at $1400+)

## The Core Approach: Behavioral Genome Compression
Replace each transformer layer with a tiny "micro-transformer" (MicroTransformerLayer) trained via KL divergence on teacher model logits. The genome IS the compressed model — no weight storage needed. This is genuinely novel — nobody else does this.

## COMPLETE RESULTS — ALL EXPERIMENTS

### Full Scoreboard (sorted by Top-10)
| # | Approach | Top-1 | Top-10 | Size | Notes |
|---|----------|-------|--------|------|-------|
| 1 | **V1 sd=128 progressive** | **20%** | **53%** | 23.9 MB | BEST top-10, 37x compression |
| 2 | V2a MultiView progressive | 26% | 48% | 16.5 MB | Most param-efficient |
| 3 | V1 sd=64 28L progressive | 16% | 47% | 9.6 MB | 91x compression |
| 4 | V1 sd=128 hybrid (cached 50K) | 27% | 46% | 23.9 MB | BEST top-1 but OVERFIT |
| 5 | Hybrid INT4-attn r128 | 15% | 45% | 117.5 MB | Disappointing |
| 6 | Hybrid INT4-attn r32 | 17% | 44% | 95.5 MB | INT4 attn + LoRA FFN |
| 7 | Hybrid INT4-attn r64 | 17% | 44% | 102.8 MB | |
| 8 | V2a MultiView cached | 18% | 44% | 16.5 MB | |
| 9 | V1 sd=128 aggressive | 23% | 51% | 23.9 MB | |
| 10 | V1 sd=256 cached | 16% | 47% | 66.1 MB | More params = same result! |
| 11 | V1 sd=128 cached | 16% | 47% | 23.9 MB | |
| 12 | V2b LoRA cached | 14% | 37% | 14.7 MB | |
| 13 | Neural DNA 512 genes | 4% | 30% | 44.4 MB | Novel but weak |
| 14 | Neural DNA 1024 genes | 3% | 29% | 89.2 MB | Bigger = same |
| 15 | Neural DNA 256 genes | 4% | 28% | 22.1 MB | |
| 16 | V1 sd=64 4L | 14% | 14% | 0.5 MB | 4-layer has 0% agreement with 28-layer |

### Overnight Experiments (ran 2026-04-11 night)

**Experiment 1: Hybrid (INT4 Attention + Genome FFN)**
- Theory: Keep real attention (critical for token routing), replace FFN with genome
- Result: 17% top-1, 44-45% top-10 at 95-117 MB — WORSE than pure genome
- Conclusion: LoRA genome FFN isn't expressive enough to replace real FFN

**Experiment 2: Neural DNA Expression (NOVEL)**
- Theory: Shared DNA bank across all layers + per-layer expression functions
- Like biology: one neuron acts like thousands based on context
- Result: 4% top-1, 28-30% top-10 at 22-89 MB
- Scaling projections: 1000T model in 0.2-0.6 GB (insanely small if quality improves)
- Conclusion: Concept works (it learns!) but needs architectural refinement

### NOT YET TESTED
- **MoE Genome** (mixture of expert layers) — code ready in `ultracompress/genome_moe.py`
- **Quantized + Genome Correction** (Q2/Q4 + LoRA correction)
- **Genome V1 Online** (fresh data each batch, no cache overfitting)
- All three ready to run: `python -u run_remaining.py 2>&1`

## Key Findings
1. **Progressive per-layer init is CRITICAL** (20-26% top-1 vs 14-18% without)
2. **Cached fine-tuning OVERFITS** (top-10 drops 53→46%). Use online distillation instead.
3. **sd=256 = same as sd=64** → bottleneck is training quality, NOT genome size
4. **Output cosine is MISLEADING** — always measure token agreement (top-1, top-10)
5. **4-layer model has 0% agreement with 28-layer** — depth matters enormously
6. **Teacher GPU fix:** Must do `teacher.embed_weight = teacher.embed_weight.to(device)` after load_weights() or training is 100x slower (1.2GB CPU→GPU transfer per step)
7. **RoPE bug fixed:** eval functions must compute `pos = torch.arange(tokens.shape[1])` from input, not use a global positions tensor

## Critical Code Files
| File | Purpose |
|------|---------|
| `ultracompress/genome_compressor.py` | Core: GenomeModel, MicroTransformerLayer, GenomeCompressor |
| `ultracompress/genome_v2.py` | V2: MultiViewGenomeLayer, LoRAGenomeLayer |
| `ultracompress/genome_moe.py` | MoE genome (untested) |
| `ultracompress/inference.py` | MiniTransformer, TransformerLayer, RoPE |
| `ultracompress/streaming_loader.py` | Shard-by-shard loading for 8B+ |
| `ultracompress/api_compressor.py` | Compress models behind APIs |
| `sandbox2_neuraldna.py` | Neural DNA Expression architecture |
| `run_remaining.py` | **RUN NEXT** — MoE + Q+Correction + V1 Online |
| `run_overnight.py` | Full 5-experiment pipeline (Hybrid + DNA done) |
| `arena.py` | Tournament framework for head-to-head comparison |
| `run_genome_8b_streaming.py` | 8B streaming compression (ready, not run) |
| `qwen3_0.6b_cache.pt` | Cached Qwen3-0.6B weights (~1.8 GB) |
| `benchmark_results.json` | 10 experiment results |
| `STATUS.md` | Project status doc |

## Architecture Details

### MicroTransformerLayer (V1)
```
Input (B, T, 1024) → down_proj (1024→128) → tiny_attention + tiny_FFN in 128-dim → up_proj (128→1024) → residual add
```
- Per layer: ~426K params at sd=128
- 28 layers: ~11.9M params = 23.9 MB

### MultiViewGenomeLayer (V2a) — Most Efficient
- 4 parallel projections ("views") each capturing different aspects
- Cross-view mixing layer
- 26% top-1, 48% top-10 at only 16.5 MB (53x compression)

### Neural DNA Expression — Novel
- Shared DNABank (n_genes × gene_dim) across ALL layers
- Per-layer ExpressionFunction: reads context → activates top-K genes
- Expressed genes modulate transformation via gating
- Duplicate param bug in optimizer was fixed (DNA bank params appeared twice)

### Training Methods
- **Progressive:** Train each layer independently on its I/O, then joint fine-tune. BEST init.
- **Cached:** Pre-compute teacher outputs, train from cache. Fast but OVERFITS.
- **Hybrid:** Progressive init + cached fine-tuning. Best top-1 but hurts top-10.
- **Online:** Fresh random tokens each batch. Prevents overfitting. Not fully tested yet.

## What To Do Next
1. **Run `python -u run_remaining.py 2>&1`** — tests MoE, Q+Correction, V1 Online
2. **Based on results, decide direction:**
   - If MoE wins → scale up expert count
   - If Q+Correction wins → different paradigm (keep quantized model + small fix)
   - If V1 Online wins → progressive init + online training is the answer
3. **Test on Qwen3-8B** to prove scaling (streaming script ready)
4. **Push quality toward 90%+ top-10** — this is the hard part
5. **Consider: we may need a fundamentally new approach** — current best is 53% top-10

## Goals (from Sip)
- 10T model → 20GB (near-zero degradation)
- 100T → 20GB
- 1000T → 20GB (moonshot)
- Replace data centers for training
- Help Athena AGI get bigger brain without hardware upgrades
- Product: sell to companies
- Think like Tesla with electricity, Einstein with spacetime
- "Not good enough" — NEVER settle for incremental
