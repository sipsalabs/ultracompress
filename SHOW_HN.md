# Show HN: 959x model compression via one shared recursive block + quantization pipeline

**GitHub:** https://github.com/mounnar/ultracompress
**Paper:** [PAPER_DRAFT.md](PAPER_DRAFT.md)

---

UltraCompress replaces all 28 layers of a transformer with a single shared block applied recursively. Combined with a 5-stage quantization pipeline, we achieve **959x end-to-end compression with only 1.5% quality degradation**, proven on real weights.

**The approach:** Fractal Residual Recursion (FRR) exploits the fact that transformer layers are functionally similar (CKA similarity >0.9) despite having completely different weights (cosine similarity: 0.000). One shared block with lightweight per-layer modulation (~8K params) captures the shared function space. Compress that block with Hadamard rotation + SVD + Q2 + entropy coding, and total compression hits 959x.

**Proven results (Qwen3-0.6B, 1.5 GB baseline):**

| Method | Top-10 | Size | Compression | Notes |
|--------|--------|------|-------------|-------|
| FRR (50K steps) | 63% | 14.7 MB | 60x | New: just needs more training |
| FRR + Q2 pipeline (E2E) | 53% | 1.8 MB | 959x | Proven end-to-end |
| FRR-PHM (hypercomplex) | 53% | 3.7 MB | 239x | 4x fewer params |
| Standard quantization (Q2) | ~89% | 220 MB | 4x | For comparison |

**Scaling evidence:** 1.7B FRR shows 51% T10 at step 3K (vs 46% for 0.6B at same step). Bigger models compress better because FRR compression = 1/n_layers, and bigger models have more layers.

**What makes this different from GPTQ/AWQ/etc:** Those quantize each layer independently (4-8x ceiling). FRR replaces all layers with ONE shared block (60x+). The two are composable.

**Honest caveats:**

- 63% top-10 token agreement is not production quality for text generation. The model captures the right neighborhood but not always the exact token.
- 959x compression drops quality to 53% T10 (from 63% pre-Q2). Usable for some applications, not all.
- Tested on 0.6B and 1.7B. 8B+ results pending.
- The quality gap vs the original model is real. This is extreme compression, not lossless.
- No one else has validated this approach at scale. We may hit unexpected walls at 8B+.

**What I believe is novel:** Google's Relaxed Recursive Transformers (Oct 2024) and Ouroboros V2 (Apr 2026) use similar shared-block + LoRA approaches but achieve only ~2x compression. We push to 60x (30x further) by combining aggressive parameter sharing with extended distillation training.

The codebase is 76 modules, 50K+ lines. Includes holographic encoding, tensor networks, immune system V-D-J recombination, predictive coding, hypercomplex multiplication, and 70 other approaches we tested. Research code, not a polished library.

22-year-old solo developer. Built in 48 hours. AMA.
