# Show HN: UltraCompress -- 42x model compression via one shared recursive block

**GitHub:** https://github.com/athena-agi/ultracompress
**Paper:** [PAPER_DRAFT.md](PAPER_DRAFT.md)

---

UltraCompress replaces all 28 layers of a transformer with a single shared block applied recursively, achieving 42x compression on Qwen3-0.6B. Instead of quantizing or pruning weights, it re-represents the entire computation through what I call Fractal Residual Recursion (FRR).

**The key insight:** Transformer layers have zero cross-layer weight redundancy (cosine similarity: 0.000), yet a single micro-transformer can learn to emulate all of them when conditioned on a layer index and residual corrections. Weight independence between layers is wasteful -- one shared block, applied 28 times with small per-layer corrections, captures most of the computation.

**Results (Qwen3-0.6B, 1.5 GB baseline):**

| Method | Top-1 | Top-10 | Size | Compression |
|--------|-------|--------|------|-------------|
| FRR (1 shared block) | 44% | 62% | 21 MB | 42x |
| HWI (holographic interference) | 35% | 57% | 12 MB | 76x |
| FRR from-scratch (toy task) | -- | -- | -- | 80.7% gen accuracy |
| Standard prune+quantize | -- | -- | 247 MB | 6x |

**Honest caveats:**

- 62% top-10 agreement is not production quality. You would not deploy this today.
- All results are on 0.6B, which is a toy model. The real test is 8B+, which I have not run yet.
- From-scratch 80.7% is on a pattern-learning task, not real LLM generation.
- Scaling projections (400x at 70B) are theoretical and unvalidated.

**What I believe is novel:** I have not found any prior work using fractal/recursive shared-block distillation as a compression method. Universal Transformers share blocks but do not use this for post-hoc compression. I did a literature review and found nothing. If I am wrong, please tell me -- I genuinely want to know.

The codebase is ~16K lines across 72 modules, covering everything from product quantization to holographic weight interference. It is messy research code, not a polished library.

Solo developer. Built over several months. AMA.
