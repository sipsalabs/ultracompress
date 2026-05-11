# Twitter / X Thread — 9-Architecture Matrix Launch (2026-05)

**1/** Today we shipped 9 transformer architectures compressed end-to-end on a single 32GB consumer GPU.

5 dense + 4 MoE. 1.7B → 405B parameters. Same pipeline. Same hardware.

Mean perplexity ratio: **1.0066**. Worst case 1.0126. Best case 1.0013.

**2/** The matrix:

Qwen3-1.7B → 1.0091
Mistral-7B-v0.3 → 1.0126
Llama-3.1-8B → 1.0071
Llama-3.1-70B → 1.0090
**Hermes-3-Llama-3.1-405B → 1.0071**
Qwen3-235B-A22B (MoE 128 exp) → 1.0038
Mixtral-8x22B (MoE 8 exp) → 1.0061
Mixtral-8x7B (MoE 8 exp) → 1.0037
**Phi-3.5-MoE-instruct → 1.0013** ← BEST

**3/** Critical detail: the 405B model is ~810 GB at bf16. The 32GB GPU never holds the full model. Per-layer streaming compression — load one decoder layer, compress it, free it, move on.

Peak VRAM during 405B compression: **29 GB**. Same single GPU does the 1.7B run too.

**4/** MoE was harder than dense. Default substring-matching of `gate_proj/up_proj/down_proj` doesn't catch Phi-MoE's `block_sparse_moe.experts.<i>.{w1,w2,w3}` naming. We caught it when first run reported `n_quantized_linears=4` instead of the expected 52.

After the fix: all 4 MoE arches converge tighter than dense.

**5/** Hardest part of the headline: this isn't an architecture trick.

Same pipeline runs across Llama (Meta), Mistral, Mixtral, Phi (Microsoft), Qwen (Alibaba), Hermes (NousResearch). Five distinct families. No per-family tuning.

Customers asked "is this just a Qwen3 thing?" Answer is now empirically no.

**6/** What this enables:

A 70B model running at sub-1% perplexity degradation on a $2K consumer GPU. A 405B model on the same. A 235B-active-22B MoE on the same. All from one pipeline.

For inference clouds (Lambda, CoreWeave, Together, Groq): your 70B+ inventory just got compressible without quality loss.

**7/** The science:

Per-layer correction overlay over 5-bit scalar quantization-quantized base. SVD warm-start from quantization residual. Per-Linear hidden-MSE training (50% of work) + full-stack logit-KL fine-tune (25% of work).

Streaming-teacher computes baselines on the same single GPU.

**8/** Open questions:

How much further can we push? 4 bpw is the next wall. 8K context PPL needs validation (we measured 1024). Long-tail fairness across rare tokens needs a per-domain split.

These are tractable research questions, not pipeline blockers.

**9/** Patent provisionals (64/049,511 + 64/049,517) filed 2026-04-25. PyPI v0.4.0 streaming compression shipped. GitHub: github.com/sipsalabs/ultracompress. HuggingFace: huggingface.co/sipsa-labs.

@SipsaLabs

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
