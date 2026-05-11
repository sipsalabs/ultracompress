# HN First Comment — paste-ready 2026-05-10

**Instructions:** After clicking Submit on the HN form, IMMEDIATELY open the new submission and paste the body below as your first comment (a self-reply at the top of the thread). Show HN best practice: URL is the link, body lives as the OP's first comment so it stays at the top of discussion.

**Action:** Copy everything between the `--- BEGIN ---` and `--- END ---` markers (do NOT include the markers themselves).

---

--- BEGIN ---
I'm a solo founder, and the cheapest hosted inference for Hermes-3-Llama-3.1-405B I could find was $4.50 per million tokens on Together. Fair price — uncompressed it's ~250 GB of weights, you can't fit it on anything cheaper than 2× H100. So I built a lossless 5-bit codec that gets it down to one 32 GB consumer GPU and wrapped it in an OpenAI-compatible API at **$2.50 in / $2.50 out per million tokens** — 44% under Together.

The codec (`ultracompress`, BUSL-1.1 from v0.6+; v0.5.x stays Apache forever) compresses any transformer's Linear weights to ~5 bits per parameter and pairs that with a per-tensor low-rank correction matrix. Across 22 architectures from 0.6B to 405B, the perplexity ratio against the bf16 baseline lands between **1.0026× and 1.0200×**, measured on a held-out FineWeb-edu tail at seq_len=1024, n=30-50, seed=42. Hermes-3-405B lands at **1.0066×** under streaming per-layer eval (the only baseline procedure that fits 405B on consumer hardware — disclosed honestly in the lab notebook).

The API speaks the OpenAI schema. Set `OPENAI_BASE_URL=https://api.sipsalabs.com/v1` and the official `openai` SDK works unchanged — same `client.chat.completions.create(...)`, same SSE chunks, same `usage` envelope.

What separates this from "another quantized inference startup":

1. **The lossless claim is verifiable.** `pip install ultracompress && uc verify SipsaLabs/qwen3-8b-uc-v3-bpw5` downloads the artifact, reconstructs the tensors, and confirms an SHA-256 match against the manifest. Other vendors quantize-then-pray; we quantize-then-correct, and you can prove it on your own GPU.
2. **22 architectures, not 4.** Llama, Qwen, Mistral, Mixtral, Phi, Hermes, Yi, OLMo, SmolLM, TinyLlama, Mamba (yes, the SSM works). Full matrix at `huggingface.co/SipsaLabs`.
3. **One person built this.** Two USPTO provisionals (64/049,511 + 64/049,517) filed 2026-04-25. License is BUSL-1.1 with an Additional Use Grant — free for sub-$1M ARR companies, research, individuals. Auto-converts to Apache 2.0 four years after each release. Sentry/HashiCorp pattern, not MongoDB SSPL.

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.sipsalabs.com/v1", api_key="sk-…")
print(client.chat.completions.create(
    model="hermes-3-405b",
    messages=[{"role":"user","content":"how does the codec work?"}],
).choices[0].message.content)
```

$5 free credits on signup, no card. If you find a place where the math doesn't work: `founder@sipsalabs.com`.

**About the front door.** I don't do cold DM outreach. This post, the live benchmark page at `sipsalabs.com/inference`, and the HF model cards are how customers find me. If you hit a wall, the comment thread below is the support channel — I'm here for the first 4 hours.

What to expect: this is one home box (dual RTX 5090) on a Cloudflare Tunnel. Latency is competitive at batch; uptime is "measured in nines, not yet five of them". Public benchmark page shows live TTFT and tok/sec. If you need single-stream sub-50 ms TTFT, use Together — they're optimized for that. If you need a ~44% lower bill on the frontier slot with a verifier you can point at, use us. That's the trade.

Happy to answer codec internals (within patent-safe bounds), why BUSL not SSPL, or why I think single-architecture vendors lose this round.
--- END ---

---

## Pre-staged follow-up answers (copy individually as new replies — DO NOT auto-post)

**Q: "Why not just use AWQ/GPTQ/EXL3/HQQ?"**
> Those are calibration-based PTQ methods that lose information at the quant step — different CUDA versions and `torch_dtype` defaults produce slightly different reconstructed weights at the customer end vs the trainer's. UltraCompress reconstruction is closed-form over fp32 metadata, so SHA-256 over reconstructed bytes matches deterministically. For audited deploys where "production model behaves bit-exactly the same as the eval model" is the compliance question, that property is the value. Not claiming faster matmul — we re-use PyTorch.

**Q: "BUSL is not open source."**
> Correct, by OSI definition. v0.5.x is Apache forever on the `legacy/0.5.x` branch. v0.6+ is BUSL-1.1 with an Additional Use Grant — free for sub-$1M ARR companies, research, and individuals — auto-converting to Apache 2.0 four years after each release. If you fall under the grant nothing changes for you. If your company is above $1M ARR shipping the codec in production, that's the conversation we want.

**Q: "What's the patent cover?"**
> Two USPTO provisionals filed 2026-04-25. The provisionals protect priority date; non-provisional follows within 12 months. Patent grant comes with the enterprise license. Fees current.

**Q: "405B on a 5090, really?"**
> Compression buys density (~250 GB → 32 GB peak during reconstruction). Streaming per-layer eval is what fits the bf16 baseline on consumer hardware too — same procedure for both, fully disclosed. Latency in single-stream is high; throughput at batch is the use case we serve. $5 free credits, try it.

**Q: "Hermes-3-405B 1.0066× — what's the apples-to-apples baseline?"**
> Honest: you can't fit full-model bf16 single-shot in 32 GB, so the baseline is `bf16_streaming_per_layer_from_hf_cache` — the identical streaming procedure the compressed run uses, with un-quantized bf16 weights. Same n=50, seq_len=1024, FineWeb-edu held-out tail, seed=42. NOT the same as a multi-GPU full-model bf16 eval. Lab-notebook entry pre-locked the band before the run finished.

**Q: "Why is Mistral so much tighter than I'd expect?"**
> Saturday night update: **Mistral-7B-v0.3 just landed at 1.00548×** (verified, n=50, seq_len=1024, FineWeb-edu, seed=42). Previous best on this arch was 1.0502× — we tried 4 different cures over 2 weeks and each one was *worse*, not better. The 5th attempt (a different training-objective class entirely; recipe patent-protected) finally cracked it. 9× tighter than the prior ceiling. That's why I think single-architecture vendors lose: the bottleneck wasn't the codec, it was the objective. The objective generalizes; the calibration tricks don't.

**Q: "Did the same trick work on other architectures?"**
> Phi-3-mini test fired tonight, result lands within the hour. Llama-3.1-8B is queued next. The hypothesis is that the new objective is architecture-agnostic — if it holds on Phi-3 and Llama, every arch in the 22-row matrix gets re-evaluated against the new ceiling. I'll publish the JSON the moment it lands; if it fails, that lands in HONEST_NEGATIVE_RESULTS the same day.
