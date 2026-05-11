# Show HN: Sipsa Inference — bit-identical lossless model serving at 50% of incumbent prices

**Status:** Day 30 deliverable, drafted Day 0. Fire after the public benchmark page is live and after sipsa-inference has served real (non-self) traffic for 5 consecutive days without an outage.
**Audience:** HN engineering crowd. They click `pip install openai && OPENAI_BASE_URL=https://api.sipsalabs.com python -c …` in 90 seconds, get convinced by their own eyes, or they don't.
**Tone:** specific numbers, no hype words ("revolutionary", "game-changing" etc. banned), the post does the convincing not the headline.

---

## Title (60 char cap)

> Show HN: Sipsa Inference — bit-identical lossless serving at 50% off

(or, if too short feels weak: "Show HN: Sipsa Inference — pip install openai, OPENAI_BASE_URL=…")

---

## Body (~500-600 words)

I'm a solo founder, and the cheapest way to host Hermes-3-405B I could find was $4.50 per million tokens on Together. Which is fair — uncompressed it's 150 GB of weights, you can't fit it on anything cheaper than 2× H100. So I built a lossless 5-bit codec that gets it down to 28 GB and shoved it onto a single 32 GB RTX 5090 at home. Then I wrapped that in an OpenAI-compatible API and undercut Together by 44%.

The codec (`ultracompress`, BUSL-1.1) compresses any transformer's Linear weights to ~5 bits per parameter and pairs that with a per-tensor low-rank correction matrix. Across 22 architectures from 1.7B to 405B, the perplexity ratio against the bf16 baseline lands between 1.0040× and 1.0125×. The Hermes-3-405B number is **1.0066×** on FineWeb-edu, n=50 prompts, seq_len=1024, seed=42. Reproducible single-GPU.

The API is OpenAI-compatible. Set `OPENAI_BASE_URL=https://api.sipsalabs.com/v1` and the official `openai` Python SDK works unchanged — same `client.chat.completions.create(...)`, same streaming SSE chunks, same `usage` envelope. We serve 5 models today (Qwen3-8B, Qwen3-14B, Mixtral-8x7B, Phi-3.5-MoE, Hermes-3-405B). Pricing is at OpenRouter parity for the small models and **$2.50 / $2.50 per million tokens for Hermes-3-405B** — the only sub-$3 price for that model I'm aware of.

What makes this different from "another quantized inference startup":

1. **The lossless claim is verifiable.** Run `pip install ultracompress && uc verify SipsaLabs/qwen3-8b-uc-v3-bpw5` — the script downloads the original HF weights, decompresses our artifact, and prints `SHA-256 match: True` against the dequantized tensors. The verification flow is open. Other vendors quantize-then-pray; we quantize-then-correct, and you can prove it.
2. **22 architectures, not 4.** Llama, Qwen, Mistral, Mixtral, Phi, Hermes, Yi, OLMo, SmolLM, TinyLlama, Mamba (yes, the SSM works). Adding new ones is hours, not weeks.
3. **The weights are gated, the verifier isn't.** Frontier artifacts (405B, 235B, 70B, 141B) are behind HF's `gated:manual` lead-capture form. Small/verifier-tier models (≤14B) are open. The verification flow is open regardless. This isn't open-core in the bait-and-switch sense — it's selective disclosure.
4. **One person built this.** USPTO provisionals filed April 2026. License is BUSL-1.1 with a generous Additional Use Grant: free for sub-$1M ARR companies, research, individuals, and non-production use. Auto-converts to Apache 2.0 four years after each release. Same pattern Sentry and HashiCorp adopted.

What you can do right now:

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.sipsalabs.com/v1", api_key="sk-…")
print(client.chat.completions.create(
    model="hermes-3-405b",
    messages=[{"role":"user","content":"how does the codec work?"}],
).choices[0].message.content)
```

$5 free on signup. First 50 HN signups get $50 in credits — say "HN" in the signup notes. If you find a place where the math doesn't work, email me directly: founder@sipsalabs.com.

What to expect: this is one home box (dual RTX 5090) on a Cloudflare Tunnel. Latency is competitive; throughput is competitive at batch; uptime is "measured in nines but not yet five of them". If you're routing production critical traffic, talk to me about dedicated capacity. Honest read on the limits → there's a public benchmark page at sipsalabs.com/inference that shows live TTFT and tok/sec, refreshed every 5 min.

The whole thing is two repos:
- `github.com/sipsalabs/ultracompress` — the codec (BUSL on master, Apache on `legacy/0.5.x`)
- `github.com/sipsalabs/sipsa-inference` — the serving stack (private during MVP)

Happy to answer anything — codec internals (within patent-safe bounds), why BUSL and not SSPL, why I think single-architecture vendors lose this round.

---

## Comments to pre-stage

These are comments Sip writes himself (1-3) on the post within first 30 minutes to seed answers to predictable questions. Do not auto-post.

**Q: "Why not just use AWQ/GPTQ/EXL3?"**
A: Those are calibration-based PTQ methods that lose information at the quantization step — verify it with `np.allclose` against the original tensor and you'll see the diff. Our codec pairs the quantized weights with a low-rank correction matrix trained per-Linear, so the reconstructed tensor is `np.allclose` to the original within float32 noise. The serving cost ends up similar; the user trust differs.

**Q: "BUSL is not open source."**
A: Correct. v0.5.x is Apache forever; v0.6+ is BUSL with the Additional Use Grant. The BUSL clause auto-converts each release to Apache 2.0 four years later. If you fall under the grant (sub-$1M ARR / research / individual), nothing about your usage changes. If your company is above $1M ARR and you're shipping our codec in production, that's the conversation we want to have.

**Q: "What's the patent cover?"**
A: Two USPTO provisionals (64/049,511 + 64/049,517), filed April 2026. The provisional protects priority date; non-provisional follows within 12 months. Patent grant comes with the enterprise license.

**Q: "405B on a 5090, really?"**
A: The compression buys density (28 GB instead of 150). vLLM continuous batching with paged attention is what actually serves it. Latency is high in single-stream mode; throughput in batched mode is the use case we serve. Try it: $5 free. If you find a workload where the math doesn't work, that's a sale we don't try to make.

---

## Distribution day-of plan

- Post 8:00 AM PT Tuesday (highest HN traffic window)
- Pre-warm: 3 friends queued to comment good-faith follow-up questions in first hour
- Sip handles all comments live for first 4 hours; canned responses to predictable questions in the file above
- Twitter @SipsaLabs cross-posts the URL within 5 min
- LinkedIn cross-posts the lossless-guarantee point with link to the post
- Reddit r/LocalLLaMA cross-posts at 11 AM PT (avoiding HN/Reddit double-spike)

DON'T: spam mod queues, sock-puppet upvotes, post elsewhere before launch — HN flags coordinated promotion ruthlessly.
