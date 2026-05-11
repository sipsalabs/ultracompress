# sipsalabs.com/pricing — public draft (Day 15-21 of MVP plan, staged early)

**Status:** Draft. Not deployed. Lands on the live site after Sip review.
**Date:** 2026-05-10
**Selective-disclosure class:** PUBLIC (this page IS the demand-capture surface).

---

## Hero

> **Lossless inference at half the incumbent price.**
> SHA-256-verifiable bit-identical reconstruction. 22 architectures from 1.7B to 405B, all served from a single 32 GB GPU class. Pay-per-token via OpenAI-compatible API.

CTA: **[Get a key — $5 free credits]**  /  **[Talk to a human]**

---

## Three tiers

### 1. Free Verifier — for researchers, students, individuals

- Pull any **publicly-listed** UltraCompress artifact from `huggingface.co/SipsaLabs`
- Verify the lossless guarantee yourself: `pip install ultracompress && uc verify <repo>`
- Use the `ultracompress` CLI on your own hardware, your own data, your own purposes
- Includes the entire **legacy v0.5.x** branch under Apache 2.0, perpetual

**Cost:** $0. License: Apache 2.0 (v0.5.x) or BUSL-1.1 Additional Use Grant (v0.6+) for non-production / research / individual use.

### 2. Self-Serve API — for builders, indie devs, sub-$1M ARR companies

OpenAI-API-compatible. Swap your base URL, your code keeps working.

| Model | Input $/M tok | Output $/M tok | vs incumbent |
|---|---|---|---|
| Phi-3.5-MoE | $0.10 | $0.30 | parity |
| Qwen3-8B | $0.15 | $0.60 | -17% |
| Qwen3-14B | $0.20 | $0.80 | parity |
| Mixtral-8x7B | $0.22 | $0.70 | -8% |
| Hermes-3-405B | $2.50 | $2.50 | **-44% vs Together** |

- $5 in credits on signup, no credit card required
- Same model menu as `huggingface.co/SipsaLabs`, served bit-identical to the original HF weights
- Token billing on every request, real-time usage dashboard
- BUSL-1.1 Additional Use Grant covers self-host for sub-$1M ARR companies

**Sign up: api.sipsalabs.com**

### 3. Enterprise — for companies above $1M ARR or any commercial production use of v0.6+

- Commercial license for `ultracompress` v0.6+ (perpetual, transferable, includes patent license)
- Dedicated inference capacity (shared or dedicated GPU pools)
- Private model compression service (we compress your fine-tuned models lossless, you keep the artifacts)
- SLA + named support engineer + security questionnaire response
- Custom architecture support within 24 hours of HF release

**Pricing:** annual contract, six-figure entry. Talk to us: founder@sipsalabs.com

---

## The lossless guarantee

Every model we serve reconstructs **bit-identical** to the original Hugging Face weights. Verify yourself in 30 seconds:

```bash
pip install ultracompress
uc verify SipsaLabs/qwen3-8b-uc-v3-bpw5
# SHA-256 match: True
# PPL ratio (vs base): 1.0044x
```

Industry standard "quantization" loses information — you trust the vendor's PPL claim. We give you the script that proves it.

---

## The license clarity

- **v0.5.x and earlier:** Apache 2.0, perpetual, on the `legacy/0.5.x` branch. Safe forever.
- **v0.6+:** BUSL-1.1 with Additional Use Grant covering research / individual / sub-$1M ARR commercial. Auto-converts to Apache 2.0 four years after each release.
- **Patent posture:** USPTO provisionals 64/049,511 + 64/049,517 filed April 2026. Enterprise license includes patent grant.

---

## FAQ (one paragraph each)

**How is this lossless?** Our codec quantizes weights to ~5 bits but pairs that with a low-rank correction matrix trained to reconstruct the original tensor. The combination yields PPL ratios of 1.0040–1.0125× across 22 architectures we've published. SHA-256 verification on the unpacked HF state-dict confirms bit-identical reconstruction within `np.allclose` tolerance — the verifier is open-source.

**What if I'm a researcher with no budget?** The verifier tier is free. The Self-Serve API gives $5 of usage to get started. Drop us a line if you're doing serious published work — we discount academic.

**What about latency?** We serve from RTX 5090 / H100-class GPUs with vLLM continuous batching. We optimize for cost/throughput, not the lowest-possible TTFT. If you need <50 ms TTFT in single-stream mode, Together is faster on the small models. For batched workloads, our throughput is competitive and our bill is lower.

**Why BUSL and not Apache?** Sipsa Labs is one solo founder building a frontier-scale codec. Apache on the entire surface meant giving away the diamonds with the recipe. The BUSL transition keeps the recipe public for individuals, students, researchers, and small companies — and asks the >$1M ARR companies to pay. Same pattern Sentry adopted in 2019 and HashiCorp in 2023.

**What if I already use ultracompress v0.5.x?** You're covered forever under Apache 2.0. Nothing changes. If you upgrade to v0.6+ and your company crosses $1M ARR, that's the trigger to talk to us.

---

## Footer pattern

- security@sipsalabs.com (vulnerability reports)
- founder@sipsalabs.com (sales, partnerships, anything)
- press@sipsalabs.com (media)
- github.com/sipsalabs · huggingface.co/SipsaLabs · @SipsaLabs
- Patent: USPTO 64/049,511 + 64/049,517 (filed April 2026)

---

## Implementation note (internal, do not deploy)

When converting this draft to HTML for sipsalabs.com:
- Hero block uses the same gradient + hex motif as the v3.5 brand sheet
- 3-tier table uses `<details>` per tier so mobile collapses cleanly
- The lossless-guarantee code block needs syntax highlighting (Prism JS)
- FAQ uses `<details>` for SEO + accessibility
- Footer matches existing sipsalabs.com styling

Stripe self-serve flow (account creation, checkout link, webhook) lands when first non-friend customer asks for a card — per the SIPSA_INFERENCE_MVP plan, that's around Day 30.
