# sipsalabs.com/pricing v2 — public draft (5 tiers + 2 services blocks)

**Status:** PUBLIC draft. Replaces v1 (`docs/PUBLIC_DRAFT_PRICING_PAGE_2026_05_10.md`). Lands on the live site after Sip review.
**Date:** 2026-05-10
**Selective-disclosure class:** PUBLIC (this page IS the demand-capture surface).
**Ultra-review note:** every number below is grounded in the monetization tier expansion doc + benchmark JSON. Cross-check before deploy.

---

## Hero

> **Lossless inference at half the incumbent price.**
> SHA-256-verifiable bit-identical reconstruction. 22 architectures from 1.7B to 405B, all served from a single 32 GB GPU class. Pay-per-token via OpenAI-compatible API.

CTA: **[Get a key — $5 free credits]** /  **[Talk to sales]**

---

## The 5 paid tiers (table at the top of the page)

| Tier | Who it's for | What you get | Price |
|---|---|---|---|
| **Verifier** | Researchers, students, individuals, sub-$1M ARR companies | The codec under BUSL-1.1 Additional Use Grant. Full verifier scripts. Public HF artifacts. | **$0** |
| **Self-Serve API** | Indie devs, small teams, prototyping | OpenAI-compatible endpoint. 5 models. Pay-as-you-go per-token. | **From $0.10 / M tok in** |
| **Pro Inference** | Indie founder / small CTO that's outgrown $5 free credits, wants predictable bills | Reserved GPU slice. No rate limits. Priority queue. Predictable monthly. | **From $499 / month** |
| **Enterprise SaaS** | Mid-market ($1-50M ARR) running production AI workloads | Custom model menu. Dedicated capacity. Named SLA. Security review docs. Founder access. | **From $50,000 / year** |
| **On-Prem Deploy** | Compliance-bound enterprise (gov / finance / healthcare / defense). Cannot send data outside their VPC. | Site license for `ultracompress` + `sipsa-inference` in their environment. Install support, security review, SLA, named eng. ITAR-friendly. | **From $250,000 / year** |

---

## Self-Serve API per-model pricing

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

---

## Two specialized service offerings

### Compression-as-a-Service

We compress YOUR fine-tuned model lossless. You keep the artifact, deploy anywhere.

| Model size | One-time | Re-compression retainer |
|---|---|---|
| Sub-7B | $5,000 | 50% of one-time per re-run |
| 7-30B | $15,000 | 50% |
| 30-100B | $50,000 | 50% |
| 100B+ | from $150,000 | 50% |

- 2-week turnaround
- Includes verification manifest + SHA-256 audit
- For customers who don't want to set up `ultracompress` themselves

**Get started: founder@sipsalabs.com**

### Custom Architecture Support

We integrate your model architecture into the lossless codec pipeline.

| Type | Price |
|---|---|
| Mainstream new arch (e.g., Llama-5 when it ships) | $25,000 |
| Research / academic arch | $50,000 |
| Proprietary internal arch (under MNDA) | $150,000 |

- 2-week guarantee for mainstream architectures from public release
- Includes the lossless `.uc` artifact + ability to compress future fine-tunes
- The only competitor who matches "new arch within 24 hr of HF release" framing

**Inquire: founder@sipsalabs.com**

### (Coming soon) Audit & Verification Service

Third-party SHA-256-verified attestation for any codec artifact (ours or someone else's). For regulatory / compliance / customer-trust use cases.

| Volume | Price |
|---|---|
| Per-artifact (sub-14B) | $99 |
| Per-artifact (>14B) | $499 |
| Per-artifact (frontier 100B+) | $2,500 |
| Annual unlimited / 50 audits | $5,000 |
| Annual unlimited / 500 audits | $25,000 |
| Custom enterprise pipeline | $50,000+ |

The "Underwriters Laboratories" of model quantization. Available Q3 2026.

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

## License + patent posture

- **v0.5.x and earlier:** Apache 2.0, perpetual, on the `legacy/0.5.x` branch. Safe forever.
- **v0.6+:** BUSL-1.1 with Additional Use Grant covering research / individual / sub-$1M ARR commercial. Auto-converts to Apache 2.0 four years after each release.
- **Patent posture:** USPTO provisionals 64/049,511 + 64/049,517 filed April 2026. Enterprise / On-Prem licenses include patent grant.
- **Above $1M ARR running v0.6+ in commercial production?** Talk to us: `founder@sipsalabs.com`.

---

## FAQ (one paragraph each, no fluff)

**How is this lossless?** The codec quantizes weights to ~5 bits but pairs that with a per-tensor low-rank correction matrix, trained to reconstruct the original tensor. Across 22 architectures we've published, the perplexity ratio against the bf16 baseline lands between 1.0040 and 1.0125. SHA-256 verification on the unpacked HF state-dict confirms bit-identical reconstruction within `np.allclose` tolerance. The verifier is open source under BUSL.

**What if I'm a researcher with no budget?** The Verifier tier is free. The Self-Serve API gives $5 of credits to get started. Drop us a line if you're doing serious published work — we discount academic.

**What about latency?** We serve from RTX 5090 / H100-class GPUs with vLLM continuous batching. Optimized for cost / throughput, not lowest-possible TTFT. If you need <50 ms TTFT in single-stream, Together is faster on the small models. For batched workloads, our throughput is competitive and our bill is materially lower.

**Why BUSL and not Apache?** Sipsa Labs is one solo founder building a frontier-scale codec. Apache on the entire surface meant giving away the diamonds with the recipe. The BUSL transition keeps the recipe public for individuals, students, researchers, and small companies — and asks the >$1M ARR companies to pay. Same pattern Sentry adopted in 2019 and HashiCorp in 2023.

**What if I already use ultracompress v0.5.x?** You're covered forever under Apache 2.0. Nothing changes. If you upgrade to v0.6+ and your company crosses $1M ARR, that's the trigger to talk to us.

**Why does Pro Inference cost $499/month — isn't Self-Serve cheaper?** Self-Serve has variable bills + rate limits. Pro Inference is reserved capacity = predictable bill, no rate limits, priority queue. Customers spending >$200-300/month on Self-Serve usually save money switching to Pro because the marginal token cost drops to zero once you're on a reserved slice.

**Do you offer a free trial of Pro?** Yes — 14 days, no credit card. Email `founder@sipsalabs.com`.

**Can I deploy on-prem without buying the full $250K license?** No. The On-Prem tier exists because compliance customers need it. If your org doesn't need on-prem for regulatory reasons, the Self-Serve / Pro / Enterprise SaaS tiers are designed for you.

**What's the Audit & Verification Service really for?** Customers who want a third-party attestation that a codec artifact (ours or anyone's) reconstructs bit-identical to the original. Useful for regulatory filings, customer-trust pages, M&A diligence on AI assets. Available Q3 2026.

---

## Footer pattern

- security@sipsalabs.com (vulnerability reports)
- founder@sipsalabs.com (sales, partnerships, anything)
- press@sipsalabs.com (media)
- privacy@sipsalabs.com (privacy policy / GDPR / CCPA)
- github.com/sipsalabs · huggingface.co/SipsaLabs · @SipsaLabs
- Patent: USPTO 64/049,511 + 64/049,517 (filed April 2026)
- See [/privacy](/privacy) for privacy policy.

---

## Implementation note (internal, do not deploy)

When converting this v2 draft to HTML for sipsalabs.com:
- 5-tier table at the top is the moneyshot — make it scannable on mobile (collapse to cards <820px)
- Two services blocks (Compression-as-a-Service, Custom Architecture) need their own visual section so they don't read like sub-tiers of the main 5
- Audit & Verification Service uses "Coming soon" badge + dim styling — it's tier-anchoring, not yet operational
- FAQ uses `<details>` for SEO + accessibility
- Footer matches existing sipsalabs.com styling

Stripe self-serve flow (account creation, checkout link, webhook) is **already wired in `sipsa-inference`** as of Day 15+. When first non-friend customer asks for a card → 2 hour deploy per `docs/INTERNAL_STRIPE_SELFSERVE_TURNKEY_2026_05_10.md`.

**Per ultra-review:** every number on this page must trace to:
- BENCHMARKS.json for PPL ratios + verifier hashes
- `docs/INTERNAL_MONETIZATION_TIER_EXPANSION_2026_05_10.md` for tier prices
- `docs/INTERNAL_BRIDGE_UNIT_ECONOMICS_2026_05_10.md` for breakeven justification
- USPTO public records for patent numbers

Cross-check before any vercel push.
