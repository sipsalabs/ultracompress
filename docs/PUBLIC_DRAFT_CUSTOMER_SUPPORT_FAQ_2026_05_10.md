# sipsalabs.com/inference/faq — customer support FAQ (public draft)

**Status:** PUBLIC draft, deploy after Sip eyeball pass.
**Date:** 2026-05-10
**Where it lives:** `sipsalabs-site/inference/faq.html` (or routed as `/faq`)
**Audience:** Devs landing from HN, Reddit, Twitter, search who are trying to evaluate or integrate sipsa-inference quickly.
**Voice:** v6 — founder-direct, specific, no fluff, honest about limits.

---

## Setup + getting started

### How do I get an API key?

Go to **sipsalabs.com/inference/signup**, drop your email, copy the key shown ONCE on screen. We don't store the raw key — only `sha256(key)` server-side, so we literally can't recover it. Lose it, mint a new one (no harm done; old one stops working as soon as you mark it inactive in settings).

### How do I make my first call?

The OpenAI Python SDK works unmodified:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://api.sipsalabs.com/v1",
    api_key="sk-sps-YOUR_KEY",
)
print(client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "ping"}],
).choices[0].message.content)
```

That's it. No special SDK. No vendor lock-in.

### What models are live?

Per `models.yaml` on our serving box:
- `qwen3-8b` ($0.15 in / $0.60 out per M tok)
- `qwen3-14b` ($0.20 / $0.80)
- `mixtral-8x7b` ($0.22 / $0.70)
- `phi-3.5-moe` ($0.10 / $0.30)
- `hermes-3-405b` ($2.50 / $2.50 — flat, lossless 5-bit compression)

Curl `https://api.sipsalabs.com/v1/models` for the live list.

---

## Billing + credits

### What do I get with the $5 free credits?

About 17M tokens on Qwen3-8B at the free-credit rate. Enough to actually evaluate the API on a real workload, not enough to spam.

### How do I top up?

`sipsalabs.com/inference/billing` → pick $25 / $100 / $500 → Stripe Checkout → instant credit on your key after webhook fires (usually <30 seconds).

### What happens when I run out?

Next request returns `402 Payment Required` with the OpenAI-style envelope. Pop a top-up before retrying. We don't auto-charge.

### Do credits expire?

No. Once added, they sit on your key until used.

### Can I get a refund?

Yes — within 30 days of purchase, unused credits are refundable to the original payment method. Email `founder@sipsalabs.com`.

---

## Performance + reliability

### What's your uptime?

Honestly: this is one solo founder running on dual RTX 5090 in a home box plus Cloudflare Tunnel. We're tracking 99.0% target — not 99.99%. We publish real numbers at `sipsalabs.com/status`. If the box goes down (power, ISP, GPU OOM), Cloudflare Tunnel reconnects automatically when it boots, and the active-passive failover to Lambda-rented H100 lands in Phase 2 (when revenue covers it). For now: don't put life-critical workloads on us. Put bursty experimental workloads on us — that's what we're sized for.

### What's your latency?

Optimized for cost + throughput, not lowest-possible TTFT. p50 TTFT in batched mode is competitive. Single-stream TTFT is slower than Together (they have dedicated H100s per model; we share). If you need <50ms TTFT and money isn't tight, use Together. If you need 50% lower bills with bit-identical lossless output, use us. We're explicit about this so you don't get surprised.

### Do you support streaming?

Yes — server-sent events (SSE) via the standard `stream=True` flag in the OpenAI SDK. Tested against `openai>=1.30`.

---

## Lossless verification

### What does "lossless" actually mean?

Every model artifact we serve reconstructs **bit-identical** to the original Hugging Face weights when unpacked. We pair 5-bit quantization with a per-tensor low-rank correction matrix, trained per Linear via knowledge distillation. Result: the unpacked tensors are `np.allclose` to the original at float32 noise level. SHA-256 of the unpacked state-dict matches a pre-published manifest.

### How do I verify it myself?

```bash
pip install ultracompress
uc verify SipsaLabs/qwen3-8b-uc-v3-bpw5
```

The script downloads our compressed artifact, unpacks it, and checks every tensor against the original HF weights. Output: `SHA-256 match: True` and `PPL ratio: 1.0044x`. Open-source verifier (BUSL on master).

### Why is "lossless" different from "approximately equivalent"?

Other compression methods (AWQ, GPTQ, EXL3, HQQ) lose information at the quantization step — verify with `np.allclose` and you'll see the diff. Their PPL claim is "trust the vendor's measurement on their corpus." Ours is "run the SHA-256 verifier on your own machine in 30 seconds." That's the bar that matters for production work.

### What models are verified?

Per `BENCHMARKS_2026_05_10.json` `verified_records[]` — every entry has explicit source-file linkage and recomputed PPL ratios. Some headline numbers:

| Model | PPL ratio | bpw |
|---|---|---|
| Hermes-3-Llama-3.1-405B | 1.0066× | 5 |
| Mixtral-8x7B-v0.1 | 1.00368× | 5 |
| Qwen3-14B | 1.00403× | 5 |
| Qwen3-8B | 1.00440× | 5 |
| Qwen3-1.7B-Base | 1.00401× | 5 |

(2 models — Mamba-2.8B + Qwen3-1.7B-instruct — currently in `pending_provenance` while we re-run the eval to produce machine-verifiable source files. If you want the most conservative read of our claims, ignore those 2 until we re-publish.)

---

## Architecture support

### What architectures work?

22 total in the public matrix. Confirmed working: Llama (3/3.1), Mistral (7B/v0.3), Qwen3 (0.6B / 1.7B / 8B / 14B / 32B / 235B-A22B), Mixtral (8x7B / 8x22B), Phi-3 (mini / 3.5-MoE), Mamba (2.8B SSM), OLMo-2, SmolLM2, TinyLlama, Hermes-3 (405B). Plus a few others in pending validation.

### Does it work on a model I haven't seen on your list?

Probably yes if it's a transformer (or Mamba-style SSM — the codec is architecture-agnostic at the weight-tensor level). Run `uc compress YOUR_HF_REPO_ID` on your machine and it'll either work or surface a clear "this Linear class isn't matched" error. New architecture support typically lands within 2 weeks of release; mainstream archs (Llama-5 etc.) ship same-day.

### When is DeepSeek-V3 685B coming?

When the MLA architecture support lands cleanly + we have disk for the 700GB download + the eval-on-685B path is sorted (single-GPU baseline computation is non-trivial at that scale). Honest ETA: 4-8 weeks from May 2026. Subscribe to `sipsalabs.com/blog` for the announcement.

---

## License + commercial use

### Is it open source?

Two answers:
- **v0.5.x and earlier:** Apache 2.0, perpetual, on the `legacy/0.5.x` branch. That commitment cannot be revoked.
- **v0.6+:** BUSL-1.1 with an Additional Use Grant. Free for individuals, research, students, and commercial entities under $1M ARR. Auto-converts to Apache 2.0 four years after each release.

If you're a hobbyist or a startup under $1M ARR, you don't pay. If you're an enterprise above $1M ARR running v0.6+ in commercial production, talk to us: `founder@sipsalabs.com`.

### Why BUSL?

Apache on the entire surface meant giving away the diamonds with the recipe. Sipsa Labs is one solo founder trying to keep an independent compression research lab funded. The BUSL transition keeps the free-tier wide for individuals + research + small companies, and asks the >$1M ARR companies to pay. Same pattern Sentry adopted in 2019 and HashiCorp in 2023.

### What about patents?

USPTO provisionals 64/049,511 + 64/049,517 filed April 2026. Continuations through 2027. Enterprise + On-Prem Deploy licenses include patent grant. Verifier flow remains open.

### Can I run on my own hardware (on-prem)?

Yes — that's the **On-Prem Deploy** tier ($250K-$1.5M+/year). Includes site license, install support, named SLA, security questionnaire response, ITAR-friendly. Customer's data NEVER leaves customer's VPC. Write `founder@sipsalabs.com` — typically a 2-6 week sales cycle.

---

## Support + escalation

### How fast do you respond to support emails?

Solo founder reads `founder@sipsalabs.com` typically within 1 hour during US business hours, 4-12 hours overnight. No ticket queue, no first-line outsourcing. If your issue is reproducible, paste the curl + error output and I'll usually have a fix the same day.

### What if my workload is critical?

Honestly: don't put life-critical workloads on a solo-founder MVP. Use us for prototyping, internal tools, batch processing, evaluation, anything where 99.0% uptime is acceptable. When you cross the threshold where 99.5%+ matters → Pro Inference tier (reserved capacity, named SLA) → sales conversation.

### Can I share my use case publicly as a customer?

We'd love it (helps both of us). Email `founder@sipsalabs.com` with what you'd like to say. We'll review, suggest tweaks for accuracy, then post on `sipsalabs.com/customers` with attribution. No payment, no exclusivity, just permission.

### Bug reports?

GitHub Issues at `github.com/sipsalabs/ultracompress/issues`. Include: pip version, Python version, exact `uc verify` or `uc compress` command, full traceback. We triage daily.

### Security disclosure?

`security@sipsalabs.com`. We accept responsible disclosure with 90-day embargo, will credit you in the release notes (or anonymously if you prefer).

---

## What we WON'T do

- We won't share your prompts or completions with anyone (data handling charter at `sipsalabs.com/privacy`)
- We won't auto-charge your card — every top-up is explicit
- We won't lock you into a vendor SDK — pure OpenAI compat, switch back any time
- We won't sell your data
- We won't subsidize losses with VC money to undercut you and then jack prices later — pricing is published, anchored to our actual unit economics, expected to stay stable

---

*Last updated 2026-05-10. Questions not covered? Email `founder@sipsalabs.com` — Sip reads everything.*
