# SipsaLabs HF Gating Policy

**Effective:** 2026-05-10
**Owner:** Sip (founder@sipsalabs.com)

## TL;DR

| Tier | Param count | HF status | License | Who pays |
|---|---|---|---|---|
| **Free** | < 10B | Public download (no gate) | BUSL-1.1 Additional Use Grant | Free for sub-$1M ARR + individuals + research |
| **Gated free** | 10B – 100B | `gated: manual` (request access) | BUSL-1.1 Additional Use Grant + manual review | Free for sub-$1M ARR + research; manual approval required |
| **Paid** | 100B+ | `gated: manual` + commercial license required | Commercial license | Above $1M ARR or production deployment |

## Rationale

- **Sub-10B (free tier):** small public artifacts are the proof-of-concept. Anyone can `pip install ultracompress && uc verify` against them. Acquisition channel for credibility + inbound leads.
- **10B – 100B (gated free):** compute-intensive artifacts. Manual gate is friction that filters serious users. Email captures = pipeline. Free use grant unchanged.
- **100B+ (paid):** flagship artifacts (Hermes-3-405B, Qwen3-235B-A22B). Production use requires commercial conversation.

## Current state — as of 2026-05-10 night

All 18 SipsaLabs/ HF model artifacts ≥10B are now `gated: manual`. The 14 freshly-gated repos this session:

**Streaming-bpw5 variants:**
- SipsaLabs/qwen3-14b-streaming-bpw5
- SipsaLabs/qwen3-32b-streaming-bpw5
- SipsaLabs/qwen2.5-72b-streaming-bpw5
- SipsaLabs/hermes-3-llama-3.1-405b-streaming-bpw5
- SipsaLabs/llama-3.1-70b-streaming-bpw5
- SipsaLabs/qwen3-235b-a22b-streaming-bpw5
- SipsaLabs/mixtral-8x22b-v0.1-streaming-bpw5
- SipsaLabs/mixtral-8x7b-v0.1-streaming-bpw5
- SipsaLabs/phi-3.5-moe-instruct-streaming-bpw5

**UC-v3-bpw5 variants:**
- SipsaLabs/qwen3-14b-uc-v3-bpw5
- SipsaLabs/mixtral-8x7b-v0.1-uc-v3-bpw5
- SipsaLabs/phi-3.5-moe-uc-v3-bpw5
- SipsaLabs/yi-1.5-9b-uc-v3-bpw5 (9B — borderline, included to standardize MoE/large family)
- SipsaLabs/phi-3.5-moe-instruct-uc-v3-bpw5

**Pre-existing gated (4 repos):**
- SipsaLabs/llama-3.1-70b-uc-v3-bpw5
- SipsaLabs/mixtral-8x22b-v0.1-uc-v3-bpw5
- SipsaLabs/qwen3-235b-a22b-uc-v3-bpw5
- SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5

## Sub-10B (intentionally NOT gated — free tier proof points)

22 SipsaLabs/ repos under 10B remain public download. These include the **Mistral-7B-v0.3 1.00548× verified record** (the recent breakthrough) — kept public as the proof-point that the codec works on a stubborn architecture. The PATENT protects the method (USPTO 64/049,511 + 64/049,517); the artifact itself is free for verification.

## Approval workflow

When someone clicks "Request access" on a gated repo:
1. HF emails Sip at the org's notification address (default: sipsalabs@gmail.com or founder@sipsalabs.com per HF account settings)
2. Sip reviews the use case (1-2 sentences from the requester)
3. Sip approves or rejects via HF Hub UI (one click)
4. Approved users get download access; rejected users get a polite decline + invite to email founder@sipsalabs.com for commercial conversation

**Auto-approval policy** (Sip can flip later):
- Approve: academic emails, individual researchers, sub-$1M ARR companies, open-source maintainers
- Manual review: corporate emails from Anthropic, OpenAI, Google, Meta, Microsoft, AWS, Together, Lambda, CoreWeave, Groq, Mistral, Cohere, AI21, Cerebras, Anyscale (commercial competitors — likely lead to enterprise license conversation)
- Decline + invite: vague use cases, no employer info

## Future tier additions

- **Mistral-7B-v0.3 v10 1.00548× HF artifact** stays public until either: (a) a competitor uses it as a benchmark to claim parity, or (b) Sip decides the breakthrough merits gating to capture leads. Default: stay public for now (this is the headline marketing win).
- **DeepSeek-V3 685B** when published: paid tier (gated + commercial license required).
- **All future >10B artifacts**: default to `gated: manual` at upload time.

## Sip-action checklist (manual)

- [ ] Verify HF notification email destination (Settings → Notifications → Org SipsaLabs → "Receive emails for access requests")
- [ ] Update HF org bio to mention gating policy: "Production-tier (>10B) artifacts require manual approval. Email founder@sipsalabs.com for commercial use."
- [ ] Add `/pricing` link target if not yet on sipsalabs.com (currently goes to TBD)
- [ ] If any false-positive rejection causes a credible user to bounce, retroactively approve + apologize
- [ ] Track requests-to-approval ratio in a Notion sheet — leads pipeline

## Why not auto-approve everything?

Auto-approve = no friction = no lead capture. The gate IS the marketing channel. Every "request access" click is a person whose email + stated use case enters our pipeline. At zero marketing spend, this is the cheapest qualified-lead funnel we can build.
