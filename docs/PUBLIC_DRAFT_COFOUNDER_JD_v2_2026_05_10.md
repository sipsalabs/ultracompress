# Cofounder, Sipsa Labs — Inference Systems

**Status:** PUBLIC draft. Awaiting Sip eyeball before paste to `sipsalabs.com/careers/cofounder`.
**Replaces:** `COFOUNDER_JD_LANDING_PAGE_2026_05_04.md` (v1). Keeps v1 in place for diff review; do not delete.
**Charter compliance:** No recipe details. No internal track names. No founder personal info beyond the public `founder@` channel.

---

**Role:** Cofounder. Inference systems lead. Equity partner.
**Location:** Remote. US tax-resident strongly preferred. Visa support is uncertain at this stage; if you need sponsorship, raise it on the first call.
**Compensation:** 5–15% common stock from the option pool, four-year vest, one-year cliff, double-trigger acceleration on change of control. Salary deferred until first $10K MRR or seed close, then $130–180K base in line with stage and reservoir of cash on hand.
**Decision speed:** Two video calls plus one shared work session, then offer or polite decline within 14 days.
**Apply:** `founder@sipsalabs.com`, subject `Cofounder — [your name]`.

---

## What Sipsa Labs is

Sipsa Labs ships infrastructure for serving and storing large neural networks at consumer-GPU economics. The codec — `ultracompress` on PyPI, BUSL-1.1 from v0.6+ with a free-for-sub-$1M-ARR Additional Use Grant — turns the weight matrices of any transformer into a compact pack with a per-tensor low-rank correction overlay. The artifact is verifiable: anyone with a 32 GB GPU can run the three-line ritual and confirm a SHA-256 match against the manifest.

```
pip install ultracompress
hf download SipsaLabs/<model>
uc verify SipsaLabs/<model>
```

What that ritual proved this month:

- **Hermes-3-Llama-3.1-405B**, a 405-billion-parameter language model, reconstructed at **PPL ratio 1.0066×** versus the bf16 baseline on a streaming per-layer eval, FineWeb-edu held-out tail, n=50, seq_len=1024, seed 42. The compressed pack reloads inside 32 GB of VRAM on a single consumer GPU. To the best of our knowledge this is the first lossless 5-bit reconstruction of a 405B model that runs end-to-end on a 32 GB consumer GPU.
- **22 architectures** validated end-to-end at five bits per weight — Llama, Mistral, Qwen3, Phi, Yi, Mamba (yes, the SSM works), Mixtral, Hermes, OLMo, SmolLM, TinyLlama, plus more. PPL ratios across the matrix sit between **1.00262× and 1.0125×** per the verified records (1.0066× on the Hermes-3-405B headline). Full table on `huggingface.co/SipsaLabs`.
- **USPTO 64/049,511 + 64/049,517** filed 2026-04-25 (provisionals; supplement filed in May). Continuations through 2027.
- **`sipsa-inference`** OpenAI-API-compatible serving endpoint shipped Day 15+ of build. Live at `sipsalabs.com/inference`. Three-command try-it: set `OPENAI_BASE_URL`, point the official `openai` SDK at it, get the same SSE chunks and the same usage envelope you'd get from any other vendor — at roughly half the headline price on the frontier slot.

YC S26 application: **In Review**. Decision June 5, 2026.

The thesis: customers will pay a premium for inference where the served weights are *the* weights — the ones the lab actually trained — and they can prove it themselves with a SHA-256 receipt. Compression density is the cost moat. The verify ritual is the trust moat.

---

## What you'd do as cofounder

You own the systems and serving half of the company. The codec exists; the engineering work to turn one home box into a real serving substrate does not.

- **Run `sipsa-inference` like a real API.** Today: one home box (dual RTX 5090) on a Cloudflare Tunnel, OpenAI-compatible. Your job over the first 90 days: continuous batching at scale, request-level metering, p50/p99 TTFT and tok/sec dashboards customers can see, statuspage that doesn't lie, on-call rotation that's literally just you and Sip.
- **Bring up Lambda H100s on the trigger ladder.** $500 MRR fires Phase-2 reserved H100. $5K MRR fires Phase-3 multi-region. The math is in `INTERNAL_BRIDGE_UNIT_ECONOMICS_2026_05_10.md`; you live it.
- **vLLM (or equivalent) integration of the v3 pack format.** Batched serving against the compressed weight format with the per-tensor correction overlay. Initially a Python integration; over time, fused dequant + matmul + correction kernels where the bandwidth math justifies it.
- **Cross-platform binary distribution.** Linux first (the serving target), then macOS / Windows wheels for the local-developer install. ARM Mac via Metal where it makes sense.
- **Customer onboarding for the IaaS segment.** When Together / Fireworks / Lambda / CoreWeave / a smaller IaaS buyer asks "show me the cost reduction in our stack" — you sit with their inference team, instrument the path, prove the per-token economics, hand off to deploy. The verify ritual is your demo. The SHA-256 receipt is your differentiator.
- **First two systems hires after seed close.** You define the role, source candidates, run interviews. The first reports report to you.
- **Anchor research-direction tradeoffs to what is shippable.** Sip drives mechanism work; you push back when "interesting" and "in production this quarter" are different things.

---

## What Sip does (so you know what's covered)

- Codec research and the lab notebook discipline that produces the headline numbers.
- USPTO patent prosecution. Provisional and supplement drafting. Continuation strategy.
- Customer-facing aerospace and defense conversations where the lossless framing is the wedge.
- Investor conversations and board-level relationships.
- Public communications: the GitHub org, the HuggingFace org, the Twitter and LinkedIn accounts, the docs site, the launch threads.

You and Sip together cover the full stack. Neither of us is doing the other's job. Neither of us is hiring an employee.

---

## What you look like

**Required.**

- **High-throughput Python API at scale.** You have personally shipped a service that survived production load. You know what it looks like when continuous batching saves you and what it looks like when it bites you. FastAPI, Starlette, or equivalent.
- **vLLM or equivalent inference engine experience.** vLLM, TensorRT-LLM, llama.cpp, MLC-LLM, ExLlamaV3, SGLang, LMDeploy. You can read the scheduler code; you can debug a memory leak in PagedAttention; you have an opinion about prefix caching strategies.
- **Linux DevOps.** Systemd units, journald, nftables, NVIDIA driver pinning, CUDA toolkit version drift. You are not surprised when a kernel update breaks driver ABI; you have a runbook.
- **AWS or GCP or Azure at production grade.** Or Lambda Labs / CoreWeave / Modal at the same level. Spinning up an H100 on demand, putting a load balancer in front of it, and having Cloudflare DNS route by latency is something you have done before, not something you would google.
- **Customer-facing comfort.** You can sit in a room with a CTO, draw on a whiteboard, and not flinch when they ask hard questions about the economics. If the idea of being on a sales call physically tightens your chest, this isn't the role.
- **Comfort with pre-funding ambiguity.** Two people for the next 6–12 months. No org chart. No middle layer. The role expands as we hire, not before.

**Nice-to-have.**

- Prior YC company, especially one that scaled past Series A.
- Prior solo-founder or 1–2-person startup experience — you know what the lonely months feel like.
- OSS contributions to AI infrastructure: vLLM, llama.cpp, TGI, ExLlamaV3, ggml, candle, MLC-LLM, SGLang, Triton kernels, NCCL bringup. Public commit history with merged PRs counts for more than a private resume.
- Latency-optimization war stories you can tell from memory: a TTFT regression you bisected, a tail-latency event you fixed, a network event you traced to BBR tuning.
- Quantization-specific knowledge: AWQ, GPTQ, HQQ, QTIP / EXL3 internals, GGUF format. You don't have to invent it; you do have to read the existing literature critically.

**Anti-fit signals — please self-screen.**

- Background is pure ML research, no production ops. Sip handles the research; this is the other half.
- "I'm interested in prompt engineering" or "I want to fine-tune models." Not the role.
- Looking for a full market salary on day one. The cash story is honest below; if it doesn't work for your runway, it's not the right time for us.
- Need a 30-person org chart to function. The first 90 days are two laptops and a shared Linear board.

---

## How the conversation goes

1. **First call (45 min).** Sip walks the codec and the serving path. You ask hard questions. We see if the technical fit is real.
2. **Second call (90 min).** Shared screen, real Sipsa work — pick a thing on the serving roadmap and pair on it for an hour. We see how we work under technical pressure.
3. **Reference checks both directions.** You talk to anyone Sip has worked with. Sip talks to anyone you've worked with.
4. **Offer or polite decline within 14 days of first call.** No drawn-out interview funnel.

---

## Compensation, honestly

**Equity.** 5–15% common from the option pool, depending on background and stage at signing. Four-year vest, one-year cliff, monthly thereafter. Double-trigger acceleration on change of control (founder-favorable industry standard). Sip retains majority and primary control through Series B at minimum.

**Cash.** Deferred until **first $10K MRR** OR seed close (whichever comes first). At that point: $130K–180K base, calibrated to your background and the cash position at that moment. Pre-trigger, we'll cover specific living-expense edges if your runway is tight — talk to Sip on the first call about the structure that works.

**Post-seed.** Market cofounder cash for a research-led infra company at our stage, plus equity refresh as the option pool reloads.

If you need market cash from day one, this is not the right role. If you can sustain three to six months of deferred salary for an outsized swing, it might be.

---

## What you get out of this

- Equity in a company whose moat is two USPTO provisionals on a codec that has shipped end-to-end on 22 architectures and a 405B-class flagship.
- Direct partnership with the founder. Two people decide everything together for the first 6+ months.
- The serving substrate for lossless inference at consumer-GPU economics. Whoever ships this first defines it.
- Public attribution: GitHub commits under your name, patent inventor credit on continuations you co-invent, conference talks, paper authorship on the ICLR submission in flight.
- First-mover position in a market that has not yet been priced.

---

## What you should know before applying

- **Sip is solo.** No previous cofounder. No team to inherit. You're the second person.
- **Operating discipline is real.** Six months of progress on roughly five thousand dollars of operating cost. That's how the cash story works at all.
- **Honest about negative results.** The lab notebook publishes the misses as prominently as the wins. If you can't hold the line on intellectual honesty under customer or investor pressure, this is not the right partnership.
- **Patent timeline matters.** The USPTO supplement filed in May; continuations land Q3–Q4 2026. The pace of engineering and the pace of prosecution are linked.
- **YC June 5 is a fork.** If we get in, the YC check unlocks the cofounder hire and the bridge. If we don't, we close the seed on the strength of revenue and the public release. The first-conversations window is open right now in either case.

---

## To apply

Send to `founder@sipsalabs.com`:

1. **Subject line:** `Cofounder — [your full name]`.
2. **GitHub profile** so Sip can read your code.
3. **One PR or merged commit** you're proud of, plus 2–3 sentences on what it does and why you made the design choice you made.
4. **One paragraph: "why this and not the next Together / Anthropic / OpenAI job."** Honest answer, not a pitch.
5. **Honest framing on cash runway** so we can match expectations on the deferral structure.
6. **One question for Sip** — anything from "deepest technical risk in the next 12 months" to "what does week-12 look like."

Response within 72 hours, every time. If we're not the right fit, we'll say so directly and quickly. If we are, we'll get on the first video call within seven days.

---

*Sipsa Labs is research-led infrastructure. The first cofounder is the most important hire of the company's existence. We are taking that seriously.*

— Sip, Founder
`founder@sipsalabs.com` · `sipsalabs.com` · `github.com/sipsalabs/ultracompress`
