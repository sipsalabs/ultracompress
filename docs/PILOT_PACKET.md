<!-- Classification: PUBLIC | Owner: Sipsa Labs | Last Reviewed: 2026-04-27 | External Sharing: allowed -->

# UltraCompress design-partner pilot

**For**: chip vendors · OEMs · AI inference platforms · edge-cloud operators · robotics + automotive teams

**From**: Sipsa Labs, Inc. (Delaware C-Corp in formation; sipsalabs.com)

**Filed IP**: U.S. provisional patent applications (filed April 2026, patent pending)

---

## The problem you have

Modern transformer language models have outgrown the hardware most of the world actually runs them on:

- **Phone-class and automotive deployments** are memory-constrained, often forcing teams to ship smaller local models than the product would otherwise want
- **In-vehicle inference** is latency-bound on memory budgets, not capability-bound on the model itself
- **Inference platforms** at scale are GPU-memory-bound on margins
- **Model registries** absorb storage + egress costs that scale linearly with fleet size

The methods that exist (bitsandbytes, GPTQ, AWQ, HQQ) hit a wall at 4 bits per weight. Below 4 bpw, **in our 6-model benchmark cohort**, every public method falls off a quality cliff.

## What we deliver

### Weight-level compression method (patent pending) — shipping now

Sub-3-bits-per-weight on a 6-model head-to-head cohort. **30% smaller than bitsandbytes NF4 at equivalent retention.** Zero catastrophic failures across the cohort — the only public method at this compression frontier with that property in the cohort we tested.

### Architectural compression method (patent pending) — v0.2 (Q3 2026)

Architectural compression beyond the published academic frontier. Combined with the weight-level method, the strongest end-to-end ratio we've measured for transformer language models in our cohort. Public-safe architectural-compression evidence at [docs/evidence/matrix.md](evidence/matrix.md).

### What ships under a pilot

- Pre-compressed model artifacts (rolling release on Hugging Face Hub through April–May 2026 — let's discuss which architecture families fit your stack)
- A reproducibility manifest (SHA-256 of every input + deterministic seed)
- A reference loader you can drop into your runtime
- A model card describing the per-task agreement / retention envelope
- Direct technical support from the founder during the pilot window

## Pilot offers

We run two pilot shapes. Both are designed to convert to a recurring license if the technology lands.

### Tier 1 — Compression Assessment ($5,000 · 2-week turnaround)

For teams who want to validate UltraCompress against a **public/open-weight representative model from your stack** before committing to a full deployment pilot.

What you get:

- Sipsa runs the internal reference pipeline and delivers the assessment on a model + benchmark of your choice
- Public-method comparison table: UltraCompress vs your current quantization stack
- Per-task retention curves (T1, T10, T32, T64, T128, T256) on the metrics you care about
- A 30-minute deep-dive call covering methodology, limits, and the v0.2 roadmap
- A written assessment report (10-15 pages) you can take to internal stakeholders

What we need from you:

- The model and the benchmark we should run against
- Two 30-minute calls (kickoff + readout)
- A signed mutual NDA before kickoff

### Tier 2 — Production Deployment Pilot ($15,000–$25,000 · 60-day pilot window)

For teams ready to put UltraCompress into a development or staging deployment surface and measure the production characteristics.

What you get:

- Three pre-compressed model artifacts selected or prepared for your target hardware profile (architectures of your choice)
- Integration support for your inference stack (vLLM, TensorRT-LLM, llama.cpp, custom) — within reason
- Daily Slack / email channel during the 60-day window
- Per-deployment performance dashboard: latency, memory, retention, customer-facing metrics
- A pilot readout deck you can use internally to evaluate go-no-go on a recurring license
- Right of first negotiation on a per-deployment SaaS license at the end of the pilot

What we need from you:

- A scoped deployment surface (one product or one internal use case is plenty)
- A technical lead on your side for daily cadence
- A signed mutual NDA before kickoff
- A signed pilot agreement (we provide a template)

## What's in scope vs out

| In scope (pilot) | Out of scope (separate license required) |
|---|---|
| Public / open-weight model assessment + benchmark | Compression of your private/proprietary models (requires NDA + commercial pilot terms) |
| Methodology deep-dives under NDA | Per-device royalty / OEM licensing structure (separate term sheet, scoped per customer) |
| Bug fixes + integration help | Custom new compression methods (separate research engagement) |
| 60-day production pilot window | Permanent production deployment (recurring license required) |

## Patent + commercial licensing path post-pilot

Both pilot tiers convert to one of three commercial license shapes (or you can walk away with the assessment report).

| License shape | Pricing posture | Best fit |
|---|---|---|
| **Per-deployment SaaS** | Starts at design-partner-friendly entry pricing; scales with deployment surface | Single product / single customer |
| **Multi-deployment SaaS** | Tiered annual; structured with the customer based on internal use-case count | Enterprise with multiple internal use cases |
| **OEM / per-device royalty** | Custom volume-tiered structure (annual license, per-device royalty, or hybrid); includes patent license | Chip vendors and device OEMs |

Patent license terms are bundled into the commercial license. Audit rights and standard commercial license terms apply, with redlines worked on a 2-3 week cycle. Specific bands are scoped per customer under NDA.

## Get started

Email **founder@sipsalabs.com** with:

- Which tier (assessment or pilot) is most useful right now
- The architecture family / specific model you want benchmarked
- Your timing window
- Your preferred call structure (1-on-1 founder, technical team, exec sponsor)

We respond same-day during US business hours and target a kickoff call within 5 business days.

---

*UltraCompress v0.1 alpha shipped 2026-04-25. Pre-compressed reference models release throughout April–May 2026. Architectural compression support and `uc compress` ship in v0.2 (Q3 2026), gated on patent prosecution timing.*

*The CLI is Apache 2.0. The pre-compressed model artifacts are licensed separately (research-free or commercial-paid). The compression methodology is patent pending.*

*sipsalabs.com · github.com/sipsalabs/ultracompress · huggingface.co/sipsalabs*

Codec internals and training procedure are patent-pending.
