# 05 — NASA SBIR / HPSC (heads-up before Phase I filing)

**To:** `sbir-sttr@nasa.gov` (program office, primary)
**Cc:** `sbir@nasa.gov` if returned undeliverable; LinkedIn intro to NASA HPSC project office (Goddard / JPL) parallel as warm path
**From:** `sipsalabs@gmail.com`
**Send window:** today/tomorrow, US business hours (program office is M-F)

> Pre-filing courtesy heads-up. Not a substitute for the formal NSPIRES submission. Per Sip's no-pre-filing-disclosure rule, no specific compression internals or numbers beyond what's already public on HuggingFace / PyPI.

---

**Subject:** Heads-up — Sipsa Labs filing Phase I imminently against ENABLE.2.S26B (HPSC), STREAM-HPSC project

Hello,

Brief courtesy heads-up. Sipsa Labs Inc. (Delaware C-corp, Stripe Atlas-incorporated this week, SAM.gov registration in flight) intends to file an SBIR Phase I proposal against **Subtopic ENABLE.2.S26B — High Performance Onboard Computing** in the next available appendix window. Project name: **STREAM-HPSC** (Streaming Lossless Transformer Compression for HPSC-class Onboard Inference).

Why we believe the topic fit is exact: as of this week, our open-source pipeline (`pip install ultracompress`) has end-to-end-validated lossless 5-bit compression across 11 model architectures, including a 405B-parameter dense transformer (Hermes-3-Llama-3.1-405B) compressed on a single 32 GB consumer-class GPU — direct analog to the HPSC compute envelope. Two public HuggingFace artifacts already pass third-party `uc verify` on download (`SipsaLabs/qwen3-1.7b-uc-v3-bpw5`, `SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5`). Patent posture: 2 USPTO provisionals filed (Apr 25, 2026), 5 additional filing tomorrow.

Two questions before the formal submission:

1. Is there a NASA technologist working on HPSC inference infrastructure who would have 15 minutes for a recorded technical preview before we lock the proposal? Letter-of-support framing not required — pre-submission alignment only.
2. Confirm the next Appendix C release window if known, so the timing is right.

Happy to send the public technical brief (`docs/NASA_SBIR_BRIEF_2026_05_07.md` equivalent) on request.

Thank you for the time.

Missipssa Ounnar
Founder, Sipsa Labs Inc.
sipsalabs.com / github.com/sipsalabs/ultracompress
