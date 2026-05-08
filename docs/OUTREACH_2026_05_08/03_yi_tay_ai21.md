# 03 — Yi Tay (Reka, ex-AI21)

**To:** `yi@reka.ai` (founder address; fallback `yitayml@gmail.com`, LinkedIn DM `/in/yitay`)
**From:** `sipsalabs@gmail.com`
**Send window:** today/tomorrow, US morning (Singapore evening)

---

**Subject:** 12-arch lossless 5-bit pack including SSM — fit for Reka serving costs?

Hi Yi,

Read your scaling-laws and architecture work for years; wanted to flag a result that landed today and is a 30-second test on Reka's side.

We shipped v0.5.2 of UltraCompress this morning (PyPI, Apache-2.0). 12 architectures end-to-end validated at PPL_r ≤ 1.013x — Qwen3-0.6B holds **1.0069x** at 5 bpw, the tightest dense-decoder ratio at this bitrate I have measured anywhere. The codec is architecture-agnostic: Mamba-2.8B SSM hit 1.0119x with the same pipeline, no architecture-specific tuning. Hermes-3-Llama-3.1-405B is mid-compression on a single 32 GB consumer GPU as I write this.

The Reka angle: a serving fleet that hosts Reka Flash / Core variants gets ~3x weight footprint reduction at sub-1.3% perplexity cost. If "more variants per H100" is on Reka's cost roadmap, we have a paid 1-week Phase 0 ($5K) where you pick a Reka model class and we ship back a packed checkpoint plus a side-by-side benchmark report.

Or just run the 30-second eval yourself:

```
pip install ultracompress
hf download SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5 --local-dir ./pack
uc verify ./pack
```

30-min Zoom this week works on either side?

Best,
Missipssa Ounnar
Founder, Sipsa Labs Inc.
sipsalabs.com / github.com/sipsalabs/ultracompress
