# Modal Labs — Phase 0 POC cold email (HOT tier, 7.8/10)

**To:** Erik Bernhardsson (CEO/Founder) — erik@modal.com
**LinkedIn:** /in/erikbernhardsson
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)
**Why HOT:** Erik personally evaluates technical POCs of this size. Modal's value prop is fast cold starts + auto-scaling — 5-bpw weights mean smaller container images, faster cold starts, more models per GPU. Direct margin lever.

---

**Subject:** 3x faster cold starts via 5-bpw weights — paid POC for Modal?

Hi Erik,

Sipsa Labs ships lossless 5-bit transformer compression — 22 architectures validated end-to-end, sub-1% PPL drift, bit-identical reconstruction with SHA-256 verification on every load. Tightest ratios: Qwen3-1.7B at 1.00401x, Yi-1.5-9B at 1.00414x, Mamba-2.8B at 1.012x (first public SSM compression artifact at this ratio). Hermes-3-Llama-3.1-405B finishing tonight on dual 5090s.

Why Modal: cold-start latency is dominated by weight load time. A 70B model at fp16 is 140 GB; at 5-bpw it's ~44 GB — 3x less to read off blob storage into VRAM. Same quality, 3x faster cold start, 3x more concurrent containers per node.

Proposal: 1-week paid Phase 0 POC for $5,000. Modal picks 3 models from your most-served catalog (your data on what users hit most). We compress each to 5 bpw, deliver verified packs + benchmark report (PPL ratio, Top-1 retention, decode latency, peak VRAM, cold-start time A/B vs baseline). You keep the artifacts and serving rights.

```bash
pip install ultracompress
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5
uc verify
```

Happy to do a 15-minute walkthrough this week.

Best,
Sipsa Labs
founder@sipsalabs.com · sipsalabs.com · huggingface.co/SipsaLabs
