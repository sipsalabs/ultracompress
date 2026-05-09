# Together AI — Phase 0 POC cold email (REFINED with Mamba-3 hook)

**To:** Dan Fu, SVP Engineering — try `dfu@together.ai` first, fallback `hello@together.ai`
**LinkedIn:** /in/danfu / Stanford CS, co-author Mamba-3 (ICLR 2026, March 17 2026)
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)
**Priority:** TOP — highest likelihood-to-convert (9.2/10 per ICP research)

**Why this hook works:** Dan Fu co-signed the Mamba-3 ICLR 2026 paper March 17. Sipsa Labs has Mamba-2.8B publicly validated at 5 bpw, sub-1% PPL drift — first published SSM compression artifact at this ratio. Direct technical alignment with a senior decision-maker, once-in-a-quarter timing.

---

**Subject:** Mamba SSM compressed to 5 bpw, sub-1% PPL drift — paid POC for Together?

Hi Dan,

Saw you co-signed the Mamba-3 ICLR paper in March — congrats on the work. Reason this is relevant: Sipsa Labs hit sub-1% PPL drift at 5 bits per weight on Mamba-2.8B, the first public SSM compression artifact at this ratio. Same substrate (per-row scalar quant + rank-32 low-rank correction) holds across 20 transformer architectures — Qwen3-1.7B at 1.00401x, Yi-1.5-9B at 1.00414x, Hermes-3-Llama-3.1-405B finishing tonight on dual 5090s.

What we're actually selling isn't the compressed bytes — it's the **signed PPL bound**. Every pack ships with SHA-256 verification plus a documented PPL ratio against bf16 baseline on the same calibration sample. Auditable.

Proposal: a 1-week paid Phase 0 POC for $5,000. Together picks 3 models from your serving catalog (Mamba-3 itself if you want a stress test). We compress each to 5 bpw, deliver verified packs + benchmark report (PPL ratio, Top-1 retention, decode latency, peak VRAM). You keep the artifacts and serving rights.

Reproducible in three commands:

```bash
pip install ultracompress
hf download SipsaLabs/mamba-2.8b-uc-v3-bpw5
uc verify
```

Happy to do a 15-minute walkthrough this week.

Sipsa Labs · founder@sipsalabs.com · sipsalabs.com · huggingface.co/SipsaLabs
