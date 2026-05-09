# Together AI — Phase 0 POC cold email

**To:** hello@together.ai (fallback: Vipul Ved Prakash, CEO — via LinkedIn DM /in/vipulvedprakash)
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)

---

**Subject:** Paid POC — lossless 5 bpw compression for Together's inference fleet

Hi,

Sipsa Labs does lossless model compression. 20+ architectures validated at 5 bits per weight, sub-1% PPL drift. Tightest ratios: Qwen3-1.7B at 1.00401x, Yi-1.5-9B at 1.00414x, Phi-3-mini at 1.00262x. Today we released Hermes-3-Llama-3.1-405B compressed to a single 32 GB consumer GPU. Every pack is bit-exact verifiable. Public artifacts: huggingface.co/SipsaLabs

Together serves inference at scale. Fitting the same model in fewer GPUs at the same quality means direct margin improvement on every served token.

Proposal: a 1-week paid Phase 0 POC for $5,000. You choose 3 models from your serving catalog. We compress each to 5 bpw, deliver verified packs plus a full benchmark report (PPL ratio, Top-1 retention, decode latency, peak VRAM). You keep the artifacts and serving rights.

Reproducible in three commands:

    pip install ultracompress
    hf download SipsaLabs/<model>-uc-v3-bpw5
    uc verify

Happy to do a 15-minute walkthrough this week.

Sipsa Labs · founder@sipsalabs.com · sipsalabs.com
