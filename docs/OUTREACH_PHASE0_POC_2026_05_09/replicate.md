# Replicate — Phase 0 POC cold email

**To:** hello@replicate.com (fallback: Ben Firshman, CEO — via LinkedIn DM /in/bfirsh or Twitter @bfirsh)
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)

---

**Subject:** Paid POC — lossless 5 bpw compression for Replicate's model hosting

Hi,

Sipsa Labs does lossless model compression. 20+ architectures validated at 5 bits per weight, sub-1% perplexity drift -- Qwen3-1.7B at 1.00401x, Yi-1.5-9B at 1.00414x, Mamba-2.8B SSM at 1.012x. Today: Hermes-3-Llama-3.1-405B running on a single 32 GB consumer GPU with bit-exact verified reconstruction. Public artifacts: huggingface.co/SipsaLabs

For Replicate, smaller resident footprints mean faster cold starts, lower GPU-seconds per prediction, and the ability to host larger models on cheaper hardware -- all without measurable quality loss.

Proposal: 1-week paid Phase 0 POC for $5,000. You pick 3 models from your most-requested catalog. We compress to 5 bpw and deliver verified packs plus benchmarks (PPL ratio, Top-1 retention, decode latency, peak VRAM). You keep the artifacts and serving rights.

Three-command repro:

    pip install ultracompress
    hf download SipsaLabs/<model>-uc-v3-bpw5
    uc verify

Happy to do a quick walkthrough this week.

Sipsa Labs · founder@sipsalabs.com · sipsalabs.com
