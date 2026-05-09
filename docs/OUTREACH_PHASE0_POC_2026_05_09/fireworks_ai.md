# Fireworks AI — Phase 0 POC cold email

**To:** contact@fireworks.ai (fallback: Lin Qiao, CEO — via LinkedIn DM /in/linqiao)
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)

---

**Subject:** Paid POC — lossless 5 bpw compression for Fireworks inference serving

Hi,

Sipsa Labs builds lossless model compression. 20+ architectures validated at 5 bits per weight, sub-1% PPL drift. Tightest: Phi-3-mini at 1.00262x, Qwen3-1.7B at 1.00401x, Yi-1.5-9B at 1.00414x. Today we shipped Hermes-3-Llama-3.1-405B compressed to a single 32 GB consumer GPU with bit-exact SHA-256 verified reconstruction. Public artifacts: huggingface.co/SipsaLabs

Fireworks optimizes inference at the serving layer. Compressing model weights to 5 bpw at sub-1% drift means fitting more concurrent models per GPU and reducing per-token memory bandwidth -- the binding constraint at scale.

Proposal: 1-week paid Phase 0 POC for $5,000. You choose 3 models from your serving fleet. We compress each to 5 bpw, deliver verified packs with full benchmarks (PPL ratio, Top-1 retention, decode latency, peak VRAM). You keep the artifacts and all serving rights.

Three-command repro:

    pip install ultracompress
    hf download SipsaLabs/<model>-uc-v3-bpw5
    uc verify

Worth a 15-minute call this week?

Sipsa Labs · founder@sipsalabs.com · sipsalabs.com
