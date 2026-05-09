# CoreWeave — Phase 0 POC cold email

**To:** info@coreweave.com (fallback: Mike Intrator, CEO — via LinkedIn DM /in/mikeintrator)
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)

---

**Subject:** Paid POC — fit 3x more model classes per GPU for CoreWeave inference customers

Hi,

Sipsa Labs builds lossless model compression. We have 20+ architectures publicly validated at 5 bits per weight with sub-1% perplexity drift, including the first public SSM compression artifact (Mamba-2.8B at 1.012x). Today we shipped Hermes-3-Llama-3.1-405B compressed to a single 32 GB consumer GPU -- bit-exact SHA-256 verified reconstruction. Public artifacts: huggingface.co/SipsaLabs

For CoreWeave's H100/H200 inference customers, this translates directly to more rentable model classes per chip. A 405B that today requires 8x H100 sharded fits on a single H100 80 GB at the same quality envelope.

Proposal: 1-week paid Phase 0 POC for $5,000. You pick 3 models from your customer fleet. We deliver compressed packs with full benchmarks (PPL ratio, Top-1 retention, decode latency, peak VRAM). You keep the artifacts and all serving rights.

Three-command repro:

    pip install ultracompress
    hf download SipsaLabs/<model>-uc-v3-bpw5
    uc verify

Worth a short call this week?

Sipsa Labs · founder@sipsalabs.com · sipsalabs.com
