# Ollama — Phase 0 POC cold email (HOT tier, 8.0/10)

**To:** Jeff Morgan (CEO) — jeff@ollama.com (best guess, fallback: hello@ollama.com)
**LinkedIn:** /in/jmorganca
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)
**Why HOT:** Ollama is a community multiplier — single integration propagates to millions of users. Their model catalog is the biggest distribution channel for compressed weights outside Hugging Face. They actively curate quality and would care about lossless 5-bpw vs the AWQ/GPTQ artifacts they ship today.

---

**Subject:** Lossless 5-bpw weights for the Ollama catalog — paid POC?

Hi Jeff,

Sipsa Labs just shipped lossless 5-bit transformer compression with sub-1% PPL drift across 22 architectures (Llama, Qwen, Mistral, Phi, Mixtral 8x7B/8x22B, Phi-3.5-MoE, Mamba SSM at 1.012x, Hermes-3-Llama-3.1-405B finishing tonight). What "lossless" means here: every pack ships with SHA-256 verification — bit-identical reconstruction every load, not "close enough to fp16." That matters for a catalog like Ollama's where users hit the same model from a thousand different machines and expect the same tokens back.

Tightest published ratios: Qwen3-1.7B at 1.00401x, Yi-1.5-9B at 1.00414x.

Proposal: 1-week paid Phase 0 POC for $5,000. Ollama picks 3 of your most-pulled models (whatever's hot in your tracking right now). We deliver compressed packs + benchmark report (PPL ratio, Top-1 retention, decode latency, peak VRAM) + verification one-liner. You decide whether to make them available alongside the existing AWQ/GGUF builds.

```bash
pip install ultracompress
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5
uc verify
```

Public artifacts: huggingface.co/SipsaLabs · code: github.com/sipsalabs/ultracompress

Best,
Sipsa Labs
founder@sipsalabs.com · sipsalabs.com
