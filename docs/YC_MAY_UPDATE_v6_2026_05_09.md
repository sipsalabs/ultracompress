# YC May Update v6 — 2026-05-09

**For submission to YC S26 application dashboard. ~250 words. Company-voice only.**

---

**Subject:** Sipsa Labs progress update — week 2 of S26 in-review window

Hi YC team,

Quick progress note from Sipsa Labs (UltraCompress — lossless 5-bit transformer compression).

**Shipped this week:**
- **22 architectures validated end-to-end at 5 bits per weight**, all sub-1% PPL drift against bf16 baseline. Full matrix at huggingface.co/SipsaLabs and benchmarks JSON at github.com/sipsalabs/ultracompress/blob/main/docs/BENCHMARKS_2026_05_08.json.
- **New tightest published ratios:** Qwen3-1.7B-Base 1.00401x (small-decoder record), Yi-1.5-9B 1.00414x (>8B parameters record at 8.8B), Phi-3-mini-instruct 1.00262x (caveat: seq_len=128).
- **Hermes-3-Llama-3.1-405B** finished compression overnight (~13h compute on dual RTX 5090), packed to v3 format (251 GB), HF upload in flight under retry-watchdog. Largest dense model compressed by Sipsa to date — single 32 GB GPU streaming inference path.
- **First public SSM compression artifact:** Mamba-2.8B at 1.012x PPL ratio. Direct technical alignment with the Mamba-3 ICLR paper authors who shipped in March.
- **v3 pack format** (mathematically lossless, SHA-256 verified, bit-identical reconstruction) shipped on PyPI as `ultracompress==0.5.3` plus `uc verify-org` + `uc status` CLI subcommands.

**Research discipline:** 13 honest negative results documented in HONEST_NEGATIVE_RESULTS_2026_05_08.md (per-Linear adaptive bpw v1 refuted apples-to-apples; rank/train_steps saturation surfaced; V18-C correction hits a real wall in deep layers). v3 cure (rank-redistribution at constant total budget) lands on Qwen3-1.7B-Base today; v4-D Multi-Pass cascade auto-fires after.

**Customer side:** Phase 0 POC offering ($5,000 / 1 week / 3 customer-picked models) drafted into 9 named outreach emails to Together AI (Mamba-3 author co-signed), Fireworks AI, Ollama, Modal, Replicate, Mistral, Glean, Harvey AI, Palantir. Sending today. Realistic conversion math: 1-2 signed POCs from this batch.

**Patents:** 64/049,511 + 64/049,517 filed April 25; 5-provisional batch filing this weekend ($325 micro-entity).

**Operations:** Stripe Atlas EIN at day 5 of 7-day window. Once EIN lands: NASA SBIR Phase 1 ($256K) + AFWERX Phase 1 ($75K) submissions fire (drafts ready).

Best,
The Sipsa Labs team
founder@sipsalabs.com · sipsalabs.com · huggingface.co/SipsaLabs · github.com/sipsalabs/ultracompress
