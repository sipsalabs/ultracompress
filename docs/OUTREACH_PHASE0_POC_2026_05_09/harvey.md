# Harvey AI — Phase 0 POC cold email (HOT tier, 7.0/10)

**To:** Engineering@harvey.ai (general) or via Winston Weinberg / Gabriel Pereyra LinkedIn
**LinkedIn (CTO):** /in/gabrielpereyra
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)
**Why HOT:** Harvey serves legal AI to AmLaw 100 firms. Bit-identical reproducibility is a compliance requirement — opposing counsel can re-run the model and verify outputs. Lossless 5-bpw is a fundamentally different deliverable than AWQ/GPTQ "close enough." Strongest reproducibility-angle prospect of the set.

---

**Subject:** Bit-identical legal-AI inference at 5 bpw — paid POC for Harvey?

Hi,

Sipsa Labs ships lossless 5-bit transformer compression. The "lossless" matters in legal AI because it means bit-identical reconstruction from compressed weights — every load gives the same forward pass, every prompt produces the same tokens, every retrieval re-ranks identically. Each pack ships with a SHA-256 manifest that opposing counsel could cryptographically verify.

22 architectures validated end-to-end at sub-1% PPL drift — Qwen3-1.7B at 1.00401x, Yi-1.5-9B at 1.00414x, Mistral / Mixtral / Llama-3 / Phi-3 families, Hermes-3-Llama-3.1-405B finishing tonight on dual 5090s.

Why this matters for Harvey: regulated-industry inference (legal, healthcare, finance) increasingly cares about reproducibility — depositions reference prior outputs, regulators ask "would the model give the same answer next week," AmLaw 100 partners hate "the AI was confident on Tuesday but uncertain on Thursday." Standard 4-bit quantization (AWQ/GPTQ) loses bit-identicality. We don't.

Proposal: 1-week paid Phase 0 POC for $5,000. Harvey picks 3 models from your serving stack. We compress each to 5 bpw, deliver verified packs + benchmark report (PPL ratio, Top-1 retention, decode latency, peak VRAM) + a SHA-256 cryptographic manifest your compliance team could include in client deliverables.

```bash
pip install ultracompress
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5
uc verify
```

Best,
Sipsa Labs
founder@sipsalabs.com · sipsalabs.com · huggingface.co/SipsaLabs
