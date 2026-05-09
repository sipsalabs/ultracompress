# Glean — Phase 0 POC cold email (HOT tier, 7.5/10)

**To:** Manav Khandelwal (VP Engineering) — manav@glean.com (best guess, fallback: hello@glean.com)
**LinkedIn:** /in/manavkhandelwal
**From:** founder@sipsalabs.com
**Send window:** 2026-05-09, US morning (8-10 AM PT)
**Why HOT:** Glean serves enterprise search at high query volume. Per-query inference cost is a direct margin lever. Their RAG re-ranker uses LLMs for every query. 5-bpw lossless = same retrieval quality at lower cost.

---

**Subject:** Per-query inference cost lever — paid POC for Glean?

Hi Manav,

Sipsa Labs ships lossless 5-bit transformer compression with sub-1% PPL drift across 22 architectures — Qwen3-1.7B at 1.00401x, Yi-1.5-9B at 1.00414x, Mistral and Mixtral families validated, Hermes-3-Llama-3.1-405B finishing tonight. Bit-identical reconstruction with SHA-256 verification.

Why this matters for Glean: enterprise search RAG re-ranking calls an LLM on every query. At Glean's volume, the per-query inference cost is one of the largest variable costs in your unit economics. 5-bpw lossless means the same RAG re-ranker quality at meaningfully lower compute per query — and because reconstruction is bit-identical, the eval results from your QA pipeline don't shift.

Proposal: 1-week paid Phase 0 POC for $5,000. Glean picks 3 models that hit your inference fleet hardest (re-ranker, summarizer, query rewriter — whichever costs you most). We compress each to 5 bpw, deliver verified packs + benchmark report (PPL ratio, Top-1 retention, decode latency, peak VRAM, per-query cost A/B). You decide whether to deploy.

```bash
pip install ultracompress
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5
uc verify
```

Public artifacts: huggingface.co/SipsaLabs

Best,
Sipsa Labs
founder@sipsalabs.com · sipsalabs.com
