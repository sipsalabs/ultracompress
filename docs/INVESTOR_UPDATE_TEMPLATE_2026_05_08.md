# Investor / Press Update Template — 2026-05-08

> Personalize the greeting, the ask, and any context-specific intro. Keep the body otherwise as-is unless a specific recipient warrants a substitution. Send from `founder@sipsalabs.com` (or `sipsalabs@gmail.com` if SES isn't yet wired). Do **not** add a Co-Author signature. Do **not** insert "world-class" / "revolutionary" copy.

---

**Subject:** Sipsa Labs — May 8 update

Hi [first name],

Quick update on where Sipsa Labs / UltraCompress sits as of tonight.

**Today's headline.** 18 model architectures now validated end-to-end at 5 bits per weight with mathematically lossless reconstruction of W_base. We hit a new tightest-PPL record on Qwen3-1.7B-Base — compressed/baseline PPL ratio of 1.0040x on WikiText-2 (N=30, seq_len=1024, bf16). All 18 packs pass `uc verify` (bit-exact round-trip check).

**In flight tonight.** Hermes-3-Llama-3.1-405B is currently being packed — 72 of 126 decoder layers done at the time of this writing, ETA around midnight MDT. If the eval lands clean, that becomes the first publicly-released lossless 5-bit compression of a 405B-parameter model that runs on a single 32 GB consumer GPU.

**Shipped this week.**
- `ultracompress` v0.5.2 live on PyPI ([pypi.org/project/ultracompress/0.5.2](https://pypi.org/project/ultracompress/0.5.2/))
- 9 public Hugging Face artifacts at [huggingface.co/SipsaLabs](https://huggingface.co/SipsaLabs), all `uc verify` PASS
- [sipsalabs.com](https://sipsalabs.com) refreshed with current stack
- 8 GitHub stars on the open-source decoder (was 0 a week ago — small number, real signal)

**Next 7 days.**
- File the 5-provisional patent batch tomorrow (2026-05-09, $325 USPTO micro-entity)
- Submit NASA HPSC and AFWERX SBIR applications once the Atlas EIN lands
- 5 cold emails out: Tri Dao, Albert Gu, Yi Tay, Lambda Labs, NASA HPSC program office

**Honest framing.** Pre-revenue. Cash-constrained — paying USPTO fees only until YC S26 decision (June 5) or first commercial customer, whichever comes first. YC application is in review. Patents 64/049,511 and 64/049,517 filed 2026-04-25 cover the core methods.

**One ask.** If you know anyone evaluating compression for frontier-model inference — Anthropic, Google, Meta, Cartesia, Together, Lambda, anyone in the cislunar / sovereign-AI / on-prem stack — an intro is worth more than capital right now.

Thanks for reading.

Missipssa
Founder, Sipsa Labs, Inc.
[sipsalabs.com](https://sipsalabs.com) · [github.com/sipsalabs](https://github.com/sipsalabs) · [huggingface.co/SipsaLabs](https://huggingface.co/SipsaLabs)
