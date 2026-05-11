# Show HN v0.5.4 — 2026-05-09

**Status:** draft for Sip to submit manually. Nothing posted automatically.

---

## Posting strategy

**Best window: weekday Tuesday-Thursday 8-10 AM ET (5-7 AM PT).**

- HN's `/newest` page traffic peaks 8-11 AM ET on weekdays as the US wakes up and EU is still active. Submissions in this window get the highest first-hour upvote velocity, which is the signal HN ranking weighs most heavily.
- Tuesday-Thursday avoids the Monday backlog effect (everyone submits Monday and dilutes the page) and the Friday afternoon / weekend tail-off.
- 4-hour rule: post needs meaningful upvotes in the first 4 hours to reach front page. Be at the keyboard.

**One shot.** HN spam filter penalises duplicate Show HN submissions from the same domain stack within 30 days. Don't delete and re-post.

**First-comment-from-OP within 30 seconds.** HN ranking weighs `comments per hour` as a freshness signal, and the first reply anchors the technical depth before drive-by skeptics frame it. The body below goes as the first reply.

---

## Title (≤ 80 chars including "Show HN: ")

> Show HN: UltraCompress 0.5.4 – lossless 5-bit packs with built-in repro harness

Length: **78 chars** including `Show HN: `. Inside HN's hard limit.

Backups if the en-dash glyph or length is rejected:

- `Show HN: UltraCompress 0.5.4 - bit-identical 5-bit LLM packs + uc bench` (70 chars)
- `Show HN: UltraCompress - lossless 5-bit transformer compression on consumer GPUs` (80 chars)

---

## URL field

> `https://pypi.org/project/ultracompress/0.5.4/`

(Direct PyPI link is the right submit target — front-loads the install command. HN auto-displays the domain. HF org `https://huggingface.co/SipsaLabs` is linked from the first comment for the artifacts.)

---

## First-comment body (~350 words, paste as first reply within 30s)

OP here — Sipsa Labs.

We're publishing the lossless 5-bit pack format and the measurement harness as `ultracompress==0.5.4`. Two things make this submission different from prior 4-bit / 5-bit Show HNs:

**(1) Bit-identical reconstruction.** AWQ, GPTQ, HQQ, and bitsandbytes-NF4 all introduce small run-to-run drift between the trainer's quantized eval and the customer's deployed inference (different CUDA versions, `torch_dtype` defaults, per-channel scale rounding paths). UltraCompress reconstruction is closed-form over fp32 metadata: `W_reconstructed = scalar_dequantize(codes, scale) + low_rank_overlay`. SHA-256 over reconstructed bytes matches at pack write time, and `uc verify` re-checks it on your machine. For audited deploys where the compliance question is "does production behave bit-exactly the same as the eval model," that property is the value.

**(2) `uc bench` as a reproducibility primitive.** The artifact ships with the measurement harness. One command on your hardware gives you TTFT, tokens/sec, and peak VRAM. No notebook trust required.

PPL records from this week (FineWeb-edu held-out, no calibration overlap, single RTX 5090):

- Phi-3-mini-4k: 1.00262x (seq_len=128 — flagged transparently, re-running at 1024)
- Mixtral-8x7B-v0.1: 1.00368x (best MoE)
- Qwen3-1.7B-Base: 1.00401x (small-decoder record)
- Qwen3-14B: 1.00403x (ties at 14B class)
- Yi-1.5-9B: 1.00414x (>8B record)
- Qwen3-8B: 1.00440x (8B-class record)

22 architectures validated end-to-end so far; many sit sub-1% PPL drift vs FP16 teacher.

Three commands to re-verify any pack:

    pip install -U ultracompress
    hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 --local-dir ./qwen3-1.7b-base
    uc verify ./qwen3-1.7b-base

**What we're not claiming.** "Lossless" here means bit-identical between trainer's compressed weights and customer's reconstructed weights — not against the bf16 base model (the 5-bit scalar quantization + low-rank correction is the lossy step). Single-GPU streaming inference today; no custom CUDA matmul kernels yet (re-uses PyTorch). MoE PPL eval still in flight for a few packs (Hermes-3-Llama-3.1-405B pack is on disk; HF upload and full-run PPL eval are running, no number to share until clean). Sub-3 bpw still hits the Qwen3-fragility wall.

USPTO provisionals 64/049,511 and 64/049,517 filed 2026-04-25. Apache-2.0 CLI; weights under research-evaluation license. Repo: `github.com/sipsalabs/ultracompress`. HF: `huggingface.co/SipsaLabs`. Site: `sipsalabs.com`.

If something doesn't reproduce on your hardware, file an issue. That's the most useful comment you could leave.

— Sipsa Labs

---

## Prepared answers (paste-ready for top HN comments)

**Q: "Lossless" with PPL ratio 1.004x is a marketing word. Pick one.**

> Fair pushback — the word is doing narrow work and we should defend it precisely. "Lossless" here means **bit-identical, byte-for-byte, between the weights the trainer evaluated during compression and the weights the customer reconstructs from the pack on their machine**. SHA-256 over reconstructed tensor bytes matches deterministically; `uc verify` checks it. That's it. The 1.004x PPL ratio is the lossy step (5-bit scalar quantization + low-rank correction) and it sits between bf16 base and the compressed model — which we are NOT calling lossless. The eval-to-deploy reconstruction is what's lossless. AWQ/GPTQ/HQQ/NF4 don't have that property because their dequant kernels have implementation freedom that produces small `Wq` drift across CUDA versions and `torch_dtype` defaults. For regulated deploys (defense, finance, medical) where the auditor signed off on a specific eval, this is the difference between "compliant" and "needs re-validation."

**Q: How does this compare to AWQ / GPTQ / HQQ at 4-bit on the same models?**

> At 4-bit, AWQ and GPTQ are competitive with us on PPL ratio for most dense architectures (often within 0.5-1%). The functional difference isn't the PPL number — it's the reproducibility property in the previous answer, plus the `uc bench` harness shipping with the artifact. We chose 5-bit because that's where bit-identical reconstruction + sub-1% PPL drift holds across most dense architectures we've tested; 4-bit lossless is on the roadmap as per-Linear-class adaptive-bpw research and not shipping until we have an apples-to-apples positive result. We document our refuted hypotheses in the repo.

**Q: What's the throughput cost? Is this slower than AWQ at inference?**

> Reconstruction materializes weights to bf16 then standard PyTorch matmul — no custom kernel in 0.5.4. So throughput is roughly bf16 minus per-layer reconstruction overhead: meaningful on prefill, near-zero on decode after first token. Versus AWQ with a fused dequant-matmul kernel on a long generation, AWQ wins on tokens/sec. Versus a vanilla HF load of bf16 weights at the same VRAM budget, we win because we fit a much bigger model. The honest framing is: today this is a memory-budget tool with audit-grade reconstruction, not a throughput tool. Fused kernel is on the roadmap. `uc bench` measures both on your hardware.

**Q: How big can you go? You mention 405B — show me the artifact.**

> Hermes-3-Llama-3.1-405B compression is complete locally — 126 layers via per-layer streaming on a single 32 GB consumer GPU, 251 GB pack on disk. `uc verify` passed locally. HF upload and full-run PPL eval are both in flight as of submission time; we deliberately didn't include a 405B PPL number in the body because the eval isn't done. When the upload lands and the eval finishes, the artifact will go up at `huggingface.co/SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5` with the full benchmark JSON. Largest currently live is `mixtral-8x22b-v0.1-uc-v3-bpw5` (100.6 GB pack, 56 .uc files) and `llama-3.1-70b-uc-v3-bpw5` (48.7 GB).

**Q: Apache-2.0 on the CLI but the weights have a research-evaluation license — that's not really open.**

> Correct framing — and we should say it that way rather than dress it up. The CLI (pack format spec, reconstruction code, `uc verify`, `uc bench`) is Apache-2.0; you can pack your own models, ship your own packs under your own license, and audit every byte of the reconstruction path. The pre-packed weights at `huggingface.co/SipsaLabs` are released under research-evaluation terms (commercial use needs a separate agreement) because they represent meaningful compute on our side and are the path to funding the research. If you want fully-open weights, pack your own with the Apache CLI — the format is fully documented and the same on both sides.

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
