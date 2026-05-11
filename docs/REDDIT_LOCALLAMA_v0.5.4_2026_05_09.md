# Reddit r/LocalLLaMA post — v0.5.4 — 2026-05-09

**Status:** draft for Sip to submit manually. Nothing posted automatically.

---

## Posting strategy

**Best window: weekday 4-6 PM ET (1-3 PM PT).** r/LocalLLaMA's most active technical discussion happens late-afternoon US weekdays as EU readers tail off and US/PT readers get off work. Avoid weekends — lower comment density on this sub.

**Self-post (text), not link.** Reddit downranks link posts in technical subs. Code block + table render properly in the body.

**Engage in the first 60 minutes.** Reddit ranking weighs early comments + OP replies heavily. Be at the keyboard for the first hour.

---

## Title (Reddit title field — keep under 300 chars)

> UltraCompress 0.5.4 — bit-identical 5-bit packs for 22 architectures, with a one-command repro harness. Re-verify any pack in 3 commands.

(140 chars. Implies reproducibility, names the format, names the count. No marketing words.)

Backup titles if the first feels long:

- `UltraCompress 0.5.4 on PyPI — lossless 5-bit packs + uc bench reproducibility (Qwen3-14B 1.00403x, Mixtral-8x7B 1.00368x)`
- `Lossless 5-bit transformer compression with built-in repro harness — UltraCompress 0.5.4 (PyPI + HF)`

---

## Body

`pip install ultracompress==0.5.4` is live. The new release ships `uc bench`, a one-command harness that measures TTFT, tokens/sec, and peak VRAM on any UC-packed model — same numbers, your hardware, no notebook trust.

What "lossless" means here, precisely: the weights you reconstruct from the pack are **bit-identical, byte-for-byte, to the weights the trainer evaluated during compression**. Reconstruction is closed-form:

```
W_reconstructed = scalar_dequantize(codes, scale) + low_rank_overlay
```

Stored: per-row k-means scalar grid (5 bpw codes), per-block fp32 absmax (block_size=64), low-rank low-rank correction `(U, V, alpha)`. SHA-256 over the reconstructed tensor bytes matches at write time and `uc verify` re-checks it on the consumer side. This is **not** a "lossless against bf16" claim — the 5-bit scalar quantization + low-rank correction is the lossy step. The lossless property is between the trainer's compressed weights and your reconstructed weights, which is what regulated-industry deploys actually need.

Many of the architectures we've validated land sub-1% PPL drift vs FP16 teacher (FineWeb-edu held-out tail, no calibration overlap):

| Model | Class | PPL ratio | Notes |
|---|---|---|---|
| Phi-3-mini-4k | 3.8B dense | **1.00262x** | seq_len=128 (flagged — not 1024) |
| Mixtral-8x7B-v0.1 | 47B MoE | **1.00368x** | best MoE we've measured |
| Qwen3-1.7B-Base | 1.7B dense | **1.00401x** | small-decoder record |
| Qwen3-14B | 14B dense | **1.00403x** | ties small-decoder at 14B class |
| Yi-1.5-9B | 9B dense | **1.00414x** | >8B record |
| Qwen3-8B | 8B dense | **1.00440x** | 8B-class record |

22 architectures validated end-to-end so far. Full ladder: `https://huggingface.co/SipsaLabs`.

Re-verify any pack in 3 commands on a single consumer GPU:

```
pip install -U ultracompress
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 --local-dir ./qwen3-1.7b-base
uc verify ./qwen3-1.7b-base
```

Expected: `VERIFY: PASS — pack format integrity confirmed; lossless reconstruction guaranteed.`

Then `uc bench ./qwen3-1.7b-base` on your hardware to get TTFT / tokens-per-second / peak VRAM in your environment.

Honest scope: single-GPU streaming inference today (no tensor-parallel kernels yet, re-uses PyTorch matmul, no custom CUDA). MoE PPL eval still in flight for a few of the bigger packs (Hermes-3-405B is the headline one — pack is on disk and verifying, HF upload + PPL eval running). Sub-3 bpw still hits the documented Qwen3-fragility wall — not claiming we beat that yet.

USPTO provisionals 64/049,511 and 64/049,517 filed 2026-04-25. Apache-2.0 CLI; compressed weights under a research-evaluation license. File issues if anything doesn't reproduce — that's the most useful thing you could do.

— Sipsa Labs

---

## Prepared answers (paste-ready for top comments)

**Q: How does this compare to AWQ / GPTQ / HQQ / bitsandbytes-NF4?**

> The functional difference isn't the PPL number (those formats are competitive at 4-bit); it's **reproducibility of the eval-to-deploy pipeline**. AWQ/GPTQ/HQQ/NF4 reconstruct via dequant kernels with implementation freedom — different CUDA versions, `torch_dtype` defaults, or per-channel scale rounding paths produce slightly different `Wq` at the customer's machine vs the trainer's. UltraCompress's reconstruction is closed-form over fp32 metadata, so SHA-256 over reconstructed bytes matches deterministically. For audited deploys (regulated industries) where "production model behaves bit-exactly the same as the eval model" is the compliance question, that property is the value. We're not claiming faster matmul — we re-use PyTorch.

**Q: What's the speed penalty / inference throughput?**

> No custom CUDA yet. Reconstruction is `scalar_dequantize(codes) * absmax + alpha * low_rank(U, V)` materialized to bf16 then standard PyTorch matmul. So throughput is roughly bf16 baseline minus the per-layer reconstruction overhead — meaningful on prefill, near-zero on decode after the first token. `uc bench` measures it on your hardware so you don't have to take our word; numbers vary materially by GPU and seq_len. Custom kernel for fused dequant-matmul is on the roadmap but not in 0.5.4.

**Q: Are the HF artifacts gated?**

> No. All packs at `huggingface.co/SipsaLabs` are public download. Pack format is documented in the repo. The CLI is Apache-2.0 — you can read every line of the reconstruction code. The compressed weights themselves are released under a research-evaluation license (commercial use requires a separate agreement) but the format is open and you can pack your own models with the CLI.

**Q: Why 5-bit and not 4-bit? Most of the open ecosystem standardized on 4-bit.**

> 5-bit is where the bit-identical-reconstruction-plus-sub-1%-PPL-drift property holds across most dense architectures we've tested. At 4-bit the low-rank correction can't carry enough signal on most architectures to stay under 1% on the hardest layers (`k_proj` outliers on Qwen3 are the canonical pain point — refuted negative result documented in our repo). 4-bit lossless is on the roadmap as per-Linear-class adaptive bpw research; not shipping yet because we don't have an apples-to-apples positive result. We catalogue the failures alongside the wins.

**Q: Phi-3-mini 1.00262x with seq_len=128 — isn't that cherry-picked? Most people report seq_len=1024 or 2048.**

> Fair flag. The Phi-3-mini number is at seq_len=128, not 1024 — we're transparent about it in the table above. Other records on the table (Qwen3, Yi, Mixtral, Qwen3-14B, Qwen3-8B) are at seq_len=1024. We're re-running Phi-3-mini at seq_len=1024 and will update; the early indication is the ratio holds within a few hundredths but we won't quote a number until the run is clean. The honest framing is "small-decoder record at seq_len=128" until we have the matched-condition number.

**Q: How does the pack format actually work? Where's the spec?**

> Repo: `github.com/sipsalabs/ultracompress`. Per-tensor metadata sidecar (fp32 absmax, U, V, alpha, grid centroids), 5-bpw packed codes in the main blob, SHA-256 over reconstructed bytes in the manifest. The reconstruction expression in the body of this post is the entire spec — there is no hidden state. If you want the long-form: `docs/HF_MODEL_CARD_HERMES_3_405B.md` walks through it on a real 405B pack.

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
