# Reddit r/LocalLLaMA post — v0.5.5 — 2026-05-10

**Status:** draft for Sip to submit manually. Nothing posted automatically.

---

## Posting strategy

**Best window: Tue 5/12 8-10 AM ET (per distribution sprint plan).** r/LocalLLaMA's most active technical discussion happens during US weekday morning windows when EU readers are still active and US/PT readers are starting their day. Avoid weekends — lower comment density on this sub.

**Self-post (text), not link.** Reddit downranks link posts in technical subs. Code block + table render properly in the body.

**Engage in the first 60 minutes.** Reddit ranking weighs early comments + OP replies heavily. Be at the keyboard for the first hour.

---

## Title (Reddit title field — keep under 300 chars)

> Hermes-3-Llama-3.1-405B compressed to 5 bpw lossless — 1.0066x PPL ratio, reloads on a single 32 GB consumer GPU. UltraCompress 0.5.5 on PyPI.

(150 chars. Names the artifact, names the bar, names the number, names the constraint that makes it interesting.)

Backup titles if the first feels long:

- `Lossless 5-bit 405B on a single 32 GB GPU — Hermes-3-405B at 1.0066x PPL, repro in 3 commands. UltraCompress 0.5.5.`
- `UltraCompress 0.5.5 — first lossless 5-bit 405B on consumer hardware (Hermes-3-405B 1.0066x), 22 architectures verified.`

---

## Body

`pip install ultracompress==0.5.5` is live.

Headline result: we compressed **Hermes-3-Llama-3.1-405B** to 5 bits per weight and the resulting pack reloads on a single 32 GB consumer GPU (RTX 5090, 27.33 GB peak during reconstruction). PPL ratio against a streaming bf16 per-layer teacher: **1.0066x** (n=50, seq_len=1024, FineWeb-edu held-out tail, seed 42).

To the best of our knowledge this is the first lossless 5-bit compression of a 405B-parameter model that runs end-to-end on a single 32 GB consumer GPU. We can't exhaustively prove the negative — there may be unpublished work in private pipelines — so we'll qualify the claim and let the artifact stand on its own.

What "lossless" means here, precisely: the weights you reconstruct from the pack are **bit-identical, byte-for-byte, to the weights the trainer evaluated during compression**. Reconstruction is closed-form:

```
W_reconstructed = scalar_dequantize(codes, scale) + low_rank_overlay
```

Stored: per-row k-means scalar grid (5 bpw codes), per-block fp32 absmax (block_size=64), low-rank low-rank correction `(U, V, alpha)`. SHA-256 over the reconstructed tensor bytes matches at write time and `uc verify` re-checks it on the consumer side. This is **not** a "lossless against bf16" claim — the 5-bit scalar quantization + low-rank correction is the lossy step. The lossless property is between the trainer's compressed weights and your reconstructed weights, which is the property regulated-industry deploys actually need (production model bit-exactly matches the eval model).

Public verifiable sub-1.010x band so far, all at n=30 or n=50, seq_len=1024, FineWeb-edu held-out tail, same seed:

| Model | Class | PPL ratio | Notes |
|---|---|---|---|
| Mixtral-8x7B-v0.1 | 47B MoE | **1.00368x** | best MoE we've measured |
| Qwen3-1.7B-Base | 1.7B dense | **1.00401x** | small-decoder record |
| Qwen3-14B | 14B dense | **1.00403x** | ties small-decoder at 14B class |
| Qwen3-8B | 8B dense | **1.00440x** | 8B-class record |
| **Hermes-3-Llama-3.1-405B** | **405B dense** | **1.0066x** | first 405B-class lossless on a single 32 GB consumer GPU |

22 architectures validated end-to-end so far. Full ladder: `https://huggingface.co/SipsaLabs`.

Re-verify the 405B pack in 3 commands on a single 32 GB GPU:

```
pip install -U ultracompress
hf download SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5 --local-dir ./hermes-3-405b
uc verify ./hermes-3-405b
```

Expected: `VERIFY: PASS — pack format integrity confirmed; lossless reconstruction guaranteed.`

Then `uc bench ./hermes-3-405b` on your hardware to get TTFT / tokens-per-second / peak VRAM in your environment.

Honest scope: single-GPU streaming inference today (no tensor-parallel kernels yet, re-uses PyTorch matmul, no custom CUDA). The 405B compression itself took ~14 hours on dual RTX 5090s, streaming per-layer so the full bf16 model never lives in memory at once. Sub-3 bpw still hits the documented Qwen3-fragility wall — not claiming we beat that yet. The 1.0066x at 405B sits in the STRONG band (<1.010x) but NOT the EXCELLENT band (<1.005x) the smaller-decoder records hit — depth-driven residual accumulation is real and visible vs the 8B/14B numbers.

USPTO provisionals 64/049,511 and 64/049,517 filed 2026-04-25. Apache-2.0 CLI; compressed weights under a research-evaluation license. File issues if anything doesn't reproduce — that's the most useful thing you could do.

— Sipsa Labs

---

## Prepared answers (paste-ready for top comments)

**Q: How does this compare to AWQ / GPTQ / HQQ / bitsandbytes-NF4?**

> The functional difference isn't the PPL number (those formats are competitive at 4-bit); it's **reproducibility of the eval-to-deploy pipeline**. AWQ/GPTQ/HQQ/NF4 reconstruct via dequant kernels with implementation freedom — different CUDA versions, `torch_dtype` defaults, or per-channel scale rounding paths produce slightly different `Wq` at the customer's machine vs the trainer's. UltraCompress's reconstruction is closed-form over fp32 metadata, so SHA-256 over reconstructed bytes matches deterministically. For audited deploys (regulated industries) where "production model behaves bit-exactly the same as the eval model" is the compliance question, that property is the value. We're not claiming faster matmul — we re-use PyTorch.

**Q: 405B at 1.0066x — what's the apples-to-apples baseline? You can't fit the full bf16 in 32 GB.**

> Correct, you can't — and that's the honest constraint that shapes the eval. The baseline is `bf16_streaming_per_layer_from_hf_cache` — same per-layer streaming reconstruction the compressed run uses, just with the un-quantized bf16 weights from the upstream NousResearch HF cache. Eval ran on cuda:1 RTX 5090 (16.86 GB peak VRAM, 7.7h). The compressed run used the identical streaming procedure (27.33 GB peak, 14.3h). Same eval harness, same n=50 prompts, same seq_len=1024, same FineWeb-edu held-out tail, same seed. This is the only baseline procedure that fits 405B on consumer hardware, and it's the same procedure used to generate the correction overlay training targets during compression — so the ratio is measured against the exact teacher the codec was asked to reproduce. It is NOT the same as a multi-GPU full-model bf16 single-shot eval. Full lab-notebook entry with HYPOTHESIS → MECHANISM → MEASUREMENT → CONCLUSION + honest disclosures is in the GitHub repo at `docs/_HERMES_405B_LAB_NOTEBOOK_ENTRY_FILLED.md`.

**Q: What's the speed penalty / inference throughput?**

> No custom CUDA yet. Reconstruction is `scalar_dequantize(codes) * absmax + alpha * low_rank(U, V)` materialized to bf16 then standard PyTorch matmul. So throughput is roughly bf16 baseline minus the per-layer reconstruction overhead — meaningful on prefill, near-zero on decode after the first token. `uc bench` measures it on your hardware so you don't have to take our word; numbers vary materially by GPU and seq_len. Custom kernel for fused dequant-matmul is on the roadmap but not in 0.5.5.

**Q: Are the HF artifacts gated?**

> No. All packs at `huggingface.co/SipsaLabs` are public download. Pack format is documented in the repo. The CLI is Apache-2.0 — you can read every line of the reconstruction code. The compressed weights themselves are released under a research-evaluation license (commercial use requires a separate agreement) but the format is open and you can pack your own models with the CLI.

**Q: Why 5-bit and not 4-bit? Most of the open ecosystem standardized on 4-bit.**

> 5-bit is where the bit-identical-reconstruction-plus-sub-1%-PPL-drift property holds across most dense architectures we've tested. At 4-bit the low-rank correction can't carry enough signal on most architectures to stay under 1% on the hardest layers (`k_proj` outliers on Qwen3 are the canonical pain point — refuted negative result documented in our repo). 4-bit lossless is on the roadmap as per-Linear-class adaptive bpw research; not shipping yet because we don't have an apples-to-apples positive result. We catalogue the failures alongside the wins.

**Q: 1.0066x is great for 405B but not as tight as your 8B/14B numbers — what gives?**

> Depth. 126 transformer layers accumulate per-layer residual quantization noise through the residual stream. The 8B/14B numbers (1.00440x / 1.00403x) are 28-40 layers; 405B is 126 layers — roughly 3-4× the layer count. The codec degrades gracefully rather than collapsing — that's the claim — but graceful degradation is not "no degradation". Pre-stated hypothesis was 1.005x to 1.015x, landed at 1.0066x, in the predicted band. Lab-notebook entry pre-locked the band before the run finished, which is the honesty we're trying to keep.

**Q: How does the pack format actually work? Where's the spec?**

> Repo: `github.com/sipsalabs/ultracompress`. Per-tensor metadata sidecar (fp32 absmax, U, V, alpha, grid centroids), 5-bpw packed codes in the main blob, SHA-256 over reconstructed bytes in the manifest. The reconstruction expression in the body of this post is the entire spec — there is no hidden state. If you want the long-form: `docs/HF_MODEL_CARD_HERMES_3_405B.md` walks through it on a real 405B pack.

**Q: Phi-3-mini 1.00262x — why isn't that on the table now?**

> It's at seq_len=128, not 1024. We're re-running it at matched conditions (seq_len=1024) before re-claiming it next to the rest. The numbers in the table above are all at seq_len=1024 at the same FineWeb-edu held-out tail with the same seed, so they're comparable to each other. That's why the 405B headline is the lead — it's the cleanest controlled measurement we have at the new scale.

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
