# r/LocalLLaMA cross-post — UltraCompress 0.5.5 — v6 — 2026-05-10

**Status:** PUBLIC draft. Sip submits manually after Show HN fires Mon and after eyeball pass. Cross-post window: Tue 2026-05-12 9:00 AM PT.
**Charter class:** PUBLIC. Voice: technical, no marketing-speak, no emojis except where they replace a noun.
**Difference vs Show HN draft:** more GPU + how-to-run, more honest comparison to AWQ/GPTQ/EXL3/HQQ, less business model. r/LocalLLaMA cares about the verifier ritual and the OSS license posture; they don't want the founder narrative.

---

## Title (~10 words, claim-first)

> Lossless 5-bit Hermes-3-405B on a single 5090 — verifier ships SHA-256 manifest

(81 chars, 11 words. Names the artifact, names the constraint that makes it interesting, names the compliance-grade property.)

Backups:

- `Lossless 5-bit 405B on a single 32 GB GPU — UltraCompress 0.5.5 on PyPI` (74 chars)
- `UltraCompress 0.5.5 — Hermes-3-405B at 1.0066x PPL on a single 5090, verifier open` (84 chars)

---

## Body (target 400-600 words; current = 553)

`pip install ultracompress==0.5.5` is live. Reproducer for the 405B headline is three commands on a single 32 GB consumer GPU:

```
pip install -U ultracompress
hf download SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5 --local-dir ./hermes-3-405b
uc verify ./hermes-3-405b
```

Expected: `VERIFY: PASS — pack format integrity confirmed; lossless reconstruction guaranteed.`

Then `uc bench ./hermes-3-405b` measures TTFT, decode tok/sec, and peak VRAM in your environment so you don't have to take our word.

**The headline:** Hermes-3-Llama-3.1-405B compressed to ~5 bpw, peak VRAM during streaming reconstruction = 27.33 GB on RTX 5090, PPL ratio = **1.0066×** (n=50, seq_len=1024, FineWeb-edu held-out tail, seed=42). Baseline is `bf16_streaming_per_layer_from_hf_cache` — same per-layer streaming procedure, un-quantized bf16 weights from the upstream NousResearch HF cache. Disclosed honestly: this is the only baseline that fits 405B on a 32 GB consumer GPU. NOT equivalent to a multi-GPU full-model bf16 single-shot eval, and we don't claim it is.

**What "lossless" means here, precisely:** the weights you reconstruct from the pack are bit-identical, byte-for-byte, to the weights the trainer evaluated during compression. Reconstruction is closed-form:

```
W_reconstructed = scalar_dequantize(codes, scale) + low_rank_overlay
```

The 5-bit scalar quantization + low-rank correction is the lossy step (~1% PPL drift). The lossless property is between the trainer's compressed weights and your reconstructed weights — which is the property regulated-industry deploys actually need (production model bit-exactly matches eval model). SHA-256 over reconstructed bytes is in the manifest; `uc verify` rechecks it on your hardware.

**Honest comparison to the rest of the stack:**

| Codec | Bit-identical client-side reconstruction? | PPL drift class | Honest read |
|---|---|---|---|
| AWQ | No — dequant kernel impl-dependent | ~0.5-1.5% at 4-bit | Faster matmul today; we re-use PyTorch and lose on TTFT |
| GPTQ | No — same | ~0.5-1.5% at 4-bit | Same — they win speed, we win audit |
| HQQ | No — same | ~0.5-2% at 4-bit | Tight on smaller models; weaker reproducibility story |
| EXL3 | No — bitshift trellis is impl-stable but not SHA-256-verifiable end-to-end | tightest sub-3 bpw of any public codec | EXL3 wins sub-3 bpw on Qwen3-14B (3 bpw / 7% drift); we don't try to compete there yet |
| llama.cpp Q5_K_M | No | ~1-3% | Largest ecosystem; we lose ecosystem, win verifier ritual |
| **UltraCompress 5 bpw** | **Yes (closed-form fp32 metadata)** | 1.0026-1.0200× across 22 archs | Win audit + breadth. No custom CUDA kernel yet. |

We are NOT claiming UltraCompress beats the others on inference throughput. We re-use PyTorch matmul. The win is **reproducibility of the eval-to-deploy pipeline** — for any deploy where compliance asks "is the model in production the same model you evaluated?", SHA-256 over reconstructed weight bytes is the answer.

**22 architectures validated end-to-end so far:**

| Model | Class | PPL ratio |
|---|---|---|
| Mixtral-8x7B-v0.1 | 47B MoE | 1.00368× |
| Qwen3-1.7B-Base | 1.7B dense | 1.00401× |
| Qwen3-14B | 14B dense | 1.00403× |
| Qwen3-8B | 8B dense | 1.00440× |
| Yi-1.5-9B | 8.8B dense | 1.00414× |
| Mistral-7B-v0.3 | 7.2B dense | 1.0100× |
| Mamba-2.8B | 2.8B SSM | 1.0119× |
| Llama-3.1-8B | 8B dense | 1.0125× |
| **Hermes-3-Llama-3.1-405B** | **405B dense** | **1.0066×** |

Full ladder: `huggingface.co/SipsaLabs`. Failures catalogued in `docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md` — sub-3 bpw still hits the documented Qwen3-fragility wall (arxiv 2505.02214); we don't beat that yet.

**License + patent posture (the part that usually bothers this sub):**

- CLI is **Apache-2.0 on the `legacy/0.5.x` branch — perpetual, no rug-pull**
- v0.6+ moves to BUSL-1.1 with an Additional Use Grant: **free for individuals, research, sub-$1M ARR commercial use**, auto-converts to Apache 2.0 four years after each release
- USPTO provisionals 64/049,511 + 64/049,517 filed 2026-04-25
- Pack format is documented in the repo; you can pack your own models with the open CLI

This is the Sentry/HashiCorp pattern, not MongoDB SSPL. If you're an individual / researcher / small company, nothing about your usage changes. If your company is above $1M ARR shipping the codec in production, that's the conversation we want.

**The ask:** if you're running on a 5090 (or any 32 GB consumer GPU), can someone confirm the Hermes-3-405B pack reproduces on your end? Run `uc verify ./hermes-3-405b` and reply with the JSON output. Confirmation from a third independent box is the most useful thing you could contribute.

— Sipsa Labs

---

## Prepared answers (paste-ready when comments land)

**Q: Why 5-bit and not 4-bit? Most of the open ecosystem standardized on 4-bit.**

> 5-bit is where the bit-identical-reconstruction-plus-sub-1%-PPL-drift property holds across most dense architectures we've tested. At 4-bit the low-rank correction can't carry enough signal on the hardest layers (`k_proj` outliers on Qwen3 are the canonical pain point — refuted negative result documented in our repo). 4-bit lossless is on the roadmap as per-Linear-class adaptive bpw research; not shipping yet because we don't have an apples-to-apples positive result. We catalogue failures alongside wins.

**Q: 1.0066× on 405B is great, but your 8B/14B numbers (1.0044×/1.0040×) are tighter. What gives?**

> Depth. 126 transformer layers in 405B vs 28-40 in the 8B/14B class — roughly 3-4× the layer count, so per-layer residual quant noise compounds further through the residual stream. Codec degrades gracefully rather than collapsing — that's the claim — but graceful degradation is not "no degradation". Pre-stated hypothesis was 1.005-1.015×; landed at 1.0066×, in the predicted band. Lab-notebook entry pre-locked the band before the run finished.

**Q: 405B compression on consumer hardware — how long?**

> ~14 hours on dual RTX 5090s, streaming per-layer so the full bf16 model never lives in memory at once. 27.33 GB peak VRAM during reconstruction. Eval baseline run was 7.7h on cuda:1 (16.86 GB peak). Both fit on a single 32 GB GPU.

**Q: Are the HF artifacts gated?**

> Verifier-tier artifacts (≤14B): public download. Frontier artifacts (405B, 235B, 70B, 141B): HF `gated:manual` lead-capture form. The pack format is documented in the repo. The CLI is Apache-2.0 on the `legacy/0.5.x` branch — you can read every line of the reconstruction code, and you can pack your own models with it. The compressed frontier weights themselves are released under a research-evaluation license; commercial use requires a separate agreement.

**Q: How does this compare to llama.cpp / vLLM / TGI?**

> Different layer of the stack. UltraCompress is a codec — it produces a compressed weight artifact. llama.cpp / vLLM / TGI are inference engines that consume weights. Today we re-use PyTorch matmul on top of the reconstructed weights, which means we don't compete with vLLM continuous batching on throughput. Custom dequant-fused matmul kernel is roadmap, not in 0.5.5. The right mental model: we're complementary to the inference engine of your choice, not a replacement.

**Q: BUSL is not OSI-approved open source.**

> Correct — that's the OSI position and we agree with the framing. If "OSS or nothing" is your filter, the `legacy/0.5.x` branch is Apache-2.0 perpetual; that's the version that's open by your definition. v0.6+ is source-available under BUSL with an Additional Use Grant that covers the actual users of the OSI definition (individuals, researchers, small companies) and asks the venture-backed >$1M ARR companies to pay. The 4-year auto-Apache clause means even v0.6 becomes OSI-compliant on its own clock. Same posture Sentry adopted in 2019. We're trying not to be MongoDB.

**Q: Reproducer didn't work on my box — what's the issue?**

> File an issue at `github.com/sipsalabs/ultracompress` with the full traceback and `uc --version`. We test against PyTorch 2.6+, CUDA 12.4+, and Python 3.10+. The Mamba-2.8b pack needs `mamba-ssm` installed separately. Bug reports are the most useful contribution; that's the support contract.

---

## Ultra-review checklist (run before submit)

- [ ] Title under 300 chars Reddit limit (84)
- [ ] Self-post, not link post (Reddit downranks link posts in technical subs)
- [ ] All PPL ratios trace to `docs/BENCHMARKS_2026_05_09.json`
- [ ] No recipe (rank, lr, train_steps, calibration set) — only seed=42 disclosed for eval reproducibility
- [ ] No personal info, no internal Track names
- [ ] Honest comparison table doesn't oversell — explicitly says where we lose (TTFT, sub-3 bpw, ecosystem)
- [ ] BUSL framing leads with the Additional Use Grant (the part that addresses r/LocalLLaMA's OSS-purity reflex)
- [ ] Specific ask at end (third-box reproduction request)
- [ ] Sip explicit eyeball — "ship it" in chat before submit
- [ ] Append to `docs/PUBLIC_ACTIONS_LOG_2026_05.md` after fire

---

## Notes for Sip (delete before submission)

- Body is 553 words (within the 400-600 target)
- Comparison table includes EXL3 honestly: they win sub-3 bpw, we don't try to compete there. r/LocalLLaMA respects this; pretending we beat EXL3 at 3 bpw on Qwen3-14B would burn us on day 1.
- The "specific ask" is the third-box reproduction — Reddit treats specific asks much better than generic "check it out". A confirmed third-box `uc verify` JSON in a top reply is reproducibility credit on a silver platter.
- Patent paragraph leads with the BUSL Additional Use Grant scope rather than the patent number — patent numbers in the lead read defensive on this sub.
- Don't cross-post to r/MachineLearning the same day — different audience, different voice draft needed.

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
