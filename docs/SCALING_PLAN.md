# Scaling Plan — proving FRR works beyond Qwen3-1.7B

## Status

- **Today (HEAD):** Verified on Qwen3-1.7B teacher at 311× (h256, 69.64% all-T10) and 734× (h128, 68.00% all-T10). **Not yet proven at other scales or model families.**
- **This document defines the experiments required to defend the claim "FRR works for any transformer."** Results table below is filled in as runs complete.

## What the claim actually requires

A reviewer or acquirer will attack "works on any model" with these three cuts:

1. **Same family, different scale.** If FRR only works at 1.7B hidden=2048, the method might be over-fit to Qwen3-1.7B's specific feature geometry. → Need a second scale from the Qwen3 family.
2. **Different family, same scale.** Qwen3 has quirks (Q/K norm, specific tokenizer, specific RoPE). If FRR only works on Qwen3-family architectures the claim is much narrower. → Need a Llama-3, Phi-3, or Gemma teacher at ~1-2B scale.
3. **Different modality.** LLMs are one thing; proving the method on ViTs or ASR models would unlock 10× the market. → Future work, not Monday.

## Experimental matrix

Legend: ✅ done · 🔄 running · ⏳ queued · ❌ blocked

| Teacher                     | Family     | Hidden | Student h | Ratio | all-T10 | Status |
|-----------------------------|------------|--------|-----------|-------|---------|--------|
| Qwen3-1.7B                  | Qwen3      | 2048   | h256      | 311×  | 69.64%  | ✅ HQ5 |
| Qwen3-1.7B                  | Qwen3      | 2048   | h128      | 734×  | 68.00%  | ✅ HQ5 |
| **Qwen3-0.6B**              | Qwen3      | 1024   | h128      | ~30×  | —       | ⏳     |
| **Qwen3-0.6B**              | Qwen3      | 1024   | h64       | ~110× | —       | ⏳     |
| **Qwen3-0.6B**              | Qwen3      | 1024   | h256      | ~7×   | —       | ⏳ (upper bound check) |
| Qwen3-4B                    | Qwen3      | 2560   | h256      | ~400× | —       | ❌ cache not yet created |
| Llama-3.2-1B                | Llama-3    | 2048   | h256      | ~230× | —       | ❌ cache not yet created |
| Phi-3-mini-4k (3.8B)        | Phi-3      | 3072   | h256      | ~900× | —       | ❌ loader not yet written |

## Priority sequencing

Given the hardware budget (2× RTX 5090, currently running HQ6 + HQ7 + combined-stack eval):

1. **Highest impact, lowest cost** — Qwen3-0.6B cross-scale. We have the cache. `run_frr_generic.py` + `scale_eval.py` ready to go. When HQ6 h256 finishes on GPU 0 (~02:00 Sunday), launch `python run_frr_generic.py --teacher_cache qwen3_0.6b_cache.pt --h 128 --steps 80000 --tag generic_0.6b_h128 --device cuda:0`. 0.6B teacher is ~2.5 GB so the GPU has plenty of headroom; training rate will be faster than 1.7B (fewer teacher FLOPs). ~4-5 hours to 80K steps.

2. **Medium priority** — Qwen3-4B cache creation + training. Qwen3-4B fp32 is ~16 GB which fits a 5090 but leaves thin margins. Worth doing for the paper/pitch — 4B is a more impressive teacher for VCs than 1.7B.

3. **Medium priority** — Llama-3.2-1B. Different family validates the core insight. `teacher_loader.py` today assumes Qwen3 naming; Llama-3 has the same naming minus `q_norm`/`k_norm`, so it will auto-detect correctly (head_dim fallback to 128 kicks in). Need to download + cache.

4. **Stretch** — Phi-3. Fused QKV projection means state-dict naming differs; needs a new loader. Hold until we have the fundamentals proven.

## Why "same family different scale" already matters a lot

If FRR at 311× on Qwen3-**1.7B** produces 69.64% all-T10, and the method at proportionally matched ratio (~400× target) on Qwen3-**0.6B** produces something comparable (65-70%), that's a **width-scaling result**: the method doesn't care about hidden size. Very few compression methods have that property — most (GPTQ, AWQ, pruning) have sharp phase transitions at small scale. A positive 0.6B result immediately strengthens the patent breadth.

If 0.6B results are comparably good, we can claim: **"Linear in width, invariant in depth"** — because the fractal block iteration count doesn't change with teacher depth or width.

## Automated regression protection

`tests/test_sanity.py` now enforces:
- Both cached teachers auto-detect correctly (1.7B → hidden=2048, 0.6B → hidden=1024).
- Teacher forward is deterministic and produces finite logits.
- HQ5 h256 checkpoint must score ≥ 68.5% all-T10 on a fixed 100-sample seed-42 draw.
- A random-init student must score < 20% all-T10 (proves training is doing something).
- Checkpoint save/load roundtrip preserves the forward pass bit-close.

Run any time: `python tests/test_sanity.py`.

## Known issues (honest disclosure)

See `docs/KNOWN_ISSUES.md`.

## Commands (copy-paste ready)

```powershell
# Cross-scale validation on 0.6B teacher (most important next run)
# Launch when a GPU frees up:
python run_frr_generic.py --teacher_cache qwen3_0.6b_cache.pt `
    --h 128 --steps 80000 --tag generic_0.6b_h128 --device cuda:0

# Evaluate on truly-disjoint WikiText-103 test
python scale_eval.py --tags generic_0.6b_h128 hq5_h256 hq5_h128 `
    --corpus wikitext --n 1000 --device cuda:0

# Run sanity tests (any time, ~3 minutes)
python tests/test_sanity.py
```
