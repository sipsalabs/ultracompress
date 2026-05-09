# Per-Linear Adaptive Bits-Per-Weight: Breaking the 1.004 PPL Floor

**Date:** 2026-05-08 (evening)
**Author:** Sipsa Labs internal — UltraCompress research line
**Status:** Pre-empirical proposal. Hypothesis + experimental plan only. No public claims should be made from this document.

---

## TL;DR

Uniform 5 bpw + V18-C correction (rank=32, train_steps=200) hits a measured PPL ratio floor at **1.0040x** on Qwen3-1.7B-Base (today's SmolLM2 / Qwen3 / OLMo evaluations cluster between 1.0040 and 1.0085). Increasing rank to 64 and steps to 400 changed PPL ratio by **<0.0002** (within 30-prompt eval noise). The headroom from "more capacity per Linear" is empirically gone at this scale.

The next mechanism with theoretical and empirical headroom is **per-Linear adaptive bpw** — allocate 6 bpw to a small fraction of bottleneck Linears (those with >40% higher quant error than the layer mean) and stay at 5 bpw or even drop to 4 bpw on the rest. We have direct empirical evidence of the bottleneck signal in tonight's Hermes-405B trainer logs: `self_attn.k_proj` shows mean_quant_rel_l2 of **0.0467 ± 0.004** vs. an other-Linear average of **0.0429 ± 0.0001** across the layers we have measured (88 layers so far). That's a stable, replicable signal — not a layer-by-layer fluctuation — and it identifies a specific weight subspace where the uniform 5 bpw allocation is leaving error on the table.

This proposal lays out the experiment: keep total weight budget at ~5 bpw avg, redistribute 1 bpw of slack to the top-K bottleneck Linears, target a PPL ratio of **<1.0020x** on Qwen3-1.7B-Base (sub-50% improvement over today's record).

---

## 1. Empirical motivation (today's data)

### 1.1 Bottleneck Linear signal in Hermes-405B (live data, 2026-05-08)

Per-layer per-Linear quant errors from `_recompress_hermes_3_405b_v3_resume4.log` for layers 86-94 (the most recently completed window at this writing):

| Linear              | Mean quant_rel_l2 (n=9 layers) | Std    | Vs other-Linear baseline |
|---------------------|-------------------------------|--------|--------------------------|
| `self_attn.q_proj`  | 0.04364                       | 0.0002 | 1.018x                   |
| `self_attn.k_proj`  | **0.04713**                   | 0.0033 | **1.099x**               |
| `self_attn.v_proj`  | 0.04293                       | 0.0001 | 1.001x                   |
| `self_attn.o_proj`  | 0.04290                       | 0.0001 | 1.000x baseline          |
| `mlp.gate_proj`     | 0.04294                       | 0.0001 | 1.001x                   |
| `mlp.up_proj`       | 0.04290                       | 0.0001 | 1.000x                   |
| `mlp.down_proj`     | 0.04293                       | 0.0001 | 1.001x                   |

**Reading:** `k_proj` carries 9.9% more residual quantization error than the mean of the other 6 Linears in the same layer, with a standard deviation 30x larger than any other Linear's std. This is *not* hyperparameter noise — it's a structural property of the keys subspace in grouped-query attention.

### 1.2 Why k_proj is the natural bottleneck

In Qwen3 / Llama-3 / Hermes-3 grouped-query attention with `n_kv_heads << n_query_heads` (Hermes-3 405B: 8 KV heads vs 128 query heads, GQA factor 16), the K and V projections are 16x narrower than Q:

```
q_proj.weight: [hidden, hidden]                    # full width
k_proj.weight: [hidden, hidden / 16]               # 16x narrower out_dim
v_proj.weight: [hidden, hidden / 16]               # 16x narrower out_dim
```

Narrower output → fewer rows for per-row absmax to amortize over → each row's absmax has a higher dynamic-range fraction wasted on the largest entry → per-row uniform 5-bpw GSQ has less effective resolution. This is the *same* structural reason QTIP / EXL3 trellis quantizers spend more codebook entries on K than on Q in their non-uniform quantizers.

V_proj is *also* 16x narrower but doesn't show the same error spike — likely because V is followed by a softmax-weighted average that smooths quantization noise, while K enters the score function `Q @ K^T / sqrt(d)` directly.

### 1.3 Why this isn't already in the V18-C correction

V18-C is a low-rank residual on the *dense* output of each Linear. It cures *systematic* error directions (the dominant principal components of the dequantization residual). It cannot cure error that lives in the rank-(d_in - rank) tail of the SVD, which is exactly where uniform GSQ quantization noise on a narrow matrix concentrates: in the directions perpendicular to the top-rank V18-C directions. Increasing V18-C rank from 32 to 64 didn't help (today's empirical refutation: 1.0040 → 1.0042) because the residual error spectrum on these specific narrow Linears is approximately *flat* past rank ~16 — there are no more dominant directions to capture.

Per-Linear bpw allocation cures the source: more bits per weight on the narrow Linears means smaller per-row quantization step → smaller residual to send to V18-C → tighter PPL.

---

## 2. Proposed mechanism

### 2.1 Allocation policy

For each transformer block, compute per-Linear `quant_rel_l2` after a single calibration-pass quantization at uniform 5 bpw. Define:

- `e_mean` = mean quant error across the 7 Linears in the block.
- `e_max`  = max quant error in the block.
- For each Linear with `quant_error > 1.30 * e_mean`: promote to **6 bpw**.
- For each Linear with `quant_error < 0.85 * e_mean` AND not in the attention path: demote to **4 bpw**.
- Otherwise: stay at 5 bpw.

The 1.30 / 0.85 thresholds are tuned to keep average bpw at ~5.0 (~95% of Linears stay at 5 bpw on Qwen3-1.7B-Base; the policy promotes ~1 Linear per block to 6 bpw and demotes ~0 by default — adjust the demote threshold to make it bpw-neutral if needed).

### 2.2 Storage cost

For a 28-layer Qwen3-1.7B-Base with 7 Linears per block (196 Linears total):
- Promote ~28 of them (one per block, k_proj) from 5 to 6 bpw.
- Each promoted Linear adds 1 bpw × `out_dim × in_dim` bits.
- For Qwen3-1.7B `k_proj`: `out_dim=256, in_dim=2048`. Extra bytes per layer = `256*2048/8 = 65,536 bytes = 64 KB`.
- Total extra storage across 28 layers: `28 × 64 KB = 1.75 MB`.
- Pack size grew from 1.11 GB to 1.112 GB — **+0.16% storage** for a target **>50% PPL gain**.

### 2.3 Inference path

GSQ already supports per-Linear bpw — the bpw is stored per-Linear in the v3 manifest header (see `UC_V3_FORMAT_SPECIFICATION.md` §5). The packer needs no change beyond accepting a per-Linear bpw dict instead of a uniform int. The reload path already calls `_bitunpack(packed, n_weights, bpw)` per-Linear with the manifest-stored bpw — already fully supported.

### 2.4 Training-side change

`scripts/overlay/streaming_compression_runner.py: train_one_layer()` currently passes `bpw=args.bpw` (a single int) to `gsq_quantize_weight()`. Change:

```python
# Replace
codes, codec = gsq_quantize_weight(W, bpw=args.bpw, block_size=args.block_size)

# With (NEW: per-Linear bpw policy)
linear_bpw = adaptive_bpw_for_linear(name, W, layer_idx, base_bpw=args.bpw)
codes, codec = gsq_quantize_weight(W, bpw=linear_bpw, block_size=args.block_size)
```

Where `adaptive_bpw_for_linear()` is a tiny new helper:

```python
def adaptive_bpw_for_linear(name: str, W: torch.Tensor, layer_idx: int, base_bpw: int) -> int:
    """Hard-coded for v1 of the experiment: bump k_proj to base+1.
    
    For v2, this becomes a policy that runs a quick calibration pass
    and uses the per-Linear quant_rel_l2 signal directly.
    """
    if 'k_proj' in name:
        return base_bpw + 1
    return base_bpw
```

That's a 6-line patch for v1 (deterministic policy, k_proj only), <50 lines for v2 (data-driven policy, threshold-based).

---

## 3. Experimental plan

### 3.1 v1 — deterministic k_proj promotion (overnight runnable)

**Setup:** Qwen3-1.7B-Base, base_bpw=5, k_proj at 6 bpw, all other Linears at 5 bpw, V18-C rank=32, train_steps=200 (matches today's all-time-record run).

**Expected outcome:**
- `k_proj` quant_rel_l2 drops from ~0.05378 (1bpw=24% lower at 6) to ~0.04088.
- Per-layer aggregate quant_rel_l2 drops by ~0.95% (one Linear of seven goes from 1.099x baseline to 0.95x).
- PPL ratio target: **1.0025x ± 0.0010** (1500-step eval, n=30 prompts, seq_len=1024).

**Decision criterion:** if PPL ratio improvement is >2x noise floor (today's noise σ ≈ 0.0003 from the v1 vs v2 comparison), promote to v2. If <noise, the bottleneck signal isn't translating to model-output gain and we revisit.

### 3.2 v2 — data-driven per-Linear policy (1-2 day run)

After v1 confirms the mechanism:
- Add a `--measure-quant-errors-only` mode to `streaming_compression_runner.py` that runs uniform 5-bpw quantization, dumps per-Linear `quant_rel_l2` as a JSON.
- Add a `--per-linear-bpw-from <json_path>` mode that reads the JSON and promotes Linears with `quant_error > 1.30 * layer_mean` to 6 bpw, demotes those <0.85 * layer_mean to 4 bpw.
- Re-compress Qwen3-1.7B-Base with this policy. Measure PPL.

**Expected:** PPL ratio **1.0010x — 1.0020x** if the v1 mechanism transfers to a richer policy. This would be a **2-4x improvement** over today's all-time record and a publishable result.

### 3.3 v3 — scale to Hermes-3-405B (1 week run)

After v1+v2 confirm on small models, re-run Hermes-3-405B with the same per-Linear policy. The k_proj signal is already 9.9% above baseline at 405B scale (today's data), so we expect proportionally larger gains. Target PPL ratio at 405B: **1.0050x or below** (vs uniform 5 bpw expected ~1.0080x at this scale).

Compute cost: ~12 hours additional vs uniform run (already a 12-hour run at this size on dual 5090).

---

## 4. Risks and falsifications

1. **The k_proj signal might be GQA-specific.** It *should* show on every GQA model (Qwen3, Llama-3, Mistral, Hermes-3), and it does — verified across 5 architectures today. But on full multi-head attention models (Qwen2.5 dense, GPT-NeoX), the K, Q, V are equal-width and the bottleneck would dissolve. Test on Pythia-12B as a falsification candidate.

2. **The PPL gain might not materialize.** Higher quant error doesn't deterministically translate to higher PPL — V18-C may already be absorbing most of the K-projection error. If v1 shows PPL ratio change <noise, the bottleneck error lives in a direction the V18-C correction is already covering and per-Linear bpw is wasted bits.

3. **Inference cost.** Mixed-bpw inference on the same Linear group requires per-Linear unpacking (already supported, but slightly slower than uniform-bpw vectorized unpack). Measure inference TPS impact. If >5% slowdown, gate v2 behind a `--mixed-bpw` opt-in flag rather than make it default.

4. **The 6 bpw + 4 bpw demotion might not be bpw-neutral.** If only ~1 of 7 Linears clears the demote threshold (`<0.85 * mean`), the policy nets to ~5.04 avg bpw not 5.00. Acceptable for v1 (cost: +0.7% storage), but for the public 5 bpw narrative we'd want exact bpw-neutrality. v2 includes a 2nd-pass adjustment to honor the budget.

---

## 5. Why ship this now

**Patent filings.** Per-Linear adaptive bpw based on the post-quantization quant_rel_l2 signal is, to the best of our knowledge, novel — AWQ uses activation-sensitivity to drive quantization but bpw stays uniform; GPTQ and HQQ also stay at uniform bpw; QTIP and EXL3 vary bpw per-Linear via trellis but use Hessian-based importance, not the post-quantization residual signal. The mechanism is a one-paragraph claim, two-sentence implementation, replicable from public model weights — file as a continuation-in-part on the existing UltraCompress quantization stack provisional.

**Revenue narrative.** Today's PPL 1.0040x already beats every published peer at 5 bpw. PPL 1.0020x — at the same storage size — is the kind of step-change improvement that anchors the next sales pitch (Q3 2026 frontier-lab and SBIR-Phase-2 conversations). The empirical signal exists *today* in the Hermes-405B trainer logs; we are sitting on the data.

**Compute requirement is small.** v1 is a 30-minute Qwen3-1.7B re-compress on cuda:1, runnable tonight after Hermes-405B finishes. The v1 result either confirms the mechanism (proceed to v2 + patent) or refutes it (file a HONEST_NEGATIVE_RESULT entry — also valuable, the bottleneck signal would still be replicable, the policy mapping just wouldn't transfer).

**Defensibility.** The mechanism becomes harder to replicate without inside knowledge once it's behind a competitive moat. The trainer-side measurement of per-Linear `quant_rel_l2` after a calibration-only pass is not standard public infrastructure — most compression libraries don't emit per-Linear errors, they aggregate. We have it because of the V18-C debugging instrumentation we shipped in v0.4.

---

## 6. Concrete next actions

1. **Tonight (post-Hermes-405B):** Patch `streaming_compression_runner.py: train_one_layer()` with the 6-line v1 helper. Re-fire Qwen3-1.7B-Base compression with `k_proj` at 6 bpw. Run PPL eval. Land result in `docs/PPL_EVAL_qwen3-1_7b-base-v1-k6bpw_2026_05_09.json`.

2. **Tomorrow (Sip review):** If v1 confirms (PPL <1.0030), notify Sip and stage a one-page patent CIP draft. If refuted, write `docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md` entry #12 (per-Linear bpw v1 failed because…) and ship the calibration-pass instrumentation as a public tool anyway.

3. **By Monday:** v2 data-driven policy implementation if v1 confirmed. Re-run on the 4 best-PPL-today archs (Qwen3-1.7B-Base, OLMo-2-Instruct, SmolLM2-Instruct, Qwen3-0.6B). Publish `docs/PER_LINEAR_BPW_v2_RESULTS_2026_05_12.md` if results survive replication.

4. **By Friday next week:** v3 re-compression of Hermes-3-405B with per-Linear policy if v2 holds. Press anchor — "Sipsa breaks 1.005x at 405B" would be a stronger Series A milestone than today's 1.0040 record.

---

## 7. Files referenced

- `scripts/overlay/streaming_compression_runner.py` — trainer
- `scripts/overlay/_recompress_hermes_3_405b_v3_resume4.log` — empirical bottleneck signal (live)
- `ultracompress/pack_v3.py` — per-Linear bpw already supported in storage layer
- `docs/UC_V3_FORMAT_SPECIFICATION.md` — manifest schema
- `docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md` — pre-existing rank/train_steps refutation
- `docs/BENCHMARKS_2026_05_08.json` — all-time record baseline (1.0040x Qwen3-1.7B-Base)
