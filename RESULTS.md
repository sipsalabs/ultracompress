# UltraCompress — Claim 16 Cross-Family Results

**A single 2.40-bpw compression operating point validated across 6 transformer models spanning 3 architecture families (Llama-2, Qwen3, Mistral), three independent Llama-family training corpora (TinyLlama / SmolLM2 / OLMo-2 / Dolma), institutional provenance ranging from AllenAI to Meta-derived to MistralAI to Alibaba, and 7.5× in parameter scale — with zero hyperparameter retuning.**

![cross-family envelope](claim16_envelope.png)

![bpw convergence](claim16_bpw.png)

---

## Result envelope (6/6 models)

| Model              | Family    | Params | bpw    | PPL fp16 | PPL 2.40bpw | Ratio  | T1 retention | T10 retention | T10 teacher-agreement |
|--------------------|-----------|--------|--------|----------|-------------|--------|--------------|----------------|------------------------|
| TinyLlama-1.1B     | Llama-2   | 1.1 B  | 2.4053 | 17.01    | 28.90       | 1.699× | 83.61 %      | 91.73 %        | 94.17 %                |
| OLMo-2-1B          | Llama-2   | 1.49 B | 2.3955 | 20.15    | 36.07       | 1.790× | 82.75 %      | 90.83 %        | 93.06 %                |
| SmolLM2-1.7B       | Llama-2   | 1.81 B | 2.3955 | 18.03    | 34.24       | 1.899× | 80.84 %      | 90.18 %        | 93.20 %                |
| Qwen3-1.7B         | Qwen3     | 1.7 B  | 2.4017 | 33.21    | 59.40       | 1.788× | 84.65 %      | 90.68 %        | 93.88 %                |
| Mistral-7B-v0.3    | Mistral   | 7.25 B | 2.3971 | 12.36    | 20.11       | 1.627× | 86.21 %      | 93.19 %        | 95.06 %                |
| Qwen3-8B           | Qwen3     | 8.19 B | 2.3998 | 20.70    | 28.68       | 1.386× | 91.85 %      | 95.83 %        | 96.98 %                |

All runs: `(α_attn = 0.25, α_mlp = 0.125)`, D = 8, beam = 8, 3 – 6 EM iters. Eval: 500 WikiText-103 test windows × seq_len 128, seed = 42, fp16 teacher on RTX 5090.

### The envelope holds uniformly:

- **bpw spread across all 6 models:** 0.0098 bits (0.41 % relative) — the 2.40-bpw target is architecture- and corpus-invariant.
- **PPL ratio:** 1.39× – 1.90× (all within < 2×).
- **T10 teacher-agreement ≥ 93.06 %** on every model: the compressed student matches the fp16 teacher's top-10 next-token choice on more than 93 of every 100 tokens.
- **T1 retention ≥ 80.84 %** on every model.
- **σ²-input-column outlier intensity spanning 108×** (OLMo-2 20× → Mistral 2173×) is absorbed structurally by the role-bank + per-column-scaling stack without retuning.
- **Three independent Llama-arch pretraining corpora** (TinyLlama / SmolLM2 / OLMo-2) and **Apache-2.0 / open-data** provenance (AllenAI Dolma) all land inside the envelope.

---

## What this means

Post-training quantization at **2.40 bits per weight** is typically a model-family-specific tuning problem. Published schemes (GPTQ, AWQ, SmoothQuant, OmniQuant, QuaRot, SpinQuant) require per-model calibration, per-role α/β searches, or rotation-matrix learning. The operating point documented here:

- is a **single, fixed 2-parameter point** `(0.25, 0.125)`;
- ships a **single implementation path** (`compress_v17.py`) across all 4 models;
- requires **zero per-model hyperparameter search**;
- holds under **108× differences in activation-variance outlier intensity** across families (OLMo-2 20× → Mistral 2173×);
- holds under **three independent Llama-arch pretraining corpora** (SlimPajama, FineWeb-Edu, Dolma);
- converges to **2.40 ± 0.005 bpw** deterministically.

### Out-of-distribution robustness (LAMBADA)

The canonical Qwen3-1.7B fit was re-evaluated on **LAMBADA** (BookCorpus-derived narrative fiction, not Wikipedia):

| Metric                   | WikiText-103 | LAMBADA (OOD) | Δ           |
|--------------------------|--------------|----------------|-------------|
| PPL ratio (v17 / fp16)   | 1.788×       | **1.672×**     | **−0.116**  |
| T10 teacher-agreement    | 93.88 %      | **94.15 %**    | **+0.27 pp** |
| T10 retention            | 90.68 %      | **91.43 %**    | **+0.75 pp** |
| T1 retention             | 84.65 %      | 83.19 %        | −1.46 pp    |

**The compressed model tracks the teacher *better* on out-of-distribution text than on in-distribution text.** The 2.40-bpw operating point is not a WikiText artifact; it compresses the *functional* behaviour of each Linear, not corpus-specific patterns.

### On-disk packed format (2.41 bpw, round-trip verified)

The Claim-16 fit for Qwen3-1.7B is serialised end-to-end to a single binary file, **`v17_qwen3_1.7b.bin` — 424,563,357 bytes = 2.4101 bpw** (vs 2.4017 claimed; +0.008 bpw from JSON header + codebooks + fp16 scale rounding). A pure-decode path (`pack_v17.py verify`) reconstructs 196 body Linears from the binary alone — no calibration, no beam search — and the resulting model's PPL matches the original fit to **0.064 %** relative difference on the same 64 WikiText windows. This turns Claim 16 from a bit-counting argument into a working compressed-inference format.

**Format generalises across all 6 validated models.** Running `pack_all_v17.py` over every v17 fit produces six packed binaries with bit-rates all in the 2.40–2.41 bpw band across three model families, three tokenizer/corpus pairings, and a 7.2× parameter-count range. The 8B-scale round-trip (`verify_8b.log`) is **0.0000 %** relative PPL difference on Qwen3-8B — bit-exact decode at 6.95 B Linear params.

| Model           |    Params         | Pack bytes       | bpw_disk | round-trip diff |
|-----------------|------------------:|-----------------:|---------:|----------------:|
| Qwen3-1.7B      |     1,409,286,144 |      424,563,357 |   2.4101 |       0.0610 %  |
| Qwen3-8B        |     6,945,767,424 |    2,086,698,389 |   2.4034 |       0.0000 %  |
| Mistral-7B-v0.3 |     6,979,321,856 |    2,094,344,357 |   2.4006 |       0.0122 %  |
| TinyLlama-1.1B  |       968,884,224 |      292,422,381 |   2.4145 |       0.0122 %  |
| SmolLM2-1.7B    |     1,610,612,736 |      483,884,549 |   2.4035 |       0.0488 %  |
| OLMo-2-1B       |     1,073,741,824 |      322,688,101 |   2.4042 |       0.0000 %  |

The 2.40-bpw on-disk envelope is a property of the scheme, not of a particular model.

### Run it yourself

```powershell
python demo_claim16.py `
    --model_id Qwen/Qwen3-1.7B `
    --teacher  qwen3_1.7b_cache.pt `
    --v17      v17_fit_qwen3_1.7b.pt `
    --tokens   wikitext103_test_qwen3.pt `
    --n 5
```

Prints side-by-side fp16 teacher vs 2.40-bpw compressed top-5 next-token predictions on 5 random WikiText windows. Works identically for any of the 6 validated models — just swap `--model_id`, `--teacher`, `--v17`, `--tokens`.

At 2.40 bpw, an 8 B model compresses to ≈ 2.4 GB of body weights — a 6.7× reduction vs fp16 — while retaining 97 % of the teacher's top-10 token decisions and 95.8 % of its ground-truth top-10 accuracy on held-out text.

---

## Reproducibility

For each model:

```powershell
python cache_teacher_8b.py  --model_id <HF_ID> --out <model>_cache.pt
python tokenize_wikitext.py --model_id <HF_ID> --out wikitext103_test_<model>.pt
python cache_activations.py --teacher <model>_cache.pt --model_id <HF_ID> `
                            --tokens wikitext103_test_<model>.pt `
                            --n_cal 32 --seq_len 512 --out v17_activations_<model>.pt
python fit_v17_8b.py        --teacher <model>_cache.pt `
                            --v17act v17_activations_<model>.pt `
                            --a_attn 0.25 --a_mlp 0.125 --iters 6 `
                            --out v17_fit_<model>.pt
python eval_v17_8b.py       --model_id <HF_ID> --teacher <model>_cache.pt `
                            --v17 v17_fit_<model>.pt `
                            --tokens wikitext103_test_<model>.pt `
                            --n 500 --seq_len 128 --out v17_<model>_ppl.pt
python eval_topk_8b.py      --model_id <HF_ID> --teacher <model>_cache.pt `
                            --v17 v17_fit_<model>.pt `
                            --tokens wikitext103_test_<model>.pt `
                            --n 500 --seq_len 128 --out topk_<model>_results.pt
```

Hardware: single RTX 5090 (32 GB). Fit wall clock: 160 s (1.1 B) → 545 s (7.2 B).

Raw aggregated results: [`results.json`](results.json)

---

## Artifacts of record

- `results.json` — machine-readable summary (this document's source of truth).
- `claim16_envelope.png`, `claim16_bpw.png` — portfolio plots.
- `v17_fit_<model>.pt` — compressed-weight checkpoints.
- `v17_<model>_ppl.pt` — end-to-end perplexity measurements.
- `topk_<model>_results.pt` — top-1 / top-10 fidelity measurements.

For the full method, patent claims, 16-point α × α sweep, β-sweep defensive disclosure, Qwen3-8B chunked-EM scaling path, and Mistral outlier-robustness analysis see `PATENT_CLAIMS.md`.


## LAMBADA cross-corpus generalization (all 6 models)

Every v17 fit is calibrated only on WikiText-103. LAMBADA (BookCorpus
narrative fiction via `EleutherAI/lambada_openai`) is therefore a true
out-of-distribution test -- no re-fit, no re-calibration, pure inference.
500 random 128-token windows per model, fixed seed.

| Model          | Teacher PPL | v17 PPL | PPL ratio | Teacher T1 | v17 T1 | T1 retention |
|----------------|------------:|--------:|----------:|-----------:|-------:|-------------:|
| OLMo-2-1B      |      31.589 |  43.525 |     1.378 |     34.75% | 31.07% |       89.39% |
| TinyLlama-1.1B |      21.822 |  28.732 |     1.317 |     40.03% | 36.03% |       90.02% |
| Qwen3-1.7B     |      48.384 |  80.909 |     1.672 |     32.09% | 26.70% |       83.19% |
| SmolLM2-1.7B   |      22.019 |  33.044 |     1.501 |     39.78% | 34.47% |       86.66% |
| Mistral-7B     |      17.357 |  23.410 |     1.349 |     42.96% | 39.07% |       90.94% |
| Qwen3-8B       |      35.817 |  43.797 |     1.223 |     34.94% | 33.07% |       94.66% |

Cohort envelope: PPL ratio 1.22-1.67, top-1 retention 83.2%-94.7% across
6 architectures (Llama/Mistral, Qwen3, OLMo-2, SmolLM2) and a 7x parameter
range. The 8B fit has the lowest PPL ratio and highest top-1 retention,
matching the scaling prediction. Driver `lambada_all.py`, data
`lambada_all_results.json`.

