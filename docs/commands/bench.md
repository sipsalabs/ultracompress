# `uc bench` (removed in v0.6.x)

`uc bench` is **not** part of the public CLI in v0.6.x. The downstream-task benchmarking flow it documented has been moved out of the package; callers are expected to run their own `lm-eval-harness` (or equivalent) against the reconstructed model and compare against the published reference numbers.

## Current public CLI surface

The v0.6.x public CLI exposes only these subcommands:

| Command | What it does |
|---|---|
| `uc verify <dir>` | Pack structure + SHA-256 / manifest download-integrity check |
| `uc audit <dir>` | JSON audit receipt: pack structure + per-file SHA-256 + PII-free host fingerprint (a structural/integrity artifact, **not** a reconstruction proof) |
| `uc try <model-id>` | Generate against a Sipsa-hosted compressed model |
| `uc catalog` | List the full compressed-model catalog + tiers |
| `uc info` | What this package is + contact/links |
| `uc version` | Print the installed package version |

## Recommended replacement flow

```bash
# 1. Download the compressed pack from Hugging Face
huggingface-cli download SipsaLabs/<repo-id> --local-dir ./<repo-id>

# 2. Verify it
uc verify ./<repo-id>

# 3. Run your own benchmark — any evaluator that loads transformer
#    checkpoints will work against the reconstructed model
lm_eval --model hf --model_args pretrained=./<repo-id> \
        --tasks hellaswag,arc_challenge --batch_size 8 --limit 500
```

## Reference numbers

Every architecture in the public catalog ships with a verified perplexity ratio (or, for vision encoders, a verified cosine similarity) in [`docs/benchmarks.json`](https://github.com/sipsalabs/ultracompress/blob/main/docs/benchmarks.json). If your local evaluator disagrees with those numbers by more than the published standard error, open an issue.

## See also

- [Downloading a compressed model](pull.md)
- [`uc info`](info.md) — inspect the artifact's manifest
- [Reproducibility](../concepts/reproducibility.md)
- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
