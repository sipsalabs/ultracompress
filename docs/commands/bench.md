# `uc bench`

Run downstream-task benchmarks on a compressed UltraCompress artifact.

## Synopsis

```
uc bench <path> [--tasks <list>] [--limit <int>] [--batch-size <int>] [--device <str>] [--output-dir <path>]
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `<path>` | yes | Path to a directory produced by `uc pull` |

## Options

| Option | Default | Description |
|---|---|---|
| `--tasks LIST` | `hellaswag,arc_challenge` | Comma-separated `lm-eval-harness` task names |
| `--limit INT` | `500` | Samples per task |
| `--batch-size INT` | `8` | Batch size |
| `--device STR` | `cuda:0` | PyTorch device |
| `--output-dir PATH` | `./bench-results` | Where to save per-sample logs and summary JSON |

## Output

```
UltraCompress v0.1.0  · https://sipsalabs.com
Extreme compression for large language models. Patent pending

→ Benchmarking ./models/sipsalabs_<model-id> on tasks: hellaswag,arc_challenge
  limit=500  batch_size=8  device=cuda:0

                 Benchmark results
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Task            ┃   acc   ┃ acc_norm ┃   stderr ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ hellaswag       │ 51.20%  │  67.60%  │ +/-2.23% │
│ arc_challenge   │ 38.40%  │  41.20%  │ +/-2.18% │
└─────────────────┴─────────┴──────────┴──────────┘
```

## Behavior

- Loads the compressed artifact via the UltraCompress reference loader.
- Runs `lm-eval-harness` for each named task.
- Writes a per-sample log (`<task>_<timestamp>.jsonl`) and a summary (`summary.json`) to `<output-dir>`.
- Prints a Rich-rendered table to stdout.

## Requirements

`uc bench` requires:

- PyTorch (`pip install "ultracompress[torch]"`)
- A CUDA GPU (default device `cuda:0`; specify `--device cpu` for CPU-only, but expect 100× slower)
- ~2-8 GB GPU memory for a 7B-parameter model at 2.798 bpw

## Examples

```bash
# Default: 2 tasks, 500 samples each, batch 8
uc bench ./models/sipsalabs_<model-id>

# Quick smoke check
uc bench ./models/sipsalabs_<model-id> --tasks hellaswag --limit 50

# Multiple tasks with bigger sample size
uc bench ./models/sipsalabs_<model-id> \
    --tasks hellaswag,arc_challenge,arc_easy,piqa,winogrande \
    --limit 1000 --batch-size 16

# CPU fallback (slow!)
uc bench ./models/sipsalabs_<model-id> --device cpu --limit 50

# Custom output directory
uc bench ./models/sipsalabs_<model-id> -o /tmp/bench-runs/run-001
```

## Reproducibility

- Every run uses a deterministic seed (default 42; configurable via `UC_BENCH_SEED` env var when supported).
- Results are deterministic up to GPU-arithmetic non-determinism (which is bounded; aggregates don't differ in practice).
- The `summary.json` includes the seed, sample count, batch size, and `lm-eval-harness` version used.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | Benchmark failed (PyTorch / CUDA / lm-eval-harness error) |
| 2 | Invalid arguments (Click default) |

## Performance tuning

- **Larger batch size** → faster, but more GPU memory. Bump `--batch-size 16` or `32` if you have memory headroom.
- **Smaller `--limit`** for quick iteration; full evaluation usually wants `--limit 1000` or more.
- **Device-specific tuning**: on H100 we recommend `--batch-size 32`; on consumer 4090/5090 `--batch-size 16`; on T4 `--batch-size 4`.

## See also

- [`uc pull`](pull.md) — get the artifact in the first place
- [`uc info`](info.md) — inspect the artifact's manifest
- [Reproducibility](../concepts/reproducibility.md) — concept page on how we ship reproducibility
- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the underlying benchmark library
