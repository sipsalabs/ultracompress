# Exit codes reference

UltraCompress follows standard Unix exit-code conventions. All commands below `uc <command>` return one of these codes.

## Exit codes

| Code | Meaning | Example |
|---|---|---|
| `0` | Success | `uc list` (even if no models published) |
| `1` | Generic runtime error | network failure, missing file, manifest corrupt |
| `2` | Invalid arguments | from `click` (typo'd subcommand, missing required arg) |
| `130` | Interrupted by Ctrl-C | user aborted |

We deliberately do not use specialized exit codes (e.g., 64-78 from `sysexits.h`). The semantics of the three codes above are sufficient for most scripting needs.

## Per-command nuances

### `uc list`

| Outcome | Code |
|---|---|
| Models listed (or no-models case) | 0 |
| Hub unreachable | 0 (we degrade gracefully — `[]` returned) |
| Bad arguments | 2 |

### `uc pull`

| Outcome | Code |
|---|---|
| Download succeeded, manifest verified | 0 |
| Download succeeded, manifest mismatch | 0 with stderr warning |
| Hub unreachable | 1 |
| Authentication required (private model) | 1 |
| Insufficient disk space | 1 |
| Bad arguments | 2 |

### `uc info`

| Outcome | Code |
|---|---|
| Manifest found and printed | 0 |
| Manifest found, SHA mismatch | 0 with stderr warning |
| No `ultracompress.json` at path | 1 |
| Path doesn't exist | 2 |
| Bad arguments | 2 |

### `uc bench`

| Outcome | Code |
|---|---|
| All tasks ran successfully | 0 |
| At least one task crashed | 1 |
| PyTorch / CUDA setup failure | 1 |
| Bad arguments | 2 |

### `uc demo`

| Outcome | Code |
|---|---|
| Demo played to completion | 0 |
| Terminal too small / control sequence rejected | 0 (still exits cleanly) |
| Bad arguments | 2 |

### `uc version` / `--version` / `-V`

Always returns 0.

## Scripting patterns

### Loop over models and pull each

```bash
uc list --json | jq -r '.[].modelId' | while read -r model; do
    uc pull "$model" || echo "[warn] failed to pull $model" >&2
done
```

### Bench all artifacts in a directory

```bash
for d in ./models/*/; do
    uc bench "$d" --tasks hellaswag --limit 100 \
        || echo "[warn] bench failed on $d" >&2
done
```

### Verify all manifests under a directory

```bash
find ./models -name 'ultracompress.json' | while read -r m; do
    uc info "$(dirname "$m")" >/dev/null 2>&1 \
        && echo "[ok]  $m" \
        || echo "[err] $m"
done
```

### Run inside CI with strict exit-on-error

```bash
set -euo pipefail
uc pull sipsalabs/<model-id>
uc info ./models/sipsalabs_<model-id>
uc bench ./models/sipsalabs_<model-id> --tasks hellaswag --limit 50
```
