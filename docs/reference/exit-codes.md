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

### Downloading via `huggingface-cli`

`uc pull` was removed in v0.6.21. Use `huggingface-cli download` for the
download step, then `uc verify` to check download integrity against the SHA-256
contract on disk.

| Outcome | Command | Code |
|---|---|---|
| Download succeeded | `huggingface-cli download` | 0 |
| Hub unreachable / auth error | `huggingface-cli download` | non-zero (set by HF CLI) |
| Verification PASS | `uc verify` | 0 |
| Verification FAIL | `uc verify` | 1 |
| Bad arguments | either | 2 |

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

### Loop over models and download each

```bash
uc catalog | awk 'NR>2 && $1 ~ /^sipsa-/ {print $1}' | while read -r model; do
    repo="SipsaLabs/${model#sipsa-}-uc-v3-bpw5"
    huggingface-cli download "$repo" --local-dir "./${model}" \
        || echo "[warn] failed to download $repo" >&2
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
huggingface-cli download SipsaLabs/<repo-id> --local-dir ./<repo-id>
uc verify ./<repo-id>
uc info ./<repo-id>
```
