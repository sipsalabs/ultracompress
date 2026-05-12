# `uc list`

List pre-compressed models available on the [Hugging Face Hub `sipsalabs` organization](https://huggingface.co/sipsalabs).

## Synopsis

```
uc list [--json]
```

## Options

| Option | Default | Description |
|---|---|---|
| `--json` | off | Emit results as JSON instead of a human-readable table. Useful for scripting. |

## Output (table mode)

```
                          Pre-compressed models from Hugging Face Hub
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Model ID                    ┃ Base               ┃    bpw ┃   Size ┃ Downloads┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ sipsalabs/<model-id> │ Qwen/Qwen3-1.7B    │  2.798 │ 1.04GB │     12.4k│
│ sipsalabs/llama2-7b-uc2p79  │ meta-llama/Ll...   │  2.798 │ 4.20GB │      8.7k│
│ sipsalabs/mistral-7b-uc2p79 │ mistralai/Mi...    │  2.798 │ 4.21GB │      4.2k│
└─────────────────────────────┴────────────────────┴────────┴────────┴──────────┘

Pull one with: uc pull sipsalabs/<model-id>
```

## Output (JSON mode)

```json
[
  {
    "modelId": "sipsalabs/<model-id>",
    "base_model": "Qwen/Qwen3-1.7B",
    "bpw": 2.798,
    "size_bytes": 1116691072,
    "size_human": "1.04GB",
    "downloads": 12421,
    "last_modified": "2026-04-22T18:32:11Z",
    "license": "research-free; commercial-required"
  },
  ...
]
```

## Behavior

- Calls the Hugging Face Hub API to list models tagged `library:ultracompress` and authored by `sipsalabs`.
- Hits the public API by default; if you've authenticated with `huggingface-cli login` your higher-rate-limit token is used automatically.
- If the Hub is unreachable, `uc list` exits 0 with a "no models" message rather than crashing.

## Examples

```bash
# Print the catalog as a table
uc list

# Pipe JSON to jq for scripting
uc list --json | jq '.[] | select(.bpw < 3) | .modelId'

# Count published models
uc list --json | jq '. | length'
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | OK (including "no models" case) |
| 2 | Invalid arguments (Click default) |

## See also

- [`uc pull`](pull.md) — download a listed model
- [`uc info`](info.md) — inspect what was downloaded
- Run `uc list` to see the live catalog at any time.
