# `uc version` / `uc --version` / `uc -V`

Print the installed UltraCompress CLI version.

## Synopsis

```
uc version
uc --version
uc -V
```

All three forms produce identical output.

## Output

```
ultracompress 0.1.0
```

(For `uc --version` / `-V`, Click's standard `--version` framing is used; `uc version` is the explicit subcommand form.)

## Exit code

Always 0.

## Programmatic version access

```python
import ultracompress_cli
print(ultracompress_cli.__version__)
# 0.1.0
```

The version string follows [Semantic Versioning](https://semver.org/) (major.minor.patch).

## How to interpret the version

| Version | Meaning |
|---|---|
| `0.1.0` | Public alpha — stable subset (`list`, `pull`, `info`, `bench`); self-compression not yet shipped |
| `0.1.x` | Bug fixes + minor additions to the alpha |
| `0.2.0` | Self-compression (`uc compress`) + native runtime exports — target Q3 2026 |
| `0.x.y` | Pre-1.0 still allows breaking changes between minor versions; check the [Changelog](../changelog.md) |
| `1.0.0` | First stable release with full v1 surface — target Q3 2027 |

## Reporting bugs with version info

When filing issues, please include:

```bash
uc --version
python --version
pip show ultracompress
```

This helps us reproduce the issue against the right version.
