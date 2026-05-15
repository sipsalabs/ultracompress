# Contributing to UltraCompress

Thanks for your interest in UltraCompress. This document covers how to contribute to the open-source CLI.

## Scope of this repository

This repository contains the **UltraCompress CLI** (`uc`, `ultracompress`) — the Apache-2.0-licensed tool for downloading and running pre-compressed language models distributed through the [official Hugging Face Hub collection](https://huggingface.co/sipsalabs) (rolling release through April–May 2026).

The repository does **not** contain:

- The compression methods themselves (the subject of pending U.S. patent applications)
- Training code, weight overlays, or architectural-compression internals
- Pre-compressed model weights (distributed via the Hugging Face Hub under their own license; rolling release through April–May 2026)

Pull requests touching the public CLI surface, documentation, packaging, CI/CD, and tests are welcome. Pull requests attempting to add or reverse-engineer the compression methods will be closed.

## Development setup

```bash
git clone https://github.com/sipsalabs/ultracompress.git
cd ultracompress
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run the test suite

```bash
pytest tests/ -v --cov=ultracompress_cli
```

## Lint and format

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Type checking

```bash
mypy src/ --ignore-missing-imports
```

## Pull request checklist

- [ ] All tests pass (`pytest`)
- [ ] Ruff is clean (`ruff check`)
- [ ] New code has tests (smoke or unit)
- [ ] No secrets, tokens, or credentials in the diff
- [ ] No proprietary compression internals or trade-secret information added
- [ ] CHANGELOG.md updated if your change is user-visible
- [ ] One commit per logical change; commit messages start with a verb in imperative mood

## Code style

- 4-space indentation
- Max line length 100 chars
- Type hints on public functions (`def foo(x: int) -> str:`)
- f-strings, not `.format()` or `%`
- `pathlib.Path` over `os.path`
- Imports grouped: stdlib, third-party, local — separated by blank lines
- Docstrings on public functions (single-line or short triple-quoted)

## Issue reporting

Use the issue templates on GitHub. Security issues — see [SECURITY.md](SECURITY.md).

## Code of Conduct

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Licensing

By contributing you agree that your contributions will be licensed under Apache-2.0 (the license of this repository). You retain copyright to your contributions.

## Sign-off

We do not currently require DCO sign-off, but please make sure you have the right to contribute the code under Apache-2.0.
