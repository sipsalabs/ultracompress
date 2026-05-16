# Contributing to UltraCompress

Thanks for your interest in contributing. This page is the rendered docs-site version of [`CONTRIBUTING.md`](https://github.com/sipsalabs/ultracompress/blob/main/CONTRIBUTING.md) in the repository.

## Scope

This repository contains the **UltraCompress CLI** (`uc`, `ultracompress`) — the Apache-2.0-licensed tool for downloading and running pre-compressed language models distributed through the [official Hugging Face Hub collection](https://huggingface.co/sipsalabs) (rolling release through April–May 2026).

The repository does **not** contain:

- The compression methods themselves (the subject of pending U.S. patent applications)
- Training code or any compression-method internals
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

## Pre-commit hooks (optional)

```bash
pip install pre-commit
pre-commit install
# Now ruff + gitleaks run on every git commit
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

## What's in scope for contributions

Welcome:
- Bug fixes
- Performance improvements
- Documentation improvements
- New CLI flags or subcommands that fit the existing surface
- Integration support for new runtimes (vLLM, llama.cpp, TensorRT-LLM, etc.)
- Test coverage improvements
- CI/CD improvements
- Translations of error messages and docs (if requested)

Not welcome:
- Reverse-engineering attempts of the compression methods
- Changes that disclose internal Sipsa Labs trade secrets
- Whitespace-only or noise PRs designed to inflate contributor count

## CLA for substantive contributions

For pull requests adding **>50 net lines of code** or new modules, we ask contributors to sign a Contributor License Agreement. The bot will post a link in your PR; signing takes ~1 minute.

For typo fixes, doc tweaks, and small bug fixes: no CLA required.

## Issue reporting

Use the issue templates on GitHub. Security issues — see [Security policy](../contributing/security.md).

## Code of Conduct

By participating you agree to follow the [Code of Conduct](code-of-conduct.md).

## Licensing

Contributions to this repository are licensed under Apache-2.0 (the license of the repository). You retain copyright to your contributions.

## Sign-off

We do not require DCO sign-off. Please ensure you have the right to contribute the code under Apache-2.0.

## Credits

Contributors are listed in the GitHub contributors page automatically. We also acknowledge significant contributors in release notes.

## Questions

- For technical questions: open a GitHub Discussion
- For product / commercial questions: `founder@sipsalabs.com`
- For security: `security@sipsalabs.com`
- For licensing / patents: `legal@sipsalabs.com`
