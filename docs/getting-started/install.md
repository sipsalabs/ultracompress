# Install

UltraCompress CLI is published on PyPI as `ultracompress`. The package supports Python 3.10, 3.11, and 3.12 on Linux, macOS, and Windows.

## Recommended (uv)

[`uv`](https://github.com/astral-sh/uv) is the fastest Python package manager. We recommend it for production environments and CI.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add ultracompress to your environment
uv add ultracompress
```

## Standard (pip)

```bash
pip install ultracompress
```

## With PyTorch (for `uc bench`)

The benchmarking command requires PyTorch. Install with the optional `torch` extra:

```bash
pip install "ultracompress[torch]"
```

If you need a specific CUDA version, install PyTorch first per [pytorch.org](https://pytorch.org/get-started/locally/), then `pip install ultracompress` (without the extra).

## Verify the install

```bash
uc --version
# UltraCompress v0.1.0  · https://sipsalabs.com
# Extreme compression for large language models. Patent pending — USPTO 64/049,511 + 64/049,517

uc --help
# Lists all sub-commands
```

## Configure access to the Hugging Face Hub

The `uc list` and `uc pull` commands query the Hugging Face Hub. By default this works without authentication for the public `sipsalabs` org. If you have a Hugging Face account with private models or higher-rate-limit access:

```bash
pip install -U huggingface_hub
huggingface-cli login
# Paste your access token from https://huggingface.co/settings/tokens
```

## Upgrade

```bash
pip install -U ultracompress
# or
uv add --upgrade ultracompress
```

Check the [Changelog](../changelog.md) for what changed.

## Uninstall

```bash
pip uninstall ultracompress
```

## Common install issues

??? question "Symbol not found / DLL load failed (Windows)"

    Ensure your Python is x86_64 (not ARM). On Windows ARM, install via `conda`:
    ```cmd
    conda install -c conda-forge ultracompress
    ```
    (Conda support arrives in v0.1.1.)

??? question "ImportError: huggingface_hub"

    Force-reinstall the dependencies:
    ```bash
    pip install --force-reinstall ultracompress
    ```

??? question "uc: command not found"

    Your Python `Scripts` (Windows) or `bin` (Unix) directory may not be on `$PATH`. Use the explicit module form:
    ```bash
    python -m ultracompress_cli --help
    ```

If none of those resolve it, [open an issue](https://github.com/mounnar/ultracompress/issues/new?template=bug_report.md).
