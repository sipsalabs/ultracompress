"""Entrypoint shim so `python -m ultracompress …` works as a fallback when the
`uc` console script is not on PATH (Jupyter notebooks, minimal Docker images,
locked-down CI environments).

The fully-supported entry points are still `uc <subcommand>` and
`ultracompress <subcommand>`. This module exists so that `python -m ultracompress`
delegates to the same CLI dispatcher.
"""
from __future__ import annotations

from ultracompress.cli import main


if __name__ == "__main__":
    main()
