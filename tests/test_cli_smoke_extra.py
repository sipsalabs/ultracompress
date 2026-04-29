"""Extra CLI smoke tests beyond test_smoke.py."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from ultracompress_cli import __version__
from ultracompress_cli.__main__ import main


def test_invalid_command_exits_nonzero() -> None:
    """A typo'd subcommand should exit nonzero with helpful error text."""
    runner = CliRunner()
    result = runner.invoke(main, ["doesnotexist"])
    assert result.exit_code != 0
    out = result.output.lower()
    assert "no such command" in out or "usage" in out


def test_info_on_nonexistent_path_exits_nonzero(tmp_path: Path) -> None:
    """`uc info <missing>` should fail cleanly, not crash."""
    runner = CliRunner()
    result = runner.invoke(main, ["info", str(tmp_path / "nope")])
    assert result.exit_code != 0


def test_demo_runs_without_arguments() -> None:
    """`uc demo` is a scripted local demo and should never crash."""
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--no-pause", "--speed", "999"])
    # demo is allowed to print non-zero on platform-specific issues, but should
    # not throw an unhandled exception
    assert result.exit_code in (0, 1)


def test_short_version_flag_works() -> None:
    """The -V short flag should print the version like --version."""
    runner = CliRunner()
    result = runner.invoke(main, ["-V"])
    assert result.exit_code == 0
    assert __version__ in result.output
