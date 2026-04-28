"""Smoke tests — verify the CLI imports and banner prints."""
from click.testing import CliRunner

from ultracompress_cli import __version__
from ultracompress_cli.__main__ import main


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "pull" in result.output


def test_list_empty() -> None:
    """Should not crash even if HF Hub returns no models or is unreachable."""
    runner = CliRunner()
    result = runner.invoke(main, ["list"])
    # Exit code 0 even with no models
    assert result.exit_code == 0
