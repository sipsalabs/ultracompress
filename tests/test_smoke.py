"""Smoke tests — verify the CLI imports and banner prints."""

import json
from unittest.mock import patch

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


def test_list_json_is_machine_readable() -> None:
    """`uc list --json` should emit parseable JSON without banner text."""
    runner = CliRunner()
    with patch("ultracompress_cli.__main__.list_published_models", return_value=[]):
        result = runner.invoke(main, ["list", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == []
