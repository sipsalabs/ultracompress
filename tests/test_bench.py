"""Smoke tests for ultracompress.bench - sales-grade throughput benchmark.

These tests verify the public surface (imports, CLI wiring, manifest parsing)
without actually loading models or touching the GPU. The full end-to-end
throughput run lives outside the test suite — it requires a real packed dir
and an idle GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_bench_module_importable():
    """`from ultracompress.bench import bench_packed` works without GPU."""
    from ultracompress.bench import BenchResult, bench_packed, cmd_bench

    assert callable(bench_packed)
    assert callable(cmd_bench)
    assert bench_packed.__doc__ is not None
    assert "throughput" in bench_packed.__doc__.lower()
    # BenchResult must serialize cleanly to JSON.
    sample = BenchResult(
        model_id="x", n_prompts=1, seq_len=8, n_new_tokens=4,
        device="cpu", cuda_device_name="cpu",
        ttft_s_mean=0.1, tps_overall=1.0, tps_decode_only=1.0,
        peak_vram_gb=0.0, timestamp="2026-05-09T00:00:00",
    )
    from dataclasses import asdict
    json.dumps(asdict(sample))


def test_bench_re_exported_from_top_level():
    """`from ultracompress import bench_packed` resolves to the bench module."""
    import ultracompress

    assert "bench_packed" in ultracompress.__all__
    assert callable(ultracompress.bench_packed)


def test_resolve_base_model_id_from_readme(tmp_path: Path):
    """README.md `base_model:` frontmatter is the fallback path."""
    from ultracompress.bench import _resolve_base_model_id

    pd = tmp_path / "fake_pack"
    pd.mkdir()
    (pd / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-1.7B-Base\n---\n# fake\n",
        encoding="utf-8",
    )
    assert _resolve_base_model_id(pd) == "Qwen/Qwen3-1.7B-Base"


def test_resolve_base_model_id_from_manifest(tmp_path: Path):
    """manifest.json `base_model_hf_id` takes precedence over README."""
    from ultracompress.bench import _resolve_base_model_id

    pd = tmp_path / "fake_pack"
    pd.mkdir()
    (pd / "manifest.json").write_text(
        json.dumps({"base_model_hf_id": "Qwen/Qwen3-1.7B"}),
        encoding="utf-8",
    )
    (pd / "README.md").write_text(
        "---\nbase_model: WrongModel\n---\n", encoding="utf-8",
    )
    assert _resolve_base_model_id(pd) == "Qwen/Qwen3-1.7B"


def test_resolve_base_model_id_missing_raises(tmp_path: Path):
    """No manifest, no README - hard fail with helpful message."""
    from ultracompress.bench import _resolve_base_model_id

    pd = tmp_path / "fake_pack"
    pd.mkdir()
    with pytest.raises(ValueError, match="base.model"):
        _resolve_base_model_id(pd)


def test_cli_bench_subparser_registered():
    """`uc bench <dir>` is a registered subcommand with the right options."""
    from ultracompress.cli import build_parser

    parser = build_parser()
    # parse_known_args lets us check option registration without running.
    args, _ = parser.parse_known_args([
        "bench", "/tmp/pack", "--n-prompts", "5", "--seq-len", "32",
        "--n-new-tokens", "8", "--device", "cpu", "--baseline",
    ])
    assert args.command == "bench"
    assert args.packed_dir == "/tmp/pack"
    assert args.n_prompts == 5
    assert args.seq_len == 32
    assert args.n_new_tokens == 8
    assert args.device == "cpu"
    assert args.baseline is True


def test_cli_bench_defaults():
    """Default values match the public spec."""
    from ultracompress.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["bench", "/tmp/pack"])
    assert args.n_prompts == 50
    assert args.seq_len == 1024
    assert args.n_new_tokens == 256
    assert args.device == "cuda:0"
    assert args.baseline is False


def test_legacy_bench_compress_still_registered():
    """The old `--model` benchmark survives as `bench-compress`."""
    from ultracompress.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "bench-compress", "--model", "Qwen/Qwen3-1.7B", "--target-bpw", "5",
    ])
    assert args.command == "bench-compress"
    assert args.model == "Qwen/Qwen3-1.7B"
    assert args.target_bpw == 5.0
