"""Run downstream lm-eval-harness benchmarks on a compressed artifact."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_benchmarks(
    artifact_path: Path,
    tasks: list[str],
    limit: int,
    batch_size: int,
    device: str,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Run lm-eval-harness on the artifact and return summarized results.

    v0.1 shells out to `python -m lm_eval` with the standard HuggingFace model loader.
    """
    try:
        import lm_eval  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "lm-eval-harness not installed; run `pip install lm-eval`"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_csv = ",".join(tasks)
    output_file = output_dir / f"bench_{artifact_path.name}.json"

    cmd = [
        sys.executable, "-u", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={artifact_path},dtype=float16",
        "--tasks", tasks_csv,
        "--batch_size", str(batch_size),
        "--device", device,
        "--limit", str(limit),
        "--output_path", str(output_file),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"lm-eval-harness exited with code {proc.returncode}\n{proc.stderr[-2000:]}"
        )

    result_files = list(output_dir.rglob("results_*.json"))
    if not result_files:
        raise RuntimeError(f"No results JSON produced under {output_dir}")
    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    payload: dict[str, Any] = json.loads(latest.read_text())

    out: dict[str, dict[str, float]] = {}
    for task, stats in payload.get("results", {}).items():
        out[task] = {
            "acc": float(stats.get("acc,none", 0.0)),
            "acc_norm": float(stats.get("acc_norm,none", stats.get("acc,none", 0.0))),
            "acc_stderr": float(stats.get("acc_stderr,none", 0.0)),
        }
    return out
