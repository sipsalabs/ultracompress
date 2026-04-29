"""Inspect a compressed artifact's metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def read_artifact_metadata(path: Path) -> dict[str, Any] | None:
    """Look for `ultracompress.json` in an artifact directory (or treat path itself as the manifest)."""
    if path.is_file() and path.suffix == ".json":
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    candidates = [
        path / "ultracompress.json",
        path / "compression_manifest.json",
        path / "config.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                data = json.loads(c.read_text())
                # Only return if this looks like an ultracompress artifact
                if "ultracompress" in data or "uc_version" in data or "bpw" in data:
                    return data
            except Exception:
                continue
    return None


def summarize_artifact(meta: dict[str, Any], console: Console) -> None:
    """Pretty-print artifact metadata to the console."""
    title = meta.get("name") or meta.get("model_id") or "UltraCompress artifact"
    content = []

    base = meta.get("base_model") or meta.get("teacher") or meta.get("source_model")
    if base:
        content.append(f"[bold]Base model:[/bold] {base}")

    bpw = meta.get("bpw") or meta.get("bits_per_weight")
    if bpw is not None:
        content.append(f"[bold]Compression:[/bold] {bpw:.3f} bits per weight")

    ratio = meta.get("compression_ratio") or meta.get("ratio")
    if ratio is not None:
        content.append(f"[bold]Ratio:[/bold] {ratio:.2f}×")

    method = meta.get("method") or meta.get("track")
    if method:
        content.append(f"[bold]Method:[/bold] {method}")

    uc_version = meta.get("uc_version") or meta.get("ultracompress_version")
    if uc_version:
        content.append(f"[bold]UC version:[/bold] {uc_version}")

    console.print(Panel("\n".join(content), title=f"[cyan]{title}[/cyan]", border_style="cyan"))

    # Benchmarks (if included in manifest)
    benches = meta.get("benchmarks") or meta.get("evaluations")
    if benches and isinstance(benches, dict):
        table = Table(title="Evaluation results", show_header=True, header_style="bold cyan")
        table.add_column("Task", style="bright_white")
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("Teacher", justify="right", style="dim")
        table.add_column("Retention", justify="right", style="green")
        for task, stats in benches.items():
            student = stats.get("acc") or stats.get("score") or 0
            teacher = stats.get("teacher_acc") or stats.get("teacher") or 0
            retention = (student / teacher * 100) if teacher else 0
            table.add_row(
                task,
                f"{student * 100:.2f}%",
                f"{teacher * 100:.2f}%",
                f"{retention:.1f}%",
            )
        console.print(table)
