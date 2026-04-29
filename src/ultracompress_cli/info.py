"""Inspect a compressed artifact's metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def find_manifest_path(path: Path) -> Path | None:
    """Return the best UltraCompress manifest path for a file or artifact directory."""
    if path.is_file() and path.suffix == ".json":
        return path

    candidates = [
        path / "ultracompress.json",
        path / "compression_manifest.json",
        path / "config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_artifact_metadata(path: Path) -> dict[str, Any] | None:
    """Look for `ultracompress.json` in an artifact directory (or treat path itself as the manifest)."""
    manifest_path = find_manifest_path(path)
    if manifest_path is None:
        return None

    try:
        data = json.loads(manifest_path.read_text())
    except Exception:
        return None

    if path.is_file() and path.suffix == ".json":
        return data

    # Only return directory-discovered metadata if this looks like an UltraCompress artifact.
    if "ultracompress" in data or "uc_version" in data or "bpw" in data:
        return data
    return None


def _format_float(value: Any, precision: int) -> str | None:
    """Format numeric manifest fields that may arrive as strings."""
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return None


def _artifact_dir(path: Path) -> Path:
    """Return the artifact directory for either a manifest file or directory path."""
    return path.parent if path.is_file() else path


def _safe_relative_path(raw_path: str) -> Path | None:
    """Reject absolute paths and parent traversal inside manifest file entries."""
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    return relative_path


def verify_artifact_files(path: Path, meta: dict[str, Any]) -> list[str]:
    """Validate manifest-declared files and return human-readable issues."""
    files = meta.get("files")
    if not isinstance(files, dict):
        return []

    base_dir = _artifact_dir(path)
    issues: list[str] = []
    for raw_name, expected in files.items():
        if not isinstance(raw_name, str) or not isinstance(expected, dict):
            issues.append(f"Invalid manifest file entry: {raw_name!r}")
            continue

        relative_path = _safe_relative_path(raw_name)
        if relative_path is None:
            issues.append(f"Unsafe manifest file path: {raw_name}")
            continue

        file_path = base_dir / relative_path
        if not file_path.exists():
            issues.append(f"Missing file: {raw_name}")
            continue
        if not file_path.is_file():
            issues.append(f"Not a file: {raw_name}")
            continue

        expected_size = expected.get("size_bytes")
        if expected_size is not None:
            try:
                if file_path.stat().st_size != int(expected_size):
                    issues.append(f"Size mismatch: {raw_name}")
            except (TypeError, ValueError):
                issues.append(f"Invalid size for: {raw_name}")

        expected_sha = expected.get("sha256")
        if expected_sha:
            actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_sha.lower() != str(expected_sha).lower():
                issues.append(f"SHA-256 mismatch: {raw_name}")

    return issues


def summarize_artifact(meta: dict[str, Any], console: Console, path: Path | None = None) -> None:
    """Pretty-print artifact metadata to the console."""
    title = meta.get("name") or meta.get("model_id") or "UltraCompress artifact"
    content = []

    base = meta.get("base_model") or meta.get("teacher") or meta.get("source_model")
    if base:
        content.append(f"[bold]Base model:[/bold] {base}")

    bpw = meta.get("bpw") or meta.get("bits_per_weight")
    if bpw is not None:
        formatted_bpw = _format_float(bpw, 3)
        content.append(f"[bold]Compression:[/bold] {formatted_bpw or bpw} bits per weight")

    ratio = meta.get("compression_ratio") or meta.get("ratio")
    if ratio is not None:
        formatted_ratio = _format_float(ratio, 2)
        content.append(f"[bold]Ratio:[/bold] {formatted_ratio or ratio}×")

    method = meta.get("method") or meta.get("track")
    if method:
        content.append(f"[bold]Method:[/bold] {method}")

    uc_version = meta.get("uc_version") or meta.get("ultracompress_version")
    if uc_version:
        content.append(f"[bold]UC version:[/bold] {uc_version}")

    verification_issues = verify_artifact_files(path, meta) if path is not None else []
    if path is not None and isinstance(meta.get("files"), dict):
        if verification_issues:
            content.append("[bold yellow]Manifest verification:[/bold yellow] warnings found")
        else:
            content.append("[bold green]Manifest verification:[/bold green] OK")

    console.print(Panel("\n".join(content), title=f"[cyan]{title}[/cyan]", border_style="cyan"))

    if verification_issues:
        console.print("[yellow]Manifest warnings:[/yellow]")
        for issue in verification_issues:
            console.print(f"  [yellow]-[/yellow] {issue}")

    files = meta.get("files")
    if isinstance(files, dict) and files:
        file_table = Table(title="Manifest files", show_header=True, header_style="bold cyan")
        file_table.add_column("Path", style="bright_white")
        file_table.add_column("Size", justify="right", style="dim")
        file_table.add_column("SHA-256", style="dim")
        for raw_name, expected in files.items():
            if not isinstance(expected, dict):
                continue
            sha = str(expected.get("sha256", ""))
            file_table.add_row(
                str(raw_name),
                str(expected.get("size_bytes", "?")),
                f"{sha[:12]}..." if sha else "?",
            )
        console.print(file_table)

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
