"""UltraCompress CLI entry point.

Commands: list, pull, info, bench, version.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__, HF_ORG, PORTFOLIO_URL
from .pull import pull_model
from .info import read_artifact_metadata, summarize_artifact
from .listing import list_published_models
from .benchmark import run_benchmarks

console = Console()


def _banner() -> None:
    console.print()
    console.print("[bold cyan]UltraCompress[/bold cyan] "
                  f"[dim]v{__version__}[/dim]  "
                  f"[dim]· {PORTFOLIO_URL}[/dim]")
    console.print("[dim]Extreme compression for large language models. Patent pending — USPTO 64/049,511 + 64/049,517[/dim]")
    console.print()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="uc")
def main() -> None:
    """UltraCompress: download and run extremely-compressed language models."""
    pass


@main.command("list")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
def cmd_list(as_json: bool) -> None:
    """List pre-compressed models available on the Hugging Face Hub."""
    _banner()
    with console.status("[cyan]Querying Hugging Face Hub..."):
        models = list_published_models()

    if as_json:
        click.echo(json.dumps(models, indent=2))
        return

    if not models:
        console.print("[yellow]No pre-compressed models published yet.[/yellow]")
        console.print(f"[dim]Check back soon at https://huggingface.co/{HF_ORG}[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Model ID", style="bright_white")
    table.add_column("Base", style="dim")
    table.add_column("bpw", justify="right", style="cyan")
    table.add_column("Size", justify="right", style="dim")
    table.add_column("Downloads", justify="right", style="dim")

    for m in models:
        table.add_row(
            m.get("modelId", "?"),
            m.get("base_model", "?"),
            f"{m.get('bpw', 0):.3f}",
            m.get("size_human", "?"),
            f"{m.get('downloads', 0):,}",
        )
    console.print(table)
    console.print(f"\n[dim]Pull one with:[/dim] [cyan]uc pull {models[0].get('modelId', '<model-id>')}[/cyan]")


@main.command("pull")
@click.argument("model_id", type=str)
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Local output directory (default: ./models/<name>)")
@click.option("--revision", type=str, default=None,
              help="Specific Hub revision (commit SHA or tag) to pull.")
def cmd_pull(model_id: str, output: str | None, revision: str | None) -> None:
    """Download a pre-compressed model to disk."""
    _banner()
    output_path = Path(output) if output else Path("models") / model_id.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]->[/cyan] Pulling [bright_white]{model_id}[/bright_white] "
                  f"to [dim]{output_path}[/dim]")
    if revision:
        console.print(f"[dim]  revision: {revision}[/dim]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("downloading...", total=None)
        local_dir = pull_model(model_id, output_path, revision=revision)
        progress.update(task, completed=True)

    console.print(f"[green]OK[/green] Saved to [bright_white]{local_dir}[/bright_white]")
    console.print(f"[dim]Inspect with:[/dim] [cyan]uc info {local_dir}[/cyan]")


@main.command("info")
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def cmd_info(path: str) -> None:
    """Show metadata for a compressed artifact."""
    _banner()
    meta = read_artifact_metadata(Path(path))
    if meta is None:
        console.print(f"[red]X[/red] No ultracompress metadata found at [bright_white]{path}[/bright_white]")
        console.print("[dim]Is this a directory produced by `uc pull`? "
                      "Or an ultracompress.json manifest?[/dim]")
        sys.exit(1)
    summarize_artifact(meta, console)


@main.command("bench")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--tasks", type=str, default="hellaswag,arc_challenge",
              help="Comma-separated list of lm-eval-harness tasks.")
@click.option("--limit", type=int, default=500, help="Samples per task (default: 500).")
@click.option("--batch-size", type=int, default=8, help="Batch size.")
@click.option("--device", type=str, default="cuda:0", help="PyTorch device.")
@click.option("--output-dir", type=click.Path(), default="./bench-results",
              help="Where to save per-sample logs and summary JSON.")
def cmd_bench(path: str, tasks: str, limit: int, batch_size: int,
              device: str, output_dir: str) -> None:
    """Benchmark a compressed artifact on downstream tasks."""
    _banner()
    console.print(f"[cyan]->[/cyan] Benchmarking [bright_white]{path}[/bright_white] "
                  f"on tasks: [cyan]{tasks}[/cyan]")
    console.print(f"[dim]  limit={limit}  batch_size={batch_size}  device={device}[/dim]")
    console.print()
    try:
        results = run_benchmarks(
            artifact_path=Path(path),
            tasks=tasks.split(","),
            limit=limit,
            batch_size=batch_size,
            device=device,
            output_dir=Path(output_dir),
        )
    except RuntimeError as e:
        console.print(f"[red]X[/red] Benchmark failed: {e}")
        sys.exit(1)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Task", style="bright_white")
    table.add_column("acc", justify="right", style="cyan")
    table.add_column("acc_norm", justify="right", style="cyan")
    table.add_column("stderr", justify="right", style="dim")
    for task, stats in results.items():
        table.add_row(
            task,
            f"{stats.get('acc', 0) * 100:.2f}%",
            f"{stats.get('acc_norm', 0) * 100:.2f}%",
            f"+/-{stats.get('acc_stderr', 0) * 100:.2f}%",
        )
    console.print(table)


@main.command("version")
def cmd_version() -> None:
    """Print version information."""
    click.echo(f"ultracompress {__version__}")


@main.command("demo")
@click.option("--speed", type=float, default=1.0, help="Playback speed multiplier")
@click.option("--no-pause", is_flag=True, help="Skip inter-step pauses")
def cmd_demo(speed: float, no_pause: bool) -> None:
    """Play a scripted demo session for screen recording.

    Realistic CLI output with timing, no HF Hub required.
    """
    from .demo import play_demo
    play_demo(speed=speed, no_pause=no_pause)


if __name__ == "__main__":
    main()
