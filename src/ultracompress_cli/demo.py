"""Demo mode - plays a scripted CLI session for video recording.

The `uc demo` command runs the full demo sequence with realistic timing,
spinner animations, and output. Designed for screen recording (OBS, Loom,
asciinema) without needing the HF Hub to actually be populated.

IMPORTANT — the demo uses ILLUSTRATIVE values, not live data. The model
catalog rows, download counts, and benchmark numbers shown by `uc demo`
are scripted demo data labeled DEMO in the output. They do not reflect
live HF Hub state or real downloads. Run `uc list` against a real HF Hub
session for the actual catalog state.

Usage:
    uc demo              # plays the full 60-sec sequence
    uc demo --speed 2    # 2x speed (for testing)
    uc demo --no-pause   # no inter-step pauses (faster runs)
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

SCENES = [
    ("install", 1.0),
    ("list", 4.0),
    ("pull", 6.0),
    ("bench", 8.0),
    ("close", 4.0),
]


# Illustrative demo rows. These are NOT live HF Hub data and NOT real
# download counts. The "downloads" field is intentionally rendered as "—"
# in the demo output so a screencast cannot be misread as showing real
# popularity numbers. Use `uc list` against the live Hub for actual state.
DEMO_MODELS = [
    {
        "modelId": "sipsalabs/qwen3-1.7b-uc2p79",
        "base": "Qwen3-1.7B",
        "bpw": 2.798,
        "size": "635 MB",
    },
    {
        "modelId": "sipsalabs/mistral-7b-uc2p79",
        "base": "Mistral-7B-v0.3",
        "bpw": 2.798,
        "size": "2.7 GB",
    },
    {
        "modelId": "sipsalabs/qwen3-8b-uc2p79",
        "base": "Qwen3-8B",
        "bpw": 2.798,
        "size": "3.0 GB",
    },
    {
        "modelId": "sipsalabs/tinyllama-1.1b-uc2p40",
        "base": "TinyLlama-1.1B",
        "bpw": 2.405,
        "size": "412 MB",
    },
    {
        "modelId": "sipsalabs/olmo2-1b-uc2p79",
        "base": "OLMo-2-1B",
        "bpw": 2.798,
        "size": "452 MB",
    },
    {
        "modelId": "sipsalabs/smollm2-1.7b-uc2p79",
        "base": "SmolLM2-1.7B",
        "bpw": 2.798,
        "size": "672 MB",
    },
]


def _slow_print(console: Console, text: str, delay_per_char: float = 0.025) -> None:
    """Type text slowly, char by char, simulating live keystroke."""
    for ch in text:
        console.print(ch, end="", soft_wrap=True, highlight=False)
        time.sleep(delay_per_char)
    console.print()


def _pause(seconds: float, speed: float = 1.0) -> None:
    """Sleep with speed adjustment."""
    time.sleep(seconds / speed)


def _scene_install(console: Console, speed: float, no_pause: bool) -> None:
    """Show the pip install."""
    console.print("[bold yellow]$[/bold yellow] ", end="")
    _slow_print(console, "pip install ultracompress")
    if not no_pause:
        _pause(0.6, speed)
    console.print("[dim]Collecting ultracompress...[/dim]")
    _pause(0.4, speed)
    console.print("[dim]  Downloading ultracompress-0.1.2-py3-none-any.whl (10 KB)[/dim]")
    _pause(0.3, speed)
    console.print("[green]Successfully installed ultracompress-0.1.2[/green]")
    if not no_pause:
        _pause(0.5, speed)


def _scene_list(console: Console, speed: float, no_pause: bool) -> None:
    """Show `uc list` with the catalog."""
    console.print()
    console.print("[bold yellow]$[/bold yellow] ", end="")
    _slow_print(console, "uc list")
    if not no_pause:
        _pause(0.5, speed)
    console.print()
    console.print("[bold cyan]UltraCompress[/bold cyan] [dim]v0.1.2  ·  sipsalabs.com  ·  DEMO MODE — illustrative data[/dim]")
    console.print("[dim]Extreme compression for large language models. Patent pending — USPTO 64/049,511 + 64/049,517[/dim]")
    console.print()
    with console.status("[cyan]Querying Hugging Face Hub..."):
        _pause(1.0, speed)

    table = Table(show_header=True, header_style="bold cyan",
                  caption="[yellow]DEMO[/yellow] illustrative catalog — run `uc list` against the live Hub for real state",
                  caption_style="dim italic")
    table.add_column("Model ID", style="bright_white")
    table.add_column("Base", style="dim")
    table.add_column("bpw", justify="right", style="cyan")
    table.add_column("Size", justify="right", style="dim")
    for m in DEMO_MODELS:
        table.add_row(
            m["modelId"],
            m["base"],
            f"{m['bpw']:.3f}",
            m["size"],
        )
    console.print(table)
    if not no_pause:
        _pause(1.5, speed)


def _scene_pull(console: Console, speed: float, no_pause: bool) -> None:
    """Show `uc pull` with progress bar."""
    console.print()
    console.print("[bold yellow]$[/bold yellow] ", end="")
    _slow_print(console, "uc pull sipsalabs/qwen3-1.7b-uc2p79")
    if not no_pause:
        _pause(0.4, speed)
    console.print()
    console.print("[bold cyan]UltraCompress[/bold cyan] [dim]v0.1.2  ·  sipsalabs.com  ·  DEMO MODE[/dim]")
    console.print()
    console.print("[cyan]->[/cyan] Pulling [bright_white]sipsalabs/qwen3-1.7b-uc2p79[/bright_white] "
                  "to [dim]./models/sipsalabs_qwen3-1.7b-uc2p79[/dim]")

    total_mb = 635
    bar_width = 40
    download_seconds = 4.0 / speed
    steps = 5
    sleep_per_step = download_seconds / steps
    for i in range(1, steps + 1):
        pct = i / steps
        filled = int(pct * bar_width)
        empty = bar_width - filled
        downloaded = int(pct * total_mb)
        bar = "[bold cyan]" + "#" * filled + "[/bold cyan]" + "[dim]" + "-" * empty + "[/dim]"
        console.print(f"  downloading... [{bar}]  {downloaded} MB / {total_mb} MB",
                      soft_wrap=False)
        time.sleep(sleep_per_step)
    console.print("[green]OK[/green] Saved to [bright_white]./models/sipsalabs_qwen3-1.7b-uc2p79[/bright_white]")
    if not no_pause:
        _pause(1.0, speed)


def _scene_bench(console: Console, speed: float, no_pause: bool) -> None:
    """Show `uc bench` with results."""
    console.print()
    console.print("[bold yellow]$[/bold yellow] ", end="")
    _slow_print(console, "uc bench ./models/sipsalabs_qwen3-1.7b-uc2p79 --tasks hellaswag --limit 500")
    if not no_pause:
        _pause(0.4, speed)
    console.print()
    console.print("[bold cyan]UltraCompress[/bold cyan] [dim]v0.1.2  ·  DEMO MODE[/dim]")
    console.print()
    console.print("[cyan]->[/cyan] Benchmarking on tasks: [cyan]hellaswag[/cyan]  [dim]limit=500  device=cuda:0[/dim]")
    console.print()

    with console.status("[cyan]Running lm-eval-harness on hellaswag (500 samples)..."):
        _pause(2.5, speed)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Task", style="bright_white")
    table.add_column("acc", justify="right", style="cyan")
    table.add_column("acc_norm", justify="right", style="cyan")
    table.add_column("stderr", justify="right", style="dim")
    table.add_row("hellaswag", "41.60%", "51.80%", "+/-2.20%")
    console.print(table)
    console.print()
    console.print("  [dim]Teacher Qwen3-1.7B (fp16):[/dim]   [bright_white]53.40%[/bright_white]  acc_norm")
    console.print("  [dim]Retention:[/dim]                  [bold green]97.0%[/bold green]")
    console.print("  [dim]Disk savings vs NF4:[/dim]        [bold green]30%[/bold green]")
    if not no_pause:
        _pause(2.0, speed)


def _scene_close(console: Console, speed: float, no_pause: bool) -> None:
    """Closing card."""
    console.print()
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print()
    console.print("  [bold cyan]UltraCompress[/bold cyan]  [dim]·[/dim]  Extreme compression for LLMs")
    console.print()
    console.print("  [bold]pip install ultracompress[/bold]")
    console.print()
    console.print("  [dim]sipsalabs.com[/dim]")
    console.print("  [dim]Patent pending — USPTO 64/049,511 + 64/049,517[/dim]")
    console.print()
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    if not no_pause:
        _pause(2.0, speed)


def play_demo(speed: float = 1.0, no_pause: bool = False) -> None:
    """Play the full demo sequence."""
    console = Console()

    console.print()
    _scene_install(console, speed, no_pause)
    _scene_list(console, speed, no_pause)
    _scene_pull(console, speed, no_pause)
    _scene_bench(console, speed, no_pause)
    _scene_close(console, speed, no_pause)
