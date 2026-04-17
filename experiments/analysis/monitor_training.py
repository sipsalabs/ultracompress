"""
Live training monitor — parses terminal output to track training progress.

Reads saved terminal output or log files and generates:
  - Summary table of all training runs
  - Best results tracking
  - Estimated time to completion
  - Trend detection (improving/plateauing/degrading)

Usage:
  python monitor_training.py                          # Show all running experiments
  python monitor_training.py --log output.log         # Parse specific log file
  python monitor_training.py --watch                  # Continuous monitoring mode
"""
import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field

os.chdir(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class StepData:
    step: int
    total_steps: int
    loss: float
    t1: float
    t10: float
    temperature: float | None = None
    elapsed: float = 0.0
    new_best: bool = False


@dataclass
class ExperimentData:
    name: str
    steps: list[StepData] = field(default_factory=list)
    trainable_params: int = 0
    is_final: bool = False
    final_t1: float = 0.0
    final_t10: float = 0.0


# ── Regex patterns ────────────────────────────────────────────────────
STEP_RE = re.compile(
    r'Step\s+(\d+)/(\d+):\s+loss=([\d.]+)\s+'
    r'T1=([\d.]+)%\s+T10=([\d.]+)%\s+'
    r'(?:T=([\d.]+)\s+)?'
    r'\((\d+)s\)'
    r'(\s+\*\*\* NEW BEST \*\*\*)?'
)

FINAL_RE = re.compile(
    r'FINAL:\s+T1=([\d.]+)%\s+T10=([\d.]+)%\s+\((\d+)s\)'
)

EXP_RE = re.compile(
    r'Experiment\s+(\d+):\s*(.*)', re.IGNORECASE
)

PARAMS_RE = re.compile(r'(?:Trainable params|FRR):\s+([\d,]+)')


def parse_output(text: str) -> list[ExperimentData]:
    """Parse training output into structured experiment data."""
    experiments = []
    current = ExperimentData(name="Main")

    for line in text.split('\n'):
        # New experiment
        m = EXP_RE.search(line)
        if m:
            if current.steps:
                experiments.append(current)
            current = ExperimentData(name=f"Exp {m.group(1)}: {m.group(2).strip()}")
            continue

        # Params
        m = PARAMS_RE.search(line)
        if m:
            current.trainable_params = int(m.group(1).replace(',', ''))
            continue

        # Step data
        m = STEP_RE.search(line)
        if m:
            sd = StepData(
                step=int(m.group(1)),
                total_steps=int(m.group(2)),
                loss=float(m.group(3)),
                t1=float(m.group(4)),
                t10=float(m.group(5)),
                temperature=float(m.group(6)) if m.group(6) else None,
                elapsed=float(m.group(7)),
                new_best=bool(m.group(8)),
            )
            current.steps.append(sd)
            continue

        # Final result
        m = FINAL_RE.search(line)
        if m:
            current.is_final = True
            current.final_t1 = float(m.group(1))
            current.final_t10 = float(m.group(2))

    if current.steps or current.is_final:
        experiments.append(current)

    return experiments


def analyze_trend(steps: list[StepData], metric: str = 't10', window: int = 3) -> str:
    """Analyze if metric is improving, plateauing, or degrading."""
    if len(steps) < 2:
        return "insufficient_data"

    values = [getattr(s, metric) for s in steps]

    if len(values) < window + 1:
        window = len(values) - 1

    recent = values[-window:]
    earlier = values[-(window * 2):-window] if len(values) >= window * 2 else values[:window]

    recent_avg = sum(recent) / len(recent)
    earlier_avg = sum(earlier) / len(earlier)

    diff = recent_avg - earlier_avg
    if abs(diff) < 0.5:
        return "PLATEAU"
    elif diff > 0:
        return "IMPROVING ↑"
    else:
        return "DECLINING ↓"


def estimate_completion(steps: list[StepData]) -> str:
    """Estimate time to completion based on current pace."""
    if len(steps) < 2:
        return "unknown"

    last = steps[-1]
    if last.step >= last.total_steps - 1:
        return "COMPLETE"

    # Use average time per step from recent data
    first = steps[0]
    if last.elapsed <= first.elapsed or last.step <= first.step:
        return "unknown"

    secs_per_step = (last.elapsed - first.elapsed) / (last.step - first.step)
    remaining_steps = last.total_steps - last.step
    remaining_secs = remaining_steps * secs_per_step

    hours = remaining_secs / 3600
    if hours > 1:
        return f"~{hours:.1f}h remaining"
    else:
        return f"~{remaining_secs/60:.0f}m remaining"


def display_experiment(exp: ExperimentData) -> None:
    """Display experiment summary."""
    print(f"\n{'─' * 60}")
    print(f"  {exp.name}")
    if exp.trainable_params:
        print(f"  Params: {exp.trainable_params:,}")
    print(f"{'─' * 60}")

    if not exp.steps:
        if exp.is_final:
            print(f"  FINAL: T1={exp.final_t1:.1f}% T10={exp.final_t10:.1f}%")
        return

    # Current status
    last = exp.steps[-1]
    progress = last.step / last.total_steps * 100
    print(f"  Progress: {last.step:,}/{last.total_steps:,} ({progress:.1f}%)")
    print(f"  Current:  loss={last.loss:.4f}  T1={last.t1:.1f}%  T10={last.t10:.1f}%")
    if last.temperature is not None:
        print(f"  Temp:     {last.temperature:.1f}")

    # Best results
    best_t10 = max(exp.steps, key=lambda s: s.t10)
    best_t1 = max(exp.steps, key=lambda s: s.t1)
    print(f"  Best T10: {best_t10.t10:.1f}% at step {best_t10.step:,}")
    print(f"  Best T1:  {best_t1.t1:.1f}% at step {best_t1.step:,}")

    # Trend
    t10_trend = analyze_trend(exp.steps, 't10')
    loss_trend = analyze_trend(exp.steps, 'loss')
    print(f"  T10 trend: {t10_trend}")
    print(f"  Loss trend: {loss_trend}")

    # ETA
    eta = estimate_completion(exp.steps)
    print(f"  ETA: {eta}")

    if exp.is_final:
        print(f"  ★ FINAL: T1={exp.final_t1:.1f}% T10={exp.final_t10:.1f}%")

    # Step history table
    print(f"\n  {'Step':>8} {'Loss':>10} {'T1':>6} {'T10':>6}", end="")
    if exp.steps[0].temperature is not None:
        print(f" {'Temp':>6}", end="")
    print(f" {'Time':>8} {'Note':>12}")
    print(f"  {'─'*8} {'─'*10} {'─'*6} {'─'*6}", end="")
    if exp.steps[0].temperature is not None:
        print(f" {'─'*6}", end="")
    print(f" {'─'*8} {'─'*12}")

    for s in exp.steps:
        note = ""
        if s.new_best:
            note = "★ NEW BEST"
        if s == best_t10 and not s.new_best:
            note = "peak T10"
        if s == best_t1 and not s.new_best and best_t1 != best_t10:
            note = "peak T1"

        print(f"  {s.step:>8,} {s.loss:>10.4f} {s.t1:>5.1f}% {s.t10:>5.1f}%", end="")
        if s.temperature is not None:
            print(f" {s.temperature:>5.1f}", end="")
        elapsed_min = s.elapsed / 60
        print(f" {elapsed_min:>7.1f}m {note:>12}")


def main():
    parser = argparse.ArgumentParser(description="Training monitor")
    parser.add_argument('--log', nargs='+', help="Log file(s) to parse")
    parser.add_argument('--dir', default='.', help="Directory to scan for log files")
    args = parser.parse_args()

    files_to_parse = []

    if args.log:
        files_to_parse = args.log
    else:
        # Scan for common log patterns
        for f in os.listdir(args.dir):
            if f.endswith('.log') or f.endswith('_output.txt'):
                files_to_parse.append(os.path.join(args.dir, f))

    if not files_to_parse:
        print("No log files found. Paste terminal output via stdin or use --log.")
        print("Reading from stdin (Ctrl+Z to finish)...")
        text = sys.stdin.read()
        experiments = parse_output(text)
    else:
        experiments = []
        for fp in files_to_parse:
            print(f"Parsing: {fp}")
            with open(fp) as f:
                text = f.read()
            experiments.extend(parse_output(text))

    if not experiments:
        print("No training data found in input.")
        return

    # ── Display ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  TRAINING MONITOR")
    print("═" * 60)

    for exp in experiments:
        display_experiment(exp)

    # ── All-time bests ────────────────────────────────────────────
    all_steps = [(exp.name, s) for exp in experiments for s in exp.steps]
    if all_steps:
        best_t10_pair = max(all_steps, key=lambda x: x[1].t10)
        best_t1_pair = max(all_steps, key=lambda x: x[1].t1)
        print(f"\n{'═' * 60}")
        print(f"  ALL-TIME BESTS (across {len(experiments)} experiments)")
        print(f"{'═' * 60}")
        print(f"  Best T10: {best_t10_pair[1].t10:.1f}% [{best_t10_pair[0]}, step {best_t10_pair[1].step:,}]")
        print(f"  Best T1:  {best_t1_pair[1].t1:.1f}% [{best_t1_pair[0]}, step {best_t1_pair[1].step:,}]")


if __name__ == "__main__":
    main()
