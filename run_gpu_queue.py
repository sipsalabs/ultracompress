"""
GPU EXPERIMENT QUEUE — Runs experiments sequentially on a specific GPU.

Keeps the GPU utilized 24/7. When one experiment finishes, the next starts
automatically. Logs results, commits to git, and sends status updates.

Usage:
  python run_gpu_queue.py --device cuda:0 --queue wave,8b
  python run_gpu_queue.py --device cuda:1 --queue 8b          # just 8B after 1.7B finishes
  python run_gpu_queue.py --list                               # show available experiments

Each experiment is defined with:
  - name: short identifier
  - script: path to the training script
  - args: command-line arguments
  - description: what it tests
  - depends_on: optional, wait for another experiment's checkpoint
"""
import lib.unbuffered
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Experiment:
    name: str
    script: str
    args: list[str] = field(default_factory=list)
    description: str = ""
    estimated_hours: float = 0
    depends_on: str | None = None


# ── Experiment Registry ───────────────────────────────────────────────
EXPERIMENTS = {
    "wave": Experiment(
        name="wave",
        script="run_stable_wave_test.py",
        description="Stable wave engine (spectral norm + magnitude clamping), 30K steps",
        estimated_hours=1.5,
    ),
    "8b": Experiment(
        name="8b",
        script="run_8b_real_text.py",
        args=["--steps", "50000"],
        description="8B teacher streaming distillation, 50K steps, real text",
        estimated_hours=12,
    ),
    "8b-long": Experiment(
        name="8b-long",
        script="run_8b_real_text.py",
        args=["--steps", "100000"],
        description="8B teacher streaming distillation, 100K steps",
        estimated_hours=24,
    ),
    "eval-1.7b-10k": Experiment(
        name="eval-1.7b-10k",
        script="eval_checkpoint.py",
        args=["checkpoints_1.7b_real_text/frr_1.7b_step10000.pt", "--teacher", "1.7b"],
        description="Full eval (HellaSwag + WikiText-2) on 1.7B step 10K checkpoint",
        estimated_hours=0.5,
    ),
    "eval-1.7b-best": Experiment(
        name="eval-1.7b-best",
        script="eval_checkpoint.py",
        args=["checkpoints_1.7b_real_text/frr_1.7b_best.pt", "--teacher", "1.7b"],
        description="Full eval on best 1.7B checkpoint",
        estimated_hours=0.5,
    ),
    "diag-1.7b-best": Experiment(
        name="diag-1.7b-best",
        script="run_diagnostic_eval.py",
        args=["checkpoints_1.7b_real_text/frr_1.7b_best.pt", "--teacher", "1.7b", "--samples", "500"],
        description="Deep diagnostic (500 samples, all positions, entropy) on best 1.7B",
        estimated_hours=0.3,
    ),
    "cyclic-1.7b": Experiment(
        name="cyclic-1.7b",
        script="run_1.7b_cyclic_temp.py",
        description="Resume 1.7B best with cyclic temp (2.0↔4.0), 50K steps",
        estimated_hours=6,
    ),
}


def list_experiments():
    print(f"\n{'=' * 70}")
    print("AVAILABLE EXPERIMENTS")
    print(f"{'=' * 70}")
    for name, exp in EXPERIMENTS.items():
        hrs = f"~{exp.estimated_hours}h" if exp.estimated_hours else "?"
        dep = f" (after {exp.depends_on})" if exp.depends_on else ""
        print(f"  {name:<20} {hrs:<8} {exp.description}{dep}")
    print(f"\nUsage: python run_gpu_queue.py --device cuda:0 --queue wave,8b")


def git_commit(message: str):
    """Commit and push results."""
    try:
        subprocess.run(["git", "add", "-A"], capture_output=True, timeout=30)
        subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "push", "origin", "master"],
            capture_output=True, timeout=60,
        )
        print(f"  [git] Committed: {message[:60]}...")
    except Exception as e:
        print(f"  [git] Commit failed: {e}")


def run_experiment(exp: Experiment, device: str) -> tuple[bool, float]:
    """Run a single experiment. Returns (success, elapsed_seconds)."""
    print(f"\n{'=' * 70}")
    print(f"STARTING: {exp.name}")
    print(f"  Script: {exp.script}")
    print(f"  Device: {device}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if exp.description:
        print(f"  {exp.description}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    # Build command
    cmd = [sys.executable, exp.script]

    # Inject --device if the script supports it
    if exp.script in ("run_8b_real_text.py", "eval_checkpoint.py"):
        cmd += ["--device", device]

    cmd += exp.args

    # Set CUDA_VISIBLE_DEVICES for scripts that use 'cuda' without device index
    env = os.environ.copy()
    if device.startswith("cuda:"):
        gpu_idx = device.split(":")[1]
        # Only set for scripts that don't take --device
        if exp.script not in ("run_8b_real_text.py", "eval_checkpoint.py"):
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx

    t0 = time.time()
    log_file = f"{exp.name}_queue_output.log"

    try:
        with open(log_file, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()

            proc.wait()

        elapsed = time.time() - t0
        success = proc.returncode == 0

        status = "COMPLETED" if success else f"FAILED (exit code {proc.returncode})"
        print(f"\n  [{exp.name}] {status} in {elapsed / 3600:.1f} hours")
        return success, elapsed

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  [{exp.name}] EXCEPTION: {e} ({elapsed / 3600:.1f} hours)")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="GPU Experiment Queue")
    parser.add_argument("--device", default="cuda:0", help="GPU to use")
    parser.add_argument(
        "--queue", type=str, default="",
        help="Comma-separated experiment names (e.g. wave,8b)",
    )
    parser.add_argument("--list", action="store_true", help="List available experiments")
    args = parser.parse_args()

    if args.list or not args.queue:
        list_experiments()
        return

    experiment_names = [n.strip() for n in args.queue.split(",") if n.strip()]
    experiments = []
    for name in experiment_names:
        if name not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment '{name}'. Use --list to see options.")
            sys.exit(1)
        experiments.append(EXPERIMENTS[name])

    total_hours = sum(e.estimated_hours for e in experiments)
    print(f"{'=' * 70}")
    print(f"GPU EXPERIMENT QUEUE")
    print(f"  Device: {args.device}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Estimated total: ~{total_hours:.1f} hours")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
    for i, exp in enumerate(experiments):
        print(f"  {i + 1}. {exp.name}: {exp.description}")
    print()

    results = []
    for i, exp in enumerate(experiments):
        print(f"\n>>> Queue item {i + 1}/{len(experiments)}: {exp.name}")

        # Check dependency
        if exp.depends_on:
            dep_result = next((r for r in results if r[0] == exp.depends_on), None)
            if dep_result and not dep_result[1]:
                print(f"  SKIPPING: dependency '{exp.depends_on}' failed")
                results.append((exp.name, False, 0))
                continue

        success, elapsed = run_experiment(exp, args.device)
        results.append((exp.name, success, elapsed))

        # Auto-commit after each experiment
        status = "completed" if success else "FAILED"
        git_commit(
            f"[queue] {exp.name} {status} ({elapsed / 3600:.1f}h)\n\n"
            f"Device: {args.device}\n"
            f"Script: {exp.script}"
        )

    # ── Final Summary ─────────────────────────────────────────────
    total_elapsed = sum(r[2] for r in results)
    print(f"\n{'=' * 70}")
    print(f"QUEUE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
    for name, success, elapsed in results:
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {name}: {elapsed / 3600:.1f}h")
    print(f"\n  Total: {total_elapsed / 3600:.1f} hours")
    print(f"  Successful: {sum(1 for _, s, _ in results if s)}/{len(results)}")

    # Final commit
    git_commit(
        f"[queue] All {len(results)} experiments done "
        f"({sum(1 for _, s, _ in results if s)}/{len(results)} passed, "
        f"{total_elapsed / 3600:.1f}h total)"
    )


if __name__ == "__main__":
    main()
