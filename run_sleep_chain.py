"""
OVERNIGHT CHAIN: Wait for MEGA, then run critical experiments.
Sip goes to bed ~10:30 PM, wakes ~6 AM.

Timeline:
  MEGA: ~5 more hours (finishes ~3:30 AM)
  E2E Proof: ~30 min
  Intermediate matching (3 best configs): ~1.25 hrs
  Multi-block FRR (2 configs): ~1 hr
  Total: finishes ~6:15 AM

Priority order (most important first):
  1. E2E proof - proves full stack works
  2. Intermediate matching - biggest quality improvement
  3. Multi-block FRR - quality breakthrough
"""
import lib.unbuffered
import subprocess
import time
import os
import sys
import psutil

MEGA_PID = 26205  # The running MEGA test

def wait_for_pid(pid, name="process", check_interval=60):
    """Wait for a process to finish."""
    print(f"Waiting for {name} (PID {pid})...")
    while True:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(check_interval)
            print(f"  [{time.strftime('%H:%M:%S')}] {name} still running...")
        except psutil.NoSuchProcess:
            break
    print(f"  {name} finished at {time.strftime('%H:%M:%S')}")


def run_script(name, script, log, timeout_hours=2):
    """Run a script with timeout and logging."""
    print(f"\n{'='*60}")
    print(f"LAUNCHING: {name}")
    print(f"Log: {log}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    sys.stdout.flush()

    try:
        with open(log, 'w') as f:
            result = subprocess.run(
                [sys.executable, '-u', script],
                stdout=f, stderr=subprocess.STDOUT,
                timeout=int(timeout_hours * 3600),
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
            )
        status = 'OK' if result.returncode == 0 else f'EXIT {result.returncode}'
    except subprocess.TimeoutExpired:
        status = 'TIMEOUT'
    except Exception as e:
        status = f'ERROR: {e}'

    elapsed = time.strftime('%H:%M:%S')
    print(f"COMPLETED: {name} at {elapsed} [{status}]")
    sys.stdout.flush()
    return status


if __name__ == '__main__':
    start = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{'='*60}")
    print(f"OVERNIGHT CHAIN - Sip's sleeping, we're working")
    print(f"Started: {start}")
    print(f"{'='*60}")

    # Step 0: Wait for MEGA to finish
    try:
        wait_for_pid(MEGA_PID, "MEGA test", check_interval=120)
    except Exception as e:
        print(f"  MEGA wait failed ({e}), continuing anyway...")

    # Step 1: E2E Proof (CRITICAL - proves the full stack)
    run_script(
        "E2E PROOF: FRR -> Compress -> Decompress -> Quality",
        "run_e2e_proof.py",
        "e2e_proof_output.log",
        timeout_hours=1.5
    )

    # Step 2: Intermediate matching (biggest quality win)
    # Modified version with only 3 configs instead of 6
    run_script(
        "INTERMEDIATE MATCHING: Hidden state distillation",
        "run_frr_intermediate_fast.py",
        "intermediate_output.log",
        timeout_hours=2.0
    )

    # Step 3: Multi-block FRR (quality breakthrough)
    run_script(
        "MULTI-BLOCK FRR: 2-3 specialized blocks",
        "run_after_mega.py",
        "after_mega_output.log",
        timeout_hours=2.5
    )

    end = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"ALL OVERNIGHT EXPERIMENTS COMPLETE")
    print(f"Started: {start}")
    print(f"Ended: {end}")
    print(f"{'='*60}")

    # Print quick summary
    print(f"\nCheck results in:")
    print(f"  mega_test_output.log")
    print(f"  e2e_proof_output.log")
    print(f"  intermediate_output.log")
    print(f"  after_mega_output.log")
    print(f"\nGood morning Sip!")
