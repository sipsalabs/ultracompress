"""COMPLETE PIPELINE — Wait for current experiments, then run EVERYTHING remaining.

Waits for run_everything_v2.py to finish, then launches:
1. MEGA test (all 15 modules)
2. FRR-BitNet (ternary 425x)
3. FRR Demo (text generation)
4. Impossible tests
5. FRR V3 (LoRA, fixed)

Nothing skipped. Nothing forgotten.
"""
import subprocess, sys, time, os

def wait_for_process(name_fragment, check_interval=30):
    """Wait until a process matching the name fragment is no longer running."""
    import re
    while True:
        result = subprocess.run(
            ['wmic', 'process', 'where', f"name like '%python%'", 'get', 'commandline'],
            capture_output=True, text=True, errors='ignore'
        )
        if name_fragment not in result.stdout:
            return
        time.sleep(check_interval)

print("=" * 60)
print("COMPLETE PIPELINE — EVERYTHING, NO EXCEPTIONS")
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Wait for current experiments to finish
print("Waiting for run_everything_v2.py to finish...")
sys.stdout.flush()
wait_for_process("run_everything_v2")
print(f"run_everything_v2 finished at {time.strftime('%H:%M:%S')}")

scripts = [
    ("MEGA TEST (ALL 15 modules individually)", "run_MEGA_test_all.py", "mega_test_output.log"),
    ("FRR MAX QUALITY (hybrid head+tail+FRR)", "run_frr_maxquality.py", "frr_maxquality_output.log"),
    ("FRR-BITNET (ternary from scratch, 425x)", "run_frr_bitnet.py", "frr_bitnet_output.log"),
    ("IMPOSSIBLE TESTS (5 wild ideas)", "run_test_impossible.py", "impossible_output.log"),
    ("FRR DEMO (text generation proof)", "run_frr_demo.py", "frr_demo_output.log"),
    ("FRR V3 (LoRA adapters, fixed)", "run_frr_v3.py", "frr_v3_output.log"),
]

for name, script, log in scripts:
    print(f"\n{'='*60}")
    print(f"LAUNCHING: {name}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    sys.stdout.flush()
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u", script],
            stdout=open(log, 'w'), stderr=subprocess.STDOUT,
            timeout=14400)
        elapsed = time.time() - t0
        print(f"COMPLETED: {name} in {elapsed/60:.0f} min [exit {result.returncode}]")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name} after 4h")
    except Exception as e:
        print(f"ERROR: {name} - {e}")
    sys.stdout.flush()

# Final dashboard
print(f"\n{'='*60}")
print(f"ALL EXPERIMENTS COMPLETE at {time.strftime('%H:%M:%S')}")
print(f"{'='*60}")
subprocess.run([sys.executable, "results_dashboard.py"])
