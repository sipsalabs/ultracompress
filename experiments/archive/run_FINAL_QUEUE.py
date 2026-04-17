"""FINAL QUEUE — Run after run_everything_v2 finishes.

Tests EVERYTHING remaining:
1. MEGA test (all 15 modules individually)
2. FRR-BitNet (ternary from scratch)
3. Evolutionary architecture search
4. FRR text generation demo
"""
import subprocess, sys, time

scripts = [
    ("MEGA TEST (ALL 15 modules)", "run_MEGA_test_all.py", "mega_test_output.log"),
    ("FRR-BITNET (ternary 425x)", "run_frr_bitnet.py", "frr_bitnet_output.log"),
    ("FRR DEMO (text generation)", "run_frr_demo.py", "frr_demo_output.log"),
    # Evolution takes ~40 hours, run last or skip if time limited
    # ("EVOLUTION (genetic search)", "run_evolve_frr.py", "evolution_output.log"),
]

print("=" * 60)
print(f"FINAL QUEUE — {len(scripts)} experiments")
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
sys.stdout.flush()

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
        print(f"COMPLETED: {name} in {(time.time()-t0)/60:.0f} min [exit {result.returncode}]")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name}")
    except Exception as e:
        print(f"ERROR: {name} - {e}")
    sys.stdout.flush()

print(f"\n{'='*60}")
print(f"ALL DONE at {time.strftime('%H:%M:%S')}")
print(f"Run: python results_dashboard.py  to see all results")
print(f"{'='*60}")
