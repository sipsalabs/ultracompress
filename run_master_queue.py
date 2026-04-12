"""MASTER QUEUE — Run everything that hasn't been tested yet.

Runs AFTER run_everything.py finishes (FRR V2 + GWE + Breakers).
This covers: FRR V3, HWI, FRR Demo, and more.
"""
import subprocess, sys, time

scripts = [
    ("ABLATION STUDY (scientific method)", "run_ablation_study.py", "ablation_output.log"),
    ("HWI (Holographic)", "run_moonshot_hwi.py", "hwi_output.log"),
    ("FRR Demo (text generation)", "run_frr_demo.py", "frr_demo_output.log"),
]

print("=" * 60)
print("MASTER QUEUE — Testing everything")
print(f"Scripts to run: {len(scripts)}")
print("=" * 60)
sys.stdout.flush()

for name, script, log in scripts:
    print(f"\n{'='*60}")
    print(f"LAUNCHING: {name}")
    print(f"Script: {script}")
    print(f"Log: {log}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u", script],
            stdout=open(log, 'w'),
            stderr=subprocess.STDOUT,
            timeout=7200,  # 2 hour max per experiment
        )
        elapsed = time.time() - t0
        print(f"COMPLETED: {name} in {elapsed/60:.0f} min (exit code {result.returncode})")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"TIMEOUT: {name} after {elapsed/60:.0f} min")
    except Exception as e:
        print(f"ERROR: {name} - {e}")
    sys.stdout.flush()

print("\n" + "=" * 60)
print("MASTER QUEUE COMPLETE")
print("=" * 60)

# Print summary of all results
print("\nChecking result files...")
import json, os
for rfile in ['frr_v3_results.json', 'hwi_results.json', 'gwe_results.json',
              'moonshot_frr_results.json', 'supertrain_results.json']:
    path = os.path.join(os.path.dirname(__file__), rfile)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        best_key = max(data, key=lambda k: data[k].get('top10', 0))
        best = data[best_key]
        print(f"  {rfile}: best={best_key} top10={best.get('top10',0)*100:.0f}%")
