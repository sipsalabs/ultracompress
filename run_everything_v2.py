"""RUN EVERYTHING V2 — All untested approaches, prioritized by impact.

Priority order:
1. Ablation study (scientific method — test each FRR enhancement individually)
2. HWI moonshot (holographic — completely different paradigm)
3. FRR Demo (text generation — the proof people can see)
4. Breakers (Swarm + Program — fixed from earlier)

GWE confirmed dead (loss flat at 168). Skipped.
"""
import subprocess, sys, time

scripts = [
    ("ABLATION STUDY (PHM + Dendritic + LoRA + HiddenSup + TempAnneal)",
     "run_ablation_study.py", "ablation_output.log"),
    ("HWI MOONSHOT (Holographic Weight Interference)",
     "run_moonshot_hwi.py", "hwi_output.log"),
    ("FRR DEMO (Text Generation)",
     "run_frr_demo.py", "frr_demo_output.log"),
    ("PARADIGM BREAKERS (Seed+Swarm+Program, fixed)",
     "run_paradigm_breakers.py", "breakers_v2_output.log"),
]

print("=" * 60)
print("RUN EVERYTHING V2 — All untested approaches")
print(f"Scripts: {len(scripts)}")
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
sys.stdout.flush()

for name, script, log in scripts:
    print(f"\n{'='*60}")
    print(f"LAUNCHING: {name}")
    print(f"Log: {log}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u", script],
            stdout=open(log, 'w'),
            stderr=subprocess.STDOUT,
            timeout=10800,  # 3 hour max per experiment
        )
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else f"EXIT {result.returncode}"
        print(f"COMPLETED: {name} in {elapsed/60:.0f} min [{status}]")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name} after 3 hours")
    except Exception as e:
        print(f"ERROR: {name} - {e}")
    sys.stdout.flush()

print(f"\n{'='*60}")
print(f"ALL EXPERIMENTS COMPLETE at {time.strftime('%H:%M:%S')}")
print(f"{'='*60}")

# Summary
import json, os
print("\nRESULTS SUMMARY:")
for rfile in ['ablation_results.json', 'hwi_results.json',
              'frr_v3_results.json', 'paradigm_breakers_results.json']:
    path = os.path.join(os.path.dirname(__file__), rfile)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data:
            best_key = max(data, key=lambda k: data[k].get('top10', 0) if isinstance(data[k], dict) else 0)
            best = data[best_key]
            if isinstance(best, dict):
                print(f"  {rfile}: best={best_key} top10={best.get('top10',0)*100:.0f}%")
