"""RUN EVERYTHING — FRR V2 + GWE + Fixed Breakers, sequential on GPU 0."""
import subprocess, sys, time

scripts = [
    ("FRR V2 (hidden supervision)", "run_frr_v2.py", "frr_v2_output.log"),
    ("GWE (genome generates weights)", "run_moonshot_gwe.py", "gwe_output.log"),
    ("Paradigm Breakers (Seed+Swarm+Program)", "run_paradigm_breakers.py", "breakers_v2_output.log"),
]

for name, script, log in scripts:
    print(f"\n{'='*60}")
    print(f"LAUNCHING: {name}")
    print(f"Log: {log}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-u", script],
        stdout=open(log, 'w'),
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - t0
    print(f"COMPLETED: {name} in {elapsed/60:.0f} min (exit code {result.returncode})")
    sys.stdout.flush()

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE")
print("="*60)
