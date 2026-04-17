"""
Extreme sweep: find the actual wall.
h=96, 64, 48, 32, 16. 8K steps each.
"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DIMS = [96, 64, 48, 32, 16]
STEPS = 8000

for h in DIMS:
    print(f"\n{'='*60}\n  Launching TinyFRR h={h}\n{'='*60}", flush=True)
    t0 = time.time()
    log = f'tinyfrr_h{h}.log'
    with open(log, 'w') as lf:
        p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                              '--h', str(h), '--steps', str(STEPS)],
                             stdout=lf, stderr=subprocess.STDOUT)
        rc = p.wait()
    if rc != 0:
        print(f"[sweep] h={h} crashed rc={rc}, retrying once", flush=True)
        with open(log, 'a') as lf:
            lf.write(f"\n=== RETRY ===\n")
            p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                                  '--h', str(h), '--steps', str(STEPS)],
                                 stdout=lf, stderr=subprocess.STDOUT)
            rc = p.wait()
    dt = time.time() - t0
    print(f"[sweep] h={h} done rc={rc} in {dt/60:.1f} min", flush=True)

print("\n[sweep3] ALL DONE", flush=True)
