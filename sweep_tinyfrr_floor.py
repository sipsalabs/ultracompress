"""
Floor sweep: push to absolute extreme. h=8, 4, 2. 8K steps each.
Runs on GPU 1 (CUDA_VISIBLE_DEVICES=1 set by caller).
"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DIMS = [8, 4]  # h=2 would need n_heads<=2 but script candidates don't include that
STEPS = 8000

env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = '1'

for h in DIMS:
    print(f"\n{'='*60}\n  Launching TinyFRR h={h} on GPU 1\n{'='*60}", flush=True)
    t0 = time.time()
    log = f'tinyfrr_h{h}.log'
    with open(log, 'w') as lf:
        p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                              '--h', str(h), '--steps', str(STEPS)],
                             stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = p.wait()
    if rc != 0:
        print(f"[floor] h={h} crashed rc={rc}, retrying once", flush=True)
        with open(log, 'a') as lf:
            lf.write(f"\n=== RETRY ===\n")
            p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                                  '--h', str(h), '--steps', str(STEPS)],
                                 stdout=lf, stderr=subprocess.STDOUT, env=env)
            rc = p.wait()
    dt = time.time() - t0
    print(f"[floor] h={h} done rc={rc} in {dt/60:.1f} min", flush=True)

print("\n[floor-sweep] ALL DONE", flush=True)
