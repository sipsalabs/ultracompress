"""
Extreme long-training sweep: 40K steps on h=32 and h=16.
Test if the long-training gap-close effect holds at the very bottom.
Runs on GPU 0 (CUDA_VISIBLE_DEVICES=0).
"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DIMS = [32, 16]
STEPS = 40000

env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = '0'

for h in DIMS:
    tag = f'h{h}_long'
    print(f"\n{'='*60}\n  Launching TinyFRR h={h} for 40K steps (tag={tag})\n{'='*60}", flush=True)
    t0 = time.time()
    log = f'tinyfrr_{tag}.log'
    with open(log, 'w') as lf:
        p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                              '--h', str(h), '--steps', str(STEPS),
                              '--tag', tag],
                             stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = p.wait()
    dt = time.time() - t0
    print(f"[extreme_long] {tag} done rc={rc} in {dt/60:.1f} min", flush=True)

print("\n[extreme_long] ALL DONE", flush=True)
