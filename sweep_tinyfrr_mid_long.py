"""
Mid long-training: 40K steps on h=64. Fills gap between h=48_long (64.86%) and
h=128_long (68.44%) in the Pareto curve. Runs on GPU 1.
"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DIMS = [64, 96]  # 96 also missing long-training
STEPS = 40000

env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = '1'

for h in DIMS:
    tag = f'h{h}_long'
    print(f"\n{'='*60}\n  Launching TinyFRR h={h} for 40K steps on GPU 1 (tag={tag})\n{'='*60}", flush=True)
    t0 = time.time()
    log = f'tinyfrr_{tag}.log'
    with open(log, 'w') as lf:
        p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                              '--h', str(h), '--steps', str(STEPS),
                              '--tag', tag],
                             stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = p.wait()
    dt = time.time() - t0
    print(f"[mid_long] {tag} done rc={rc} in {dt/60:.1f} min", flush=True)

print("\n[mid_long] ALL DONE", flush=True)
