"""
Supervisor that waits for both long-training sweeps to finish, then runs PTQ
on the best checkpoints automatically.

Expected flow:
  - Poll nvidia-smi until sweep_tinyfrr_extreme_long (GPU 0) and
    sweep_tinyfrr_mid_long (GPU 1) finish.
  - Then run PTQ on all long checkpoints + h=512 (best main-sweep).
  - Also run hires eval on the new long checkpoints.
"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Checkpoints we want to wait for
TARGETS_GPU0 = ['h32_long', 'h16_long']  # sequential
TARGETS_GPU1 = ['h64_long', 'h96_long']

def done(tag):
    p = f'checkpoints_1.7b_tinyfrr_{tag}/best.pt'
    if not os.path.exists(p): return False
    # Training is done when log has "DONE tag:" line
    lf = f'tinyfrr_{tag}.log'
    if not os.path.exists(lf): return False
    try:
        with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return f'DONE {tag}:' in content
    except: return False

print("[wait_ptq] Waiting for all long-training to finish...", flush=True)
while True:
    missing = [t for t in TARGETS_GPU0 + TARGETS_GPU1 if not done(t)]
    if not missing:
        break
    print(f"[wait_ptq] {time.strftime('%H:%M:%S')}  pending: {missing}", flush=True)
    time.sleep(120)

print("\n[wait_ptq] All sweeps done. Running hires eval on new long ckpts.", flush=True)
new_tags = TARGETS_GPU0 + TARGETS_GPU1
rc = subprocess.call([sys.executable, 'eval_tinyfrr_hires.py',
                      '--tags'] + new_tags + ['--n', '500'])
print(f"[wait_ptq] hires eval rc={rc}", flush=True)

print("\n[wait_ptq] Running PTQ on best checkpoints (GPU 0).", flush=True)
# Include a range: best (h512), champion (h128_long), mid (h48_long), extreme (h16_long)
ptq_tags = ['h512', 'h128_long', 'h48_long', 'h16_long']
env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = '0'
rc = subprocess.call([sys.executable, 'ptq_tinyfrr.py',
                      '--tags'] + ptq_tags + ['--bits', '8', '4', '--n', '300',
                      '--device', 'cuda:0'], env=env)
print(f"[wait_ptq] PTQ rc={rc}", flush=True)

print("\n[wait_ptq] Training tied-projection variants at h=128 and h=64 (GPU 0).", flush=True)
for h in [128, 64]:
    rc = subprocess.call([sys.executable, 'run_1.7b_tinyfrr_tied.py',
                          '--h', str(h), '--steps', '8000',
                          '--tag', f'h{h}_tied'], env=env)
    print(f"[wait_ptq] tied h={h} rc={rc}", flush=True)

print("\n[wait_ptq] Running HQ training (80K steps, T-schedule, forward+reverse-KL) on h=128.", flush=True)
rc = subprocess.call([sys.executable, 'run_1.7b_tinyfrr_hq.py',
                      '--h', '128', '--steps', '80000',
                      '--tag', 'h128_hq', '--device', 'cuda:0'], env=env)
print(f"[wait_ptq] h128_hq rc={rc}", flush=True)

print("\n[wait_ptq] Final quality benchmark on all best checkpoints.", flush=True)
rc = subprocess.call([sys.executable, 'bench_tinyfrr_quality.py',
                      '--tags', 'h128_hq', 'h128_long', 'h512', 'h48_long',
                      'h16_long', 'h128_tied',
                      '--n', '200', '--device', 'cuda:0'], env=env)
print(f"[wait_ptq] final quality bench rc={rc}", flush=True)

print("\n[wait_ptq] ALL POST-SWEEP WORK COMPLETE", flush=True)
