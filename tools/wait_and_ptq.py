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

# HQ2: hidden-state matching on top of HQ. This is the real "match teacher
# latent, not just token distribution" step. Warm-starts from h128_hq.
print("\n[wait_ptq] Running HQ2 (hidden-state matching, 80K) on h=128.", flush=True)
rc = subprocess.call([sys.executable, 'run_1.7b_tinyfrr_hq2.py',
                      '--h', '128', '--steps', '80000',
                      '--tag', 'h128_hq2', '--device', 'cuda:0'], env=env)
print(f"[wait_ptq] h128_hq2 rc={rc}", flush=True)

# Also try a smaller dim w/ hidden-state matching — if this tracks hq2
# performance it means the representation-matching is doing the heavy lift
# and inner-dim can drop much further.
print("\n[wait_ptq] HQ2 on h=64 (test if hidden-state matching scales down).", flush=True)
rc = subprocess.call([sys.executable, 'run_1.7b_tinyfrr_hq2.py',
                      '--h', '64', '--steps', '60000',
                      '--tag', 'h64_hq2', '--device', 'cuda:0'], env=env)
print(f"[wait_ptq] h64_hq2 rc={rc}", flush=True)

print("\n[wait_ptq] Final quality benchmark on all best checkpoints.", flush=True)
rc = subprocess.call([sys.executable, 'bench_tinyfrr_quality.py',
                      '--tags', 'h128_hq2', 'h128_hq', 'h64_hq2', 'h128_long',
                      'h512', 'h48_long', 'h16_long', 'h128_tied',
                      '--n', '200', '--device', 'cuda:0'], env=env)
print(f"[wait_ptq] final quality bench rc={rc}", flush=True)

# ==================== NEXT-GEN: Joint body+ASVD head ====================
# These are the highest-impact experiments. GPU 0 does joint training,
# the second GPU (if available) does HQ3 multi-layer matching.
print("\n[wait_ptq] ===== NEXT-GEN PHASE 1 =====", flush=True)

env1 = dict(os.environ); env1['CUDA_VISIBLE_DEVICES'] = '1'

# GPU 0: Joint h=128, r=512
print("[wait_ptq] GPU 0: Joint body+ASVD head h=128 r=512 (80K steps)", flush=True)
p_joint = subprocess.Popen(
    [sys.executable, 'run_1.7b_tinyfrr_joint.py',
     '--h', '128', '--r', '512', '--steps', '80000', '--seq_len', '128',
     '--tag', 'h128_r512_joint', '--device', 'cuda:0'],
    env=env,
    stdout=open('tinyfrr_h128_r512_joint.log', 'w'),
    stderr=subprocess.STDOUT)
print(f"  PID={p_joint.pid}", flush=True)

# GPU 1: HQ3 multi-layer matching h=128
print("[wait_ptq] GPU 1: HQ3 multi-layer h=128 (100K steps)", flush=True)
p_hq3 = subprocess.Popen(
    [sys.executable, 'run_1.7b_tinyfrr_hq3.py',
     '--h', '128', '--steps', '100000',
     '--tag', 'h128_hq3', '--device', 'cuda:0'],
    env=env1,
    stdout=open('tinyfrr_h128_hq3.log', 'w'),
    stderr=subprocess.STDOUT)
print(f"  PID={p_hq3.pid}", flush=True)

for name, proc in [('h128_r512_joint', p_joint), ('h128_hq3', p_hq3)]:
    rc = proc.wait()
    print(f"  [{name}] rc={rc}", flush=True)

# ==================== NEXT-GEN PHASE 2: More compression variants ====================
print("\n[wait_ptq] ===== NEXT-GEN PHASE 2 =====", flush=True)

# GPU 0: Joint h=128, r=256 (more head compression)
print("[wait_ptq] GPU 0: Joint h=128 r=256 (80K steps)", flush=True)
p2a = subprocess.Popen(
    [sys.executable, 'run_1.7b_tinyfrr_joint.py',
     '--h', '128', '--r', '256', '--steps', '80000', '--seq_len', '128',
     '--tag', 'h128_r256_joint', '--device', 'cuda:0'],
    env=env,
    stdout=open('tinyfrr_h128_r256_joint.log', 'w'),
    stderr=subprocess.STDOUT)
print(f"  PID={p2a.pid}", flush=True)

# GPU 1: Joint h=64, r=512 (more body compression)
print("[wait_ptq] GPU 1: Joint h=64 r=512 (80K steps)", flush=True)
p2b = subprocess.Popen(
    [sys.executable, 'run_1.7b_tinyfrr_joint.py',
     '--h', '64', '--r', '512', '--steps', '80000', '--seq_len', '128',
     '--tag', 'h64_r512_joint', '--device', 'cuda:0'],
    env=env1,
    stdout=open('tinyfrr_h64_r512_joint.log', 'w'),
    stderr=subprocess.STDOUT)
print(f"  PID={p2b.pid}", flush=True)

for name, proc in [('h128_r256_joint', p2a), ('h64_r512_joint', p2b)]:
    rc = proc.wait()
    print(f"  [{name}] rc={rc}", flush=True)

# ==================== Final combined evaluation ====================
print("\n[wait_ptq] Final combined evaluation (all variants).", flush=True)
all_tags = [
    'h128_r512_joint', 'h128_r256_joint', 'h64_r512_joint',
    'h128_hq3', 'h128_hq2', 'h128_hq', 'h64_hq2',
    'h128_long', 'h512', 'h48_long', 'h16_long', 'h128_tied',
]
existing = [t for t in all_tags
            if os.path.exists(f'checkpoints_1.7b_tinyfrr_{t}/best.pt')]
if existing:
    rc = subprocess.call([sys.executable, 'eval_combined.py',
                          '--tags'] + existing + ['--n', '200', '--device', 'cuda:0'], env=env)
    print(f"[wait_ptq] combined eval rc={rc}", flush=True)

print("\n[wait_ptq] ALL POST-SWEEP WORK COMPLETE (incl. next-gen)", flush=True)
