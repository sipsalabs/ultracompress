"""
Launch next-gen experiments after current pipeline finishes.

GPU 0: TinyFRR+ASVD joint training (h=128, r=512 and r=256)
GPU 1: HQ3 multi-layer matching (h=128, 100K steps)

Can be run immediately — scripts warm-start from best checkpoints.
If the long sweeps haven't finished yet, these will grab whatever
checkpoints are currently best.
"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print(" NEXT-GEN EXPERIMENT LAUNCHER")
print("=" * 60)

# Check what checkpoints exist for warm-starting
for tag in ['h128_hq2', 'h128_hq', 'h128_long', 'h128']:
    p = f'checkpoints_1.7b_tinyfrr_{tag}/best.pt'
    exists = os.path.exists(p)
    print(f"  {tag:20s}: {'EXISTS' if exists else 'missing'}")

processes = []

# GPU 0: Joint body+head (the big experiment)
print("\n[GPU 0] Launching TinyFRR+ASVD joint h=128 r=512 (80K steps, seq=128)...")
env0 = dict(os.environ); env0['CUDA_VISIBLE_DEVICES'] = '0'
p0 = subprocess.Popen(
    [sys.executable, 'run_1.7b_tinyfrr_joint.py',
     '--h', '128', '--r', '512', '--steps', '80000', '--seq_len', '128',
     '--tag', 'h128_r512_joint', '--device', 'cuda:0'],
    env=env0,
    stdout=open('tinyfrr_h128_r512_joint.log', 'w'),
    stderr=subprocess.STDOUT
)
processes.append(('h128_r512_joint', p0))
print(f"  PID={p0.pid}")

# GPU 1: HQ3 multi-layer matching
print("[GPU 1] Launching HQ3 multi-layer h=128 (100K steps)...")
env1 = dict(os.environ); env1['CUDA_VISIBLE_DEVICES'] = '1'
p1 = subprocess.Popen(
    [sys.executable, 'run_1.7b_tinyfrr_hq3.py',
     '--h', '128', '--steps', '100000',
     '--tag', 'h128_hq3', '--device', 'cuda:0'],  # cuda:0 within CUDA_VISIBLE_DEVICES=1
    env=env1,
    stdout=open('tinyfrr_h128_hq3.log', 'w'),
    stderr=subprocess.STDOUT
)
processes.append(('h128_hq3', p1))
print(f"  PID={p1.pid}")

print(f"\nBoth experiments launched. Waiting for completion...")
print(f"Logs: tinyfrr_h128_r512_joint.log, tinyfrr_h128_hq3.log")

# Wait for both
for tag, proc in processes:
    rc = proc.wait()
    print(f"  [{tag}] finished with rc={rc}", flush=True)

print("\n[Phase 2] Joint h=128 r=256 (more compression) on GPU 0...")
p2 = subprocess.Popen(
    [sys.executable, 'run_1.7b_tinyfrr_joint.py',
     '--h', '128', '--r', '256', '--steps', '80000', '--seq_len', '128',
     '--tag', 'h128_r256_joint', '--device', 'cuda:0'],
    env=env0,
    stdout=open('tinyfrr_h128_r256_joint.log', 'w'),
    stderr=subprocess.STDOUT
)
print(f"  PID={p2.pid}")

print("[Phase 2] Joint h=64 r=512 (smaller body) on GPU 1...")
p3 = subprocess.Popen(
    [sys.executable, 'run_1.7b_tinyfrr_joint.py',
     '--h', '64', '--r', '512', '--steps', '80000', '--seq_len', '128',
     '--tag', 'h64_r512_joint', '--device', 'cuda:0'],
    env=env1,
    stdout=open('tinyfrr_h64_r512_joint.log', 'w'),
    stderr=subprocess.STDOUT
)
print(f"  PID={p3.pid}")

for tag, proc in [('h128_r256_joint', p2), ('h64_r512_joint', p3)]:
    rc = proc.wait()
    print(f"  [{tag}] finished with rc={rc}", flush=True)

# Final comparison bench
print("\n[Final] Running quality benchmark on all variants...")
bench_tags = [
    'h128_r512_joint', 'h128_r256_joint', 'h64_r512_joint',
    'h128_hq3', 'h128_hq2', 'h128_hq', 'h128_long', 'h512',
]
existing_tags = [t for t in bench_tags
                 if os.path.exists(f'checkpoints_1.7b_tinyfrr_{t}/best.pt')]
if existing_tags:
    rc = subprocess.call([
        sys.executable, 'bench_tinyfrr_quality.py',
        '--tags'] + existing_tags + ['--n', '200', '--device', 'cuda:0'])
    print(f"  bench rc={rc}")

print("\nALL NEXT-GEN EXPERIMENTS COMPLETE")
