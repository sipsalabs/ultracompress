"""
Long-training sweep: most promising TinyFRR sizes for 40K steps each.
h=128 (734x, sweet spot), h=48 (2203x, ultra-compression target).
Tests if longer training closes the gap to baseline's 68.47% ceiling.
Loads from existing 8K checkpoint to save time.
"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Copy existing 8K best -> latest so training resumes from there
import torch, shutil
for h in [128, 48]:
    src_dir = f'checkpoints_1.7b_tinyfrr_h{h}'
    best_path = f'{src_dir}/best.pt'
    latest_path = f'{src_dir}/latest.pt'
    if os.path.exists(best_path) and not os.path.exists(latest_path):
        # Need to add 'opt' state_dict... but we don't have it. Start fresh opt from best weights.
        ck = torch.load(best_path, map_location='cpu', weights_only=False)
        ck_latest = {'state_dict': ck['state_dict'], 'step': 8000, 'best': ck['best']}
        # No opt - trainer will start optimizer fresh at resumed step
        # But the trainer tries to opt.load_state_dict which will fail.
        # Instead: just delete any existing latest, let it train fresh from start
        # BUT we want to preserve the weights. Better approach:
        # Keep latest.pt with state_dict only, but trainer needs opt too.
        # Simplest: skip resume, just train 40K fresh.
        pass

# Just run fresh 40K on each
DIMS_STEPS = [(128, 40000), (48, 40000)]

for h, steps in DIMS_STEPS:
    tag = f'h{h}_long'
    # Create symlink-style tag by renaming the checkpoint dir target
    print(f"\n{'='*60}\n  Launching TinyFRR h={h} for {steps} steps\n{'='*60}", flush=True)
    t0 = time.time()
    log = f'tinyfrr_{tag}.log'
    # We pass --tag to change the checkpoint dir (new dir, fresh train)
    with open(log, 'w') as lf:
        p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                              '--h', str(h), '--steps', str(steps), '--tag', tag],
                             stdout=lf, stderr=subprocess.STDOUT)
        rc = p.wait()
    if rc != 0:
        print(f"[sweep] h={h} crashed rc={rc}, retrying with resume", flush=True)
        with open(log, 'a') as lf:
            lf.write(f"\n=== RETRY ===\n")
            p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                                  '--h', str(h), '--steps', str(steps), '--tag', tag],
                                 stdout=lf, stderr=subprocess.STDOUT)
            rc = p.wait()
    dt = time.time() - t0
    print(f"[sweep] h={h} long done rc={rc} in {dt/60:.1f} min", flush=True)

print("\n[sweep_long] ALL DONE", flush=True)
