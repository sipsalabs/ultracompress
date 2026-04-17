"""
Supervisor: keeps hmkgt running until it reaches STEPS=50000.
Auto-restarts on silent crashes (Windows). Checks step count from latest ckpt
to decide when to stop.
"""
import subprocess, time, os, sys, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CKPT = 'checkpoints_1.7b_hmkgt/hmkgt_latest.pt'
TARGET_STEPS = 50000
MAX_RESTARTS = 30
LOG = 'hmkgt.log'

def cur_step():
    if not os.path.exists(CKPT): return 0
    try:
        c = torch.load(CKPT, map_location='cpu', weights_only=False)
        return c.get('step', 0)
    except Exception:
        return 0

restarts = 0
while True:
    s = cur_step()
    print(f"[sup] current step from ckpt: {s} / {TARGET_STEPS}  (restarts={restarts})", flush=True)
    if s >= TARGET_STEPS:
        print("[sup] TARGET REACHED, exiting", flush=True)
        break
    if restarts >= MAX_RESTARTS:
        print("[sup] max restarts hit, exiting", flush=True)
        break

    # Launch training subprocess, appending to log
    print(f"[sup] launching run_1.7b_hmkgt.py (attempt {restarts+1})...", flush=True)
    with open(LOG, 'a') as lf:
        lf.write(f"\n\n=== [sup] restart {restarts+1} at step {s} ===\n")
        lf.flush()
        p = subprocess.Popen(
            [sys.executable, 'run_1.7b_hmkgt.py'],
            stdout=lf, stderr=subprocess.STDOUT,
        )
    rc = p.wait()
    print(f"[sup] subprocess exited rc={rc}", flush=True)
    restarts += 1
    time.sleep(3)  # brief pause before relaunch

print("[sup] DONE", flush=True)
