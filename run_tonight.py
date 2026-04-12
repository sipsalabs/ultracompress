"""
TONIGHT'S EXPERIMENTS: Launch when Sip gets home.
Runs on GPU 0 after evening chain finishes.

1. Born-again distillation (if frr_100k_best.pt exists)
2. Optimized training (2x batch, 1.5x LR, T warmup)

Both ~2.5 hours each. Total: ~5 hours overnight.
1.7B 100K continues on GPU 1.
"""
import lib.unbuffered
import subprocess, time, os, sys

scripts = [
    ('DUAL-OBJECTIVE TEST (Sips T1/T10 split)', 'run_dual_objective_test.py', 'dual_objective_output.log', 2.5),
    ('TOP-1 LOSS TEST', 'run_top1_test.py', 'top1_output.log', 2.5),
    ('OPTIMIZED TRAINING 50K', 'run_optimized_train.py', 'optimized_output.log', 4.0),
]

# Only run born-again if we have a trained model
if os.path.exists('frr_100k_best.pt'):
    scripts.append(('BORN-AGAIN DISTILLATION', 'run_born_again.py', 'born_again_output.log', 8.0))

start = time.strftime('%Y-%m-%d %H:%M:%S')
print(f"{'='*60}")
print(f"TONIGHT'S EXPERIMENTS")
print(f"Start: {start}")
print(f"{'='*60}")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cwd = os.path.dirname(os.path.abspath(__file__))

for name, script, log, timeout_hrs in scripts:
    print(f"\n[{time.strftime('%H:%M:%S')}] Launching {name} -> {log}")
    sys.stdout.flush()
    try:
        with open(os.path.join(cwd, log), 'w') as f:
            result = subprocess.run(
                [sys.executable, '-u', script],
                stdout=f, stderr=subprocess.STDOUT,
                timeout=int(timeout_hrs * 3600),
                cwd=cwd
            )
        print(f"[{time.strftime('%H:%M:%S')}] {name}: exit {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"[{time.strftime('%H:%M:%S')}] {name}: TIMEOUT")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] {name}: ERROR {e}")
    sys.stdout.flush()

print(f"\n{'='*60}")
print(f"ALL DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
