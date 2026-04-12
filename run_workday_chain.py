"""
WORKDAY CHAIN: Run while Sip is at work.
1. PHM on best config (10K + 30K steps) — ~1 hour
2. 1.7B scaling test (15K steps) — ~1-2 hours
3. 100K step training (100K steps) — ~2.5 hours
Total: ~5-6 hours
"""
import lib.unbuffered
import subprocess, time, os, sys

scripts = [
    ('PHM BEST CONFIG', 'run_phm_best_config.py', 'phm_best_output.log', 2.0),
    ('1.7B SCALING TEST', 'run_1.7b_scaling.py', 'scaling_1.7b_output.log', 3.0),
    ('100K LONG TRAINING', 'run_100k_train.py', '100k_train_output.log', 4.0),
]

start = time.strftime('%Y-%m-%d %H:%M:%S')
print(f"{'='*60}")
print(f"WORKDAY CHAIN — Sip at work, GPU grinding")
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
