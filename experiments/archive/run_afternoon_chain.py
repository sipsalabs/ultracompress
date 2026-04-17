"""
AFTERNOON CHAIN: Run after workday experiments finish.
1. Speed benchmark (5 min)
2. Standard eval on teacher (10 min)
3. MoL test (1 hour)
4. Optimized training 50K (2.5 hours)
"""
import lib.unbuffered
import subprocess, time, os, sys

scripts = [
    ('SPEED BENCHMARK', 'run_speed_benchmark.py', 'speed_output.log', 0.5),
    ('MOL TEST', 'run_mol_test.py', 'mol_output.log', 2.0),
    ('OPTIMIZED TRAINING 50K', 'run_optimized_train.py', 'optimized_output.log', 4.0),
]

start = time.strftime('%Y-%m-%d %H:%M:%S')
print(f"{'='*60}")
print(f"AFTERNOON CHAIN")
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
