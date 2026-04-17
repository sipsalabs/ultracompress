"""
EVENING CHAIN: Run when Sip gets home.
Uses GPU 0 (freed from controller test).

1. Speculative decoding benchmark (5 min) — prove 2x speedup
2. Standard eval on teacher + FRR (10 min) — real benchmark numbers
3. MoL test (1 hour) — token-conditional LoRA routing
"""
import lib.unbuffered
import subprocess, time, os, sys

scripts = [
    ('SPEC DECODE BENCH', 'run_spec_decode_bench.py', 'spec_decode_output.log', 0.5),
    ('4D BLOCK TEST (Sips idea)', 'run_4d_block_test.py', '4d_block_output.log', 1.5),
    ('STANDARD EVAL', 'run_standard_eval.py', 'standard_eval_output.log', 0.5),
    ('MOL TEST', 'run_mol_test.py', 'mol_output.log', 2.0),
]

start = time.strftime('%Y-%m-%d %H:%M:%S')
print(f"{'='*60}")
print(f"EVENING CHAIN")
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
