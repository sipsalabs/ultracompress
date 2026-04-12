"""
OVERNIGHT PIPELINE V2: Chain experiments on GPU 0.
1. Wait for MEGA test to finish (if running)
2. Multi-block FRR (quality breakthrough)
3. Quantization-aware FRR (Q2-friendly outputs)
"""
import subprocess, time, os, sys

scripts = [
    ('E2E PROOF (FRR + Pipeline + Quality)', 'run_e2e_proof.py', 'e2e_proof_output.log'),
    ('INTERMEDIATE MATCHING (quality breakthrough)', 'run_frr_intermediate.py', 'intermediate_output.log'),
    ('MULTI-BLOCK FRR', 'run_after_mega.py', 'after_mega_output.log'),
]

start = time.strftime('%Y-%m-%d %H:%M:%S')
print(f"{'='*60}")
print(f"OVERNIGHT PIPELINE V2")
print(f"Start: {start}")
print(f"{'='*60}")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

for name, script, log in scripts:
    print(f"\n{'='*60}")
    print(f"LAUNCHING: {name}")
    print(f"Log: {log}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    sys.stdout.flush()

    try:
        with open(log, 'w') as f:
            result = subprocess.run(
                [sys.executable, '-u', script],
                stdout=f, stderr=subprocess.STDOUT,
                timeout=3*3600,  # 3 hour max per script
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        status = 'OK' if result.returncode == 0 else f'EXIT {result.returncode}'
    except subprocess.TimeoutExpired:
        status = 'TIMEOUT'
    except Exception as e:
        status = f'ERROR: {e}'

    print(f"COMPLETED: {name} [{status}]")
    sys.stdout.flush()

print(f"\n{'='*60}")
print(f"ALL DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
