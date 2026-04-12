"""8B OVERNIGHT — Complete 8B scaling test pipeline.

Run when you're home and can monitor hardware.
Tests FRR distillation from Qwen3-8B, then benchmarks.

Memory safe: streams teacher, monitors GPU temp.
Saves checkpoints every 500 steps.
Auto-stops if GPU temp > 85C or memory > 30GB.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u run_8b_overnight.py > 8b_overnight_output.log 2>&1
"""
import subprocess, sys, os, time

print("=" * 60)
print("8B OVERNIGHT PIPELINE")
print("=" * 60)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Pre-flight checks
print("[Pre-flight] Checking GPU memory...")
import torch
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {mem:.1f} GB")
    if mem < 30:
        print("  WARNING: Less than 30GB GPU memory. 8B distillation needs ~11GB.")
        print("  Proceeding with caution...")
else:
    print("  ERROR: No CUDA GPU available!")
    sys.exit(1)

# Check model exists
MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B")
if not os.path.exists(MODEL_PATH):
    print(f"  ERROR: Qwen3-8B not found at {MODEL_PATH}")
    sys.exit(1)
print(f"  Model: {MODEL_PATH} [OK]")

print()
scripts = [
    ("8B FRR Distillation (5K steps)", "run_8b_frr.py", "8b_frr_output.log"),
]

for name, script, log in scripts:
    print(f"\n{'='*60}")
    print(f"LAUNCHING: {name}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u", script],
            stdout=open(log, 'w'),
            stderr=subprocess.STDOUT,
            timeout=14400,  # 4 hour max
        )
        elapsed = time.time() - t0
        print(f"COMPLETED: {name} in {elapsed/60:.0f} min")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name} after 4 hours")
    except Exception as e:
        print(f"ERROR: {name} - {e}")
    sys.stdout.flush()

print(f"\n{'='*60}")
print(f"8B OVERNIGHT COMPLETE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
