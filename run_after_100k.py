"""
AUTO-CHAIN: Wait for 0.6B 100K to finish, then launch experiments on GPU 0.
1. Speed benchmark (~5 min)
2. 8B FRR test (15K steps, ~2-3 hours)
"""
import lib.unbuffered
import subprocess, time, os, sys
import psutil

# Find the 100K training process
def find_process(name_contains):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info.get('cmdline', []) or [])
            if name_contains in cmdline:
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

print("=" * 60)
print("AUTO-CHAIN: Waiting for 0.6B 100K to finish")
print("=" * 60)

# Wait for 100K to finish
pid = find_process('run_100k_train.py')
if pid:
    print(f"Found 100K process: PID {pid}")
    while True:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(60)
            print(f"  [{time.strftime('%H:%M:%S')}] Still running...")
        except psutil.NoSuchProcess:
            break
    print(f"100K finished at {time.strftime('%H:%M:%S')}")
else:
    print("100K not running. Proceeding immediately.")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cwd = os.path.dirname(os.path.abspath(__file__))

# 1. Speed benchmark
print(f"\n[{time.strftime('%H:%M:%S')}] Running speed benchmark...")
try:
    with open(os.path.join(cwd, 'speed_output.log'), 'w') as f:
        subprocess.run([sys.executable, '-u', 'run_speed_benchmark.py'],
                      stdout=f, stderr=subprocess.STDOUT, timeout=600, cwd=cwd)
    print(f"[{time.strftime('%H:%M:%S')}] Speed benchmark done")
except Exception as e:
    print(f"Speed benchmark failed: {e}")

# 2. 8B FRR
if os.path.exists(os.path.join(cwd, 'qwen3_8b_cache.pt')):
    print(f"\n[{time.strftime('%H:%M:%S')}] Running 8B FRR test...")
    try:
        with open(os.path.join(cwd, '8b_frr_output.log'), 'w') as f:
            subprocess.run([sys.executable, '-u', 'run_8b_dual_gpu.py'],
                          stdout=f, stderr=subprocess.STDOUT, timeout=4*3600, cwd=cwd)
        print(f"[{time.strftime('%H:%M:%S')}] 8B FRR done")
    except Exception as e:
        print(f"8B FRR failed: {e}")
else:
    print("8B cache not found. Skipping 8B test.")

print(f"\nAll done at {time.strftime('%H:%M:%S')}")
