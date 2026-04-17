"""
Resume h64_long which crashed at step 32001 due to OOM.
Waits for GPU memory to free up, then resumes training.
"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def gpu_mem_free(gpu_id=0):
    """Return free MB on given GPU."""
    import subprocess
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
         f'--id={gpu_id}'], text=True)
    return int(out.strip())

print("[resume_h64] Waiting for GPU 0 to have enough memory...", flush=True)
while True:
    free = gpu_mem_free(0)
    print(f"[resume_h64] {time.strftime('%H:%M:%S')}  GPU 0 free={free}MB", flush=True)
    if free > 14000:  # Need ~13GB for teacher + student
        break
    time.sleep(60)

print(f"[resume_h64] GPU 0 has {free}MB free. Launching h64_long resume.", flush=True)

env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = '0'

# Resume from checkpoint (run_1.7b_tinyfrr.py supports resume via latest.pt)
log = 'tinyfrr_h64_long_resume.log'
with open(log, 'w') as lf:
    p = subprocess.Popen([sys.executable, 'run_1.7b_tinyfrr.py',
                          '--h', '64', '--steps', '40000',
                          '--tag', 'h64_long'],
                         stdout=lf, stderr=subprocess.STDOUT, env=env)
    rc = p.wait()

print(f"[resume_h64] h64_long done rc={rc}", flush=True)

# Append DONE to original log so supervisor can detect completion
if rc == 0:
    resume_log = open(log).read()
    # Find the DONE line
    for line in resume_log.split('\n'):
        if 'DONE h64_long:' in line:
            with open('tinyfrr_h64_long.log', 'a') as f:
                f.write('\n' + line + '\n')
            print(f"[resume_h64] Appended DONE to original log", flush=True)
            break
