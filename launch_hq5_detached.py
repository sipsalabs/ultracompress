"""
Detached launcher for dual-GPU HQ5 runs.
HQ5 = HQ4 objective with stronger signal:
  - h128: latent floor 0.3 (vs HQ4's 0.1) — less aggressive release, more stable
  - h256: entropy_power 1.5 (vs HQ4's 1.0) — harder focus on hard tokens
Warm-starts from HQ4 best.pt (picked up automatically by WARM_CANDIDATES).
"""
import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000
FLAGS = (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
         | CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW)

PY = sys.executable

RUNS = [
    # h128: try a higher latent floor to reduce oscillation
    {'h': 128, 'gpu': '0', 'log': 'hq5_h128.log', 'tag': 'hq5_h128',
     'extra': ['--latent_w_final', '0.3', '--entropy_power', '1.0']},
    # h256: strengthen hard-token focus
    {'h': 256, 'gpu': '1', 'log': 'hq5_h256.log', 'tag': 'hq5_h256',
     'extra': ['--latent_w_final', '0.1', '--entropy_power', '1.5']},
]

for r in RUNS:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = r['gpu']
    env['PYTHONUNBUFFERED'] = '1'
    logf = open(os.path.join(HERE, r['log']), 'ab')
    cmd = [PY, '-u', 'run_hq4_ceiling_break.py',
           '--h', str(r['h']), '--steps', '80000',
           '--device', 'cuda:0', '--tag', r['tag']] + r['extra']
    logf.write(f"\n===== HQ5 detached launch at {time.ctime()} =====\n".encode())
    logf.write(f"cmd: {' '.join(cmd)}\n".encode())
    logf.flush()
    p = subprocess.Popen(
        cmd,
        cwd=HERE,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=logf,
        stderr=subprocess.STDOUT,
        creationflags=FLAGS,
        close_fds=True,
    )
    print(f"launched h={r['h']} gpu={r['gpu']} pid={p.pid} tag={r['tag']} -> {r['log']}")

print("parent exiting; children detached.")
