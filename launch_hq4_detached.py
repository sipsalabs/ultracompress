"""
Detached launcher for dual-GPU HQ4 runs.
Uses Windows DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP + CREATE_BREAKAWAY_FROM_JOB
so the children survive the parent shell / VS Code terminal dying.
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

PY = sys.executable  # current python

RUNS = [
    {'h': 128, 'gpu': '0', 'log': 'hq4_h128.log'},
    {'h': 256, 'gpu': '1', 'log': 'hq4_h256.log'},
]

for r in RUNS:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = r['gpu']
    env['PYTHONUNBUFFERED'] = '1'
    logf = open(os.path.join(HERE, r['log']), 'ab')
    cmd = [PY, '-u', 'run_hq4_ceiling_break.py',
           '--h', str(r['h']), '--steps', '80000',
           '--device', 'cuda:0', '--tag', f"hq4_h{r['h']}"]
    logf.write(f"\n===== detached launch at {time.ctime()} =====\n".encode())
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
    print(f"launched h={r['h']} gpu={r['gpu']} pid={p.pid} -> {r['log']}")

print("parent exiting; children detached.")
