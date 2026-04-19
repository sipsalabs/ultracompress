"""
Detached launcher for HQ6 — keep pushing past HQ5's 70.0% quality / 57.0% peak T1.

Two complementary directions:
  - h256: entropy_power 2.0 (HQ5 was 1.5 → +1.1 quality; extrapolating +0.5-1.0 more).
          Warm-started from HQ5 h256 best.pt.
  - h384: fresh capacity test. New width. entropy_power 1.5, latent floor 0.1.
          Warm-start fallback chain (no HQ5_h384 exists, so starts from HQ3/TinyFRR h384 if any,
          else fresh random init via run_hq4_ceiling_break.py's fallback).
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
    # h256: push entropy_power higher still
    {'h': 256, 'gpu': '0', 'log': 'hq6_h256.log', 'tag': 'hq6_h256',
     'extra': ['--latent_w_final', '0.1', '--entropy_power', '2.0']},
    # h384: capacity test at the winning HQ5-h256 objective
    {'h': 384, 'gpu': '1', 'log': 'hq6_h384.log', 'tag': 'hq6_h384',
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
    logf.write(f"\n===== HQ6 detached launch at {time.ctime()} =====\n".encode())
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
