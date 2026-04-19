"""
Wait for hires_results_hq5.pt to appear, then launch combined_stack_eval.py
detached on GPU 1. Polls every 30s.
"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SENTINEL = os.path.join(HERE, 'hires_results_hq5.pt')
LOG = os.path.join(HERE, 'combined_stack_eval.log')

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000
FLAGS = (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
         | CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW)

print(f"waiting for {SENTINEL} ...")
while not os.path.exists(SENTINEL):
    time.sleep(30)

# small extra pause so the hires process fully releases GPU memory
time.sleep(20)

cmd = [
    sys.executable, '-u', 'combined_stack_eval.py',
    '--body', 'hq5_h256',
    '--heads', 'asvd_r1024_ft', 'asvd_r512_ft', 'asvd_r256_ft',
    '--n', '1000',
    '--seq_len', '128',
    '--seed', '42',
    '--device', 'cuda:1',
    '--out_prefix', 'combined_stack_results_hq5',
]
f = open(LOG, 'wb')
p = subprocess.Popen(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                     close_fds=True, creationflags=FLAGS)
print(f"combined_stack_eval launched pid={p.pid}  log={LOG}")
