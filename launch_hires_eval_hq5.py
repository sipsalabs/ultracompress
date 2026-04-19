"""
Detached launcher for hires_eval.py on HQ5 flagship checkpoints.

Evaluates HQ5 h256 and HQ5 h128 on 1000 samples, SEQ_LEN=128, seed 42,
draws positions from the tail 50M tokens of fineweb_edu_500M_tokens.pt.
Shares GPU 1 with ongoing HQ6 h384 training (14GB headroom).

Output:
  hires_results.pt / hires_results.json
  hires_eval.log
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
log = os.path.join(HERE, 'hires_eval.log')

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000
FLAGS = (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
         | CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW)

cmd = [
    sys.executable, '-u', 'hires_eval.py',
    '--tags', 'hq5_h256', 'hq5_h128',
    '--n', '1000',
    '--seq_len', '128',
    '--seed', '42',
    '--device', 'cuda:1',
    '--out_prefix', 'hires_results_hq5',
]

f = open(log, 'wb')
p = subprocess.Popen(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                     close_fds=True, creationflags=FLAGS)
print(f"hires eval pid={p.pid}  log={log}")
print(f"cmd: {' '.join(cmd)}")
