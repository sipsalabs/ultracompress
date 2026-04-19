"""
Sequential Monday-readiness pipeline on GPU 1.

Runs in order:
  1. hires_eval.py on HQ5 h256 + h128 (1000 samples, 95% CIs)
  2. combined_stack_eval.py body=hq5_h256 heads=[asvd_r1024_ft, asvd_r512_ft, asvd_r256_ft]
     (the actual end-to-end compressed-inference numbers)

Shares GPU 1 with HQ6 h384 training (14GB headroom).
Writes one unified log: monday_eval.log
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(HERE, 'monday_eval.log')

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000
FLAGS = (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
         | CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW)

# A tiny PowerShell driver that serializes the two python commands and appends
# to the same log. Using ps so each child process closes and frees GPU before
# the next one starts.
py = sys.executable
driver = f"""
Set-Location '{HERE}'
'[monday_eval] start ' + (Get-Date).ToString('s') | Out-File -Append -Encoding utf8 '{log_path}'
& '{py}' -u hires_eval.py --tags hq5_h256 hq5_h128 --n 1000 --seq_len 128 --seed 42 --device cuda:1 --out_prefix hires_results_hq5 *>> '{log_path}'
'[monday_eval] hires done ' + (Get-Date).ToString('s') | Out-File -Append -Encoding utf8 '{log_path}'
& '{py}' -u combined_stack_eval.py --body hq5_h256 --heads asvd_r1024_ft asvd_r512_ft asvd_r256_ft --n 1000 --seq_len 128 --seed 42 --device cuda:1 --out_prefix combined_stack_results_hq5 *>> '{log_path}'
'[monday_eval] combined done ' + (Get-Date).ToString('s') | Out-File -Append -Encoding utf8 '{log_path}'
"""

cmd = ['powershell', '-NoProfile', '-Command', driver]
f = open(log_path, 'wb')
p = subprocess.Popen(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                     close_fds=True, creationflags=FLAGS)
print(f"monday eval pipeline pid={p.pid}  log={log_path}")
