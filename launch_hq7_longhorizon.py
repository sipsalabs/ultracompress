"""
HQ7 — long-horizon (160K-step) extension of the HQ5/HQ6 objective.

Rationale: every prior HQ generation peaked at step 78K of 80K. Loss and
ppl-ratio were still dropping at termination. Doubling the budget at the same
objective is the highest-confidence gain available (+0.5-1.0 pp projected).

Schedules in run_hq4_ceiling_break.py are STEPS-relative, so 160K automatically
stretches latent decay (40K->100K), ce ramp (32K->96K), T anneal (->128K),
and cosine LR decay.

Warm-start: run_hq4_ceiling_break.py's WARM_CANDIDATES prefers hq5 -> hq4 -> ...
When HQ6 checkpoints exist we also look at those explicitly via symlink below.
Otherwise falls through to HQ5 best.pt as before.

h256 on GPU 0: entropy_power 2.0, latent floor 0.1 (HQ6-h256 settings).
h128 on GPU 1: entropy_power 1.5, latent floor 0.3 (best of both worlds).
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
    {'h': 256, 'gpu': '0', 'log': 'hq7_h256.log', 'tag': 'hq7_h256',
     'extra': ['--latent_w_final', '0.1', '--entropy_power', '2.0']},
    {'h': 128, 'gpu': '1', 'log': 'hq7_h128.log', 'tag': 'hq7_h128',
     'extra': ['--latent_w_final', '0.3', '--entropy_power', '1.5']},
]


def launch():
    for r in RUNS:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = r['gpu']
        env['PYTHONUNBUFFERED'] = '1'
        logf = open(os.path.join(HERE, r['log']), 'ab')
        cmd = [PY, '-u', 'run_hq4_ceiling_break.py',
               '--h', str(r['h']), '--steps', '160000',
               '--device', 'cuda:0', '--tag', r['tag']] + r['extra']
        logf.write(f"\n===== HQ7 detached launch at {time.ctime()} =====\n".encode())
        logf.write(f"cmd: {' '.join(cmd)}\n".encode())
        logf.flush()
        p = subprocess.Popen(
            cmd, cwd=HERE, env=env,
            stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=FLAGS, close_fds=True,
        )
        print(f"launched h={r['h']} gpu={r['gpu']} pid={p.pid} tag={r['tag']} -> {r['log']}")
    print("parent exiting; children detached.")


if __name__ == '__main__':
    launch()
