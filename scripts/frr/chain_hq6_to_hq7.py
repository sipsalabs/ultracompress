"""
Auto-chain: wait until both HQ6 runs emit DONE markers, then launch HQ7.

Usage (detached):
    python chain_hq6_to_hq7.py &

Polls hq6_h256.log and hq6_h384.log every 5 minutes for "DONE hq6_" lines.
When BOTH are done, starts launch_hq7_longhorizon.py in the same detached
style. Writes chain status to chain_hq6_to_hq7.log.
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

POLL_SECS = 300  # 5 minutes
LOG = os.path.join(HERE, 'chain_hq6_to_hq7.log')
WATCH = ['hq6_h256.log', 'hq6_h384.log']
DONE_MARKERS = {'hq6_h256.log': 'DONE hq6_h256',
                'hq6_h384.log': 'DONE hq6_h384'}


def log(msg):
    line = f"[{time.ctime()}] {msg}\n"
    with open(LOG, 'ab') as f:
        f.write(line.encode())
    print(line, end='')


def is_done(path, marker):
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except Exception:
        return False
    text = raw.decode('utf-8', errors='ignore')
    return marker in text


def main():
    log(f"chain daemon started; watching {WATCH}")
    while True:
        statuses = {p: is_done(p, DONE_MARKERS[p]) for p in WATCH}
        log(f"status: {statuses}")
        if all(statuses.values()):
            log("both HQ6 runs DONE — launching HQ7 long-horizon")
            p = subprocess.Popen(
                [sys.executable, '-u', 'launch_hq7_longhorizon.py'],
                cwd=HERE,
                stdin=subprocess.DEVNULL,
                stdout=open(os.path.join(HERE, 'chain_hq7_launch.log'), 'ab'),
                stderr=subprocess.STDOUT,
                creationflags=FLAGS,
                close_fds=True,
            )
            log(f"HQ7 launcher kicked off (pid={p.pid}); chain daemon exiting")
            return
        time.sleep(POLL_SECS)


if __name__ == '__main__':
    main()
