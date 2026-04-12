"""
Fix Windows stdout buffering permanently.
Import this at the top of any script:
    import lib.unbuffered
All print() calls will flush immediately after.
"""
import sys
import os

# Force unbuffered stdout/stderr
os.environ['PYTHONUNBUFFERED'] = '1'

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)
