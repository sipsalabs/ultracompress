"""
Fix Windows stdout buffering permanently.
Import this at the top of any script:
    import lib.unbuffered
All print() calls will flush immediately after.

Also auto-detects log file from script name and writes directly to it,
bypassing the OS pipe buffer entirely.
"""
import sys
import os

os.environ['PYTHONUNBUFFERED'] = '1'

class DirectLogger:
    """Writes to both console AND log file with immediate flush.
    Bypasses OS pipe buffer by writing directly via Python file I/O."""
    def __init__(self, stream, log_path=None):
        self.stream = stream
        self.log_file = None
        if log_path:
            self.log_file = open(log_path, 'a', encoding='utf-8')
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if self.log_file:
            self.log_file.write(data)
            self.log_file.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
        if self.log_file:
            self.log_file.writelines(datas)
            self.log_file.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

# Auto-detect log file: script_name.py -> script_name_output.log
_script = os.path.basename(sys.argv[0]) if sys.argv[0] else ''
_log_path = None
if _script.endswith('.py'):
    _log_name = _script.replace('.py', '_output.log').replace('run_', '')
    _log_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), _log_name)

sys.stdout = DirectLogger(sys.__stdout__, _log_path)
sys.stderr = DirectLogger(sys.__stderr__)
