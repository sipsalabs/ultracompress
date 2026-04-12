"""Auto-results knowledge base builder.

Scans ALL .log and *results*.json files, extracts structured records,
saves to results_knowledge_base.json, and prints a summary sorted by top10.
"""
import os, re, json, glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

records = []


def parse_size_mb(text):
    """Extract size in MB from text like '21.0MB' or 'Size=66.1MB'."""
    m = re.search(r'(\d+\.?\d*)\s*MB', text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_compression(text):
    """Extract compression ratio from text like '42x' or 'compression=37'."""
    m = re.search(r'(\d+\.?\d*)\s*x\b', text)
    if m:
        return float(m.group(1))
    m = re.search(r'compression[=:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_params(text):
    """Extract param count from text like 'params=11,934,720'."""
    m = re.search(r'params[=:\s]+([\d,]+)', text)
    if m:
        return int(m.group(1).replace(',', ''))
    return None


def add_record(method, params=None, size_mb=None, compression=None,
               top1=None, top10=None, source_file=None, extra=None):
    """Add a result record, skip if no useful data."""
    if top1 is None and top10 is None:
        return
    rec = {
        'method': method,
        'params': params,
        'size_mb': size_mb,
        'compression': compression,
        'top1': top1,
        'top10': top10,
        'source_file': os.path.basename(source_file) if source_file else None,
    }
    if extra:
        rec.update(extra)
    records.append(rec)


# ─── SCAN JSON RESULTS FILES ───────────────────────────────────────────

json_files = glob.glob(os.path.join(BASE_DIR, '*results*.json'))
json_files += glob.glob(os.path.join(BASE_DIR, '**/*results*.json'), recursive=True)
json_files = list(set(json_files))
# Exclude our own output
json_files = [f for f in json_files if 'results_knowledge_base' not in f]

for jf in json_files:
    try:
        with open(jf, 'r') as f:
            data = json.load(f)
    except Exception:
        continue

    fname = os.path.basename(jf)

    if isinstance(data, list):
        # List of result dicts (benchmark_results.json style)
        for entry in data:
            if isinstance(entry, dict):
                add_record(
                    method=entry.get('name', fname),
                    params=entry.get('params'),
                    size_mb=entry.get('size_mb'),
                    compression=entry.get('compression'),
                    top1=entry.get('top1'),
                    top10=entry.get('top10'),
                    source_file=jf,
                    extra={'time': entry.get('time'), 'timestamp': entry.get('timestamp')},
                )
    elif isinstance(data, dict):
        # Dict of named experiments (supertrain_results.json style)
        # Check if it's a single result or nested
        if 'top1' in data or 'top10' in data:
            add_record(
                method=fname.replace('_results.json', '').replace('.json', ''),
                top1=data.get('top1'),
                top10=data.get('top10'),
                size_mb=data.get('size_mb'),
                compression=data.get('compression'),
                source_file=jf,
            )
        else:
            for key, val in data.items():
                if isinstance(val, dict) and ('top1' in val or 'top10' in val):
                    add_record(
                        method=key,
                        params=val.get('params'),
                        size_mb=val.get('size_mb'),
                        compression=val.get('compression'),
                        top1=val.get('top1'),
                        top10=val.get('top10'),
                        source_file=jf,
                        extra={'time': val.get('time'), 'best_t10': val.get('best_t10')},
                    )

# ─── SCAN LOG FILES ────────────────────────────────────────────────────

log_files = glob.glob(os.path.join(BASE_DIR, '*.log'))
log_files += glob.glob(os.path.join(BASE_DIR, '**/*.log'), recursive=True)
log_files = list(set(log_files))

# Pattern: RESULT <name>: Top1=XX% Top10=YY% [Size=ZZ.ZMB] [NNx] [Time=NNNs]
result_pattern = re.compile(
    r'RESULT\s+(.+?):\s+Top1=(\d+)%\s+Top10=(\d+)%'
    r'(?:\s+Size=([\d.]+)MB)?'
    r'(?:\s+(\d+)x)?'
    r'(?:\s+Compression=([\d.]+)x)?'
    r'(?:\s+Time=(\d+)s)?'
)

# Pattern: <Name>: Top1=XX% Top10=YY% params=N time=Ns
inline_pattern = re.compile(
    r'(\S+[\w\-]+):\s+Top1=(\d+)%\s+Top10=(\d+)%'
    r'(?:\s+params=([\d,]+))?'
    r'(?:\s+time=(\d+)s)?'
)

# Pattern for summary lines: <Name>  Top1=XX% Top10=YY% ... Size=ZZ.ZMB
summary_pattern = re.compile(
    r'^\s+(?:\d+\w+:\s+)?(\S+[\w\-]+)\s+Top1=\s*(\d+)%\s+Top10=\s*(\d+)%'
    r'(?:.*?Size=\s*([\d.]+)MB)?'
    r'(?:.*?(\d+\.?\d*)x)?'
)

for lf in log_files:
    try:
        with open(lf, 'r', errors='replace') as f:
            content = f.read()
    except Exception:
        continue

    fname = os.path.basename(lf)
    seen_methods = set()

    for line in content.split('\n'):
        # Try RESULT pattern first (most specific)
        m = result_pattern.search(line)
        if m:
            method = m.group(1).strip()
            if method in seen_methods:
                continue
            seen_methods.add(method)
            size = float(m.group(4)) if m.group(4) else None
            comp = float(m.group(5)) if m.group(5) else None
            if not comp and m.group(6):
                comp = float(m.group(6))
            # Try to extract compression from the line if not matched
            if not comp and size:
                comp_m = re.search(r'(\d+)x', line)
                if comp_m:
                    comp = float(comp_m.group(1))
            add_record(
                method=method,
                size_mb=size,
                compression=comp,
                top1=int(m.group(2)) / 100.0,
                top10=int(m.group(3)) / 100.0,
                source_file=lf,
            )
            continue

        # Try inline pattern (V1-MicroTF: Top1=14% Top10=37% params=...)
        m = inline_pattern.search(line)
        if m and 'Step' not in line and 'step' not in line:
            method = m.group(1).strip().rstrip(':')
            if method in seen_methods or method.startswith('Step'):
                continue
            seen_methods.add(method)
            params = int(m.group(4).replace(',', '')) if m.group(4) else None
            size = None
            if params:
                size = round(params * 4 / 1024**2, 2)
            comp = parse_compression(line)
            add_record(
                method=method,
                params=params,
                size_mb=parse_size_mb(line) or size,
                compression=comp,
                top1=int(m.group(2)) / 100.0,
                top10=int(m.group(3)) / 100.0,
                source_file=lf,
            )

# ─── DEDUPLICATE ───────────────────────────────────────────────────────
# Keep best result per method name
best = {}
for r in records:
    key = r['method']
    if key not in best or (r.get('top10') or 0) > (best[key].get('top10') or 0):
        best[key] = r

records = list(best.values())

# ─── ESTIMATE MISSING FIELDS ──────────────────────────────────────────
# Qwen3-0.6B reference: ~592M params total, ~1761.9 MB FP32 layer params
ORIGINAL_SIZE_MB = 1761.9

for r in records:
    if r.get('size_mb') and not r.get('compression'):
        r['compression'] = round(ORIGINAL_SIZE_MB / r['size_mb'], 1)
    if r.get('compression') and not r.get('size_mb'):
        r['size_mb'] = round(ORIGINAL_SIZE_MB / r['compression'], 2)
    if r.get('params') and not r.get('size_mb'):
        r['size_mb'] = round(r['params'] * 4 / 1024**2, 2)

# ─── SORT BY TOP10 AND SAVE ───────────────────────────────────────────
records.sort(key=lambda x: (x.get('top10') or 0), reverse=True)

out_path = os.path.join(BASE_DIR, 'results_knowledge_base.json')
with open(out_path, 'w') as f:
    json.dump(records, f, indent=2)

# ─── PRINT SUMMARY ────────────────────────────────────────────────────
print("=" * 90)
print("RESULTS KNOWLEDGE BASE -- All experiments sorted by Top-10 accuracy")
print("=" * 90)
print(f"{'#':>3} {'Method':<35} {'Top1':>6} {'Top10':>6} {'SizeMB':>8} {'Compress':>9} {'Source':<25}")
print(f"{'---':>3} {'-' * 35} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 9} {'-' * 25}")

for i, r in enumerate(records, 1):
    t1 = f"{r['top1']:.0%}" if r.get('top1') is not None else '-'
    t10 = f"{r['top10']:.0%}" if r.get('top10') is not None else '-'
    sz = f"{r['size_mb']:.1f}" if r.get('size_mb') is not None else '-'
    comp = f"{r['compression']:.0f}x" if r.get('compression') is not None else '-'
    src = r.get('source_file', '-') or '-'
    print(f"{i:>3} {r['method']:<35} {t1:>6} {t10:>6} {sz:>8} {comp:>9} {src:<25}")

print(f"\nTotal records: {len(records)}")
print(f"Saved to: {out_path}")
