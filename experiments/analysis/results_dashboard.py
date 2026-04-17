"""RESULTS DASHBOARD — Collect and display ALL experiment results.

Scans all result JSON files and log files, builds a unified leaderboard.
Run anytime to see the current state of all experiments.

Usage: python results_dashboard.py
"""
import json, os, re, sys
from datetime import datetime

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))

# Collect from JSON result files
json_results = {}
for f in os.listdir(RESULT_DIR):
    if f.endswith('_results.json') or f == 'benchmark_results.json':
        path = os.path.join(RESULT_DIR, f)
        try:
            with open(path) as fp:
                data = json.load(fp)
            for name, r in data.items():
                key = f"{f.replace('_results.json','')}:{name}"
                json_results[key] = r
        except:
            pass

# Collect from log files (grep for RESULT lines)
log_results = {}
for f in os.listdir(RESULT_DIR):
    if f.endswith('_output.log') or f.endswith('.log'):
        path = os.path.join(RESULT_DIR, f)
        try:
            with open(path) as fp:
                for line in fp:
                    m = re.search(r'RESULT\s+(\S+):\s+Top1=(\d+)%\s+Top10=(\d+)%', line)
                    if m:
                        name = m.group(1)
                        t1 = int(m.group(2)) / 100
                        t10 = int(m.group(3)) / 100
                        # Extract size if present
                        size_m = re.search(r'Size=([0-9.]+)MB', line)
                        comp_m = re.search(r'(\d+\.?\d*)x', line)
                        key = f"{f.replace('_output.log','').replace('.log','')}:{name}"
                        log_results[key] = {
                            'top1': t1, 'top10': t10,
                            'size_mb': float(size_m.group(1)) if size_m else 0,
                            'compression': float(comp_m.group(1)) if comp_m else 0,
                            'source': f,
                        }
        except:
            pass

# Merge (prefer JSON over log)
all_results = {}
all_results.update(log_results)
for k, v in json_results.items():
    if isinstance(v, dict) and 'top10' in v:
        all_results[k] = v

# Print dashboard
print("=" * 80)
print(f"  ULTRACOMPRESS RESULTS DASHBOARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 80)

if not all_results:
    print("  No results found.")
    sys.exit(0)

# Sort by top10
sorted_all = sorted(all_results.items(), key=lambda x: x[1].get('top10', 0), reverse=True)

# Top results
print(f"\n  TOP 10 RESULTS (by Top-10 accuracy):")
print(f"  {'Rank':<5} {'Name':<40} {'Top1':>5} {'Top10':>6} {'Size':>8} {'Compress':>9}")
print(f"  {'-'*5} {'-'*40} {'-'*5} {'-'*6} {'-'*8} {'-'*9}")

for i, (name, r) in enumerate(sorted_all[:10]):
    t1 = r.get('top1', 0)
    t10 = r.get('top10', 0)
    size = r.get('size_mb', 0)
    comp = r.get('compression', 0)
    rank = f"#{i+1}"
    print(f"  {rank:<5} {name:<40} {t1*100:>4.0f}% {t10*100:>5.0f}% {size:>7.1f}MB {comp:>8.0f}x")

# By approach
print(f"\n  RESULTS BY APPROACH:")
approaches = {}
for name, r in sorted_all:
    # Guess approach from name
    if 'frr' in name.lower() or 'FRR' in name:
        cat = 'FRR (Fractal)'
    elif 'genome' in name.lower() or 'V1' in name:
        cat = 'Genome'
    elif 'alg' in name.lower() or 'Alg' in name:
        cat = 'Algebraic'
    elif 'codec' in name.lower() or 'Codec' in name:
        cat = 'Weight Codec'
    elif 'nerf' in name.lower() or 'NeRF' in name:
        cat = 'NeRF'
    elif 'proc' in name.lower() or 'Proc' in name:
        cat = 'Procedural'
    elif 'seed' in name.lower() or 'Seed' in name:
        cat = 'Seed'
    elif 'swarm' in name.lower() or 'Swarm' in name:
        cat = 'Swarm'
    elif 'prog' in name.lower() or 'Prog' in name:
        cat = 'Program'
    elif 'dna' in name.lower() or 'DNA' in name:
        cat = 'DNA'
    elif 'hwi' in name.lower() or 'HWI' in name:
        cat = 'Holographic'
    elif 'gwe' in name.lower() or 'GWE' in name:
        cat = 'GWE'
    else:
        cat = 'Other'

    if cat not in approaches:
        approaches[cat] = []
    approaches[cat].append((name, r))

for cat in sorted(approaches.keys()):
    results = approaches[cat]
    best = max(results, key=lambda x: x[1].get('top10', 0))
    print(f"\n  {cat}:")
    for name, r in sorted(results, key=lambda x: x[1].get('top10', 0), reverse=True)[:3]:
        t10 = r.get('top10', 0)
        size = r.get('size_mb', 0)
        print(f"    {name:<38} Top10={t10*100:>4.0f}% {size:>6.1f}MB")

# Summary stats
total_experiments = len(all_results)
best_name, best = sorted_all[0]
print(f"\n  SUMMARY:")
print(f"    Total experiments: {total_experiments}")
print(f"    Best result: {best_name}")
print(f"    Best Top-10: {best.get('top10',0)*100:.0f}%")
print(f"    Best compression: {max(r.get('compression',0) for r in all_results.values()):.0f}x")
print(f"    Smallest model: {min(r.get('size_mb',999) for r in all_results.values() if r.get('size_mb',0) > 0):.1f}MB")

# Running experiments
print(f"\n  RUNNING EXPERIMENTS:")
for f in os.listdir(RESULT_DIR):
    if f.endswith('_output.log'):
        path = os.path.join(RESULT_DIR, f)
        mtime = os.path.getmtime(path)
        age = (datetime.now().timestamp() - mtime) / 60
        if age < 30:  # Modified in last 30 min
            print(f"    {f} (updated {age:.0f} min ago)")

print(f"\n{'='*80}")
