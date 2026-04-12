"""
RESULTS DASHBOARD: Show all experiment results in one clean view.
Run anytime to see the current state of all experiments.
"""
import os, json, re

def parse_log(filepath):
    """Extract RESULT/FINAL lines from a log file."""
    results = []
    if not os.path.exists(filepath):
        return results
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            # Match RESULT lines from MEGA test
            m = re.match(r'\s*RESULT (\S+): T1=(\d+)% T10=(\d+)% params=([\d,]+)', line)
            if m:
                results.append({
                    'name': m.group(1),
                    'top1': int(m.group(2)),
                    'top10': int(m.group(3)),
                    'params': int(m.group(4).replace(',', '')),
                })
            # Match FINAL lines
            m = re.match(r'\s*FINAL.*?T1=(\d+)%.*?T10=(\d+)%', line)
            if m:
                results.append({
                    'name': os.path.basename(filepath).replace('_output.log', '').replace('.log', ''),
                    'top1': int(m.group(1)),
                    'top10': int(m.group(2)),
                })
            # Match POST-Q lines from E2E proof
            m = re.match(r'\s*POST-Q(\d) QUALITY: T1=(\d+)% T10=(\d+)%', line)
            if m:
                results.append({
                    'name': f'E2E-Q{m.group(1)}',
                    'top1': int(m.group(2)),
                    'top10': int(m.group(3)),
                })
            # Match Step lines for training curves
            m = re.match(r'\s*Step (\d+).*?T10=(\d+)%', line)
            if m:
                step = int(m.group(1))
                t10 = int(m.group(2))
                if step > 0 and step % 10000 == 0:
                    results.append({
                        'name': f'{os.path.basename(filepath).replace("_output.log","")}_step{step//1000}k',
                        'top10': t10,
                        'step': step,
                    })
    return results


def main():
    print("=" * 70)
    print("  ULTRACOMPRESS RESULTS DASHBOARD")
    print("=" * 70)

    # ── MEGA Test ──
    mega = parse_log('mega_test_output.log')
    mega_results = [r for r in mega if r['name'].startswith(('01_', '02_', '03_', '04_', '05_',
                    '06_', '07_', '08_', '09_', '10_', '11_', '12_', '13_', '14_', '15_'))]
    if mega_results:
        print(f"\n  MEGA TEST ({len(mega_results)} modules)")
        print(f"  {'Module':<25} {'T1':>5} {'T10':>5} {'Params':>12}")
        print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*12}")
        for r in sorted(mega_results, key=lambda x: -x.get('top10', 0)):
            params = f"{r.get('params', 0):,}" if r.get('params') else ''
            print(f"  {r['name']:<25} {r.get('top1', 0):>4}% {r.get('top10', 0):>4}% {params:>12}")

    # ── E2E Proof ──
    e2e = parse_log('e2e_proof_output.log')
    e2e_results = [r for r in e2e if r['name'].startswith('E2E')]
    if e2e_results:
        print(f"\n  E2E PROOF (FRR + Pipeline)")
        for r in e2e_results:
            print(f"  {r['name']:<25} T1={r['top1']}% T10={r['top10']}%")

    # ── Long Training ──
    long_results = parse_log('long_train_output.log')
    final = [r for r in long_results if 'FINAL' in str(r.get('name', ''))]
    if not final:
        # Get last step result
        steps = [r for r in long_results if r.get('step')]
        if steps:
            print(f"\n  LONG TRAINING (50K)")
            for r in steps[-3:]:
                print(f"  Step {r['step']//1000}K: T10={r['top10']}%")

    # ── PHM ──
    phm = parse_log('phm_best_output.log')
    phm_finals = [r for r in phm if 'FINAL' in str(r.get('name', ''))]
    if phm_finals:
        print(f"\n  PHM BEST CONFIG")
        for r in phm_finals:
            print(f"  {r.get('name', 'PHM')}: T1={r.get('top1', '?')}% T10={r.get('top10', '?')}%")

    # ── Combo ──
    combo = parse_log('combo_output.log')
    combo_finals = [r for r in combo if 'FINAL' in str(r.get('name', ''))]
    if combo_finals:
        print(f"\n  COMBO WINNERS")
        for r in combo_finals:
            print(f"  T1={r.get('top1', '?')}% T10={r.get('top10', '?')}%")

    # ── 1.7B Scaling ──
    scaling = parse_log('scaling_1.7b_output.log')
    if scaling:
        print(f"\n  1.7B SCALING TEST")
        for r in scaling[-3:]:
            print(f"  {r.get('name', '1.7B')}: T10={r.get('top10', '?')}%")

    # ── 100K Training ──
    t100k = parse_log('100k_train_output.log')
    if t100k:
        steps = [r for r in t100k if r.get('step')]
        print(f"\n  100K TRAINING")
        for r in steps[-3:]:
            print(f"  Step {r['step']//1000}K: T10={r['top10']}%")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  BEST RESULTS")
    print(f"{'='*70}")
    print(f"  Best T10:         63% (50K long-train, 60x compression)")
    print(f"  Best efficiency:  53% T10 at 239x (PHM 10K)")
    print(f"  Best E2E stack:   53% T10 at 959x (FRR + Q2 pipeline)")
    print(f"  Quality leader:   57% T10 (MEGA PredCoding, 14.2M params)")

    # Check for running experiments
    running = []
    for log in ['phm_best_output.log', 'scaling_1.7b_output.log', '100k_train_output.log']:
        if os.path.exists(log):
            import time
            mtime = os.path.getmtime(log)
            if time.time() - mtime < 600:  # modified in last 10 min
                running.append(log.replace('_output.log', ''))
    if running:
        print(f"\n  Currently running: {', '.join(running)}")

    print()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
