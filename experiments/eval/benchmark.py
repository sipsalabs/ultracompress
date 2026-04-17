"""Automated benchmarking — run all experiments and log results."""
import torch, json, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_FILE = "benchmark_results.json"

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []

def save_result(result):
    results = load_results()
    results.append(result)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Result saved ({len(results)} total)")

def print_leaderboard():
    results = load_results()
    if not results:
        print("No results yet.")
        return

    # Sort by top10 descending
    results.sort(key=lambda r: r.get('top10', 0), reverse=True)

    print(f"\n{'='*70}")
    print(f"  ULTRACOMPRESS LEADERBOARD ({len(results)} experiments)")
    print(f"{'='*70}")
    print(f"{'#':>3} {'Method':30s} {'Top1':>6} {'Top10':>6} {'Params':>10} {'Comp':>6} {'Time':>6}")
    print(f"{'-'*70}")

    for i, r in enumerate(results):
        print(f"{i+1:>3} {r.get('name','?'):30s} "
              f"{r.get('top1',0)*100:>5.0f}% "
              f"{r.get('top10',0)*100:>5.0f}% "
              f"{r.get('params',0):>10,} "
              f"{r.get('compression',0):>5.0f}x "
              f"{r.get('time',0):>5.0f}s")
    print(f"{'='*70}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--add", nargs=6, metavar=('NAME','TOP1','TOP10','PARAMS','COMP','TIME'),
                       help="Add result: name top1 top10 params compression time")
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
    elif args.add:
        save_result({
            'name': args.add[0],
            'top1': float(args.add[1]),
            'top10': float(args.add[2]),
            'params': int(args.add[3]),
            'compression': float(args.add[4]),
            'time': float(args.add[5]),
            'timestamp': time.strftime('%Y-%m-%d %H:%M'),
        })
    else:
        print_leaderboard()
