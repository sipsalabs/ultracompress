"""Merge head-to-head n=500 results and emit Claim-20 summary tables."""
import json, statistics

small = json.load(open("h2h_n500_small.json"))
large = json.load(open("h2h_n500_large.json"))
rows = small + large
json.dump(rows, open("h2h_n500_full.json", "w"), indent=2)
print(f"merged rows: {len(rows)}  -> h2h_n500_full.json")
print()

ix = {(r["name"], r["method"]): r for r in rows}
models = []
for r in rows:
    if r["name"] not in models:
        models.append(r["name"])

methods_order = [
    "bnb_nf4", "bnb_int8",
    "hqq_2bit_g64", "hqq_2bit_g16", "hqq_3bit_g64", "hqq_4bit_g64",
    "our_fp8_2p79", "our_mixed_2p79",
]

def pr_fmt(p):
    if p is None: return "     n/a"
    if p < 100: return f"{p:8.3f}"
    return f"{p:8.1e}"

print("=" * 110)
print("PER-MODEL TABLE (n=500, LAMBADA, seq_len=128)")
print("=" * 110)
for n in models:
    print()
    print(f"### {n}")
    print(f"{'method':18s}  {'bpw':>6s}  {'t1_ret':>8s}  {'ppl_r':>8s}  {'t1_stu':>7s}  {'t1_tch':>7s}  {'t1vsT':>7s}")
    for m in methods_order:
        r = ix.get((n, m))
        if r is None: continue
        ret_s = f"{r['t1_ret']*100:6.2f}%"
        print(f"{m:18s}  {r['effective_bpw']:6.3f}  {ret_s:>8s}  {pr_fmt(r['ppl_ratio']):>8s}  "
              f"{r['student_t1']*100:6.2f}%  {r['teacher_t1']*100:6.2f}%  {r['student_t1_vs_teacher']*100:6.2f}%")

print()
print("=" * 110)
print("COHORT AVERAGES (all 6 models, by method)")
print("=" * 110)
print(f"{'method':18s}  {'bpw_avg':>7s}  {'t1_ret_avg':>10s}  {'ppl_r_med':>10s}  {'n_models':>8s}")
for m in methods_order:
    rs = [ix[(n,m)] for n in models if (n,m) in ix]
    if not rs: continue
    bpw = statistics.mean(r["effective_bpw"] for r in rs)
    ret_avg = statistics.mean(r["t1_ret"] for r in rs)
    ppl_med = statistics.median(r["ppl_ratio"] for r in rs)
    print(f"{m:18s}  {bpw:7.3f}  {ret_avg*100:9.2f}%  {ppl_med:10.3f}  {len(rs):8d}")

print()
print("=" * 110)
print("CROSS COMPARISONS (ours vs external, per model)")
print("=" * 110)
print(f"{'model':22s}  {'comparison':38s}  {'d_bpw':>7s}  {'d_t1ret_pp':>10s}  {'pplR_ours':>10s}  {'pplR_ext':>10s}")
for n in models:
    for ours_key in ["our_fp8_2p79", "our_mixed_2p79"]:
        ours = ix.get((n, ours_key))
        if ours is None: continue
        for ext_m in ["bnb_nf4", "bnb_int8", "hqq_4bit_g64", "hqq_3bit_g64", "hqq_2bit_g16", "hqq_2bit_g64"]:
            ext = ix.get((n, ext_m))
            if ext is None: continue
            dret = (ours["t1_ret"] - ext["t1_ret"]) * 100
            db = ours["effective_bpw"] - ext["effective_bpw"]
            tag = f"{ours_key[4:]} vs {ext_m}"
            print(f"{n:22s}  {tag:38s}  {db:+7.3f}  {dret:+10.2f}  {ours['ppl_ratio']:10.3f}  {ext['ppl_ratio']:10.3f}")
    print()

print("=" * 110)
print("CATASTROPHIC FAILURE CASES (ppl_ratio > 10)")
print("=" * 110)
for r in rows:
    if r.get("ppl_ratio") and r["ppl_ratio"] > 10:
        print(f"  {r['name']:22s}  {r['method']:18s}  bpw={r['effective_bpw']:.3f}  "
              f"t1_ret={r['t1_ret']*100:.2f}%  ppl_r={r['ppl_ratio']:.2e}")
