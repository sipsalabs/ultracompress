"""
eval_claim16_topk.py — measure top-1 / top-10 next-token metrics at the
Claim-16 operating point (alpha_attn=0.25, alpha_mlp=0.125), comparing
fp16 teacher vs compressed student on identical WikiText-103 windows.

Metrics (all per-token, averaged across all positions of all windows):
  t1_gt      = P(argmax logits == ground-truth next token)
  t10_gt     = P(ground-truth next token in top-10 logits)
  t1_agree   = P(student argmax == teacher argmax)
  t10_agree  = P(teacher argmax in student top-10)
"""
from __future__ import annotations
import argparse, gc, time
import torch
import torch.nn.functional as F

from compress_v17 import v17_compress
from compress_v16 import DEFAULT_ROLE_K
from per_role_alpha_sweep import make_dict
from eval_v16_ppl import reset_teacher
from eval_v17_ppl import substitute_v17


@torch.no_grad()
def measure_topk(model, tokens, starts, seq_len, device, teacher_topk=None):
    """Returns (t1_gt, t10_gt, t1_agree, t10_agree, teacher_topk_cache).
    If teacher_topk is provided (list indexed by window), use it for agreement
    metrics; else compute teacher_topk = None (agreement metrics NaN)."""
    model.eval()
    t1_gt = 0; t10_gt = 0; n = 0
    t1_ag = 0; t10_ag = 0
    cache = []
    for i, s in enumerate(starts.tolist()):
        win = tokens[s:s + seq_len + 1].to(device=device, dtype=torch.long)
        inp = win[:-1].unsqueeze(0)
        tgt = win[1:]
        logits = model(inp).logits[0]          # [L, V]
        top1 = logits.argmax(-1)               # [L]
        top10 = logits.topk(10, dim=-1).indices  # [L, 10]
        # ground truth
        t1_gt += (top1 == tgt).sum().item()
        t10_gt += (top10 == tgt.unsqueeze(-1)).any(-1).sum().item()
        n += tgt.numel()
        if teacher_topk is not None:
            t_top1 = teacher_topk[i]["top1"].to(device)
            t1_ag += (top1 == t_top1).sum().item()
            t10_ag += (top10 == t_top1.unsqueeze(-1)).any(-1).sum().item()
        else:
            cache.append({"top1": top1.cpu()})
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(starts)}] t1_gt={t1_gt/n*100:.2f}%  "
                  f"t10_gt={t10_gt/n*100:.2f}%")
    out = {
        "t1_gt": t1_gt / n, "t10_gt": t10_gt / n, "n": n,
        "t1_agree": t1_ag / n if teacher_topk is not None else float("nan"),
        "t10_agree": t10_ag / n if teacher_topk is not None else float("nan"),
    }
    return out, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--activations", default="v17_activations.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--a_attn", type=float, default=0.25)
    ap.add_argument("--a_mlp", type=float, default=0.125)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="claim16_topk_results.pt")
    args = ap.parse_args()
    device = args.device

    print(f"[topk] loading teacher")
    teacher_sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in teacher_sd: teacher_sd = teacher_sd["state_dict"]

    print(f"[topk] loading tokens")
    all_tokens = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, all_tokens.numel() - args.seq_len - 1,
                           (args.n,), generator=g)

    print(f"[topk] building model on {device}")
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(teacher_sd, strict=False)
    model = model.to(device).eval()

    results = {"args": vars(args)}

    # ---- fp16 teacher ----
    print("\n[topk] === fp16 TEACHER ===")
    t = time.time()
    teach_metrics, teacher_cache = measure_topk(
        model, all_tokens, starts, args.seq_len, device, teacher_topk=None)
    print(f"[topk] teacher  t1_gt={teach_metrics['t1_gt']*100:.2f}%  "
          f"t10_gt={teach_metrics['t10_gt']*100:.2f}%  ({time.time()-t:.0f}s)")
    results["teacher"] = teach_metrics

    # ---- fit Claim-16 at (a_attn, a_mlp) ----
    print(f"\n[topk] === fitting Claim 16 at a_attn={args.a_attn} "
          f"a_mlp={args.a_mlp} ===")
    alpha_dict = make_dict(args.a_attn, args.a_mlp)
    r = v17_compress(args.teacher, args.activations, DEFAULT_ROLE_K, args.D,
                     alpha=args.a_attn, alpha_per_role=alpha_dict,
                     iters=args.iters, beam=8, device=device)
    v17 = {"banks": r["banks"], "s_col": r["s_col"],
           "alpha": args.a_attn, "alpha_per_role": alpha_dict,
           "global_bpw": r["global_bpw"],
           "overhead_bpw": r["overhead_bpw"]}
    reset_teacher(model, teacher_sd)
    substitute_v17(model, teacher_sd, v17, device, args.D)

    print("\n[topk] === Claim 16 COMPRESSED ===")
    t = time.time()
    comp_metrics, _ = measure_topk(model, all_tokens, starts, args.seq_len,
                                    device, teacher_topk=teacher_cache)
    print(f"[topk] claim16  t1_gt={comp_metrics['t1_gt']*100:.2f}%  "
          f"t10_gt={comp_metrics['t10_gt']*100:.2f}%  "
          f"t1_agree={comp_metrics['t1_agree']*100:.2f}%  "
          f"t10_agree={comp_metrics['t10_agree']*100:.2f}%  "
          f"({time.time()-t:.0f}s)")
    results["claim16"] = comp_metrics
    results["alpha_per_role"] = alpha_dict

    # summary
    print("\n" + "=" * 72)
    print(f"CLAIM 16 TOP-K FIDELITY (n={args.n} windows, seq_len={args.seq_len})")
    print("=" * 72)
    print(f"  fp16 teacher    T1(gt) = {teach_metrics['t1_gt']*100:6.2f}%   "
          f"T10(gt) = {teach_metrics['t10_gt']*100:6.2f}%")
    print(f"  2.40 bpw stack  T1(gt) = {comp_metrics['t1_gt']*100:6.2f}%   "
          f"T10(gt) = {comp_metrics['t10_gt']*100:6.2f}%")
    print(f"                  T1(agree) = {comp_metrics['t1_agree']*100:6.2f}%"
          f"   T10(agree) = {comp_metrics['t10_agree']*100:6.2f}%")
    print(f"  retention T1 = {comp_metrics['t1_gt']/teach_metrics['t1_gt']*100:5.2f}%"
          f"   T10 = {comp_metrics['t10_gt']/teach_metrics['t10_gt']*100:5.2f}%")

    torch.save(results, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
