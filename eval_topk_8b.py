"""eval_topk_8b.py — T1/T10 eval on 8B using precomputed v17 fit (no in-process compression)."""
from __future__ import annotations
import argparse, time
import torch
from eval_v17_ppl import substitute_v17
from eval_claim16_topk import measure_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-8B")
    ap.add_argument("--teacher", default="qwen3_8b_cache.pt")
    ap.add_argument("--v17", default="v17_fit_8b.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="topk_8b_results.pt")
    args = ap.parse_args()
    device = args.device

    print(f"[topk] teacher {args.teacher}", flush=True)
    sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in sd: sd = sd["state_dict"]

    print(f"[topk] tokens {args.tokens}", flush=True)
    toks = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, toks.numel() - args.seq_len - 1, (args.n,), generator=g)

    print(f"[topk] building {args.model_id}", flush=True)
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    print("\n[topk] fp16 teacher", flush=True)
    t = time.time()
    teach, teacher_cache = measure_topk(model, toks, starts, args.seq_len, device, teacher_topk=None)
    print(f"[topk] teacher T1={teach['t1_gt']*100:.2f}% T10={teach['t10_gt']*100:.2f}% "
          f"({time.time()-t:.0f}s)", flush=True)

    print(f"\n[topk] loading {args.v17}", flush=True)
    v17 = torch.load(args.v17, map_location="cpu", weights_only=False)
    substitute_v17(model, sd, v17, device, args.D)

    print("\n[topk] compressed", flush=True)
    t = time.time()
    comp, _ = measure_topk(model, toks, starts, args.seq_len, device, teacher_topk=teacher_cache)
    print(f"[topk] compressed T1={comp['t1_gt']*100:.2f}% T10={comp['t10_gt']*100:.2f}% "
          f"T1ag={comp['t1_agree']*100:.2f}% T10ag={comp['t10_agree']*100:.2f}% "
          f"({time.time()-t:.0f}s)", flush=True)

    retT1 = comp['t1_gt']/teach['t1_gt']*100
    retT10 = comp['t10_gt']/teach['t10_gt']*100
    print("\n" + "="*60, flush=True)
    print(f"CLAIM 16 TOP-K on {args.model_id}  (n={args.n}, seq_len={args.seq_len})", flush=True)
    print("="*60, flush=True)
    print(f"  fp16    T1={teach['t1_gt']*100:.2f}%  T10={teach['t10_gt']*100:.2f}%", flush=True)
    print(f"  2.40bpw T1={comp['t1_gt']*100:.2f}%  T10={comp['t10_gt']*100:.2f}%", flush=True)
    print(f"  agreement T1={comp['t1_agree']*100:.2f}%  T10={comp['t10_agree']*100:.2f}%", flush=True)
    print(f"  retention T1={retT1:.2f}%  T10={retT10:.2f}%", flush=True)

    torch.save({"model_id": args.model_id, "teacher": teach, "compressed": comp,
                "retention_t1": retT1, "retention_t10": retT10}, args.out)
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
