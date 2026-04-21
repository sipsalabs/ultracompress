"""eval_v17_8b.py — model-agnostic Claim 16 eval (8B or any Qwen3).
Baseline fp16 PPL + v17-compressed PPL on wikitext103 windows.
"""
from __future__ import annotations
import argparse, gc, time
import torch
from eval_v17_ppl import substitute_v17
from eval_v16_ppl import measure_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-8B")
    ap.add_argument("--v17", default="v17_fit_8b.pt")
    ap.add_argument("--teacher", default="qwen3_8b_cache.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v17_8b_ppl.pt")
    args = ap.parse_args()
    device = args.device

    print(f"[eval] teacher {args.teacher}", flush=True)
    sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in sd: sd = sd["state_dict"]

    print(f"[eval] tokens {args.tokens}", flush=True)
    toks = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, toks.numel() - args.seq_len - 1, (args.n,), generator=g)

    print(f"[eval] building {args.model_id}", flush=True)
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    print("\n[eval] baseline fp16", flush=True)
    t = time.time()
    ppl_b, nll_b = measure_ppl(model, toks, starts, args.seq_len, device)
    print(f"[eval] baseline PPL = {ppl_b:.4f} ({time.time()-t:.0f}s)", flush=True)

    print(f"\n[eval] loading {args.v17}", flush=True)
    v17 = torch.load(args.v17, map_location="cpu", weights_only=False)
    rwm = v17.get("rel_w_final_mean", v17.get("rel_w_mean"))
    rwx = v17.get("rel_w_final_max", v17.get("rel_w_max"))
    print(f"[eval]   rel-W mean={rwm:.4f} max={rwx:.4f}  "
          f"bpw={v17.get('global_bpw',0):.3f}", flush=True)
    substitute_v17(model, sd, v17, device, args.D)
    t = time.time()
    ppl_17, nll_17 = measure_ppl(model, toks, starts, args.seq_len, device)
    print(f"[eval] v17 PPL = {ppl_17:.4f} (ratio {ppl_17/ppl_b:.3f}x) "
          f"({time.time()-t:.0f}s)", flush=True)

    out = {"model_id": args.model_id, "ppl_baseline": ppl_b,
           "ppl_v17": ppl_17, "ratio": ppl_17/ppl_b,
           "rel_w_final_mean": v17.get("rel_w_final_mean"),
           "global_bpw": v17.get("global_bpw"),
           "overhead_bpw": v17.get("overhead_bpw")}
    torch.save(out, args.out)
    print(f"\nSaved {args.out}", flush=True)
    print(f"SUMMARY: {args.model_id}  baseline {ppl_b:.4f}  "
          f"v17 {ppl_17:.4f}  ratio {ppl_17/ppl_b:.3f}x", flush=True)


if __name__ == "__main__":
    main()
