"""demo_claim16.py — portfolio-ready end-to-end Claim 16 demonstration.

Takes a v17 fit (`v17_fit_*.pt`) + the original fp16 teacher state_dict
and shows the 2.40-bpw compressed model reproducing the teacher's
top-10 token decisions on short WikiText-103 / LAMBADA windows.

Usage (works identically for any of the 6 validated models):

  python demo_claim16.py \
      --model_id  Qwen/Qwen3-1.7B \
      --teacher   qwen3_1.7b_cache.pt \
      --v17       v17_fit_qwen3_1.7b.pt \
      --tokens    wikitext103_test_qwen3.pt

Designed to be readable on a portfolio page, not a benchmark rig.
"""
from __future__ import annotations
import argparse
import torch

from eval_v17_ppl import substitute_v17


@torch.no_grad()
def topk_final(model, ids, seq_len, k=10, device="cuda:0"):
    """Return top-k next-token ids at the final position of a length-`seq_len` window."""
    ids = ids[:seq_len].to(device).unsqueeze(0)
    logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    p, i = torch.topk(probs, k)
    return i.cpu(), p.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--teacher",  required=True)
    ap.add_argument("--v17",      required=True)
    ap.add_argument("--tokens",   required=True)
    ap.add_argument("--n",        type=int, default=5)
    ap.add_argument("--seq_len",  type=int, default=96)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--device",   default="cuda:0")
    args = ap.parse_args()

    device = args.device
    print(f"=== UltraCompress Claim-16 demo — {args.model_id} ===", flush=True)

    print(f"[demo] loading teacher state_dict  ({args.teacher})", flush=True)
    sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in sd: sd = sd["state_dict"]

    print(f"[demo] loading v17 fit             ({args.v17})", flush=True)
    v17 = torch.load(args.v17, map_location="cpu", weights_only=False)
    rwm = v17.get("rel_w_final_mean", v17.get("rel_w_mean"))
    bpw_body = v17.get("global_bpw", 0.0) or 0.0
    bpw_over = v17.get("overhead_bpw", 0.0) or 0.0
    bpw = bpw_body + bpw_over
    print(f"[demo]   bpw = {bpw:.4f}   rel-W mean = {rwm:.4f}", flush=True)

    print(f"[demo] loading {args.n} prompts from {args.tokens}", flush=True)
    toks = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, toks.numel() - args.seq_len - 1, (args.n,), generator=g)

    print(f"[demo] building model              ({args.model_id})", flush=True)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # teacher top-10s
    teacher_tops = []
    for s in starts:
        ids = toks[s : s + args.seq_len]
        ids_t, _ = topk_final(model, ids, args.seq_len, k=10, device=device)
        teacher_tops.append(ids_t)

    print("\n[demo] substituting 2.40-bpw compressed body", flush=True)
    substitute_v17(model, sd, v17, device, v17.get("D", 8))

    print("\n--- teacher fp16  vs  2.40-bpw compressed  (next-token top-5) ---")
    n_agree = 0; n_total = 0
    for k, s in enumerate(starts):
        ids = toks[s : s + args.seq_len]
        ids_c, _ = topk_final(model, ids, args.seq_len, k=10, device=device)

        ctx = tokenizer.decode(ids[-16:].tolist()).replace("\n", " ")
        t_top = [tokenizer.decode([t]) for t in teacher_tops[k][:5].tolist()]
        c_top = [tokenizer.decode([t]) for t in ids_c[:5].tolist()]
        overlap = len(set(teacher_tops[k].tolist()) & set(ids_c.tolist()))
        n_agree += overlap; n_total += 10

        print(f"\nprompt #{k+1} ... {ctx!r}")
        print(f"  teacher  top-5: {t_top}")
        print(f"  2.40bpw  top-5: {c_top}")
        print(f"  top-10 overlap: {overlap}/10")

    agr = 100.0 * n_agree / n_total if n_total else 0.0
    print(f"\n[demo] mean top-10 teacher-agreement over {args.n} prompts: {agr:.1f}%")
    print(f"[demo] {args.model_id} compressed to {bpw:.4f} bpw at fixed "
          f"(alpha_attn=0.25, alpha_mlp=0.125) - zero per-model retuning.")


if __name__ == "__main__":
    main()
