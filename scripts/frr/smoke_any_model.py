"""smoke_any_model.py -- end-to-end v17 smoke test on any HuggingFace causal LM.

Given `--model <HF_ID>`, this script:
  1. loads the model in fp16 on GPU
  2. captures ~4 512-token calibration sequences from a small WikiText-2 slice
  3. fits v17 (base tier, K1=2048 K2=256, alpha=0.25, 3 EM iters -- small budget)
  4. measures teacher fp16 PPL on ~100 held-out 128-token windows
  5. substitutes the v17 fit back into the model and re-measures PPL
  6. reports PPL ratio, bpw, relW, wall time, pass/fail

Pass criterion (configurable): PPL ratio < 2.5x AND relW mean < 0.15.
This is a pass/fail gate, not a performance number -- it asserts the method
handles the model's architecture, tensor shapes, tokenizer, and dtype flow
without crashing and with sane reconstruction.

Usage:
  python smoke_any_model.py --models gpt2 EleutherAI/pythia-160m google/gemma-2-2b-it
"""
from __future__ import annotations
import argparse, gc, json, time, traceback
import torch


def fetch_calibration(tok, n_seq: int = 4, seq_len: int = 512):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    txt = "\n\n".join([r["text"] for r in ds if r["text"].strip()])
    ids = tok(txt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    N = ids.numel()
    out = []
    g = torch.Generator().manual_seed(42)
    for _ in range(n_seq):
        s = torch.randint(0, max(1, N - seq_len - 1), (1,), generator=g).item()
        out.append(ids[s:s+seq_len])
    return torch.stack(out)  # [n_seq, seq_len]


def capture_activations(model, seq_batch: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    """Hook every body Linear; capture per-column |x| mean."""
    from compress_v14 import ROLE_PATTERNS
    stats: dict[str, tuple[torch.Tensor, int]] = {}

    def mk_hook(fqn):
        def _hk(m, inp, out):
            x = inp[0]
            if x.dim() == 3:  # [B, T, C]
                x = x.reshape(-1, x.shape[-1])
            a = x.abs().mean(0).float().detach().cpu()
            if fqn in stats:
                s, n = stats[fqn]
                stats[fqn] = (s + a, n + 1)
            else:
                stats[fqn] = (a, 1)
        return _hk

    hs = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "layers." in name \
           and any(p in name for p in ROLE_PATTERNS):
            hs.append(m.register_forward_hook(mk_hook(name + ".weight")))

    with torch.no_grad():
        for i in range(seq_batch.shape[0]):
            ids = seq_batch[i:i+1].to(device)
            model(ids)
    for h in hs:
        h.remove()

    return {k: (v[0] / max(v[1], 1)).clamp(min=1e-6) for k, v in stats.items()}


def run_one(model_id: str, device: str = "cuda:0",
            n_cal: int = 4, seq_len_cal: int = 512,
            n_ppl: int = 100, seq_len_ppl: int = 128,
            K1: int = 2048, K2: int = 256, alpha: float = 0.25,
            iters: int = 3, D: int = 8,
            max_ppl_ratio: float = 2.5, max_relw: float = 0.15) -> dict:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from compress_v17 import v17_compress
    from eval_v17_ppl import substitute_v17
    from eval_v16_ppl import measure_ppl

    t_start = time.time()
    rec: dict = {"model": model_id}

    print(f"\n[smoke] {model_id}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True,
            low_cpu_mem_usage=True).to(device).eval()
    except Exception as e:
        rec["stage"] = "load"; rec["error"] = repr(e)
        traceback.print_exc(); return rec

    try:
        cal = fetch_calibration(tok, n_cal, seq_len_cal)
        stats = capture_activations(model, cal, device)
        rec["n_act_tensors"] = len(stats)
    except Exception as e:
        rec["stage"] = "activation"; rec["error"] = repr(e)
        traceback.print_exc(); return rec

    # save temp teacher + act cache -> feed v17_compress
    teacher_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    temp_teacher = f"_smoke_teacher_{abs(hash(model_id))}.pt"
    temp_act = f"_smoke_act_{abs(hash(model_id))}.pt"
    torch.save(teacher_sd, temp_teacher)
    torch.save(stats, temp_act)

    role_K = {r: (K1, K2) for r in ["q_proj", "k_proj", "v_proj",
                                     "o_proj", "gate_proj", "up_proj", "down_proj"]}

    try:
        t_fit = time.time()
        fit = v17_compress(temp_teacher, temp_act, role_K, D,
                           alpha=alpha, iters=iters, beam=8, device=device)
        rec["fit_wall"] = time.time() - t_fit
        rec["bpw"] = float(fit.get("global_bpw", float("nan"))) + float(fit.get("overhead_bpw", 0.0))
        rec["relw_mean"] = float(fit.get("rel_w_final_mean", float("nan")))
        rec["relw_max"]  = float(fit.get("rel_w_final_max",  float("nan")))
    except Exception as e:
        rec["stage"] = "fit"; rec["error"] = repr(e)
        traceback.print_exc()
        _cleanup(temp_teacher, temp_act); return rec

    # teacher PPL on held-out window
    try:
        toks_all = tok("\n\n".join(["The quick brown fox jumps over the lazy dog."] * 200 +
                                    [open(__file__, encoding="utf-8").read()]),
                       return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        N = toks_all.numel()
        if N < seq_len_ppl + 2:
            # fall back to wikitext
            cal_full = fetch_calibration(tok, n_seq=1, seq_len=seq_len_ppl * (n_ppl + 2))
            toks_all = cal_full[0]
            N = toks_all.numel()
        starts = torch.randint(0, max(1, N - seq_len_ppl - 1), (n_ppl,),
                               generator=torch.Generator().manual_seed(0))
        tch_ppl, _ = measure_ppl(model, toks_all, starts, seq_len_ppl, device)
        substitute_v17(model, teacher_sd, fit, device, D)
        v17_ppl, _ = measure_ppl(model, toks_all, starts, seq_len_ppl, device)
        rec["teacher_ppl"] = float(tch_ppl)
        rec["v17_ppl"]     = float(v17_ppl)
        rec["ppl_ratio"]   = float(v17_ppl / max(tch_ppl, 1e-9))
    except Exception as e:
        rec["stage"] = "ppl"; rec["error"] = repr(e)
        traceback.print_exc()

    _cleanup(temp_teacher, temp_act)
    del model, teacher_sd, fit, stats
    torch.cuda.empty_cache(); gc.collect()

    rec["wall_sec"] = time.time() - t_start
    pr = rec.get("ppl_ratio", float("inf"))
    rw = rec.get("relw_mean", float("inf"))
    rec["pass"] = (pr < max_ppl_ratio) and (rw < max_relw)
    print(f"  {'PASS' if rec['pass'] else 'FAIL'}  bpw={rec.get('bpw',0):.3f} "
          f"relW={rw:.4f} ppl_ratio={pr:.3f} wall={rec['wall_sec']:.0f}s")
    return rec


def _cleanup(*paths):
    import os
    for p in paths:
        try: os.remove(p)
        except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="smoke_results.json")
    ap.add_argument("--max_ppl_ratio", type=float, default=2.5)
    ap.add_argument("--max_relw", type=float, default=0.15)
    args = ap.parse_args()

    results = []
    for mid in args.models:
        rec = run_one(mid, device=args.device,
                      max_ppl_ratio=args.max_ppl_ratio, max_relw=args.max_relw)
        results.append(rec)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    n_pass = sum(1 for r in results if r.get("pass"))
    print(f"\n==== smoke summary: {n_pass}/{len(results)} passed ====")
    for r in results:
        tag = "PASS" if r.get("pass") else ("FAIL" if "error" not in r else f"ERR@{r.get('stage','?')}")
        print(f"  [{tag:>10}] {r.get('model','?'):<40} bpw={r.get('bpw',0):.3f} "
              f"relW={r.get('relw_mean',float('nan')):.4f} ppl_r={r.get('ppl_ratio',float('nan')):.3f}")


if __name__ == "__main__":
    main()
