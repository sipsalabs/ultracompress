"""lambada_all.py -- LAMBADA cross-corpus generalization across all 6 v17 fits.

For each of the 6 validated models, (1) tokenize LAMBADA with the model's own
tokenizer if not already cached, (2) measure teacher fp16 PPL + T1 + T10,
(3) substitute the v17 fit, (4) measure v17 PPL + T1 + T10, (5) write a row to
lambada_all_results.json.

LAMBADA (EleutherAI/lambada_openai) is narrative fiction from BookCorpus,
completely disjoint from WikiText-103's encyclopedic style. A Claim-16 v17
fit trained on WikiText-103 calibration tokens but retaining LAMBADA PPL +
T1 to within a consistent envelope proves the 2.40-bpw operating point is
a property of the *model body*, not the eval corpus.
"""
from __future__ import annotations
import argparse, gc, json, os, time, traceback
import torch

MODELS = [
    # (display_name, hf_model_id, teacher_cache, v17_fit, lambada_tokens_path)
    ("OLMo-2-1B",      "allenai/OLMo-2-0425-1B",              "olmo2_1b_cache.pt",         "v17_fit_olmo2.pt",       "lambada_test_olmo2.pt"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  "tinyllama_1.1b_cache.pt",   "v17_fit_tinyllama.pt",   "lambada_test_tinyllama.pt"),
    ("Qwen3-1.7B",     "Qwen/Qwen3-1.7B",                     "qwen3_1.7b_cache.pt",       "v17_fit_qwen3_1.7b.pt",  "lambada_test_qwen3.pt"),
    ("SmolLM2-1.7B",   "HuggingFaceTB/SmolLM2-1.7B",          "smollm2_1.7b_cache.pt",     "v17_fit_smollm2.pt",     "lambada_test_smollm2.pt"),
    ("Mistral-7B",     "mistralai/Mistral-7B-v0.3",           "mistral_7b_v0.3_cache.pt",  "v17_fit_mistral.pt",     "lambada_test_mistral.pt"),
    ("Qwen3-8B",       "Qwen/Qwen3-8B",                       "qwen3_8b_cache.pt",         "v17_fit_8b.pt",          "lambada_test_qwen3.pt"),  # shares Qwen3 tokenizer
]


def tokenize_if_missing(model_id: str, out_path: str):
    if os.path.exists(out_path):
        return
    from datasets import load_dataset
    from transformers import AutoTokenizer
    print(f"[tok] {out_path} <- LAMBADA via {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    text = "\n\n".join([r["text"] for r in ds if r["text"].strip()])
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    torch.save(ids.to(torch.int32), out_path)


def run_one(name: str, model_id: str, teacher_path: str, v17_path: str,
            tokens_path: str, n: int, seq_len: int, device: str) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    from eval_v16_ppl import measure_ppl, reset_teacher
    from eval_v17_ppl import substitute_v17
    from eval_claim16_topk import measure_topk

    print(f"\n[{name}] teacher -> {teacher_path}", flush=True)
    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    toks = torch.load(tokens_path, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(42)
    starts = torch.randint(0, toks.numel() - seq_len - 1, (n,), generator=g)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    t0 = time.time()
    tch_topk, tch_cache = measure_topk(model, toks, starts, seq_len, device, teacher_topk=None)
    tch_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] teacher  PPL={tch_ppl:.4f}  T1={tch_topk['t1_gt']*100:.2f}%  T10={tch_topk['t10_gt']*100:.2f}%   ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    D = v17.get("D", 8)
    substitute_v17(model, sd, v17, device, D)
    v17_topk, _ = measure_topk(model, toks, starts, seq_len, device, teacher_topk=tch_cache)
    v17_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] v17 fit  PPL={v17_ppl:.4f}  T1={v17_topk['t1_gt']*100:.2f}%  T10={v17_topk['t10_gt']*100:.2f}%   "
          f"T1_vs_teacher={v17_topk['t1_agree']*100:.2f}%  ({time.time()-t1:.0f}s)", flush=True)

    ppl_ratio = v17_ppl / tch_ppl
    del model, v17, sd, toks, tch_cache
    torch.cuda.empty_cache(); gc.collect()

    return {
        "name": name, "model_id": model_id, "n": n, "seq_len": seq_len,
        "teacher_ppl": tch_ppl, "teacher_t1": tch_topk["t1_gt"], "teacher_t10": tch_topk["t10_gt"],
        "v17_ppl": v17_ppl, "v17_t1": v17_topk["t1_gt"], "v17_t10": v17_topk["t10_gt"],
        "v17_t1_vs_teacher": v17_topk["t1_agree"],
        "ppl_ratio": ppl_ratio,
        "t1_ret": v17_topk["t1_gt"] / tch_topk["t1_gt"] if tch_topk["t1_gt"] > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="lambada_all_results.json")
    args = ap.parse_args()

    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {r["name"] for r in results}

    for name, mid, teacher, v17, tokens in MODELS:
        if name in done:
            print(f"[skip] {name}")
            continue
        try:
            tokenize_if_missing(mid, tokens)
            rec = run_one(name, mid, teacher, v17, tokens, args.n, args.seq_len, args.device)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            rec = {"name": name, "model_id": mid, "error": repr(ex)}
        results.append(rec)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[lambada_all] wrote {args.out} ({len(results)})", flush=True)

    print("\n================ summary ================")
    print(f"{'model':18s} {'teacher_ppl':>12s} {'v17_ppl':>10s} {'ratio':>7s} "
          f"{'tchT1':>7s} {'v17T1':>7s} {'T1_ret':>8s}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:16s}  ERROR")
            continue
        print(f"{r['name']:18s} {r['teacher_ppl']:>12.4f} {r['v17_ppl']:>10.4f} "
              f"{r['ppl_ratio']:>7.3f} {r['teacher_t1']*100:>6.2f}% {r['v17_t1']*100:>6.2f}% "
              f"{r['t1_ret']*100:>7.2f}%")


if __name__ == "__main__":
    main()
