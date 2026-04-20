"""
eval_v16_ppl.py — End-to-end PPL validation of compressed Qwen3-1.7B body.

This is the first evaluator that measures downstream quality (not rel-W) of
the v10-v16 compression stack: load the full Qwen3-1.7B HF model, substitute
each body Linear's weight with its stack-reconstructed counterpart, and
measure perplexity on WikiText-103 test.

Three configurations in one run so differences are directly comparable:
  baseline  : raw fp16 teacher weights, no substitution.
  v10_greedy: one global codebook (K1=2048, K2=256, D=8), greedy residual
              PQ, no rotation, no weighting — the first published data
              point in this series.
  v16       : full stack (rotation + role banks + weighted EM + beam-8 +
              asymmetric o_proj K) — ALREADY FIT in v16_result.pt; this
              re-derives (idx1, idx2) from the stored codebooks at eval
              time and reconstructs W_q.

Protocol (matches wikitext_eval.py for comparability):
  seed 42, seq_len 128, n=1000 windows sampled uniformly from the full
  WikiText-103 test stream (~245K tokens).

Output:
  baseline PPL, v10_greedy PPL, v16 PPL, and the *PPL ratio* v16/baseline
  and v10/baseline. Honest accuracy retention on a standard public set.

Usage:
  python eval_v16_ppl.py --v16 v16_result.pt --n 1000 --device cuda:0
"""
from __future__ import annotations
import argparse
import gc
import math
import time

import torch
import torch.nn.functional as F

from universal_v9 import kmeans_init
from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from compress_v15 import beam_assign


# ---------------------------------------------------------------------------
# helpers (shared by v10 and v16 reconstruction paths)
# ---------------------------------------------------------------------------
def _chunked_argmin(X: torch.Tensor, C: torch.Tensor, bs: int = 300_000) -> torch.Tensor:
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    C_nrm = (C * C).sum(-1)
    for s in range(0, X.shape[0], bs):
        e = min(s + bs, X.shape[0])
        d = C_nrm.unsqueeze(0) - 2.0 * (X[s:e] @ C.T) + (X[s:e] * X[s:e]).sum(-1, keepdim=True)
        out[s:e] = d.argmin(-1)
        del d
    return out


def _reconstruct_v16(W_fp16: torch.Tensor, role: str, bank: dict, D: int,
                     rot: torch.Tensor, device: str) -> torch.Tensor:
    """Decode W under the v16 stack: rotate, group, beam-assign to (cb1,cb2),
    scale by per-row rs, unrotate. Returns fp16 weight on CPU."""
    W = W_fp16.to(device=device, dtype=torch.float32)
    Wrot = W @ rot                               # [O, I]
    O, I = Wrot.shape
    rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
    g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
    cb1 = bank["cb1"].to(device); cb2 = bank["cb2"].to(device)
    idx1, idx2, _ = beam_assign(g, cb1, cb2, beam=8)
    gh = cb1[idx1] + cb2[idx2]
    Wq_rot = (gh.view(O, I // D, D).reshape(O, I)) \
             * rs.expand(O, I // D * D)
    Wq = Wq_rot @ rot.T
    del W, Wrot, g, cb1, cb2, idx1, idx2, gh, Wq_rot
    return Wq.to(dtype=torch.float16, device="cpu")


def _reconstruct_v10_greedy(W_fp16: torch.Tensor, cb1: torch.Tensor, cb2: torch.Tensor,
                            D: int, device: str) -> torch.Tensor:
    """Decode under v10 baseline: NO rotation, NO row-scale, global codebooks,
    GREEDY residual PQ. This is what the first universal-PQ paper would do."""
    W = W_fp16.to(device=device, dtype=torch.float32)
    O, I = W.shape
    g = W.view(O, I // D, D).reshape(-1, D)
    idx1 = _chunked_argmin(g, cb1)
    R1 = g - cb1[idx1]
    idx2 = _chunked_argmin(R1, cb2)
    gh = cb1[idx1] + cb2[idx2]
    Wq = gh.view(O, I // D, D).reshape(O, I)
    del W, g, R1, idx1, idx2, gh
    return Wq.to(dtype=torch.float16, device="cpu")


def _fit_v10_global_codebook(state_dict: dict, D: int, K1: int, K2: int,
                             device: str, pool_sz: int = 200_000,
                             kmeans_iters: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ONE global (cb1, cb2) pair across ALL body linears. The v10 baseline."""
    pool_list = []
    for k, v in state_dict.items():
        if not ("layers." in k and any(p in k for p in ROLE_PATTERNS)):
            continue
        if v.ndim != 2 or v.shape[1] % D != 0:
            continue
        W = v.to(device=device, dtype=torch.float32)
        O, I = W.shape
        g = W.view(O, I // D, D).reshape(-1, D)
        # take a stratified sample from each tensor
        take = min(20000, g.shape[0])
        idx = torch.randperm(g.shape[0], device=device)[:take]
        pool_list.append(g[idx].clone())
        del W, g
    pool = torch.cat(pool_list, 0)
    del pool_list
    # subsample to pool_sz for kmeans init
    if pool.shape[0] > pool_sz:
        pool = pool[torch.randperm(pool.shape[0], device=device)[:pool_sz]]
    cb1 = kmeans_init(pool, K1, iters=kmeans_iters)
    idx1 = _chunked_argmin(pool, cb1)
    R1 = pool - cb1[idx1]
    cb2 = kmeans_init(R1, K2, iters=kmeans_iters)
    del pool, R1
    torch.cuda.empty_cache()
    return cb1, cb2


# ---------------------------------------------------------------------------
# PPL measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_ppl(model, tokens: torch.Tensor, starts: torch.Tensor,
                seq_len: int, device: str) -> tuple[float, float]:
    """Teacher-forced NLL over `len(starts)` random windows. Returns (ppl, nll)."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for i, s in enumerate(starts.tolist()):
        win = tokens[s:s + seq_len + 1].to(device=device, dtype=torch.long)
        inp = win[:-1].unsqueeze(0)
        tgt = win[1:].unsqueeze(0)
        out = model(inp).logits   # [1, L, V]
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                               tgt.reshape(-1), reduction="sum")
        total_nll += loss.item()
        total_tokens += tgt.numel()
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(starts)}] running ppl "
                  f"{math.exp(total_nll / total_tokens):.4f}")
    nll = total_nll / total_tokens
    return math.exp(nll), nll


def substitute_v16(model, state_dict: dict, v16: dict, device: str, D: int):
    """For each body Linear, compute v16-reconstructed weight and copy into model."""
    rots = {}
    dims = sorted({v.shape[1] for k, v in state_dict.items()
                   if "layers." in k and any(p in k for p in ROLE_PATTERNS)
                   and v.ndim == 2 and v.shape[1] % D == 0})
    for I in dims:
        rots[I] = build_rotation(I, device, seed=42 + I)
    banks = v16["banks"]
    # Map HF key → nn.Linear module in model
    hf_keys = [k for k in state_dict.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and state_dict[k].ndim == 2
               and state_dict[k].shape[1] % D == 0]
    print(f"  substituting {len(hf_keys)} body Linears (v16 stack)")
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        W_new = _reconstruct_v16(state_dict[k], role, bank, D, rots[state_dict[k].shape[1]], device)
        # find the module
        mod = model
        parts = k.replace(".weight", "").split(".")
        for p in parts:
            mod = getattr(mod, p)
        mod.weight.data.copy_(W_new.to(mod.weight.device, dtype=mod.weight.dtype))
        del W_new
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    torch.cuda.empty_cache(); gc.collect()


def substitute_v10(model, state_dict: dict, cb1: torch.Tensor, cb2: torch.Tensor,
                   device: str, D: int):
    hf_keys = [k for k in state_dict.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and state_dict[k].ndim == 2
               and state_dict[k].shape[1] % D == 0]
    print(f"  substituting {len(hf_keys)} body Linears (v10 greedy)")
    for n, k in enumerate(hf_keys):
        W_new = _reconstruct_v10_greedy(state_dict[k], cb1, cb2, D, device)
        mod = model
        for p in k.replace(".weight", "").split("."):
            mod = getattr(mod, p)
        mod.weight.data.copy_(W_new.to(mod.weight.device, dtype=mod.weight.dtype))
        del W_new
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    torch.cuda.empty_cache(); gc.collect()


def reset_teacher(model, teacher_sd: dict):
    """Copy raw fp16 teacher weights back into every body Linear."""
    for k, v in teacher_sd.items():
        if not ("layers." in k and any(p in k for p in ROLE_PATTERNS)):
            continue
        if not k.endswith(".weight") or v.ndim != 2:
            continue
        mod = model
        for p in k.replace(".weight", "").split("."):
            mod = getattr(mod, p)
        mod.weight.data.copy_(v.to(mod.weight.device, dtype=mod.weight.dtype))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v16", default="v16_result.pt")
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--K1", type=int, default=2048)
    ap.add_argument("--K2", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_v10", action="store_true",
                    help="Skip v10 baseline — only measure fp16 vs v16.")
    ap.add_argument("--out", default="v16_ppl_results.pt")
    args = ap.parse_args()
    device = args.device

    print(f"[eval] loading teacher state_dict from {args.teacher}")
    teacher_sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in teacher_sd:
        teacher_sd = teacher_sd["state_dict"]
    print(f"[eval] teacher has {len(teacher_sd)} tensors")

    print(f"[eval] loading tokens from {args.tokens}")
    all_tokens = torch.load(args.tokens, weights_only=True).to(torch.long)
    print(f"[eval] {all_tokens.numel()/1e3:.1f}K WikiText-103 test tokens")
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, all_tokens.numel() - args.seq_len - 1,
                           (args.n,), generator=g)

    print(f"[eval] building Qwen3-1.7B HF model on {device}")
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    missing, unexpected = model.load_state_dict(teacher_sd, strict=False)
    print(f"[eval]   missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[eval]   missing sample: {missing[:3]}")
    model = model.to(device).eval()

    results: dict[str, dict] = {}

    # --- baseline ---
    print("\n[eval] === BASELINE (fp16 teacher) ===")
    t = time.time()
    ppl_base, nll_base = measure_ppl(model, all_tokens, starts, args.seq_len, device)
    results["baseline"] = {"ppl": ppl_base, "nll": nll_base, "wall_s": time.time() - t}
    print(f"[eval] baseline PPL = {ppl_base:.4f}  ({time.time()-t:.0f}s)")

    # --- v10 greedy ---
    if not args.skip_v10:
        print("\n[eval] === v10_greedy (K1=2048 K2=256 D=8, no rotation, no weighting) ===")
        t = time.time()
        cb1_g, cb2_g = _fit_v10_global_codebook(teacher_sd, args.D, args.K1, args.K2, device)
        print(f"[eval]   fit global codebook ({time.time()-t:.0f}s)")
        substitute_v10(model, teacher_sd, cb1_g, cb2_g, device, args.D)
        ppl_v10, nll_v10 = measure_ppl(model, all_tokens, starts, args.seq_len, device)
        results["v10_greedy"] = {"ppl": ppl_v10, "nll": nll_v10,
                                 "wall_s": time.time() - t}
        print(f"[eval] v10_greedy PPL = {ppl_v10:.4f}  "
              f"(ratio {ppl_v10/ppl_base:.3f}×)  ({time.time()-t:.0f}s)")
        # restore
        reset_teacher(model, teacher_sd)
        del cb1_g, cb2_g
        torch.cuda.empty_cache(); gc.collect()

    # --- v16 stacked ---
    print(f"\n[eval] === v16 stacked (loading banks from {args.v16}) ===")
    t = time.time()
    v16 = torch.load(args.v16, map_location="cpu", weights_only=False)
    print(f"[eval]   v16 has {len(v16['banks'])} role banks")
    substitute_v16(model, teacher_sd, v16, device, args.D)
    ppl_v16, nll_v16 = measure_ppl(model, all_tokens, starts, args.seq_len, device)
    results["v16"] = {"ppl": ppl_v16, "nll": nll_v16, "wall_s": time.time() - t,
                      "global_bpw": v16.get("global_bpw", None)}
    print(f"[eval] v16 PPL = {ppl_v16:.4f}  "
          f"(ratio {ppl_v16/ppl_base:.3f}×)  ({time.time()-t:.0f}s)")

    # --- summary ---
    print("\n" + "=" * 60)
    print("SUMMARY  (WikiText-103 test, n=%d, seq_len=%d)" % (args.n, args.seq_len))
    print("=" * 60)
    print(f"  baseline fp16    : PPL {results['baseline']['ppl']:.4f}  (ratio 1.000×)")
    if "v10_greedy" in results:
        r = results["v10_greedy"]["ppl"] / results["baseline"]["ppl"]
        print(f"  v10_greedy 2.38  : PPL {results['v10_greedy']['ppl']:.4f}  "
              f"(ratio {r:.3f}×, +{(r-1)*100:.1f}%)")
    r16 = results["v16"]["ppl"] / results["baseline"]["ppl"]
    bpw = results["v16"].get("global_bpw") or 2.396
    print(f"  v16       {bpw:.3f}  : PPL {results['v16']['ppl']:.4f}  "
          f"(ratio {r16:.3f}×, +{(r16-1)*100:.1f}%)")

    torch.save(results, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
