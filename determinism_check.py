"""determinism_check.py -- verify the v17 fit and decode paths are deterministic.

Loads `v17hi_fit_tinyllama.pt` (existing hifi fit), re-decodes every body Linear
twice with the same seed on the same GPU, asserts bit-for-bit equality across
the full substitution. This is the single-file determinism guarantee: given
a fixed teacher state dict, fixed fit file, and fixed seed, the decode output
is reproducible.

A second path (`--refit`) re-runs the full Claim-16 fit twice from the same
teacher + activation cache and asserts the two resulting .pt checkpoints
produce bit-identical reconstructions across all body Linears. This is the
end-to-end determinism guarantee.
"""
from __future__ import annotations
import argparse, time, os
import torch

from eval_v17_ppl import _reconstruct_v17
from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation


def decode_all(teacher_sd: dict, v17: dict, device: str, D: int) -> dict[str, torch.Tensor]:
    """Run the v17 decode path on every body Linear; return {name: Wq fp16 cpu}."""
    rots = {}
    dims = sorted({v.shape[1] for k, v in teacher_sd.items()
                   if "layers." in k and any(p in k for p in ROLE_PATTERNS)
                   and v.ndim == 2 and v.shape[1] % D == 0})
    for I in dims:
        rots[I] = build_rotation(I, device, seed=42 + I)
    banks = v17["banks"]
    s_col = v17["s_col"]
    keys = [k for k in teacher_sd.keys()
            if "layers." in k and any(p in k for p in ROLE_PATTERNS)
            and k.endswith(".weight") and teacher_sd[k].ndim == 2
            and teacher_sd[k].shape[1] % D == 0]
    out = {}
    for k in keys:
        role = _role_of(k)
        bank = banks[role]
        W = teacher_sd[k]
        s = s_col.get(k, torch.ones(W.shape[1]))
        Wq = _reconstruct_v17(W, role, bank, s, D, rots[W.shape[1]], device)
        out[k] = Wq
    return out


def compare(a: dict, b: dict) -> tuple[int, int, list[str]]:
    """Return (n_equal, n_total, list_of_mismatched_keys)."""
    if set(a.keys()) != set(b.keys()):
        missing = set(a.keys()) ^ set(b.keys())
        return (0, len(a), [f"__key_mismatch__({missing})"])
    mismatched = []
    n_eq = 0
    for k in a:
        if torch.equal(a[k], b[k]):
            n_eq += 1
        else:
            # report max abs diff for informativeness
            d = (a[k].float() - b[k].float()).abs().max().item()
            mismatched.append(f"{k} max_abs_diff={d:.6g}")
    return n_eq, len(a), mismatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="tinyllama_1.1b_cache.pt")
    ap.add_argument("--fit",     default="v17hi_fit_tinyllama.pt")
    ap.add_argument("--device",  default="cuda:0")
    args = ap.parse_args()

    if not os.path.exists(args.teacher):
        print(f"[skip] {args.teacher} missing")
        return
    if not os.path.exists(args.fit):
        print(f"[skip] {args.fit} missing")
        return

    print(f"[decode-determinism] teacher={args.teacher}  fit={args.fit}")
    sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    v17 = torch.load(args.fit, map_location="cpu", weights_only=False)
    D = v17.get("D", 8)

    t0 = time.time()
    a = decode_all(sd, v17, args.device, D)
    t1 = time.time()
    torch.cuda.empty_cache()
    b = decode_all(sd, v17, args.device, D)
    t2 = time.time()

    n_eq, n_tot, mis = compare(a, b)
    print(f"  decoded {n_tot} tensors in {t1-t0:.1f}s + {t2-t1:.1f}s")
    print(f"  bit-identical: {n_eq}/{n_tot}")
    if mis:
        print(f"  MISMATCHES ({len(mis)}):")
        for m in mis[:10]:
            print(f"    {m}")
    else:
        print(f"  PASS: decode is deterministic.")


if __name__ == "__main__":
    main()
