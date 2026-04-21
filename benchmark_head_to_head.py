from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any

import torch

from lambada_overlay import MODELS as OVERLAY_MODELS
from lambada_overlay import run_one as run_fp16_overlay
from lambada_overlay_fp8 import run_one as run_fp8_overlay
from lambada_overlay_mixed import run_one as run_mixed_overlay


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    params: dict[str, Any]


def _model_matches(name: str, only_tokens: set[str] | None) -> bool:
    if not only_tokens:
        return True
    lname = name.lower()
    return any(tok in lname for tok in only_tokens)


def _load_teacher_state(path: str) -> dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _load_tokens(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True).to(torch.long)


def _random_starts(toks: torch.Tensor, n: int, seq_len: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, toks.numel() - seq_len - 1, (n,), generator=g)


def _measure_teacher(model, toks: torch.Tensor, starts: torch.Tensor, seq_len: int, device: str):
    from eval_claim16_topk import measure_topk
    from eval_v16_ppl import measure_ppl

    topk, cache = measure_topk(model, toks, starts, seq_len, device, teacher_topk=None)
    ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    return topk, cache, ppl


def _measure_student(model, toks: torch.Tensor, starts: torch.Tensor, seq_len: int, device: str, teacher_cache):
    from eval_claim16_topk import measure_topk
    from eval_v16_ppl import measure_ppl

    topk, _ = measure_topk(model, toks, starts, seq_len, device, teacher_topk=teacher_cache)
    ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    return topk, ppl


def _run_bnb_baseline(
    name: str,
    model_id: str,
    tokens_path: str,
    n: int,
    seq_len: int,
    device: str,
    quant_bits: int,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.modeling_utils import no_init_weights

    if quant_bits not in (4, 8):
        raise ValueError(f"quant_bits must be 4 or 8, got {quant_bits}")

    toks = _load_tokens(tokens_path)
    starts = _random_starts(toks, n=n, seq_len=seq_len, seed=42)

    # Teacher model loaded from full precision config+weights, same protocol as our overlay scripts.
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with no_init_weights():
        teacher = AutoModelForCausalLM.from_config(
            cfg,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    t0 = time.time()
    tch_topk, tch_cache, tch_ppl = _measure_teacher(teacher, toks, starts, seq_len, device)
    teacher_wall = time.time() - t0

    del teacher
    torch.cuda.empty_cache()
    gc.collect()

    if quant_bits == 4:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        method = "bnb_nf4"
    else:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        method = "bnb_int8"

    dev_index = int(device.split(":", 1)[1]) if ":" in device else 0
    student = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map={"": dev_index},
    ).eval()

    t1 = time.time()
    q_topk, q_ppl = _measure_student(student, toks, starts, seq_len, device, teacher_cache=tch_cache)
    student_wall = time.time() - t1

    del student, toks
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "name": name,
        "model_id": model_id,
        "method": method,
        "n": n,
        "seq_len": seq_len,
        "teacher_ppl": float(tch_ppl),
        "teacher_t1": float(tch_topk["t1_gt"]),
        "teacher_t10": float(tch_topk["t10_gt"]),
        "student_ppl": float(q_ppl),
        "student_t1": float(q_topk["t1_gt"]),
        "student_t10": float(q_topk["t10_gt"]),
        "student_t1_vs_teacher": float(q_topk["t1_agree"]),
        "ppl_ratio": float(q_ppl / tch_ppl),
        "t1_ret": float(q_topk["t1_gt"] / tch_topk["t1_gt"] if tch_topk["t1_gt"] > 0 else 0.0),
        "t10_ret": float(q_topk["t10_gt"] / tch_topk["t10_gt"] if tch_topk["t10_gt"] > 0 else 0.0),
        "effective_bpw": float(4.0 if quant_bits == 4 else 8.0),
        "tier": "external-bnb",
        "teacher_wall_s": float(teacher_wall),
        "student_wall_s": float(student_wall),
    }


def _run_ours(
    method: MethodSpec,
    name: str,
    model_id: str,
    teacher_path: str,
    fit_path: str,
    tokens_path: str,
    n: int,
    seq_len: int,
    device: str,
) -> dict[str, Any]:
    if method.kind == "fp16_overlay":
        rec = run_fp16_overlay(
            name,
            model_id,
            teacher_path,
            fit_path,
            tokens_path,
            n,
            seq_len,
            device,
            method.params["rho"],
            score_mode=method.params.get("score_mode", "weighted"),
            tier="head2head+fp16-overlay",
        )
    elif method.kind == "fp8_overlay":
        rec = run_fp8_overlay(
            name,
            model_id,
            teacher_path,
            fit_path,
            tokens_path,
            n,
            seq_len,
            device,
            method.params["rho"],
            score_mode=method.params.get("score_mode", "weighted"),
            tier="head2head+fp8-overlay",
        )
    elif method.kind == "mixed_overlay":
        rec = run_mixed_overlay(
            name,
            model_id,
            teacher_path,
            fit_path,
            tokens_path,
            n,
            seq_len,
            device,
            method.params["rho_hi"],
            method.params["rho_lo"],
            score_mode=method.params.get("score_mode", "weighted"),
            tier="head2head+mixed-overlay",
        )
    else:
        raise ValueError(f"Unknown our method kind: {method.kind}")

    return {
        "name": rec["name"],
        "model_id": rec["model_id"],
        "method": method.name,
        "n": rec["n"],
        "seq_len": rec["seq_len"],
        "teacher_ppl": rec["teacher_ppl"],
        "teacher_t1": rec["teacher_t1"],
        "teacher_t10": rec["teacher_t10"],
        "student_ppl": rec["v17_ppl"],
        "student_t1": rec["v17_t1"],
        "student_t10": rec["v17_t10"],
        "student_t1_vs_teacher": rec["v17_t1_vs_teacher"],
        "ppl_ratio": rec["ppl_ratio"],
        "t1_ret": rec["t1_ret"],
        "t10_ret": rec["t10_ret"],
        "effective_bpw": rec.get("effective_bpw", 0.0),
        "tier": rec.get("tier", "head2head+overlay"),
        "overlay_bpw": rec.get("overlay_bpw", 0.0),
        "base_bpw": rec.get("base_bpw", 0.0),
        "meta": {
            k: v
            for k, v in method.params.items()
            if k in {"rho", "rho_hi", "rho_lo", "score_mode"}
        },
    }


def _run_hqq_baseline(
    name: str,
    model_id: str,
    tokens_path: str,
    n: int,
    seq_len: int,
    device: str,
    nbits: int,
    group_size: int = 64,
) -> dict[str, Any]:
    """HQQ (Half-Quadratic Quantization) baseline.

    Uses group-wise affine quantization with the HQQ algorithm. Pure PyTorch,
    so it runs on Windows without triton.

    Effective bpw accounting (meta not quantized): per group of ``group_size``
    weights we store ``group_size * nbits`` quantized bits plus two fp16 scalars
    (scale + zero) = 32 bits of metadata. So bpw = nbits + 32/group_size.
    """
    from hqq.core.quantize import BaseQuantizeConfig
    from hqq.models.hf.base import AutoHQQHFModel
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights

    toks = _load_tokens(tokens_path)
    starts = _random_starts(toks, n=n, seq_len=seq_len, seed=42)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with no_init_weights():
        _ = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    t0 = time.time()
    tch_topk, tch_cache, tch_ppl = _measure_teacher(teacher, toks, starts, seq_len, device)
    teacher_wall = time.time() - t0

    del teacher
    torch.cuda.empty_cache()
    gc.collect()

    student = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    quant_config = BaseQuantizeConfig(
        nbits=nbits,
        group_size=group_size,
        quant_zero=False,
        quant_scale=False,
        offload_meta=False,
        axis=1,
    )
    AutoHQQHFModel.quantize_model(
        student,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device=device,
    )
    student.eval()

    effective_bpw = float(nbits) + 32.0 / float(group_size)

    t1 = time.time()
    q_topk, q_ppl = _measure_student(student, toks, starts, seq_len, device, teacher_cache=tch_cache)
    student_wall = time.time() - t1

    del student, toks
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "name": name,
        "model_id": model_id,
        "method": f"hqq_{nbits}bit_g{group_size}",
        "n": n,
        "seq_len": seq_len,
        "teacher_ppl": float(tch_ppl),
        "teacher_t1": float(tch_topk["t1_gt"]),
        "teacher_t10": float(tch_topk["t10_gt"]),
        "student_ppl": float(q_ppl),
        "student_t1": float(q_topk["t1_gt"]),
        "student_t10": float(q_topk["t10_gt"]),
        "student_t1_vs_teacher": float(q_topk["t1_agree"]),
        "ppl_ratio": float(q_ppl / tch_ppl),
        "t1_ret": float(q_topk["t1_gt"] / tch_topk["t1_gt"] if tch_topk["t1_gt"] > 0 else 0.0),
        "t10_ret": float(q_topk["t10_gt"] / tch_topk["t10_gt"] if tch_topk["t10_gt"] > 0 else 0.0),
        "effective_bpw": effective_bpw,
        "tier": "external-hqq",
        "teacher_wall_s": float(teacher_wall),
        "student_wall_s": float(student_wall),
        "meta": {"nbits": nbits, "group_size": group_size},
    }


def _default_methods() -> list[MethodSpec]:
    return [
        MethodSpec("our_fp16_2p79", "fp16_overlay", {"rho": 0.002, "score_mode": "weighted"}),
        MethodSpec("our_fp16_2p83", "fp16_overlay", {"rho": 0.005, "score_mode": "weighted"}),
        MethodSpec("our_fp8_2p79", "fp8_overlay", {"rho": 0.005, "score_mode": "weighted"}),
        MethodSpec("our_fp8_2p83", "fp8_overlay", {"rho": 0.012, "score_mode": "weighted"}),
        MethodSpec(
            "our_mixed_2p79",
            "mixed_overlay",
            {"rho_hi": 0.001, "rho_lo": 0.003, "score_mode": "weighted"},
        ),
        MethodSpec(
            "our_mixed_2p83",
            "mixed_overlay",
            {"rho_hi": 0.002, "rho_lo": 0.008, "score_mode": "weighted"},
        ),
        MethodSpec("bnb_nf4", "bnb4", {}),
        MethodSpec("bnb_int8", "bnb8", {}),
        MethodSpec("hqq_4bit_g64", "hqq", {"nbits": 4, "group_size": 64}),
        MethodSpec("hqq_3bit_g64", "hqq", {"nbits": 3, "group_size": 64}),
        MethodSpec("hqq_2bit_g64", "hqq", {"nbits": 2, "group_size": 64}),
        MethodSpec("hqq_2bit_g16", "hqq", {"nbits": 2, "group_size": 16}),
    ]


def _print_summary(rows: list[dict[str, Any]]):
    if not rows:
        print("No rows yet.")
        return
    print("\n================ head-to-head summary ================")
    print(f"{'model':<18} {'method':<16} {'t1_ret':>8} {'ppl_r':>8} {'eff_bpw':>8}")
    for r in rows:
        print(
            f"{r['name']:<18} {r['method']:<16} "
            f"{r['t1_ret']*100:>7.2f}% {r['ppl_ratio']:>7.3f} {r.get('effective_bpw',0):>8.4f}"
        )


def main():
    ap = argparse.ArgumentParser(description="Head-to-head benchmark for our overlay methods vs external baselines.")
    ap.add_argument("--out", default="head_to_head_results.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--only", default="", help="comma-separated model tokens")
    ap.add_argument(
        "--methods",
        default="",
        help="comma-separated method names from default set; blank means all",
    )
    args = ap.parse_args()

    wanted_models = {s.strip().lower() for s in args.only.split(",") if s.strip()} or None
    default_methods = _default_methods()
    methods_by_name = {m.name: m for m in default_methods}

    if args.methods.strip():
        methods = []
        for name in [s.strip() for s in args.methods.split(",") if s.strip()]:
            if name not in methods_by_name:
                raise ValueError(f"Unknown method: {name}. Choices: {sorted(methods_by_name)}")
            methods.append(methods_by_name[name])
    else:
        methods = default_methods

    results: list[dict[str, Any]] = []
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    done = {
        (r.get("name"), r.get("method"), r.get("n"), r.get("seq_len"))
        for r in results
    }

    for name, model_id, teacher, fit, tokens in OVERLAY_MODELS:
        if not _model_matches(name, wanted_models):
            continue

        for method in methods:
            key = (name, method.name, args.n, args.seq_len)
            if key in done:
                print(f"[skip] {name} {method.name} already present")
                continue

            print(f"\n[{name}] method={method.name}")
            try:
                if method.kind == "bnb4":
                    rec = _run_bnb_baseline(name, model_id, tokens, args.n, args.seq_len, args.device, quant_bits=4)
                elif method.kind == "bnb8":
                    rec = _run_bnb_baseline(name, model_id, tokens, args.n, args.seq_len, args.device, quant_bits=8)
                elif method.kind == "hqq":
                    rec = _run_hqq_baseline(
                        name,
                        model_id,
                        tokens,
                        args.n,
                        args.seq_len,
                        args.device,
                        nbits=method.params["nbits"],
                        group_size=method.params.get("group_size", 64),
                    )
                else:
                    rec = _run_ours(method, name, model_id, teacher, fit, tokens, args.n, args.seq_len, args.device)

                results.append(rec)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"[head2head] wrote {args.out} ({len(results)} rows)")
            except Exception as exc:
                print(f"[FAIL] {name} {method.name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()

    _print_summary(results)


if __name__ == "__main__":
    main()
