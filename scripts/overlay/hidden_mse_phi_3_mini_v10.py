"""Phi-3-mini-4k-instruct v10 cure runner — per-layer hidden-state distillation.

Cross-architecture generalization test of the v10 recipe validated on
Mistral-7B-v0.3 (PPL ratio 1.0055x).  Phi-3-mini is 3.8B, 32 layers,
uses Phi3DecoderLayer + Phi3RotaryEmbedding.  Architecture differs from
Mistral (SuRoPE + block-sparse attention vs standard GQA).

Hypothesis: the per-layer hidden-state objective generalizes across
architectures. Phi-3-mini already has a published 1.00262x record at
seq_len=128 (v0.5.x); this run evals at seq_len=1024 for apples-to-apples
comparability with the rest of the scaling matrix.

Pipeline per layer i:
  1. Cache teacher hidden INPUT to layer i  (= output of layer i-1)
  2. Cache teacher hidden OUTPUT of layer i (= target for student)
  3. Quantize layer i (GSQ K=block_size, --bpw bits)
  4. Wrap each TARGET_SUBS Linear with CorrectionMatrixC (rank=--rank, SVD warm-start)
  5. Train V18-C: min || student_layer_i(teacher_input_i) - teacher_output_i ||^2
  6. Probe KL_init/KL_final on held-out validation batch
  7. Save per-layer .pt + freeze + advance

After all layers compressed, run end-to-end PPL on n_eval prompts.

FIRE COMMAND:
    python scripts/overlay/hidden_mse_phi_3_mini_v10.py --device cuda:0 --seed 42 \
        --bpw 5 --rank 48 --train_steps 300 \
        --out_json docs/HIDDEN_MSE_PHI_3_MINI_4K_v10_RESULTS.json

INTERNAL research code.
"""
from __future__ import annotations

import sys
import io

# CRITICAL: Force UTF-8 stdio on Windows so emoji/Rich output from HF/tqdm
# doesn't crash on cp1252.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

import argparse
import datetime
import json
import math
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # .../ultracompress
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask

# Reuse the production primitives unchanged so this runner stays a thin
# objective-swap on the v6b base, not a fork.
from streaming_compression_runner import (
    CorrectionMatrixC,
    gsq_quantize_weight,
    get_model_classes,
    free_memory,
    vram_gb,
    vram_report,
)

# ---------------------------------------------------------------------------
# Phi-3-mini-4k-instruct specific config
# ---------------------------------------------------------------------------
HF_ID = "microsoft/Phi-3-mini-4k-instruct"
N_LAYERS = 32
DTYPE = torch.bfloat16

# Same TARGET_SUBS subset as the Mistral v10 runner for apples-to-apples.
# Phi-3-mini uses Llama-style attention + MLP with gate; no MoE, no MLA.
TARGET_SUBS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

DATA_CANDIDATES = [
    _ROOT / "fineweb_edu_10M_tokens_phi_3_mini_4k_instruct.pt",
    _ROOT / "fineweb_edu_100M_tokens_phi_3_mini_4k_instruct.pt",
    _ROOT / "fineweb_edu_500M_tokens_phi_3_mini_4k_instruct.pt",
]


# ---------------------------------------------------------------------------
# Cache teacher hiddens (input + output) for layer i
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_teacher_hiddens_for_layer(
    model: nn.Module,
    layer_idx: int,
    calibration_ids: list[torch.Tensor],
    config,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run teacher forward on calib set, capture (input, output) at layer_idx.

    Returns:
        teacher_in:  [n_prompts, seq_len, hidden_dim] on CPU (bf16)
        teacher_out: [n_prompts, seq_len, hidden_dim] on CPU (bf16)
    """
    model.train(False)
    embed_dev = model.model.embed_tokens.weight.device

    _, RotaryEmbClass = get_model_classes(HF_ID)
    rotary_emb = RotaryEmbClass(config=config, device=device)

    in_list: list[torch.Tensor] = []
    out_list: list[torch.Tensor] = []

    for prompt_ids in calibration_ids:
        ids = prompt_ids.unsqueeze(0).long().to(embed_dev)
        bs, seq_len = ids.shape
        hidden = model.model.embed_tokens(ids).to(dtype)

        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(bs, -1)
        causal_mask = create_causal_mask(
            config=config,
            input_embeds=hidden,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = rotary_emb(hidden, position_ids)

        # Run prefix layers 0..layer_idx-1 (teacher state, frozen)
        for li in range(layer_idx):
            lo = model.model.layers[li](
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = lo[0] if isinstance(lo, tuple) else lo

        # teacher_in for layer i = hidden after running layers 0..i-1
        in_list.append(hidden.cpu())

        # teacher_out for layer i = run layer i on teacher_in (still uncompressed)
        lo = model.model.layers[layer_idx](
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        h_out = lo[0] if isinstance(lo, tuple) else lo
        out_list.append(h_out.cpu())

    teacher_in = torch.cat(in_list, dim=0)
    teacher_out = torch.cat(out_list, dim=0)
    del in_list, out_list, rotary_emb
    free_memory()
    return teacher_in, teacher_out


# ---------------------------------------------------------------------------
# Cache teacher logits (one-time, for KL_init/KL_final reporting)
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_teacher_logits(
    model: nn.Module,
    calibration_ids: list[torch.Tensor],
    device: torch.device,
    n_kl_probe: int = 8,
) -> torch.Tensor:
    """Cache teacher log-probs on a small probe set for per-layer KL reporting."""
    model.train(False)
    embed_dev = model.model.embed_tokens.weight.device

    probe_ids = calibration_ids[:n_kl_probe]
    out_list = []
    for ids in probe_ids:
        x = ids.unsqueeze(0).long().to(embed_dev)
        out = model(input_ids=x, use_cache=False, return_dict=True)
        lp = F.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        out_list.append(lp.half().cpu())
        del out, lp
    free_memory()
    return torch.cat(out_list, dim=0)


# ---------------------------------------------------------------------------
# Probe: KL between current (compressed-so-far) model and teacher logprobs
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_kl(
    model: nn.Module,
    calibration_ids: list[torch.Tensor],
    teacher_logprobs_probe: torch.Tensor,
    device: torch.device,
    n_kl_probe: int,
) -> float:
    """Run model forward on probe prompts, return mean KL vs teacher_logprobs_probe."""
    model.train(False)
    embed_dev = model.model.embed_tokens.weight.device

    kls: list[float] = []
    for pi in range(n_kl_probe):
        ids = calibration_ids[pi].unsqueeze(0).long().to(embed_dev)
        out = model(input_ids=ids, use_cache=False, return_dict=True)
        student_lp = F.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        target_lp = teacher_logprobs_probe[pi:pi + 1].to(device).float()
        kl = F.kl_div(
            student_lp.view(-1, student_lp.size(-1)),
            target_lp.view(-1, target_lp.size(-1)),
            log_target=True,
            reduction="batchmean",
        ).item()
        kls.append(kl)
        del out, student_lp, target_lp
    free_memory()
    return sum(kls) / max(1, len(kls))


# ---------------------------------------------------------------------------
# Compress one layer with hidden-state distillation objective
# ---------------------------------------------------------------------------
def compress_single_layer_hidden_mse(
    model: nn.Module,
    config,
    layer_idx: int,
    calibration_ids: list[torch.Tensor],
    teacher_logprobs_probe: torch.Tensor,
    output_dir: Path,
    bpw: int,
    block_size: int,
    rank: int,
    train_steps: int,
    train_lr: float,
    train_bs: int,
    n_kl_probe: int,
    device: torch.device,
) -> dict[str, Any]:
    """Compress layer_idx with per-layer hidden-state distillation objective.

    Pipeline:
      1. Cache (teacher_in, teacher_out) for layer i
      2. Quantize layer i + wrap with V18-C (SVD warm-start)
      3. Train: run layer i (compressed) on teacher_in, MSE vs teacher_out
      4. Probe KL on validation batch BEFORE training (init) and AFTER (final)
      5. Save artifact, freeze
    """
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = next(model.parameters()).dtype

    # ---- Step 1: KL_init probe (BEFORE quantizing this layer) ----
    t_kl0 = time.time()
    kl_init_pre_layer = probe_kl(
        model, calibration_ids, teacher_logprobs_probe, device, n_kl_probe
    )
    t_kl0_elapsed = time.time() - t_kl0

    # ---- Step 2: Cache teacher hiddens for layer i ----
    t_cache = time.time()
    teacher_in, teacher_out = cache_teacher_hiddens_for_layer(
        model, layer_idx, calibration_ids, config, device, dtype
    )
    cache_time = time.time() - t_cache
    in_mb = teacher_in.nbytes / 1e6
    out_mb = teacher_out.nbytes / 1e6
    print(f"    teacher hiddens: in={teacher_in.shape} ({in_mb:.0f}MB) "
          f"out={teacher_out.shape} ({out_mb:.0f}MB)  cache={cache_time:.1f}s")

    # ---- Step 3: Quantize layer i ----
    layer = model.model.layers[layer_idx]

    original_weights: dict[str, torch.Tensor] = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBS):
            original_weights[name] = mod.weight.data.clone()

    n_quantized = 0
    quant_errors: dict[str, float] = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBS):
            with torch.no_grad():
                W = mod.weight.data.float()
                Wq = gsq_quantize_weight(W, bpw, block_size)
                rel_l2 = (W - Wq).norm() / W.norm()
                quant_errors[name] = rel_l2.item()
                mod.weight.data.copy_(Wq.to(dtype))
                del W, Wq
            n_quantized += 1
    free_memory()

    # ---- Step 4: Wrap Linears with V18-C (SVD warm-start) ----
    correction_modules: dict[str, CorrectionMatrixC] = {}

    def wrap_linears_with_correction(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(s in name for s in TARGET_SUBS):
                cm = CorrectionMatrixC(
                    child.weight.data,
                    child.bias.data if child.bias is not None else None,
                    rank=rank,
                )
                if full_name in original_weights:
                    cm.init_from_svd(original_weights[full_name].to(device))
                setattr(module, name, cm)
                correction_modules[full_name] = cm
            else:
                wrap_linears_with_correction(child, full_name)

    wrap_linears_with_correction(layer)
    del original_weights
    free_memory()

    # Freeze everything; unfreeze ONLY this layer's V/U/alpha
    for p in model.parameters():
        p.requires_grad = False
    for cm in correction_modules.values():
        for p in cm.V.parameters():
            p.requires_grad = True
        for p in cm.U.parameters():
            p.requires_grad = True
        cm.alpha.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable_params)

    opt = torch.optim.AdamW(trainable_params, lr=train_lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_steps)

    # ---- Step 5: Pre-build position state (constant across steps) ----
    seq_len = calibration_ids[0].shape[0]
    _, RotaryEmbClass = get_model_classes(HF_ID)
    rotary_emb = RotaryEmbClass(config=config, device=device)

    n_prompts = len(calibration_ids)
    losses: list[float] = []

    @torch.no_grad()
    def _fixed_batch_mse() -> float:
        idx = torch.arange(min(train_bs, n_prompts))
        h_in = teacher_in[idx].to(device=device, dtype=dtype)
        h_target = teacher_out[idx].to(device=device, dtype=dtype)
        bs = h_in.shape[0]
        cp = torch.arange(seq_len, device=device)
        pi = cp.unsqueeze(0).expand(bs, -1)
        cm = create_causal_mask(
            config=config, input_embeds=h_in, attention_mask=None,
            cache_position=cp, past_key_values=None, position_ids=pi,
        )
        pe = rotary_emb(h_in, pi)
        lo = model.model.layers[layer_idx](
            h_in, attention_mask=cm, position_ids=pi,
            past_key_values=None, use_cache=False,
            cache_position=cp, position_embeddings=pe,
        )
        h_pred = lo[0] if isinstance(lo, tuple) else lo
        return F.mse_loss(h_pred.float(), h_target.float()).item()

    mse_init = _fixed_batch_mse()

    # ---- Step 6: Training loop ----
    for step in range(train_steps):
        batch_idx = torch.randint(0, n_prompts, (train_bs,))
        h_in = teacher_in[batch_idx].to(device=device, dtype=dtype)
        h_target = teacher_out[batch_idx].to(device=device, dtype=dtype)
        h_in.requires_grad_(False)

        bs = h_in.shape[0]
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(bs, -1)
        causal_mask = create_causal_mask(
            config=config, input_embeds=h_in, attention_mask=None,
            cache_position=cache_position, past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = rotary_emb(h_in, position_ids)

        layer_out = model.model.layers[layer_idx](
            h_in,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        h_pred = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # MSE in fp32 for numerical stability
        loss = F.mse_loss(h_pred.float(), h_target.float())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        sched.step()

        losses.append(loss.item())

        del h_in, h_target, h_pred, layer_out
        if (step + 1) % 25 == 0:
            avg = sum(losses[-25:]) / len(losses[-25:])
            print(f"      step {step + 1}/{train_steps}  MSE={avg:.6e}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  {vram_report()}")
            free_memory()

    mse_final = _fixed_batch_mse()
    del rotary_emb, teacher_in, teacher_out
    free_memory()

    # ---- Step 7: KL_final probe (AFTER training this layer) ----
    t_kl1 = time.time()
    kl_final_post_layer = probe_kl(
        model, calibration_ids, teacher_logprobs_probe, device, n_kl_probe
    )
    t_kl1_elapsed = time.time() - t_kl1

    # ---- Step 8: Save compressed layer artifact ----
    save_dict: dict[str, Any] = {
        "layer_idx": layer_idx,
        "bpw": bpw,
        "block_size": block_size,
        "rank": rank,
        "objective": "hidden_state_distill",
        "state_dict": {k: v.cpu() for k, v in layer.state_dict().items()},
        "corrections": {},
    }
    for name, cm in correction_modules.items():
        save_dict["corrections"][name] = {
            "V_weight": cm.V.weight.data.cpu(),
            "U_weight": cm.U.weight.data.cpu(),
            "alpha": cm.alpha.data.cpu().item(),
        }
    output_path = output_dir / f"layer_{layer_idx:03d}.pt"
    torch.save(save_dict, output_path)

    # Freeze correction params before moving on to next layer
    for cm in correction_modules.values():
        cm.V.weight.requires_grad = False
        cm.U.weight.requires_grad = False
        cm.alpha.requires_grad = False

    mean_quant_error = sum(quant_errors.values()) / max(1, len(quant_errors))
    alphas = {n: cm.alpha.data.item() for n, cm in correction_modules.items()}
    mean_alpha = sum(alphas.values()) / max(1, len(alphas))
    peak_vram = vram_gb(device, peak=True)
    elapsed = time.time() - t0

    metrics: dict[str, Any] = {
        "layer_idx": layer_idx,
        "objective": "hidden_state_distill",
        "MSE_init": mse_init,
        "MSE_final": mse_final,
        "MSE_reduction": mse_init - mse_final,
        "KL_init": kl_init_pre_layer,
        "KL_final": kl_final_post_layer,
        "KL_delta": kl_final_post_layer - kl_init_pre_layer,
        "train_loss_init": losses[0] if losses else float("nan"),
        "train_loss_final": (sum(losses[-20:]) / max(1, len(losses[-20:]))),
        "mean_quant_rel_l2": mean_quant_error,
        "n_quantized_linears": n_quantized,
        "n_train_params": n_train_params,
        "mean_alpha": mean_alpha,
        "peak_vram_gb": peak_vram,
        "compress_time_s": elapsed,
        "teacher_cache_time_s": cache_time,
        "kl_probe_time_s": t_kl0_elapsed + t_kl1_elapsed,
        "per_linear_quant_errors": quant_errors,
    }

    torch.cuda.reset_peak_memory_stats(device)
    free_memory()
    return metrics


# ---------------------------------------------------------------------------
# PPL measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_ppl(
    model: nn.Module,
    prompts: list[torch.Tensor],
    device: torch.device,
    label: str,
) -> float:
    model.train(False)
    embed_dev = model.model.embed_tokens.weight.device
    nlls: list[float] = []
    for pi, prompt_ids in enumerate(prompts):
        ids = prompt_ids.unsqueeze(0).long().to(embed_dev)
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = ids[:, 1:].contiguous().to(shift_logits.device)
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        ).item()
        nlls.append(nll)
        if (pi + 1) % 10 == 0:
            running = math.exp(sum(nlls) / len(nlls))
            print(f"    {label} {pi + 1}/{len(prompts)}  "
                  f"running_PPL={running:.4f}  {vram_report()}")
    return math.exp(sum(nlls) / len(nlls))


# ---------------------------------------------------------------------------
# Save partial results (crash resilience)
# ---------------------------------------------------------------------------
def save_partial(
    path: Path,
    per_layer_metrics: list[dict],
    args_dict: dict,
    label: str = "partial",
) -> None:
    clean = []
    for m in per_layer_metrics:
        cm = {
            k: v for k, v in m.items()
            if isinstance(v, (int, float, str, bool, list, dict, type(None)))
        }
        clean.append(cm)
    payload = {
        "experiment": "hidden_state_distill_phi_3_mini_4k_v10",
        "status": label,
        "model": "phi-3-mini-4k-instruct",
        "hf_id": HF_ID,
        "n_layers_completed": len(clean),
        "config": args_dict,
        "per_layer": clean,
        "per_layer_KL_init": [m.get("KL_init") for m in clean],
        "per_layer_KL_final": [m.get("KL_final") for m in clean],
        "per_layer_MSE_init": [m.get("MSE_init") for m in clean],
        "per_layer_MSE_final": [m.get("MSE_final") for m in clean],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"  [CHECKPOINT] {label}: {len(clean)} layers -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phi-3-mini-4k-instruct v10 hidden-state distillation runner.")
    ap.add_argument("--bpw", type=int, default=5,
                    help="GSQ bits per weight (default 5)")
    ap.add_argument("--block_size", type=int, default=64,
                    help="GSQ block size K (default 64)")
    ap.add_argument("--rank", type=int, default=48,
                    help="V18-C correction rank (default 48)")
    ap.add_argument("--train_steps", type=int, default=300,
                    help="Training steps per layer (default 300)")
    ap.add_argument("--train_lr", type=float, default=5e-4,
                    help="Learning rate for V18-C (default 5e-4)")
    ap.add_argument("--train_bs", type=int, default=4,
                    help="Training batch size (default 4)")
    ap.add_argument("--n_calib", type=int, default=128,
                    help="Number of calibration prompts (default 128)")
    ap.add_argument("--n_eval", type=int, default=50,
                    help="Number of prompts for final PPL (default 50)")
    ap.add_argument("--seq_len", type=int, default=512,
                    help="Calibration sequence length (default 512)")
    ap.add_argument("--eval_seq_len", type=int, default=1024,
                    help="Eval sequence length for PPL (default 1024)")
    ap.add_argument("--n_kl_probe", type=int, default=8,
                    help="Number of probe prompts for per-layer KL reporting "
                         "(default 8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0",
                    help="Default cuda:0.")
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    docs_dir = _ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    output_dir = _HERE / "_e2e_phi_3_mini_4k_v10_hidden_mse"
    partial_json = docs_dir / "HIDDEN_MSE_PHI_3_MINI_4K_v10_PARTIAL.json"

    if args.out_json is None:
        args.out_json = docs_dir / "HIDDEN_MSE_PHI_3_MINI_4K_v10_RESULTS.json"

    print("=" * 78)
    print("HIDDEN-STATE DISTILL PHI-3-MINI-4K-INSTRUCT v10 RUNNER")
    print("=" * 78)
    print(f"  hf_id={HF_ID}  n_layers={N_LAYERS}")
    print(f"  bpw={args.bpw}  block={args.block_size}  rank={args.rank}")
    print(f"  train_steps={args.train_steps}  lr={args.train_lr}  bs={args.train_bs}")
    print(f"  n_calib={args.n_calib}  n_eval={args.n_eval}  n_kl_probe={args.n_kl_probe}")
    print(f"  calib_seq_len={args.seq_len}  eval_seq_len={args.eval_seq_len}")
    print(f"  device={device}  seed={args.seed}")
    print(f"  output_dir={output_dir}")
    print(f"  out_json={args.out_json}")
    print(f"  Objective: per-layer hidden-state distillation (LOCAL)")
    print(f"  Time: {datetime.datetime.now().isoformat()}")
    print("=" * 78)

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA devices visible")

    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)

    # ---- [1/5] Load FineWeb-edu tokens (Phi-3-mini vocab) ----
    print(f"\n[1/5] Loading FineWeb-edu tokens ({HF_ID} vocab)...")
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        sys.exit(
            "ERROR: no Phi-3-mini FineWeb-edu token cache found at:\n  "
            + "\n  ".join(str(p) for p in DATA_CANDIDATES)
            + "\n\nRun: python scripts/data/tokenize_fineweb_for_model.py "
            f"--model {HF_ID} --n_tokens 10000000 "
            f"--output {DATA_CANDIDATES[0]}"
        )
    all_tokens = torch.load(data_path, weights_only=True)
    total = all_tokens.numel()
    print(f"  Loaded {data_path.name} ({total / 1e6:.0f}M tokens)")

    g = torch.Generator().manual_seed(args.seed)
    tail_size = min(50_000_000, total // 5)
    tail_start = max(args.seq_len + 1, total - tail_size)

    calib_high = max(args.seq_len + 1, tail_start - args.seq_len - 1)
    calib_starts = torch.randint(0, calib_high, (args.n_calib,), generator=g)
    calibration_ids: list[torch.Tensor] = [
        all_tokens[int(s):int(s) + args.seq_len].long()
        for s in calib_starts.tolist()
    ]

    measure_starts = torch.randint(
        tail_start, total - args.eval_seq_len - 1, (args.n_eval,), generator=g
    )
    measure_prompts: list[torch.Tensor] = [
        all_tokens[int(s):int(s) + args.eval_seq_len].long()
        for s in measure_starts.tolist()
    ]
    del all_tokens

    calib_tokens = args.n_calib * args.seq_len
    print(f"  {args.n_calib} calib prompts ({calib_tokens / 1e3:.0f}K tokens, "
          f"seq={args.seq_len})")
    print(f"  {args.n_eval} measure prompts (seq={args.eval_seq_len})")

    # ---- [2/5] Load teacher ----
    print(f"\n[2/5] Loading {HF_ID} (full bf16)...")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        HF_ID,
        dtype=DTYPE,
        device_map={"": device},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.train(False)
    config = model.config
    config._attn_implementation = "eager"
    load_time = time.time() - t_load
    print(f"  Teacher loaded in {load_time:.1f}s  {vram_report()}")

    # ---- [3/5] Baseline PPL ----
    if not args.skip_baseline:
        print("\n[3/5] Baseline PPL (uncompressed teacher)...")
        t_bp = time.time()
        baseline_ppl = measure_ppl(model, measure_prompts, device, "baseline")
        baseline_time = time.time() - t_bp
        print(f"  Baseline PPL: {baseline_ppl:.4f} ({baseline_time:.1f}s)")
    else:
        baseline_ppl = float("nan")
        baseline_time = 0.0

    # ---- [3.5/5] Cache teacher logits for KL probe ----
    print(f"\n[3.5/5] Caching teacher logits on {args.n_kl_probe} probe prompts...")
    t_pc = time.time()
    teacher_logprobs_probe = cache_teacher_logits(
        model, calibration_ids, device, n_kl_probe=args.n_kl_probe
    )
    probe_cache_time = time.time() - t_pc
    print(f"  Probe logits cached: {teacher_logprobs_probe.shape}  "
          f"({probe_cache_time:.1f}s)")

    # ---- [4/5] Layer-wise compression ----
    print("\n[4/5] Layer-wise hidden-state distillation compression...")
    t_compress = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    per_layer_metrics: list[dict] = []
    args_dict = {
        "bpw": args.bpw, "block_size": args.block_size, "rank": args.rank,
        "train_steps": args.train_steps, "train_lr": args.train_lr,
        "train_bs": args.train_bs, "n_calib": args.n_calib,
        "seq_len": args.seq_len, "n_kl_probe": args.n_kl_probe,
        "seed": args.seed, "objective": "hidden_state_distill",
    }

    for i in range(N_LAYERS):
        print(f"\n  --- Layer {i}/{N_LAYERS - 1} ---")
        m = compress_single_layer_hidden_mse(
            model=model,
            config=config,
            layer_idx=i,
            calibration_ids=calibration_ids,
            teacher_logprobs_probe=teacher_logprobs_probe,
            output_dir=output_dir,
            bpw=args.bpw,
            block_size=args.block_size,
            rank=args.rank,
            train_steps=args.train_steps,
            train_lr=args.train_lr,
            train_bs=args.train_bs,
            n_kl_probe=args.n_kl_probe,
            device=device,
        )
        per_layer_metrics.append(m)
        print(f"    Layer {i}: MSE_init={m['MSE_init']:.4e} "
              f"MSE_final={m['MSE_final']:.4e}  "
              f"KL_init={m['KL_init']:.6f} KL_final={m['KL_final']:.6f}  "
              f"alpha={m['mean_alpha']:.3f}  time={m['compress_time_s']:.1f}s")

        if (i + 1) % 5 == 0:
            save_partial(partial_json, per_layer_metrics, args_dict)

    compress_time = time.time() - t_compress
    peak_vram = max(m["peak_vram_gb"] for m in per_layer_metrics)
    print(f"\n  Compression complete: {compress_time:.1f}s  peak={peak_vram:.2f}GB")

    save_partial(partial_json, per_layer_metrics, args_dict, label="all_layers_done")

    del teacher_logprobs_probe
    free_memory()

    # ---- [5/5] Compressed PPL ----
    print("\n[5/5] Compressed model PPL...")
    t_m = time.time()
    compressed_ppl = measure_ppl(model, measure_prompts, device, "compressed")
    measure_time = time.time() - t_m
    measure_peak_vram = vram_gb(device, peak=True)
    print(f"  Compressed PPL: {compressed_ppl:.4f}  ({measure_time:.1f}s)")

    ppl_ratio = (
        compressed_ppl / baseline_ppl if not math.isnan(baseline_ppl) else float("nan")
    )
    total_time = (load_time + baseline_time + probe_cache_time
                  + compress_time + measure_time)

    results = {
        "experiment": "hidden_state_distill_phi_3_mini_4k_v10",
        "model": "phi-3-mini-4k-instruct",
        "hf_id": HF_ID,
        "n_layers": N_LAYERS,
        "objective": "hidden_state_distill",
        "bpw": args.bpw,
        "block_size": args.block_size,
        "rank": args.rank,
        "train_steps": args.train_steps,
        "train_lr": args.train_lr,
        "train_bs": args.train_bs,
        "n_calib": args.n_calib,
        "n_eval": args.n_eval,
        "n_kl_probe": args.n_kl_probe,
        "seq_len": args.seq_len,
        "eval_seq_len": args.eval_seq_len,
        "calib_tokens": calib_tokens,
        "seed": args.seed,
        "baseline_ppl": baseline_ppl,
        "compressed_ppl": compressed_ppl,
        "ppl_ratio": ppl_ratio,
        "final_ppl": compressed_ppl,
        "peak_vram_compress_gb": peak_vram,
        "peak_vram_measure_gb": measure_peak_vram,
        "total_compress_time_s": compress_time,
        "total_time_s": total_time,
        "per_layer_KL_init": [m["KL_init"] for m in per_layer_metrics],
        "per_layer_KL_final": [m["KL_final"] for m in per_layer_metrics],
        "per_layer_MSE_init": [m["MSE_init"] for m in per_layer_metrics],
        "per_layer_MSE_final": [m["MSE_final"] for m in per_layer_metrics],
        "per_layer_mean_alpha": [m["mean_alpha"] for m in per_layer_metrics],
        "per_layer_quant_rel_l2": [m["mean_quant_rel_l2"] for m in per_layer_metrics],
        "per_layer_compress_time_s": [m["compress_time_s"] for m in per_layer_metrics],
        "per_layer_full": per_layer_metrics,
        "timestamp": datetime.datetime.now().isoformat(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.out_json}")
    print(f"  baseline_ppl={baseline_ppl:.4f}  compressed_ppl={compressed_ppl:.4f}  "
          f"ppl_ratio={ppl_ratio:.4f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
