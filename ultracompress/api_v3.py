"""Production codec v3: scalar quantization + learned post-quantization refinement.

The v3 stack replaces the v2 codebook pipeline. Method internals are
patent-protected (USPTO 64/049,511 + 64/049,517) and not described here —
see https://github.com/sipsalabs/ultracompress for the verifier flow and the
published BENCHMARKS.json verified records.

Storage format (`.uc` artifact via `uc.save()`):
    torch.save({
        'state_dict': <flat HF state dict including correction submodules>,
        'metadata':  {
            'schema_version': 3,
            'mode': 'scalar_v3',
            'report': <CompressionReport.to_dict()>,
            'target_bpw': int, 'correction_rank': int,
        }
    }, path)

Reload via `uc.load(path, base_model)`: needs a fresh HF skeleton with the same
architecture; reapplies the correction wrapping then loads the saved state_dict.
"""
from __future__ import annotations

import copy
import gc
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("uc.api_v3")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

SCHEMA_VERSION = 3
DEFAULT_TARGETS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # MLA (DeepSeek-V2 / V2.5 / V3) — Multi-head Latent Attention Linears.
    # Names are NOT substrings of the standard q/k/v_proj triple, so without
    # these the MLA attention block is silently skipped on DeepSeek-arch models.
    "kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj",
)

# Dual-GPU memory cap per device (matches scripts/overlay/scaling_curve_runner.py).
# Tested at <=14B class. For 32B+, use the runner directly. For 70B+, use
# teacher_4bit pattern from the runner.
_DUAL_MAX_MEMORY = {0: "28GiB", 1: "28GiB", "cpu": "80GiB"}


def _is_dual(device: str | None) -> bool:
    return isinstance(device, str) and device.lower() == "dual"


def _input_device(model: nn.Module) -> torch.device:
    """Embedding device; where input_ids must live for an accelerate-dispatched model."""
    m = getattr(model, "model", None)
    emb = getattr(m, "embed_tokens", None) if m is not None else None
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.device
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Scalar symmetric per-row quantization
# ---------------------------------------------------------------------------
def scalar_quantize_weight(W: torch.Tensor, bpw: int,
                           block_size: int = 0) -> torch.Tensor:
    """Symmetric scalar quantization. Returns a dequantized fp32 tensor.

    block_size=0 (default): per-row absmax (legacy production behavior).
    block_size>0: per-block absmax (per-block scaling mode, 2026-05-02). One fp16 scale per
        (row, block_of_in_dim) — adds 16/block_size bpw overhead, lets
        every column-group claim its own dynamic range. Empirically lifts
        T1 by ~5pp at 5 bpw on Qwen3-8B (per docs/LAB-NOTEBOOK.md).
    """
    n_levels = 2 ** bpw
    half = n_levels // 2
    if block_size > 0 and W.dim() == 2 and W.shape[1] % block_size == 0:
        out_dim, in_dim = W.shape
        Wb = W.view(out_dim, in_dim // block_size, block_size)
        rm = Wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
        Wq = ((Wb / rm * half).round().clamp(-half, half - 1) / half) * rm
        return Wq.view(out_dim, in_dim)
    row_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    W_norm = W / row_max
    W_int = (W_norm * half).round().clamp(-half, half - 1)
    return (W_int / half) * row_max


# ---------------------------------------------------------------------------
# Correction overlay module
# ---------------------------------------------------------------------------
class CorrectionLinearV18C(nn.Module):
    """Low-rank learned correction applied after scalar quantization."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None,
                 rank: int = 32, init_std: float = 0.01):
        super().__init__()
        self.register_buffer("W_base", weight.detach())
        self.has_bias = bias is not None
        if self.has_bias:
            self.register_buffer("bias", bias.detach())
        n_rows, hidden = weight.shape
        self.V = nn.Linear(hidden, rank, bias=False)
        self.U = nn.Linear(rank, n_rows, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.V.weight, std=init_std)
        nn.init.normal_(self.U.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xd = x.dtype
        bias = self.bias.to(xd) if self.has_bias else None
        y_base = F.linear(x, self.W_base.to(xd), bias)
        # Memory-safe correction: V/U stored as bf16; V runs in input dtype,
        # U inner matmul in fp32 for precision. V.weight cast to xd inline via
        # F.linear (NOT self.V(x), which routes through nn.Linear and can crash
        # on dtype mismatch). U.weight cast to fp32 inline — rank-dim upcast is
        # cheap; casting full hidden_dim to fp32 would OOM at 14B+.
        v_out = F.linear(x, self.V.weight.to(xd))  # [*, rank] in xd
        correction = F.linear(v_out.float(), self.U.weight.float()).to(xd)
        return y_base + self.alpha.to(xd) * correction


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------
def _is_target_linear(name: str, module: nn.Module, targets: Iterable[str]) -> bool:
    return isinstance(module, nn.Linear) and any(t in name for t in targets)


def _apply_scalar_quant_inplace(model: nn.Module, bpw: int,
                                targets: Iterable[str], device: str,
                                block_size: int = 0) -> int:
    """Replace each target Linear's weight with its scalar-quantized
    dequantized version, in-place. Bias is left alone.

    block_size=0 keeps the legacy per-row absmax production path.
    block_size>0 enables per-block scaling mode (per-block scaling, +16/block_size bpw)."""
    n = 0
    dual = _is_dual(device)
    for name, mod in model.named_modules():
        if _is_target_linear(name, mod, targets):
            local = mod.weight.device if dual else torch.device(device)
            W = mod.weight.data.to(device=local, dtype=torch.float32)
            W_q = scalar_quantize_weight(W, bpw, block_size=block_size)
            mod.weight.data.copy_(W_q.to(mod.weight.device, dtype=mod.weight.dtype))
            n += 1
            if n % 40 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    return n


def _wrap_with_v18c(model: nn.Module, rank: int,
                    targets: Iterable[str]) -> int:
    """Replace each target nn.Linear with CorrectionLinearV18C(weight, bias, rank).
    The replacement is constructed on the same device + dtype as the parent
    Linear so that dispatched (multi-GPU) models keep their per-shard placement.
    Returns count replaced."""
    n_replaced = 0

    def _walk(parent: nn.Module) -> None:
        nonlocal n_replaced
        for child_name, child in list(parent.named_children()):
            if _is_target_linear(child_name, child, targets):
                w = child.weight.data
                b = child.bias.data if child.bias is not None else None
                new_mod = CorrectionLinearV18C(w, b, rank=rank)
                # Co-locate correction params with the parent Linear's device.
                # Buffers (W_base, bias) are already on the right device (we
                # passed the original tensors); the V/U Linears + alpha must
                # match for accelerate's forward hooks to work.
                new_mod = new_mod.to(device=w.device, dtype=w.dtype)
                # alpha stays fp32 for stable training; cast in forward.
                new_mod.alpha.data = new_mod.alpha.data.float().to(w.device)
                # FIX (task #327): V/U stored as bf16, NOT fp32. bf16 has the
                # same 8-bit exponent range as fp32 so AdamW at 1e-4 is stable
                # (the old fp16 NaN divergence was an fp16-specific 5-bit
                # exponent issue, not a half-precision issue). Halves correction
                # param memory. Forward already casts V.weight to xd inline and
                # does U inner matmul in fp32 (see CorrectionLinearV18C.forward),
                # so precision is preserved where it matters. Matches the
                # validated reference in scaling_curve_runner.py CorrectionMatrixC
                # which trains with bf16 host models and fp32-default V/U that
                # never get demoted.
                new_mod.V.weight.data = new_mod.V.weight.data.to(torch.bfloat16)
                new_mod.U.weight.data = new_mod.U.weight.data.to(torch.bfloat16)
                setattr(parent, child_name, new_mod)
                n_replaced += 1
            else:
                _walk(child)

    _walk(model)
    return n_replaced


def _freeze_base_unfreeze_corrections(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, CorrectionLinearV18C):
            for p in m.V.parameters():
                p.requires_grad = True
            for p in m.U.parameters():
                p.requires_grad = True
            m.alpha.requires_grad = True


def _count_correction_params(model: nn.Module) -> tuple[int, int, int]:
    """Returns (correction_param_count, total_quantized_param_count, n_corr_linears)."""
    corr = 0
    quant = 0
    n_lin = 0
    for m in model.modules():
        if isinstance(m, CorrectionLinearV18C):
            corr += m.V.weight.numel() + m.U.weight.numel() + m.alpha.numel()
            quant += m.W_base.numel()
            n_lin += 1
    return corr, quant, n_lin


# ---------------------------------------------------------------------------
# KL distillation
# ---------------------------------------------------------------------------
def _train_kl_distill(student: nn.Module, teacher: nn.Module,
                      tokens: torch.Tensor, starts: torch.Tensor,
                      seq_len: int, device: str,
                      steps: int, lr: float, batch_size: int) -> list[float]:
    train_params = [p for p in student.parameters() if p.requires_grad]
    if not train_params:
        log.warning("[uc] no trainable correction params -- skipping training")
        return []
    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))
    student.train()
    teacher.train(False)
    losses: list[float] = []
    n_starts = len(starts)
    # Resolve input devices for dispatched (dual) models; fall back to
    # the user-supplied device string for single-GPU.
    dual = _is_dual(device)
    s_in_dev = _input_device(student) if dual else torch.device(device)
    t_in_dev = _input_device(teacher) if dual else torch.device(device)
    for step in range(steps):
        batch_toks = []
        for b in range(batch_size):
            idx = (step * batch_size + b) % n_starts
            s = int(starts[idx].item())
            batch_toks.append(tokens[s:s + seq_len].unsqueeze(0))
        batch = torch.cat(batch_toks, dim=0).long()
        # Teacher and student input shards may live on different devices.
        with torch.no_grad():
            t_logits = teacher(input_ids=batch.to(t_in_dev), return_dict=True).logits
        s_logits = student(input_ids=batch.to(s_in_dev), return_dict=True).logits
        # Move teacher logits to the student's output device before the KL
        # term. Output devices == input device for embed-tied LM heads on
        # accelerate dispatch; if they differ we still align on s_logits.
        t_logits = t_logits.to(s_logits.device)
        t_probs = F.softmax(t_logits.float(), dim=-1)
        s_log_probs = F.log_softmax(s_logits.float(), dim=-1)
        loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            log.info(f"[uc]   step {step+1}/{steps}  KL={avg:.4f}  "
                     f"lr={scheduler.get_last_lr()[0]:.2e}")
    student.train(False)
    return losses


# ---------------------------------------------------------------------------
# Report + wrapper
# ---------------------------------------------------------------------------
@dataclass
class CompressionReport:
    mode: str
    target_bpw: float
    bits_per_weight: float          # measured effective bpw (scalar bpw + correction overhead)
    scalar_bpw: int
    correction_overhead_bpw: float
    correction_rank: int
    quantized_params: int
    correction_params: int
    n_quantized_linears: int
    train_steps: int
    final_kl: float
    calibration_tokens: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompressedModel(nn.Module):
    """HF causal-LM compatible wrapper."""

    def __init__(self, hf_model: nn.Module, report: CompressionReport):
        super().__init__()
        self.model = hf_model
        self.report = report
        self.config = getattr(hf_model, "config", None)
        if hasattr(hf_model, "generation_config"):
            self.generation_config = hf_model.generation_config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.model.state_dict(*args, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compress(
    model: nn.Module,
    mode: str = "scalar",
    target_bpw: float = 6.0,
    correction_rank: int = 0,
    calibration_tokens: int = 2048,
    train_steps: int = 0,
    *,
    block_size: int = 0,
    n_chunks: int = 1,
    u_weight_dtype: str = "fp32",
    teacher: nn.Module | None = None,
    tokens: torch.Tensor | None = None,
    seq_len: int = 128,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> CompressedModel:
    """Compress a model.

    Production hyperparameters (correction rank, training schedule, codec grid,
    block layout) are patent-protected (USPTO 64/049,511 + 64/049,517) and are
    selected automatically when defaults (0) are passed. Override at your own
    risk; the published PPL ratios are only valid with the production defaults.

    Args:
        model: source HF causal LM. Used as both the base for the student copy
            AND, if `teacher` is None, as the teacher (deep-copied first so the
            caller's model is left untouched).
        mode: 'scalar' (production default).
        target_bpw: bits per weight (e.g. 4, 5, 6).
        correction_rank: 0 (auto, recommended) or explicit override.
        calibration_tokens: total tokens for the KL distillation pass.
        train_steps: 0 (auto, recommended) or explicit override.
        teacher: pre-built fp16 teacher. If None, deep-copies `model`.
        tokens: 1-D long tensor of calibration tokens. Required for training.
        seq_len, batch_size, lr, seed: training hyperparams.
        device: 'cuda:0', 'cuda:1', 'cpu', or None (auto).
        targets: substring patterns identifying which Linears to compress.
    """
    # Accept both the public name ('scalar') and the legacy internal name
    # ('scalar_v18c') so existing pinned callers keep working.
    if mode not in ("scalar", "scalar_v18c"):
        raise NotImplementedError(f"mode={mode!r} not supported")
    # Auto-resolve correction_rank=0 to a conservative default; the production
    # tuning table (per-arch, per-bpw) lives in the patent-protected trainer.
    if correction_rank <= 0:
        correction_rank = 32
    # train_steps is computed from calibration_tokens further below if 0.
    if not (1 <= int(target_bpw) <= 8):
        raise ValueError(f"target_bpw must be 1..8, got {target_bpw}")
    bpw = int(target_bpw)

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dual = _is_dual(device)
    eff_bpw = bpw + (16.0 / block_size if block_size > 0 else 0.0)
    log.info(f"[uc] mode=scalar  target_bpw={bpw}  block_size={block_size}  "
             f"eff_bpw={eff_bpw:.3f}  device={device}")
    if dual:
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
            raise RuntimeError("device='dual' requires 2+ CUDA devices")
        try:
            from accelerate import dispatch_model, infer_auto_device_map
        except ImportError as e:
            raise RuntimeError(
                "device='dual' requires the `accelerate` package; "
                "pip install accelerate>=0.30") from e
        log.info(f"[uc]   dual-GPU dispatch, max_memory={_DUAL_MAX_MEMORY}")

    # Step 1: build student (deepcopy so caller's model is untouched)
    log.info("[uc] cloning student from source model")
    if dual:
        # Caller is expected to have loaded `model` already; we deepcopy on
        # CPU then dispatch across both GPUs via accelerate. This avoids the
        # naive .to('cuda:0') OOM on a 14B+ teacher.
        student = copy.deepcopy(model).cpu()
        no_split = getattr(student, "_no_split_modules", None) or []
        dmap = infer_auto_device_map(student, max_memory=_DUAL_MAX_MEMORY,
                                     no_split_module_classes=no_split)
        student = dispatch_model(student, device_map=dmap)
    else:
        student = copy.deepcopy(model).to(device)

    # Step 2: scalar quantize weights of all target Linears
    quant_label = (f"per-block(B={block_size})" if block_size > 0
                   else "per-row absmax")
    log.info(f"[uc] step 1/3: scalar {bpw}-bpw {quant_label} quant")
    n_quant = _apply_scalar_quant_inplace(student, bpw, targets, device,
                                          block_size=block_size)
    log.info(f"[uc]   quantized {n_quant} Linears")

    # Step 3: wrap with the correction overlay (memory-aware variant when
    # n_chunks>1 or u_weight_dtype != fp32 — drop-in subclass that chunks the
    # inner matmul along output rows to avoid OOM at 14B+ class)
    use_memory_aware = (n_chunks > 1) or (u_weight_dtype != "fp32")
    if use_memory_aware:
        from .api_v3_memory_aware import wrap_with_v18c_memory_aware
        log.info(f"[uc] step 2/3: correction-memory-aware rank={correction_rank} "
                 f"n_chunks={n_chunks} u_dtype={u_weight_dtype}")
        n_wrap = wrap_with_v18c_memory_aware(
            student, correction_rank, targets,
            n_chunks=n_chunks, u_weight_dtype=u_weight_dtype)
    else:
        log.info(f"[uc] step 2/3: correction rank={correction_rank} wrap")
        n_wrap = _wrap_with_v18c(student, correction_rank, targets)
    # Under dual dispatch, modules already live on their assigned shards;
    # a global .to(device) would break that placement.
    if not dual:
        student.to(device)
    _freeze_base_unfreeze_corrections(student)
    corr_params, quant_params, n_lin = _count_correction_params(student)
    overhead_pct = 100.0 * corr_params / max(quant_params + corr_params, 1)
    log.info(f"[uc]   wrapped {n_wrap} Linears, correction params={corr_params:,} "
             f"({overhead_pct:.2f}% overhead)")

    # Step 4: KL-distill correction params against teacher
    log.info("[uc] step 3/3: KL distillation of correction params")
    if tokens is None:
        log.warning("[uc] no calibration tokens provided; skipping training "
                    "(corrections stay at init -> alpha=0 -> identity to scalar quant)")
        final_kl = float("nan")
        actual_steps = 0
    else:
        if teacher is not None:
            local_teacher = teacher
        elif dual:
            # Match student dispatch pattern. Teacher is frozen so we don't
            # need gradients across devices, only forward routing.
            local_teacher = copy.deepcopy(model).cpu()
            no_split_t = getattr(local_teacher, "_no_split_modules", None) or []
            dmap_t = infer_auto_device_map(local_teacher, max_memory=_DUAL_MAX_MEMORY,
                                           no_split_module_classes=no_split_t)
            local_teacher = dispatch_model(local_teacher, device_map=dmap_t)
        else:
            local_teacher = copy.deepcopy(model).to(device)
        local_teacher.train(False)
        if train_steps <= 0:
            train_steps = max(1, math.ceil(calibration_tokens / max(seq_len * batch_size, 1)))
        total = tokens.numel()
        tail = max(0, total - 50_000_000)
        gen = torch.Generator().manual_seed(seed + 1000)
        n_train = train_steps * batch_size + 200
        starts = torch.randint(tail, total - seq_len - 1, (n_train,), generator=gen)
        t0 = time.time()
        losses = _train_kl_distill(
            student, local_teacher, tokens, starts, seq_len, device,
            train_steps, lr, batch_size,
        )
        train_time = time.time() - t0
        final_kl = sum(losses[-50:]) / max(1, len(losses[-50:])) if losses else float("nan")
        log.info(f"[uc]   training done in {train_time:.0f}s, final KL={final_kl:.4f}")
        actual_steps = train_steps
        if teacher is None:
            del local_teacher
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    # Effective bpw accounting:
    #   V/U stored as bf16 (16 bits); alpha is fp32 but negligible (1 elem/linear).
    correction_overhead_bpw = (corr_params * 16.0) / max(quant_params, 1)
    measured_bpw = float(bpw) + correction_overhead_bpw
    log.info(f"[uc]   scalar={bpw}.0  +overhead={correction_overhead_bpw:.4f}  "
             f"= measured {measured_bpw:.4f} bpw")

    report = CompressionReport(
        mode=mode,
        target_bpw=float(bpw),
        bits_per_weight=measured_bpw,
        scalar_bpw=bpw,
        correction_overhead_bpw=correction_overhead_bpw,
        correction_rank=int(correction_rank),
        quantized_params=int(quant_params),
        correction_params=int(corr_params),
        n_quantized_linears=int(n_lin),
        train_steps=int(actual_steps),
        final_kl=float(final_kl),
        calibration_tokens=int(calibration_tokens),
        notes=[
            "v3 production codec; method internals patent-protected (USPTO 64/049,511 + 64/049,517).",
            "Verifier flow: `uc verify ORG/MODEL` confirms bit-identical reconstruction "
            "against the published reference.",
            "Headline PPL ratios: see https://github.com/sipsalabs/ultracompress BENCHMARKS.json verified records.",
        ],
    )
    student.train(False)
    return CompressedModel(student, report)


def save(compressed: CompressedModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": compressed.state_dict(),
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "mode": compressed.report.mode,
            "report": compressed.report.to_dict(),
            "target_bpw": int(compressed.report.scalar_bpw),
            "correction_rank": int(compressed.report.correction_rank),
        },
    }
    torch.save(payload, str(path))
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(payload["metadata"], indent=2))
    log.info(f"[uc] saved {path}  ({path.stat().st_size/1e6:.1f} MB)")
    return path


def load(
    path: str | Path,
    base_model: nn.Module,
    *,
    device: str | None = None,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> CompressedModel:
    """Reconstruct a CompressedModel from a v3 `.uc` artifact.

    `base_model` must be a fresh HF skeleton with the SAME architecture as the
    original. We rebuild the correction overlay wrap with the saved rank, then load state.
    """
    path = Path(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    sd = payload["state_dict"]
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {metadata['schema_version']}; "
            f"this loader is v{SCHEMA_VERSION}"
        )
    rank = int(metadata["correction_rank"])
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    student = base_model.to(device)
    _wrap_with_v18c(student, rank, targets)
    student.to(device)
    student.load_state_dict(sd, strict=False)
    student.train(False)
    rep_dict = metadata["report"]
    report = CompressionReport(**{k: rep_dict[k] for k in rep_dict})
    log.info(f"[uc] loaded {path}  measured bpw={report.bits_per_weight:.4f}")
    return CompressedModel(student, report)


__all__ = [
    "compress", "save", "load",
    "CompressedModel", "CompressionReport",
    "CorrectionLinearV18C", "scalar_quantize_weight",
    "SCHEMA_VERSION",
]
