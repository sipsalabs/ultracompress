"""Memory-aware variant of the production correction module for 14B+ scaling on a single 32GB GPU.

The production correction class (see `ultracompress.api_v3`) trades a
small fp32 activation buffer for numerical stability. At 14B+ scales that buffer
becomes the single-device OOM constraint. This module preserves the *exact*
inner numerics of the production class but caps peak activation memory by
splitting the inner projection into row-disjoint sub-projections written into
a single pre-allocated output buffer.

A secondary opt-in dtype flag activates a precision-tradeoff fallback for
extreme-memory regimes (e.g. 70B on a single device). Default is the
zero-precision-loss path matched to production.

Usage (drop-in replacement via api_v3 surgery pattern):

    from ultracompress.api_v3_memory_aware import CorrectionLinearV18CMemoryAware
    new_mod = CorrectionLinearV18CMemoryAware(
        weight=child.weight.data,
        bias=child.bias.data if child.bias is not None else None,
        rank=correction_rank,  # tuned per arch; see model card for value
        n_chunks=4,
        u_weight_dtype="fp32",
    )

License: Apache-2.0 (matches api_v3.py).
"""
from __future__ import annotations

import logging
from typing import Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.api_v3 import CorrectionLinearV18C

log = logging.getLogger("uc.api_v3_memory_aware")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


_UWeightDtype = Literal["fp32", "bf16", "fp16"]


def _resolve_u_weight_dtype(name: _UWeightDtype) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"u_weight_dtype must be 'fp32'|'bf16'|'fp16', got {name!r}")


class CorrectionLinearV18CMemoryAware(CorrectionLinearV18C):
    """Memory-aware drop-in for `CorrectionLinearV18C`.

    Public API:
      - Same constructor signature as the parent, plus `n_chunks` and
        `u_weight_dtype`.
      - Subclass of `CorrectionLinearV18C`, so existing `_wrap_with_v18c`,
        `_count_correction_params`, and `_freeze_base_unfreeze_corrections`
        hooks in api_v3.py continue to recognize it (isinstance check).

    Mechanism:
      - V projection: identical to parent (input dtype, memory-light).
      - U projection: split along output (row) dim into `n_chunks` slices.
        Each chunk is computed in fp32, downcast to xd, and written into a
        pre-allocated xd output tensor. Peak fp32 footprint per forward:
        ~`[B, T, n_rows / n_chunks]` (vs. `[B, T, n_rows]` in parent).
      - Optional `u_weight_dtype="bf16"`/"fp16": skip the fp32 cast entirely,
        run inner in input dtype, save half the U.weight storage too.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int = 32,
        init_std: float = 0.01,
        *,
        n_chunks: int = 4,
        u_weight_dtype: _UWeightDtype = "fp32",
    ) -> None:
        super().__init__(weight=weight, bias=bias, rank=rank, init_std=init_std)
        if n_chunks < 1:
            raise ValueError(f"n_chunks must be >=1, got {n_chunks}")
        self.n_chunks = int(n_chunks)
        self.u_weight_dtype_name: _UWeightDtype = u_weight_dtype
        # Recast U.weight if Option-B fallback was requested. We keep V at
        # fp32 because rank-dim cost is negligible.
        target_u_dtype = _resolve_u_weight_dtype(u_weight_dtype)
        if self.U.weight.dtype != target_u_dtype:
            with torch.no_grad():
                self.U.weight.data = self.U.weight.data.to(target_u_dtype)

        # Precompute row-slice boundaries once. Use roughly-equal chunks; the
        # last slice absorbs any remainder so the algebra stays exact.
        n_rows = self.U.weight.shape[0]
        self.n_chunks = max(1, min(self.n_chunks, n_rows))
        base = n_rows // self.n_chunks
        rem = n_rows % self.n_chunks
        bounds: list[int] = [0]
        for k in range(self.n_chunks):
            bounds.append(bounds[-1] + base + (1 if k < rem else 0))
        # Stored as a python list; tiny, never on GPU.
        self._chunk_bounds: list[int] = bounds

    @property
    def u_inner_fp32(self) -> bool:
        """Whether U inner matmul should run in fp32 (default path).

        True when u_weight_dtype_name is "fp32" (the default): U.weight is
        stored as bf16 but cast to fp32 inline for precision.  False only
        when the user explicitly requested Option-B (bf16/fp16 inner).
        """
        return self.u_weight_dtype_name == "fp32"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xd = x.dtype
        bias = self.bias.to(xd) if self.has_bias else None
        y_base = F.linear(x, self.W_base.to(xd), bias)

        # V projection in input dtype (cheap, rank-dim output).
        v_out = F.linear(x, self.V.weight.to(xd))  # [..., rank] in xd

        if self.u_inner_fp32:
            # Default path: U inner matmul in fp32 for precision.
            # U.weight stored as bf16, cast to fp32 inline.
            v_out_f = v_out.float()
            u_w = self.U.weight.float()  # bf16 -> fp32 inline

            if self.n_chunks == 1:
                # Same numerical contract as the parent class.
                correction = F.linear(v_out_f, u_w).to(xd)
            else:
                # Pre-allocate the full xd output buffer so we never hold
                # multiple fp32 chunks simultaneously. Each chunk lifetime is
                # bounded by one F.linear + one .to(xd) cast.
                out_shape = v_out.shape[:-1] + (u_w.shape[0],)
                correction = torch.empty(out_shape, dtype=xd, device=v_out.device)
                for k in range(self.n_chunks):
                    a, b = self._chunk_bounds[k], self._chunk_bounds[k + 1]
                    if a == b:
                        continue
                    # F.linear(v_out_f, u_w[a:b]) -> [..., b-a] in fp32
                    chunk_out = F.linear(v_out_f, u_w[a:b])
                    correction[..., a:b] = chunk_out.to(xd)
                    del chunk_out
        else:
            # Option-B fallback: user explicitly requested bf16/fp16 inner.
            # Cast U.weight to xd inline (handles bf16-weight on fp16-host
            # mismatch) and keep everything memory-light.
            u_w = self.U.weight.to(xd)
            correction = F.linear(v_out, u_w)

        return y_base + self.alpha.to(xd) * correction


# ---------------------------------------------------------------------------
# Convenience surgery helpers (mirror api_v3 patterns; opt-in)
# ---------------------------------------------------------------------------
def _is_target_linear(name: str, module: nn.Module, targets: Iterable[str]) -> bool:
    return isinstance(module, nn.Linear) and any(t in name for t in targets)


def wrap_with_v18c_memory_aware(
    model: nn.Module,
    rank: int,
    targets: Iterable[str],
    *,
    n_chunks: int = 4,
    u_weight_dtype: _UWeightDtype = "fp32",
) -> int:
    """Replace each target nn.Linear with CorrectionLinearV18CMemoryAware.

    Mirrors api_v3._wrap_with_v18c but constructs the memory-aware variant. The
    replacement is co-located on the parent Linear's device + dtype, so this
    is safe for accelerate-dispatched (multi-GPU) models. Returns count
    replaced.
    """
    n_replaced = 0

    def _walk(parent: nn.Module) -> None:
        nonlocal n_replaced
        for child_name, child in list(parent.named_children()):
            if _is_target_linear(child_name, child, targets):
                w = child.weight.data
                b = child.bias.data if child.bias is not None else None
                new_mod = CorrectionLinearV18CMemoryAware(
                    weight=w,
                    bias=b,
                    rank=rank,
                    n_chunks=n_chunks,
                    u_weight_dtype=u_weight_dtype,
                )
                new_mod = new_mod.to(device=w.device, dtype=w.dtype)
                # alpha stays fp32 for stable training (matches api_v3).
                new_mod.alpha.data = new_mod.alpha.data.float().to(w.device)
                # FIX (task #327): V/U stored as bf16 by default (matches
                # api_v3 fix). Option-B u_weight_dtype overrides U only.
                if u_weight_dtype != "fp32":
                    target_u_dtype = _resolve_u_weight_dtype(u_weight_dtype)
                    new_mod.U.weight.data = new_mod.U.weight.data.to(target_u_dtype)
                else:
                    new_mod.U.weight.data = new_mod.U.weight.data.to(torch.bfloat16)
                # V stored as bf16; forward casts to xd inline.
                new_mod.V.weight.data = new_mod.V.weight.data.to(torch.bfloat16)
                setattr(parent, child_name, new_mod)
                n_replaced += 1
            else:
                _walk(child)

    _walk(model)
    return n_replaced


__all__ = [
    "CorrectionLinearV18CMemoryAware",
    "wrap_with_v18c_memory_aware",
]
