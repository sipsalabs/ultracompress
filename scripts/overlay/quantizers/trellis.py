"""QTIP-style trellis-coded weight quantizer adapter for UltraCompress.

Wraps the reference `bitshift_codebook` from `external/qtip/lib/codebook/bitshift.py`
and exposes a `trellis_quantize_weight(W, bpw, ...)` function that returns a dense
fp32 reconstruction in the *original* basis (random Hadamard transform applied
internally to flatten the weight distribution, then exactly inverted before return).

Returning Wq in the original basis is critical: the low-rank correction adapter
overlay in `correction_adapter.py` learns to absorb (W - Wq) and assumes both are in
the same basis. The trellis quantizer is therefore a drop-in replacement for any
other `*_quantize_weight` function in `scaling_curve_runner.py`.

Design notes
------------
* QTIP's `bitshift.py` imports siblings (`lib.utils.matmul_had`, etc.) that pull
  in `fast_hadamard_transform` — a CUDA kernel package that does not compile under
  our PyTorch 2.11 + CUDA-13.2-system / CUDA-12.8-runtime mismatch. We side-step
  this by stubbing the unavailable submodules in `sys.modules` *before* loading
  `bitshift.py` via `importlib.util`. We only need the `bitshift_codebook` class
  (Viterbi + reconstruction); the CUDA-bound `BitshiftLinear` wrapper is unused.
* `@torch.compile` on `bitshift_codebook.update` is monkey-patched away — Triton
  compile under Windows is brittle and we don't need the speed-up for a one-shot
  weight quantization pass.
* Random Hadamard transform (RHT) uses a pure-PyTorch radix-2 FWHT (no external
  deps). Skipped with a one-shot warning if either dim of W is not a power of 2.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import warnings
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Reference impl loader (one-shot, cached)
# ---------------------------------------------------------------------------
_QTIP_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external', 'qtip')
)
_BITSHIFT_PATH = os.path.join(_QTIP_ROOT, 'lib', 'codebook', 'bitshift.py')

_codebook_cache: dict = {}
_rht_warned: bool = False


def _stub_module(name: str, **attrs) -> types.ModuleType:
    """Insert a placeholder module so QTIP's transitive imports don't crash."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load_bitshift_module() -> types.ModuleType:
    """Dynamically load QTIP's bitshift.py without its CUDA-kernel siblings.

    We provide stub `lib`, `lib.codebook`, `lib.utils.*` modules that satisfy the
    `from lib.codebook import kdict` etc. imports at the top of bitshift.py.
    """
    cached = _codebook_cache.get('module')
    if cached is not None:
        return cached

    if not os.path.isfile(_BITSHIFT_PATH):
        raise FileNotFoundError(
            f'QTIP bitshift.py not found at {_BITSHIFT_PATH}. '
            f'Clone the QTIP reference impl into external/qtip first.'
        )

    # Stub out the QTIP package surface that bitshift.py touches at import time.
    # `kdict` is referenced but only used by other codebook modules; an empty
    # module is sufficient. `kernel_check.has_kernel` is called inside
    # BitshiftLinear which we never instantiate, so a no-op is safe.
    # `kernel_decompress.decode_compressed` and `matmul_had.matmul_had{U,Ut}_cuda`
    # are likewise only referenced by BitshiftLinear.
    _stub_module('lib')
    _stub_module('lib.codebook', kdict=types.SimpleNamespace())
    _stub_module('lib.codebook.kdict')
    _stub_module(
        'lib.utils',
        kernel_check=types.SimpleNamespace(has_kernel=lambda *a, **k: False),
        kernel_decompress=types.SimpleNamespace(
            decode_compressed=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError('CUDA kernel path is not available in this env')
            )
        ),
        matmul_had=types.SimpleNamespace(
            matmul_hadU_cuda=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError('CUDA Hadamard path is not available')
            ),
            matmul_hadUt_cuda=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError('CUDA Hadamard path is not available')
            ),
        ),
    )
    _stub_module(
        'lib.utils.kernel_check',
        has_kernel=lambda *a, **k: False,
    )
    _stub_module(
        'lib.utils.kernel_decompress',
        decode_compressed=lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('CUDA kernel path is not available')
        ),
    )
    _stub_module(
        'lib.utils.matmul_had',
        matmul_hadU_cuda=lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('CUDA Hadamard path is not available')
        ),
        matmul_hadUt_cuda=lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('CUDA Hadamard path is not available')
        ),
    )

    spec = importlib.util.spec_from_file_location(
        'sipsa_qtip_bitshift', _BITSHIFT_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build module spec for {_BITSHIFT_PATH}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['sipsa_qtip_bitshift'] = mod
    spec.loader.exec_module(mod)

    # Disable @torch.compile on the inner update step. Triton compile under
    # Windows is unreliable and we don't need the speed-up for one-shot quant.
    if hasattr(mod.bitshift_codebook, 'update'):
        original = mod.bitshift_codebook.update
        # `@torch.compile` wraps the raw fn; the original is reachable via
        # `_torchdynamo_orig_callable` on recent PyTorch, fall back to the
        # wrapper itself which still runs (just without compile speed-ups).
        raw = getattr(original, '_torchdynamo_orig_callable', None)
        if raw is not None:
            mod.bitshift_codebook.update = raw

    _codebook_cache['module'] = mod
    return mod


# ---------------------------------------------------------------------------
# Pure-PyTorch fast Walsh-Hadamard transform (radix-2, normalized)
# ---------------------------------------------------------------------------
def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fwht(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """In-place radix-2 FWHT along `axis`, normalized so H is orthogonal.

    Length along `axis` must be a power of 2. Input is cloned to avoid
    surprising the caller. Operates in fp32 for stability.
    """
    if axis != -1:
        x = x.transpose(axis, -1)
    orig_shape = x.shape
    n = orig_shape[-1]
    assert _is_pow2(n), f'FWHT requires power-of-2 length, got {n}'
    y = x.contiguous().clone().to(torch.float32).reshape(-1, n)
    h = 1
    while h < n:
        # Iterate in slices of width 2h: pair (i, i+h) -> (i+i+h, i-(i+h))
        # Vectorized form: reshape to (..., n//(2h), 2, h) and apply butterfly.
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=2).view(-1, n)
        h *= 2
    y = y * (n ** -0.5)  # normalize so H @ H.T == I
    y = y.view(orig_shape)
    if axis != -1:
        y = y.transpose(axis, -1)
    return y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _maybe_warn_rht_skipped(reason: str) -> None:
    global _rht_warned
    if not _rht_warned:
        warnings.warn(
            f'[trellis] random Hadamard transform skipped: {reason}. '
            f'Trellis-only reconstruction error will be higher than the '
            f'RHT-enabled path.',
            stacklevel=2,
        )
        _rht_warned = True


def _get_codebook(L: int, K: int, V: int, tlut_bits: int, decode_mode: str,
                  device: torch.device):
    """Build (and cache per-config) a `bitshift_codebook` on `device`."""
    key = (L, K, V, tlut_bits, decode_mode, str(device))
    cb = _codebook_cache.get(key)
    if cb is not None:
        return cb
    mod = _load_bitshift_module()
    cb = mod.bitshift_codebook(
        L=L, K=K, V=V, tlut_bits=tlut_bits, decode_mode=decode_mode
    )
    cb = cb.to(device)
    _codebook_cache[key] = cb
    return cb


def trellis_quantize_weight(
    W: torch.Tensor,
    bpw: int,
    *,
    L: int = 16,
    V_state: int = 2,
    decode_mode: str = '3inst',
    group_size: int = 64,
    block_size: int = 64,
    use_rht: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Trellis-quantize a Linear weight and return a dense fp32 reconstruction.

    Pipeline (in order)
    -------------------
    1. (optional) Random Hadamard transform along the input axis: random ±1
       sign flip + normalized FWHT. This Gaussianizes the per-row distribution
       (sum of 2048 ± flipped weights -> CLT). After RHT every entry of
       W_rot is roughly N(0, ||W[r,:]||^2 / N) regardless of the original
       outlier structure.
    2. (optional) Per-block absmax scaling along the input axis (POST-RHT):
       split the rotated weight into chunks of `block_size` columns, divide
       each chunk by its absmax. After RHT all block absmaxes are tightly
       clustered (~3 sigma of the same Gaussian), so post-scaling
       W_rot_norm lies in [-1, 1] uniformly across blocks. On structured-
       outlier weights (real LLMs) this captures any residual non-stationarity
       RHT failed to remove. On iid-Gaussian smoke inputs the gain over
       vanilla is tiny (RHT alone already Gaussianized everything).
    3. Per-row STD normalization (calibration to codebook) + trellis Viterbi
       quantization. Critical detail: QTIP's `recons_state` codebook
       (decode_3inst at L=16/K=3/V=1) has empirical std 1.244 -- the Viterbi
       cost (recons_state - input)^2 is only minimized correctly when the
       input distribution matches the codebook. Per-row absmax normalization
       (the obvious choice) puts inputs at std ~0.28, which wastes ~60% of
       the codewords on out-of-range values. Std-normalization to match the
       codebook recovers the missing capacity (~2x better rel-Frob).
    4. Multiply trellis output by per-row scales (in the rotated, per-block-
       normalized basis).
    5. Multiply by per-block scales (in the rotated basis).
    6. Undo RHT (FWHT + sign flip) -> back to ORIGINAL basis.

    Why post-RHT per-block scaling (not pre-RHT)
    --------------------------------------------
    Pre-RHT per-block scaling injects piecewise-constant block_scales into
    the signal that FWHT then mixes into ALL rotated positions, raising
    ||W_rot||_F above ||W||_F (outlier blocks get amplified). The trellis
    then sees a noisier signal AND the un-scaling step amplifies the trellis
    error in outlier blocks.

    Post-RHT per-block scaling sees an already-Gaussianized signal where all
    blocks have nearly the same magnitude. The per-block scales are tightly
    clustered, the trellis sees a clean signal, and un-scaling barely
    amplifies the error since every S_block is similar.

    Both orderings are mathematically invertible -- the difference is in the
    QUALITY of the trellis approximation step. Empirically, post-RHT order
    + std-normalized calibration beats pre-RHT order by 2-3x rel-Frob on
    Gaussian smoke; the gap is expected to narrow on real LLM weights but
    post-RHT is never worse.

    Parameters
    ----------
    W : torch.Tensor
        Float weight tensor of shape `[out_features, in_features]`.
    bpw : int
        Target bits-per-weight; passed as the `K` parameter of `bitshift_codebook`
        (bits emitted per V_state-symbol step). Practically 2, 3, or 4.
    L : int, default 16
        Trellis state register width (log2 of number of states).
    V_state : int, default 2
        Bits per emitted symbol. For `decode_mode='3inst'` the codebook *requires*
        V==1; this argument is therefore ignored when decode_mode is 1mad/2mad/3inst
        (we silently force V=1 in that case to match the assertion in bitshift.py).
    decode_mode : str, default '3inst'
        QTIP decode mode. '3inst' is the cheapest GPU-decoder path and the QTIP
        default for low-bit Llama runs.
    group_size : int, default 64
        Number of *output rows* processed per Viterbi batch. Limits peak memory
        for the DP table (`O(2^L * batch * in_features / V)` int32). Lower values
        trade speed for memory; 64 fits comfortably in a few GB at L=16.
    block_size : int, default <NDA>
        Per-block absmax scaling (applied post-RHT). If > 0 the
        rotated weight is split into chunks of `block_size` columns; each
        chunk is normalized to [-1, 1] by its absmax BEFORE the Viterbi pass
        and rescaled afterwards. Set to 0 to recover vanilla QTIP (per-row
        absmax only). For non-divisible shapes the residual tail is left at
        per-row scaling.
    use_rht : bool, default True
        If True and BOTH dims of W are powers of 2, apply random ±1 sign flip
        + FWHT to the input axis FIRST (before per-block scaling and Viterbi),
        and exactly invert the rotation in the returned Wq.
    seed : int, default 0
        Seed for the random sign vector used by RHT.

    Returns
    -------
    Wq : torch.Tensor
        Dense fp32 reconstruction with the SAME shape, dtype-promoted to float32,
        in the SAME basis as `W`. Caller is expected to cast back to the original
        weight dtype (the caller in `scaling_curve_runner.py` already does this).
    """
    assert W.dim() == 2, f'expected 2D weight, got shape {tuple(W.shape)}'
    out_features, in_features = W.shape
    device = W.device
    orig_dtype = W.dtype

    # Validate decode-mode / V_state combination against bitshift.py asserts.
    if decode_mode in ('1mad', '2mad', '3inst'):
        V = 1
    else:
        V = V_state

    # tlut_bits == L matches QTIP's default and is required by the 'lut' branch
    # (we don't use that branch, but it's the convention).
    tlut_bits = L

    # ----- Step 1: random Hadamard transform on the original W ----------
    apply_rht = use_rht and _is_pow2(in_features) and _is_pow2(out_features)
    if use_rht and not apply_rht:
        _maybe_warn_rht_skipped(
            f'shape ({out_features}, {in_features}) has non-power-of-2 dim'
        )

    if apply_rht:
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        sign_in = (torch.randint(0, 2, (in_features,), generator=gen) * 2 - 1).to(
            device=device, dtype=torch.float32
        )
        # W_rot = FWHT(W * sign_in along input axis). H is orthogonal so to
        # recover W from Wq_rot we apply: FWHT(Wq_rot) * sign_in (FWHT is its
        # own inverse when normalized).
        W_work = (W.to(torch.float32) * sign_in.unsqueeze(0))
        W_work = _fwht(W_work, axis=-1)
    else:
        W_work = W.to(torch.float32).contiguous()
        sign_in = None

    # ----- Step 2: per-block absmax scaling on the ROTATED W (scalar-quant style)
    # Operating post-RHT means all block absmaxes are similar (RHT
    # Gaussianized the signal), so the per-block normalization is uniform
    # across blocks and the trellis sees a clean [-1, 1] input.
    use_block_scaling = block_size > 0 and in_features >= block_size
    if use_block_scaling:
        n_full_blocks = in_features // block_size
        block_in = n_full_blocks * block_size
        # Use reshape() in case the column slice is non-contiguous.
        head = W_work[:, :block_in].reshape(out_features, n_full_blocks, block_size)
        block_scales = head.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        # Build a working tensor where the head is per-block normalized to
        # [-1, 1] and the tail (if any) is unchanged.
        W_work_scaled = W_work.clone()
        W_work_scaled[:, :block_in] = (head / block_scales).reshape(
            out_features, block_in
        )
    else:
        n_full_blocks = 0
        block_in = 0
        block_scales = None
        W_work_scaled = W_work

    # ----- Step 3+4: per-row-batch trellis encode -----------------------
    # bitshift_codebook.quantize expects X of shape (in_features, batch_rows)
    # because internally it does X = X.T.contiguous().to(float16) and treats
    # the first dim as time (T) and the second as parallel batch (B).
    cb = _get_codebook(L=L, K=bpw, V=V, tlut_bits=tlut_bits,
                        decode_mode=decode_mode, device=device)

    # The Viterbi requires T (= in_features after our transpose) divisible by V.
    if in_features % V != 0:
        raise ValueError(
            f'in_features={in_features} not divisible by V={V}; '
            f'choose a different decode_mode/V_state.'
        )

    # Calibration: the Viterbi cost is `(recons_state - input)^2`, so the
    # input distribution must MATCH the codeword distribution for the trellis
    # to use its codewords efficiently. Per-row absmax normalization (the
    # obvious choice) puts inputs in [-1, 1] with std ~0.28 -- much narrower
    # than the codebook output distribution (~1.24 std for decode_3inst at
    # L=16). Result: ~2x worse rel-Frob than calibrated. We compute the
    # codebook std directly from `cb.recons_state` and scale every row's
    # std to match it. This works for any `decode_mode` / `L` / `K` combo.
    codebook_std = float(cb.recons_state.float().std().item())
    Wq_rot_scaled = torch.empty_like(W_work_scaled)
    n_rows = out_features
    bs = max(1, int(group_size))
    for r0 in range(0, n_rows, bs):
        r1 = min(n_rows, r0 + bs)
        block = W_work_scaled[r0:r1]                           # (rows, in)
        # Per-row std-normalization to match codebook calibration.
        rm = block.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)
        rm = rm / codebook_std
        # quantize() expects (in_features, rows) layout (T, B). Internally it
        # transposes back, so we feed (in, rows).
        x = (block / rm).T.contiguous()                        # (in, rows)
        hatX, _state = cb.quantize(x)                          # hatX: (in, rows)
        Wq_rot_scaled[r0:r1] = hatX.T.contiguous().to(torch.float32) * rm

    # ----- Step 5: undo per-block scaling (still in rotated basis) ------
    if use_block_scaling:
        head_q = Wq_rot_scaled[:, :block_in].reshape(
            out_features, n_full_blocks, block_size
        )
        head_q = head_q * block_scales                          # broadcast (out, n_blk, 1)
        Wq_rot = Wq_rot_scaled.clone()
        Wq_rot[:, :block_in] = head_q.reshape(out_features, block_in)
    else:
        Wq_rot = Wq_rot_scaled

    # ----- Step 6: undo RHT -> back to ORIGINAL basis -------------------
    if apply_rht:
        Wq = _fwht(Wq_rot, axis=-1) * sign_in.unsqueeze(0)
    else:
        Wq = Wq_rot

    return Wq.to(torch.float32)
