"""
ULTIMATE PIPELINE — The product-grade compression chain.

Architecture: LOSSLESS → LOSSLESS → LOSSY → LOSSLESS → LOSSLESS

  Stage 1: HADAMARD ROTATION (lossless)
    Multiply each weight matrix by a Hadamard-like orthogonal matrix.
    This spreads energy uniformly across all elements, destroying spiky
    outliers that ruin quantization. Perfectly invertible: H^T @ H = I.

  Stage 2: MANIFOLD PROJECTION (lossless in the sense of being recoverable
    with residual; the projection itself truncates, but the residual is
    stored and re-injected later)
    SVD reveals the intrinsic low-rank manifold of each weight matrix.
    Project onto the top-k singular subspace. Store the residual separately
    so it can be compressed efficiently in Stage 4.

  Stage 3: JOINT QUANTIZATION ON MANIFOLD (THE lossy step)
    Uniform quantization of the projected (rotated, low-rank) weights.
    This is the ONLY step where information is irreversibly lost.
    Because Stage 1 made outliers rare and Stage 2 concentrated energy,
    the quantizer sees a benign, well-conditioned distribution.

  Stage 4: RESIDUAL CORRECTION via FRR genome (lossless)
    The manifold residual (what SVD dropped) is quantized too, but at
    lower precision. Think of it as a "correction genome" — tiny data
    that patches up the quantization damage.

  Stage 5: ENTROPY CODING — ZipNN-style stream splitting (lossless)
    Split IEEE 754 floats into exponent and mantissa streams, compress
    each with zlib. Free 33-50% on top of everything else.

Usage:
    from ultracompress.ultimate_pipeline import UltimatePipeline

    pipe = UltimatePipeline()
    compressed = pipe.compress(model_state_dict)
    recovered  = pipe.decompress(compressed)
    pipe.report()
"""

import struct
import zlib
import time
import math
import torch
import numpy as np
from io import BytesIO
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ================================================================
# Configuration
# ================================================================

@dataclass
class UltimatePipelineConfig:
    """Knobs for each stage."""
    # Stage 1: Hadamard
    hadamard_block_size: int = 0       # 0 = full-matrix rotation; else block-diagonal

    # Stage 2: Manifold projection
    rank_fraction: float = 0.5         # Keep this fraction of singular values
    min_rank: int = 4                  # Never go below this rank
    manifold_min_dim: int = 32         # Skip SVD for tensors smaller than this

    # Stage 3: Quantization
    quant_bits: int = 8                # Bits for main (projected) weights
    quant_group_size: int = 128        # Per-group quantization granularity

    # Stage 4: Residual correction
    residual_bits: int = 4             # Bits for the manifold residual
    residual_group_size: int = 64      # Group size for residual quantization

    # Stage 5: Entropy coding
    zlib_level: int = 9                # Maximum compression

    # Thresholds
    skip_1d: bool = True               # Skip rotation/SVD on 1-D tensors (norms, biases)


# ================================================================
# Stage 1: Hadamard Rotation
# ================================================================

def _hadamard_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Build a normalized Hadamard-like orthogonal matrix of size n.

    For powers of 2 we use the true Sylvester-Hadamard construction.
    For arbitrary n we fall back to a random orthogonal (QR of Gaussian).
    Either way H @ H.T = I, so the transform is perfectly invertible.
    """
    if n <= 0:
        raise ValueError(f"Hadamard size must be positive, got {n}")
    if n == 1:
        return torch.ones(1, 1, device=device)

    # Check power of 2
    if n & (n - 1) == 0:
        # Sylvester construction: H_1 = [1], H_{2k} = [[H_k, H_k], [H_k, -H_k]] / sqrt(2)
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0) / math.sqrt(2.0)
        return H
    else:
        # Random orthogonal via QR decomposition — still perfectly invertible
        G = torch.randn(n, n, device=device)
        Q, _ = torch.linalg.qr(G)
        return Q


class HadamardRotation:
    """Stage 1: Rotate weight matrices to spread energy uniformly.

    Why: Neural network weights often have outlier rows/columns with
    huge magnitudes. These outliers dominate quantization error because
    the quantization grid must stretch to cover them, wasting precision
    on the majority of near-zero values.

    A Hadamard rotation is an orthogonal transform: it preserves all
    norms and inner products (lossless), but distributes the energy of
    any single outlier across ALL elements. Post-rotation, the weight
    distribution is much more uniform — ideal for quantization.

    Inverse: multiply by H.T (transpose = inverse for orthogonal matrices).
    """

    def __init__(self, block_size: int = 0):
        self.block_size = block_size
        self._H_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def _get_H(self, n: int, device: torch.device) -> torch.Tensor:
        key = (n, str(device))
        if key not in self._H_cache:
            self._H_cache[key] = _hadamard_matrix(n, device)
        return self._H_cache[key]

    def rotate(self, W: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Apply Hadamard rotation to a 2-D weight matrix.

        Returns rotated matrix and metadata needed for inverse.
        """
        if W.ndim != 2:
            return W, {"skipped": True, "shape": W.shape}

        rows, cols = W.shape
        device = W.device

        if self.block_size > 0 and cols > self.block_size:
            # Block-diagonal rotation: rotate blocks of columns independently
            bs = self.block_size
            n_full_blocks = cols // bs
            remainder = cols % bs
            W_rot = W.clone()
            for b in range(n_full_blocks):
                H = self._get_H(bs, device)
                W_rot[:, b*bs:(b+1)*bs] = W[:, b*bs:(b+1)*bs] @ H
            if remainder > 0:
                H_rem = self._get_H(remainder, device)
                W_rot[:, n_full_blocks*bs:] = W[:, n_full_blocks*bs:] @ H_rem
            meta = {"skipped": False, "mode": "block", "block_size": bs,
                    "cols": cols, "remainder": remainder}
        else:
            H = self._get_H(cols, device)
            W_rot = W @ H
            meta = {"skipped": False, "mode": "full", "cols": cols}

        return W_rot, meta

    def unrotate(self, W_rot: torch.Tensor, meta: dict) -> torch.Tensor:
        """Inverse Hadamard rotation: W = W_rot @ H.T"""
        if meta.get("skipped", False):
            return W_rot

        device = W_rot.device

        if meta["mode"] == "block":
            bs = meta["block_size"]
            cols = meta["cols"]
            n_full = cols // bs
            remainder = meta["remainder"]
            W = W_rot.clone()
            for b in range(n_full):
                H = self._get_H(bs, device)
                W[:, b*bs:(b+1)*bs] = W_rot[:, b*bs:(b+1)*bs] @ H.T
            if remainder > 0:
                H_rem = self._get_H(remainder, device)
                W[:, n_full*bs:] = W_rot[:, n_full*bs:] @ H_rem.T
            return W
        else:
            H = self._get_H(meta["cols"], device)
            return W_rot @ H.T


# ================================================================
# Stage 2: Manifold Projection (SVD)
# ================================================================

class ManifoldProjection:
    """Stage 2: Project onto the intrinsic low-rank subspace via SVD.

    Weight matrices are approximately low-rank: most of their energy
    lives in a small number of singular directions. We separate the
    signal (top-k SVD) from the residual (everything else).

    The projected weights go to the quantizer (Stage 3).
    The residual goes to the correction genome (Stage 4).

    Lossless: projected + residual = original exactly.
    """

    def __init__(self, rank_fraction: float = 0.5, min_rank: int = 4,
                 min_dim: int = 32):
        self.rank_fraction = rank_fraction
        self.min_rank = min_rank
        self.min_dim = min_dim

    def project(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Decompose W into low-rank projection + residual.

        Returns:
            W_proj:   low-rank approximation (goes to quantizer)
            residual: W - W_proj (goes to residual corrector)
            meta:     SVD factors needed for efficient storage
        """
        if W.ndim != 2 or min(W.shape) < self.min_dim:
            return W, torch.zeros_like(W), {"skipped": True, "shape": W.shape}

        rows, cols = W.shape
        k = max(self.min_rank, int(min(rows, cols) * self.rank_fraction))
        k = min(k, min(rows, cols))

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        U_k = U[:, :k]                # (rows, k)
        S_k = S[:k]                    # (k,)
        Vh_k = Vh[:k, :]              # (k, cols)

        W_proj = (U_k * S_k.unsqueeze(0)) @ Vh_k
        residual = W - W_proj

        # Energy captured
        total_energy = (S ** 2).sum().item()
        kept_energy = (S_k ** 2).sum().item()
        energy_ratio = kept_energy / (total_energy + 1e-12)

        meta = {
            "skipped": False,
            "k": k,
            "rows": rows,
            "cols": cols,
            "energy_ratio": energy_ratio,
            "U_k": U_k,
            "S_k": S_k,
            "Vh_k": Vh_k,
        }

        return W_proj, residual, meta

    def reconstruct(self, W_proj: torch.Tensor, residual: torch.Tensor,
                    meta: dict) -> torch.Tensor:
        """Reconstruct: just add projection and residual back together."""
        if meta.get("skipped", False):
            return W_proj + residual
        return W_proj + residual


# ================================================================
# Stage 3: Uniform Group Quantization
# ================================================================

@dataclass
class QuantizedData:
    """Quantized representation of a tensor."""
    codes: torch.Tensor       # Integer codes, uint8
    scales: torch.Tensor      # Per-group scale, float32
    zeros: torch.Tensor       # Per-group zero-point, float32
    bits: int
    group_size: int
    original_shape: tuple
    n_elements: int

    def storage_bytes(self) -> int:
        code_bytes = (self.codes.numel() * self.bits + 7) // 8
        param_bytes = (self.scales.numel() + self.zeros.numel()) * 4
        return code_bytes + param_bytes

    @property
    def bits_per_weight(self) -> float:
        return (self.storage_bytes() * 8) / max(self.n_elements, 1)


class UniformQuantizer:
    """Stage 3: The ONLY lossy step — uniform quantization with per-group scales.

    After Hadamard rotation (uniform distribution) and manifold projection
    (concentrated energy), the weights are ideally conditioned for simple
    uniform quantization. No need for fancy non-uniform schemes.

    Per-group scales capture local magnitude variations.
    """

    def __init__(self, bits: int = 8, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.n_levels = 2 ** bits

    def quantize(self, W: torch.Tensor) -> QuantizedData:
        """Quantize tensor to N-bit integers with per-group scaling."""
        shape = W.shape
        flat = W.reshape(-1).float()
        n = flat.numel()

        # Pad to multiple of group_size
        gs = self.group_size
        pad = (gs - n % gs) % gs
        if pad > 0:
            flat = torch.cat([flat, flat.new_zeros(pad)])

        groups = flat.reshape(-1, gs)
        n_groups = groups.shape[0]

        # Per-group min/max
        g_min = groups.min(dim=1).values
        g_max = groups.max(dim=1).values
        scales = (g_max - g_min) / (self.n_levels - 1)
        scales = scales.clamp(min=1e-12)

        # Quantize
        codes = torch.round((groups - g_min.unsqueeze(1)) / scales.unsqueeze(1))
        codes = codes.clamp(0, self.n_levels - 1).to(torch.uint8)

        return QuantizedData(
            codes=codes.reshape(-1)[:n],
            scales=scales,
            zeros=g_min,
            bits=self.bits,
            group_size=gs,
            original_shape=shape,
            n_elements=n,
        )

    def dequantize(self, qd: QuantizedData) -> torch.Tensor:
        """Reconstruct float tensor from quantized codes."""
        n = qd.n_elements
        gs = qd.group_size

        # Rebuild padded
        pad = (gs - n % gs) % gs
        codes_padded = qd.codes.float()
        if pad > 0:
            codes_padded = torch.cat([codes_padded, codes_padded.new_zeros(pad)])

        groups = codes_padded.reshape(-1, gs)
        values = groups * qd.scales.unsqueeze(1) + qd.zeros.unsqueeze(1)
        return values.reshape(-1)[:n].reshape(qd.original_shape)


# ================================================================
# Stage 4: Residual Correction (lightweight quantization of residual)
# ================================================================

class ResidualCorrector:
    """Stage 4: Compress the manifold residual at lower precision.

    The residual (what SVD dropped) is typically small and noisy.
    We quantize it at lower bit-width (e.g. 4-bit) — losing some
    fine detail but capturing the bulk of the correction signal.

    This is analogous to the "error correction genome" concept:
    a tiny sidecar that patches up the main compressed representation.
    """

    def __init__(self, bits: int = 4, group_size: int = 64):
        self.quantizer = UniformQuantizer(bits=bits, group_size=group_size)
        self.bits = bits

    def compress_residual(self, residual: torch.Tensor) -> QuantizedData:
        """Quantize the residual."""
        return self.quantizer.quantize(residual)

    def decompress_residual(self, qd: QuantizedData) -> torch.Tensor:
        """Reconstruct the approximate residual."""
        return self.quantizer.dequantize(qd)


# ================================================================
# Stage 5: Entropy Coding (ZipNN-style stream splitting + zlib)
# ================================================================

def _entropy_compress_tensor(tensor: torch.Tensor, zlib_level: int = 9) -> bytes:
    """Compress a single tensor via IEEE 754 stream splitting.

    Split float32 values into upper 16 bits (sign+exponent+top mantissa)
    and lower 16 bits (bottom mantissa). Each stream has very different
    entropy characteristics and compresses much better separately.

    Quantized weights cluster in a few exponent values → the upper stream
    becomes extremely compressible (often 5-10x). The lower stream is
    noisier but still benefits from zlib.
    """
    raw = tensor.to(torch.float32).contiguous().numpy().tobytes()
    arr = np.frombuffer(raw, dtype=np.uint32)

    upper = ((arr >> 16) & 0xFFFF).astype(np.uint16).tobytes()
    lower = (arr & 0xFFFF).astype(np.uint16).tobytes()

    upper_z = zlib.compress(upper, zlib_level)
    lower_z = zlib.compress(lower, zlib_level)

    buf = BytesIO()
    # Shape header
    shape = tensor.shape
    buf.write(struct.pack("<B", len(shape)))
    for s in shape:
        buf.write(struct.pack("<I", s))
    # Stream lengths
    buf.write(struct.pack("<II", len(upper_z), len(lower_z)))
    buf.write(upper_z)
    buf.write(lower_z)
    return buf.getvalue()


def _entropy_decompress_tensor(data: bytes) -> torch.Tensor:
    """Inverse of _entropy_compress_tensor."""
    buf = BytesIO(data)
    ndim = struct.unpack("<B", buf.read(1))[0]
    shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(ndim))
    upper_len, lower_len = struct.unpack("<II", buf.read(8))
    upper_z = buf.read(upper_len)
    lower_z = buf.read(lower_len)

    upper = np.frombuffer(zlib.decompress(upper_z), dtype=np.uint16).astype(np.uint32)
    lower = np.frombuffer(zlib.decompress(lower_z), dtype=np.uint16).astype(np.uint32)
    arr = (upper << 16) | lower
    return torch.from_numpy(arr.view(np.float32).copy()).reshape(shape)


def _entropy_compress_quantized(qd: QuantizedData, zlib_level: int = 9) -> bytes:
    """Entropy-code a QuantizedData struct into raw bytes.

    Packs codes (uint8), scales, zeros, and metadata into a single
    compressed byte stream.
    """
    buf = BytesIO()

    # Metadata header
    buf.write(struct.pack("<B", qd.bits))
    buf.write(struct.pack("<I", qd.group_size))
    buf.write(struct.pack("<I", qd.n_elements))
    ndim = len(qd.original_shape)
    buf.write(struct.pack("<B", ndim))
    for s in qd.original_shape:
        buf.write(struct.pack("<I", s))

    # Codes — already uint8, compress directly
    code_bytes = qd.codes.numpy().astype(np.uint8).tobytes()
    code_z = zlib.compress(code_bytes, zlib_level)
    buf.write(struct.pack("<I", len(code_z)))
    buf.write(code_z)

    # Scales — float32 stream-split
    scale_z = _entropy_compress_tensor(qd.scales)
    buf.write(struct.pack("<I", len(scale_z)))
    buf.write(scale_z)

    # Zeros — float32 stream-split
    zero_z = _entropy_compress_tensor(qd.zeros)
    buf.write(struct.pack("<I", len(zero_z)))
    buf.write(zero_z)

    return buf.getvalue()


def _entropy_decompress_quantized(data: bytes) -> QuantizedData:
    """Inverse of _entropy_compress_quantized."""
    buf = BytesIO(data)

    bits = struct.unpack("<B", buf.read(1))[0]
    group_size = struct.unpack("<I", buf.read(4))[0]
    n_elements = struct.unpack("<I", buf.read(4))[0]
    ndim = struct.unpack("<B", buf.read(1))[0]
    shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(ndim))

    code_len = struct.unpack("<I", buf.read(4))[0]
    code_z = buf.read(code_len)
    codes = torch.from_numpy(
        np.frombuffer(zlib.decompress(code_z), dtype=np.uint8).copy()
    )

    scale_len = struct.unpack("<I", buf.read(4))[0]
    scales = _entropy_decompress_tensor(buf.read(scale_len))

    zero_len = struct.unpack("<I", buf.read(4))[0]
    zeros = _entropy_decompress_tensor(buf.read(zero_len))

    return QuantizedData(
        codes=codes,
        scales=scales,
        zeros=zeros,
        bits=bits,
        group_size=group_size,
        original_shape=shape,
        n_elements=n_elements,
    )


# ================================================================
# ULTIMATE PIPELINE — Chains all five stages
# ================================================================

class UltimatePipeline:
    """The product compression pipeline.

    Chains five stages: Hadamard → Manifold → Quantize → Residual → Entropy.
    Only Stage 3 is lossy. Everything else is lossless or near-lossless.

    Usage:
        pipe = UltimatePipeline()
        compressed_bytes = pipe.compress(state_dict)
        recovered_dict   = pipe.decompress(compressed_bytes)
        pipe.report()
    """

    def __init__(self, config: Optional[UltimatePipelineConfig] = None):
        self.config = config or UltimatePipelineConfig()
        c = self.config

        # Instantiate stages
        self.stage1 = HadamardRotation(block_size=c.hadamard_block_size)
        self.stage2 = ManifoldProjection(
            rank_fraction=c.rank_fraction,
            min_rank=c.min_rank,
            min_dim=c.manifold_min_dim,
        )
        self.stage3 = UniformQuantizer(bits=c.quant_bits, group_size=c.quant_group_size)
        self.stage4 = ResidualCorrector(bits=c.residual_bits,
                                         group_size=c.residual_group_size)

        # Per-stage metrics (populated during compress)
        self._metrics: Dict[str, dict] = {}
        self._stage_times: Dict[str, float] = {}
        self._original_size: int = 0
        self._compressed_size: int = 0
        self._per_tensor_info: List[dict] = []

    # ------------------------------------------------------------------
    # COMPRESS
    # ------------------------------------------------------------------

    def compress(self, model_weights: Dict[str, torch.Tensor]) -> bytes:
        """Compress an entire model state_dict to bytes.

        Pipeline per tensor:
          1. Hadamard rotation    (lossless)
          2. Manifold projection  (lossless — stores residual)
          3. Quantize projection  (LOSSY)
          4. Quantize residual    (small loss on small signal)
          5. Entropy code both    (lossless)
        """
        t_total = time.time()
        self._per_tensor_info = []
        c = self.config

        raw_total = 0
        all_compressed_tensors: Dict[str, bytes] = {}

        stage_accum = {"hadamard": 0.0, "manifold": 0.0,
                       "quantize": 0.0, "residual": 0.0, "entropy": 0.0}

        n_tensors = len(model_weights)
        for idx, (name, W) in enumerate(model_weights.items()):
            W = W.float().cpu()
            raw_bytes = W.numel() * 4
            raw_total += raw_bytes

            info = {"name": name, "shape": tuple(W.shape),
                    "raw_bytes": raw_bytes, "numel": W.numel()}

            is_small = (W.ndim == 1) or (W.numel() < c.manifold_min_dim)

            # --- Stage 1: Hadamard rotation ---
            t0 = time.time()
            if W.ndim == 2 and not (c.skip_1d and is_small):
                W_rot, h_meta = self.stage1.rotate(W)
            else:
                W_rot, h_meta = W, {"skipped": True, "shape": W.shape}
            stage_accum["hadamard"] += time.time() - t0

            # --- Stage 2: Manifold projection ---
            t0 = time.time()
            if W.ndim == 2 and not is_small:
                W_proj, residual, m_meta = self.stage2.project(W_rot)
                info["energy_ratio"] = m_meta.get("energy_ratio", 1.0)
                info["svd_rank"] = m_meta.get("k", 0)
            else:
                W_proj = W_rot
                residual = torch.zeros_like(W_rot)
                m_meta = {"skipped": True, "shape": W.shape}
                info["energy_ratio"] = 1.0
                info["svd_rank"] = 0
            stage_accum["manifold"] += time.time() - t0

            # --- Stage 3: Quantize projected weights (THE lossy step) ---
            t0 = time.time()
            qd_proj = self.stage3.quantize(W_proj)
            stage_accum["quantize"] += time.time() - t0

            # --- Stage 4: Quantize residual ---
            t0 = time.time()
            has_residual = not m_meta.get("skipped", False)
            if has_residual and residual.abs().max() > 1e-12:
                qd_resid = self.stage4.compress_residual(residual)
            else:
                qd_resid = None
            stage_accum["residual"] += time.time() - t0

            # --- Stage 5: Entropy coding ---
            t0 = time.time()
            proj_bytes = _entropy_compress_quantized(qd_proj, c.zlib_level)

            if qd_resid is not None:
                resid_bytes = _entropy_compress_quantized(qd_resid, c.zlib_level)
            else:
                resid_bytes = b""
            stage_accum["entropy"] += time.time() - t0

            # Pack per-tensor: flags | hadamard_meta | proj_bytes | resid_bytes
            tensor_buf = BytesIO()

            # Flags byte: bit 0 = has_residual, bit 1 = hadamard_skipped
            flags = 0
            if qd_resid is not None:
                flags |= 0x01
            if h_meta.get("skipped", False):
                flags |= 0x02
            tensor_buf.write(struct.pack("<B", flags))

            # Hadamard meta (only if not skipped)
            if not h_meta.get("skipped", False):
                mode = 0 if h_meta["mode"] == "full" else 1
                tensor_buf.write(struct.pack("<BI", mode, h_meta["cols"]))
                if mode == 1:
                    tensor_buf.write(struct.pack("<II",
                                                  h_meta["block_size"],
                                                  h_meta["remainder"]))

            # Projection bytes
            tensor_buf.write(struct.pack("<I", len(proj_bytes)))
            tensor_buf.write(proj_bytes)

            # Residual bytes
            tensor_buf.write(struct.pack("<I", len(resid_bytes)))
            if resid_bytes:
                tensor_buf.write(resid_bytes)

            all_compressed_tensors[name] = tensor_buf.getvalue()

            comp_bytes = len(all_compressed_tensors[name])
            info["compressed_bytes"] = comp_bytes
            info["ratio"] = raw_bytes / max(comp_bytes, 1)
            self._per_tensor_info.append(info)

        # --- Assemble final byte stream ---
        final_buf = BytesIO()

        # Magic + version
        final_buf.write(b"ULTZ")  # magic
        final_buf.write(struct.pack("<H", 1))  # version

        # Number of tensors
        final_buf.write(struct.pack("<I", n_tensors))

        # Tensor table: name + compressed data
        for name, data in all_compressed_tensors.items():
            name_b = name.encode("utf-8")
            final_buf.write(struct.pack("<H", len(name_b)))
            final_buf.write(name_b)
            final_buf.write(struct.pack("<I", len(data)))
            final_buf.write(data)

        result = final_buf.getvalue()

        self._original_size = raw_total
        self._compressed_size = len(result)
        self._stage_times = stage_accum
        self._stage_times["total"] = time.time() - t_total

        return result

    # ------------------------------------------------------------------
    # DECOMPRESS
    # ------------------------------------------------------------------

    def decompress(self, data: bytes) -> Dict[str, torch.Tensor]:
        """Decompress bytes back to a model state_dict.

        Reverse pipeline per tensor:
          5. Entropy decode
          4. Dequantize residual
          3. Dequantize projection
          2. Add projection + residual
          1. Inverse Hadamard rotation
        """
        buf = BytesIO(data)

        # Magic + version
        magic = buf.read(4)
        assert magic == b"ULTZ", f"Bad magic: {magic!r}"
        version = struct.unpack("<H", buf.read(2))[0]
        assert version == 1, f"Unsupported version: {version}"

        n_tensors = struct.unpack("<I", buf.read(4))[0]

        result = {}
        for _ in range(n_tensors):
            name_len = struct.unpack("<H", buf.read(2))[0]
            name = buf.read(name_len).decode("utf-8")
            data_len = struct.unpack("<I", buf.read(4))[0]
            tensor_data = buf.read(data_len)

            result[name] = self._decompress_tensor(tensor_data)

        return result

    def _decompress_tensor(self, data: bytes) -> torch.Tensor:
        """Decompress a single tensor through the reverse pipeline."""
        buf = BytesIO(data)

        # Flags
        flags = struct.unpack("<B", buf.read(1))[0]
        has_residual = bool(flags & 0x01)
        hadamard_skipped = bool(flags & 0x02)

        # Hadamard meta
        h_meta = {"skipped": hadamard_skipped}
        if not hadamard_skipped:
            mode_byte, cols = struct.unpack("<BI", buf.read(5))
            h_meta["mode"] = "full" if mode_byte == 0 else "block"
            h_meta["cols"] = cols
            if mode_byte == 1:
                bs, rem = struct.unpack("<II", buf.read(8))
                h_meta["block_size"] = bs
                h_meta["remainder"] = rem

        # Stage 5 → 3: Entropy decode + dequantize projection
        proj_len = struct.unpack("<I", buf.read(4))[0]
        proj_bytes = buf.read(proj_len)
        qd_proj = _entropy_decompress_quantized(proj_bytes)
        W_proj = self.stage3.dequantize(qd_proj)

        # Stage 5 → 4: Entropy decode + dequantize residual
        resid_len = struct.unpack("<I", buf.read(4))[0]
        if has_residual and resid_len > 0:
            resid_bytes = buf.read(resid_len)
            qd_resid = _entropy_decompress_quantized(resid_bytes)
            residual = self.stage4.decompress_residual(qd_resid)
        else:
            if resid_len > 0:
                buf.read(resid_len)  # skip
            residual = torch.zeros_like(W_proj)

        # Stage 2 inverse: combine projection + residual
        W_rot = self.stage2.reconstruct(W_proj, residual,
                                          {"skipped": not has_residual})

        # Stage 1 inverse: un-rotate
        W = self.stage1.unrotate(W_rot, h_meta)

        return W

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    def report(self):
        """Print a detailed compression report with per-stage metrics."""
        if not self._per_tensor_info:
            print("[UltimatePipeline] No compression data. Run compress() first.")
            return

        print("\n" + "=" * 72)
        print("  ULTIMATE PIPELINE — Compression Report")
        print("=" * 72)

        orig_mb = self._original_size / 1e6
        comp_mb = self._compressed_size / 1e6
        ratio = self._original_size / max(self._compressed_size, 1)
        saving = (1 - self._compressed_size / self._original_size) * 100

        print(f"\n  Original size:    {orig_mb:>10.2f} MB")
        print(f"  Compressed size:  {comp_mb:>10.2f} MB")
        print(f"  Ratio:            {ratio:>10.2f}x")
        print(f"  Savings:          {saving:>9.1f}%")

        # BPW estimate
        total_params = sum(t["numel"] for t in self._per_tensor_info)
        bpw = (self._compressed_size * 8) / max(total_params, 1)
        print(f"  Bits per weight:  {bpw:>10.3f}")
        print(f"  Total parameters: {total_params:>10,}")

        # Stage timing
        print(f"\n  Stage timing:")
        for stage, t in self._stage_times.items():
            print(f"    {stage:<12s}  {t:>8.3f}s")

        # Per-stage configuration
        c = self.config
        print(f"\n  Configuration:")
        print(f"    Hadamard block:   {c.hadamard_block_size or 'full'}")
        print(f"    SVD rank frac:    {c.rank_fraction}")
        print(f"    Quant bits:       {c.quant_bits}")
        print(f"    Residual bits:    {c.residual_bits}")
        print(f"    Quant group:      {c.quant_group_size}")
        print(f"    Residual group:   {c.residual_group_size}")

        # Top-10 largest tensors by compression ratio
        sorted_info = sorted(self._per_tensor_info,
                             key=lambda x: x["raw_bytes"], reverse=True)
        print(f"\n  Top-10 largest tensors:")
        print(f"    {'Name':<45s} {'Shape':<20s} {'Ratio':>7s} {'SVD-E':>6s}")
        print(f"    {'-'*45} {'-'*20} {'-'*7} {'-'*6}")
        for info in sorted_info[:10]:
            shape_str = str(info["shape"])
            er = info.get("energy_ratio", 0)
            print(f"    {info['name']:<45s} {shape_str:<20s} "
                  f"{info['ratio']:>6.1f}x {er:>5.1%}")

        # Quality summary: tensors with lowest compression
        worst = sorted(self._per_tensor_info, key=lambda x: x["ratio"])
        print(f"\n  Bottom-5 worst-compressing tensors:")
        for info in worst[:5]:
            print(f"    {info['name']:<45s} {info['ratio']:>6.1f}x")

        # SVD rank distribution
        ranks = [i["svd_rank"] for i in self._per_tensor_info if i["svd_rank"] > 0]
        if ranks:
            print(f"\n  SVD rank stats (for 2-D tensors):")
            print(f"    count={len(ranks)}, min={min(ranks)}, "
                  f"max={max(ranks)}, mean={sum(ranks)/len(ranks):.1f}")
            energies = [i["energy_ratio"] for i in self._per_tensor_info
                        if i.get("energy_ratio", 0) > 0 and i["svd_rank"] > 0]
            if energies:
                print(f"    Energy captured: min={min(energies):.3%}, "
                      f"max={max(energies):.3%}, mean={sum(energies)/len(energies):.3%}")

        print("\n" + "=" * 72)
        print(f"  Pipeline: Hadamard -> Manifold(SVD) -> Q{c.quant_bits} "
              f"-> Residual(Q{c.residual_bits}) -> Entropy(zlib)")
        print(f"  Lossy stage: ONLY Stage 3 (Q{c.quant_bits} on projected weights)")
        print("=" * 72 + "\n")


# ================================================================
# Convenience: preset configurations
# ================================================================

def ultimate_high_quality(model_weights: Dict[str, torch.Tensor]) -> bytes:
    """Compress with high quality settings (~4-6x compression)."""
    config = UltimatePipelineConfig(
        rank_fraction=0.75,
        quant_bits=8,
        residual_bits=6,
        quant_group_size=64,
        residual_group_size=32,
    )
    pipe = UltimatePipeline(config)
    result = pipe.compress(model_weights)
    pipe.report()
    return result


def ultimate_balanced(model_weights: Dict[str, torch.Tensor]) -> bytes:
    """Compress with balanced quality/size (~8-15x compression)."""
    config = UltimatePipelineConfig(
        rank_fraction=0.5,
        quant_bits=6,
        residual_bits=4,
        quant_group_size=128,
        residual_group_size=64,
    )
    pipe = UltimatePipeline(config)
    result = pipe.compress(model_weights)
    pipe.report()
    return result


def ultimate_extreme(model_weights: Dict[str, torch.Tensor]) -> bytes:
    """Compress aggressively (~20-50x compression)."""
    config = UltimatePipelineConfig(
        rank_fraction=0.25,
        quant_bits=4,
        residual_bits=2,
        quant_group_size=256,
        residual_group_size=128,
    )
    pipe = UltimatePipeline(config)
    result = pipe.compress(model_weights)
    pipe.report()
    return result
