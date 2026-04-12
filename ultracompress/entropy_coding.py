"""
Lossless entropy coding — FREE 33-50% compression on quantized weights.

Based on ZipNN insight (arxiv 2411.05239): separate float exponent/mantissa
streams before compressing, since they have very different entropy profiles.
Quantized weights cluster in few exponent values -> extreme compressibility.

Zero quality loss. Stacks on top of any quantization method.
"""

import struct
import zlib
import torch
import numpy as np
from io import BytesIO


class EntropyCompressor:
    """Compress a single tensor via exponent/mantissa stream splitting."""

    def compress(self, tensor: torch.Tensor) -> bytes:
        raw = tensor.to(torch.float32).numpy().tobytes()
        arr = np.frombuffer(raw, dtype=np.uint32)
        # Split IEEE 754: sign+exponent (top 9 bits) vs mantissa (low 23 bits)
        exponents = ((arr >> 16) & 0xFFFF).astype(np.uint16).tobytes()
        mantissas = (arr & 0xFFFF).astype(np.uint16).tobytes()
        exp_z = zlib.compress(exponents, 9)
        man_z = zlib.compress(mantissas, 9)
        buf = BytesIO()
        # Header: shape ndim, shape, dtype, then two compressed streams
        shape = tensor.shape
        buf.write(struct.pack("<B", len(shape)))
        for s in shape:
            buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<B", _dtype_to_id(tensor.dtype)))
        buf.write(struct.pack("<I", len(exp_z)))
        buf.write(struct.pack("<I", len(man_z)))
        buf.write(exp_z)
        buf.write(man_z)
        return buf.getvalue()

    def decompress(self, data: bytes) -> torch.Tensor:
        buf = BytesIO(data)
        ndim = struct.unpack("<B", buf.read(1))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(ndim))
        dtype_id = struct.unpack("<B", buf.read(1))[0]
        exp_len = struct.unpack("<I", buf.read(4))[0]
        man_len = struct.unpack("<I", buf.read(4))[0]
        exp_z = buf.read(exp_len)
        man_z = buf.read(man_len)
        exponents = np.frombuffer(zlib.decompress(exp_z), dtype=np.uint16).astype(np.uint32)
        mantissas = np.frombuffer(zlib.decompress(man_z), dtype=np.uint16).astype(np.uint32)
        arr = (exponents << 16) | mantissas
        tensor = torch.from_numpy(arr.view(np.float32).copy()).reshape(shape)
        return tensor.to(_id_to_dtype(dtype_id))


class ModelEntropyCompressor:
    """Compress an entire model state_dict with per-tensor entropy coding."""

    def __init__(self):
        self.tc = EntropyCompressor()

    def compress_model(self, state_dict: dict, verbose: bool = False) -> bytes:
        raw_size = sum(t.nelement() * t.element_size() for t in state_dict.values())
        buf = BytesIO()
        buf.write(struct.pack("<I", len(state_dict)))
        for name, tensor in state_dict.items():
            name_b = name.encode("utf-8")
            buf.write(struct.pack("<H", len(name_b)))
            buf.write(name_b)
            compressed = self.tc.compress(tensor)
            buf.write(struct.pack("<I", len(compressed)))
            buf.write(compressed)
        result = buf.getvalue()
        ratio = raw_size / len(result) if result else 0
        saving = (1 - len(result) / raw_size) * 100 if raw_size else 0
        if verbose:
            print(f"[EntropyCodec] {raw_size:,}B -> {len(result):,}B "
                  f"({ratio:.2f}x, {saving:.1f}% saved)")
        return result

    def decompress_model(self, data: bytes) -> dict:
        buf = BytesIO(data)
        n = struct.unpack("<I", buf.read(4))[0]
        state_dict = {}
        for _ in range(n):
            name_len = struct.unpack("<H", buf.read(2))[0]
            name = buf.read(name_len).decode("utf-8")
            comp_len = struct.unpack("<I", buf.read(4))[0]
            state_dict[name] = self.tc.decompress(buf.read(comp_len))
        return state_dict


# --- dtype mapping ---
_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8]
def _dtype_to_id(dt): return _DTYPES.index(dt) if dt in _DTYPES else 0
def _id_to_dtype(i): return _DTYPES[i] if i < len(_DTYPES) else torch.float32
