"""
FIELD COMPRESSION — Weights as wave equations.

Instead of storing each weight as a number, represent the entire
weight FIELD as a superposition of standing waves.

Like how a guitar string's vibration = sum of harmonics.
A weight matrix = sum of 2D standing waves (Fourier modes).

Store only the AMPLITUDES of each mode (tiny).
Reconstruct full matrix by summing the modes.

Low-frequency modes capture global structure.
High-frequency modes capture details.
Truncate high frequencies = lossy but smooth compression.

This is DCT (Discrete Cosine Transform) applied to weight matrices,
but framed as PHYSICS: the weights are a vibrating membrane,
and we store the resonant frequencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FieldEncoder(nn.Module):
    """Encode a weight matrix as a superposition of wave modes.

    Keep only the top-K modes (lowest frequency first).
    K modes for an NxM matrix: K << N*M = compression.
    """
    def __init__(self, n_modes=64):
        super().__init__()
        self.n_modes = n_modes

    def encode(self, weight_matrix):
        """Encode weight matrix into wave amplitudes."""
        # 2D DCT (via FFT)
        # Real FFT of rows then columns
        freq = torch.fft.rfft2(weight_matrix.float())

        # Keep top-K modes by magnitude
        flat = freq.reshape(-1)
        magnitudes = flat.abs()
        if len(magnitudes) > self.n_modes:
            topk = magnitudes.topk(self.n_modes)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[topk.indices] = True
            sparse_freq = torch.where(mask, flat, torch.zeros_like(flat))
            # Store as sparse: indices + values
            indices = topk.indices
            values = flat[indices]
            return {'indices': indices, 'values': values, 'shape': freq.shape,
                    'orig_shape': weight_matrix.shape}
        else:
            return {'full': freq, 'orig_shape': weight_matrix.shape}

    def decode(self, encoded):
        """Reconstruct weight matrix from wave amplitudes."""
        if 'full' in encoded:
            return torch.fft.irfft2(encoded['full'], s=encoded['orig_shape'])

        freq = torch.zeros(encoded['shape'], dtype=torch.complex64,
                          device=encoded['values'].device).reshape(-1)
        freq[encoded['indices']] = encoded['values']
        freq = freq.reshape(encoded['shape'])
        return torch.fft.irfft2(freq, s=encoded['orig_shape'])

    def compression_ratio(self, weight_matrix):
        """How much compression for this matrix?"""
        original = weight_matrix.numel()
        compressed = self.n_modes * 2 + self.n_modes  # values (complex) + indices
        return original / compressed


def compress_model_as_field(state_dict, n_modes=64):
    """Compress an entire model as wave fields."""
    encoder = FieldEncoder(n_modes)
    compressed = {}
    total_orig = 0
    total_comp = 0

    for name, weight in state_dict.items():
        if weight.dim() >= 2:
            encoded = encoder.encode(weight)
            compressed[name] = encoded
            orig = weight.numel()
            comp = n_modes * 3  # approximate
            total_orig += orig
            total_comp += comp
        else:
            compressed[name] = {'raw': weight}
            total_orig += weight.numel()
            total_comp += weight.numel()

    ratio = total_orig / total_comp
    print(f"Field compression: {total_orig:,} -> {total_comp:,} ({ratio:.1f}x)")
    return compressed, ratio
