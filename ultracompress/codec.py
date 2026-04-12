"""
ULTRACOMPRESS CODEC — Four novel compression approaches for existing model weights.

No retraining. Pure compression. Take weights in, smaller representation out.

Approach 1: ALGEBRAIC V2
  SVD with verbatim norms + low-rank basis vectors.
  Fix from tournament: norms stored exactly, basis vectors themselves compressed.

Approach 2: NEURAL WEIGHT CODEC (JPEG for weights)
  Transform to frequency domain (DCT), quantize coefficients, entropy code.
  Like JPEG but designed for weight tensor statistics.

Approach 3: WEIGHT DNA
  Find repeating patterns in weights, encode as compact rules.
  Like run-length encoding but for mathematical patterns.
  Detects: cross-layer deltas, low-rank structure, periodic patterns, scale relationships.

Approach 4: STACKED PIPELINE
  Quantization (base) + SVD (cross-layer) + sparse corrections.
  Each stage compresses what the previous one missed.
  Like video codecs: keyframes + motion vectors + residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ================================================================
# APPROACH 1: ALGEBRAIC V2 (SVD + verbatim norms + low-rank basis)
# ================================================================

@dataclass
class AlgebraicV2Result:
    """Compressed representation of a model's weights."""
    norms: Dict[str, torch.Tensor]  # Stored verbatim (tiny, critical)
    bases: Dict[str, Tuple[torch.Tensor, torch.Tensor]]  # Low-rank: (U, V) per type
    coefficients: Dict[str, torch.Tensor]  # Per-layer mixing coefficients
    sparse_indices: Dict[str, torch.Tensor]  # Top-k residual positions
    sparse_values: Dict[str, torch.Tensor]  # Top-k residual values
    metadata: dict  # Shapes, config, etc.


class AlgebraicV2:
    """SVD compression with verbatim norms and low-rank basis vectors."""

    def __init__(self, n_basis=8, basis_rank=32, sparse_ratio=0.01):
        self.n_basis = n_basis
        self.basis_rank = basis_rank
        self.sparse_ratio = sparse_ratio

    def compress(self, matrix_stacks: Dict[str, torch.Tensor],
                 norm_weights: Dict[str, torch.Tensor]) -> AlgebraicV2Result:
        """Compress organized weight matrices.

        matrix_stacks: {type_name: (n_layers, rows, cols)} — big weight matrices
        norm_weights: {key: tensor} — norm weights stored verbatim
        """
        bases = {}
        coefficients = {}
        sparse_indices = {}
        sparse_values = {}
        metadata = {}

        total_original = 0
        total_compressed = 0

        for mtype, stack in matrix_stacks.items():
            n_layers, rows, cols = stack.shape
            total_original += stack.numel()

            # Flatten layers: (n_layers, rows*cols)
            flat = stack.reshape(n_layers, -1)

            # SVD across layers
            U, S, Vh = torch.linalg.svd(flat, full_matrices=False)

            # Keep top n_basis components
            k = min(self.n_basis, U.shape[1])
            coeffs = U[:, :k] * S[:k].unsqueeze(0)  # (n_layers, k)
            basis_full = Vh[:k]  # (k, rows*cols)

            # NOW: compress the basis vectors themselves via low-rank factorization
            # Each basis vector (1, rows*cols) reshaped to (rows, cols) then SVD'd
            basis_U_list = []
            basis_V_list = []
            for b in range(k):
                bmat = basis_full[b].reshape(rows, cols)
                bU, bS, bVh = torch.linalg.svd(bmat, full_matrices=False)
                r = min(self.basis_rank, bU.shape[1])
                # Store as (rows, r) and (r, cols)
                basis_U_list.append(bU[:, :r] * bS[:r].unsqueeze(0).sqrt())
                basis_V_list.append(bVh[:r] * bS[:r].unsqueeze(1).sqrt())

            basis_U = torch.stack(basis_U_list)  # (k, rows, r)
            basis_V = torch.stack(basis_V_list)  # (k, r, cols)

            # Reconstruct and compute residual
            reconstructed = torch.zeros_like(flat)
            for b in range(k):
                bmat = (basis_U[b] @ basis_V[b]).reshape(-1)  # (rows*cols,)
                reconstructed += coeffs[:, b:b+1] * bmat.unsqueeze(0)

            residual = flat - reconstructed

            # Sparse: keep top-p% of residual per layer
            n_sparse_per_layer = max(1, int(rows * cols * self.sparse_ratio))
            sp_idx_list = []
            sp_val_list = []
            for li in range(n_layers):
                _, top_idx = residual[li].abs().topk(n_sparse_per_layer)
                sp_idx_list.append(top_idx)
                sp_val_list.append(residual[li][top_idx])

            bases[mtype] = (basis_U, basis_V)
            coefficients[mtype] = coeffs
            sparse_indices[mtype] = torch.stack(sp_idx_list)
            sparse_values[mtype] = torch.stack(sp_val_list)

            # Count compressed size
            comp_size = (basis_U.numel() + basis_V.numel() +
                        coeffs.numel() +
                        sparse_indices[mtype].numel() + sparse_values[mtype].numel())
            total_compressed += comp_size

            # Report per-type
            recon_with_sparse = reconstructed.clone()
            for li in range(n_layers):
                recon_with_sparse[li][sp_idx_list[li]] += sp_val_list[li]
            rmse = (flat - recon_with_sparse).pow(2).mean().sqrt().item()
            ratio = stack.numel() / comp_size
            print(f"  {mtype}: {ratio:.1f}x compression, RMSE={rmse:.6f}")

        # Norms are tiny — count them
        norm_size = sum(v.numel() for v in norm_weights.values())
        total_compressed += norm_size

        metadata['total_original'] = total_original
        metadata['total_compressed'] = total_compressed
        metadata['norm_size'] = norm_size

        overall_ratio = total_original / total_compressed
        print(f"  Overall: {overall_ratio:.1f}x ({total_compressed*4/1e6:.1f} MB)")
        print(f"  Norms: {norm_size:,} params ({norm_size*4/1e3:.1f} KB) — stored verbatim")

        return AlgebraicV2Result(
            norms=norm_weights,
            bases=bases,
            coefficients=coefficients,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            metadata=metadata,
        )

    def decompress(self, result: AlgebraicV2Result, matrix_shapes: dict) -> Dict[str, torch.Tensor]:
        """Reconstruct all weight matrices from compressed representation."""
        reconstructed = {}

        for mtype in result.bases:
            basis_U, basis_V = result.bases[mtype]
            coeffs = result.coefficients[mtype]
            sp_idx = result.sparse_indices[mtype]
            sp_val = result.sparse_values[mtype]

            k = coeffs.shape[1]
            n_layers = coeffs.shape[0]
            rows, cols = matrix_shapes[mtype]

            for li in range(n_layers):
                # Reconstruct from low-rank basis
                mat = torch.zeros(rows, cols, device=coeffs.device)
                for b in range(k):
                    mat += coeffs[li, b] * (basis_U[b] @ basis_V[b])

                # Add sparse correction
                flat = mat.reshape(-1)
                flat[sp_idx[li]] += sp_val[li]
                mat = flat.reshape(rows, cols)

                reconstructed[f'{mtype}_layer{li}'] = mat

        return reconstructed


# ================================================================
# APPROACH 2: NEURAL WEIGHT CODEC (JPEG for weights)
# ================================================================

class NeuralWeightCodec:
    """JPEG-style compression for neural network weight matrices.

    Pipeline per matrix:
    1. Block decomposition (split into 8x8 or 16x16 blocks)
    2. DCT transform (like JPEG) — converts spatial patterns to frequency
    3. Quantize DCT coefficients (aggressive for high-freq, gentle for low-freq)
    4. The quantized coefficients ARE the compressed representation

    Key insight: Weight matrices have most energy in low frequencies
    (smooth spatial patterns). High-frequency noise can be thrown away.
    """

    def __init__(self, block_size=16, quality=50):
        """
        block_size: size of DCT blocks
        quality: 1-100, like JPEG quality. Lower = more compression, more loss.
        """
        self.block_size = block_size
        self.quality = quality

    def _dct2d(self, block):
        """2D DCT of a block (using torch FFT)."""
        # Type-II DCT via FFT
        N = block.shape[0]
        M = block.shape[1]
        # Row-wise DCT
        v = torch.fft.fft(block, dim=1)
        result = v.real * torch.cos(torch.arange(M, device=block.device).float() * math.pi / (2*M)).unsqueeze(0)
        # Simplified: use matrix multiplication with DCT basis
        return self._dct_matrix(block)

    def _dct_matrix(self, block):
        """Compute 2D DCT using explicit basis (more accurate than FFT approach)."""
        N, M = block.shape
        # Build DCT basis matrices
        n = torch.arange(N, device=block.device).float()
        k = torch.arange(N, device=block.device).float()
        dct_rows = torch.cos(math.pi * (2*n.unsqueeze(1) + 1) * k.unsqueeze(0) / (2*N))
        dct_rows[0] *= 1/math.sqrt(N)
        dct_rows[1:] *= math.sqrt(2/N)

        m = torch.arange(M, device=block.device).float()
        l = torch.arange(M, device=block.device).float()
        dct_cols = torch.cos(math.pi * (2*m.unsqueeze(1) + 1) * l.unsqueeze(0) / (2*M))
        dct_cols[0] *= 1/math.sqrt(M)
        dct_cols[1:] *= math.sqrt(2/M)

        return dct_rows.T @ block @ dct_cols

    def _idct_matrix(self, coeffs):
        """Inverse 2D DCT."""
        N, M = coeffs.shape
        n = torch.arange(N, device=coeffs.device).float()
        k = torch.arange(N, device=coeffs.device).float()
        dct_rows = torch.cos(math.pi * (2*n.unsqueeze(1) + 1) * k.unsqueeze(0) / (2*N))
        dct_rows[0] *= 1/math.sqrt(N)
        dct_rows[1:] *= math.sqrt(2/N)

        m = torch.arange(M, device=coeffs.device).float()
        l = torch.arange(M, device=coeffs.device).float()
        dct_cols = torch.cos(math.pi * (2*m.unsqueeze(1) + 1) * l.unsqueeze(0) / (2*M))
        dct_cols[0] *= 1/math.sqrt(M)
        dct_cols[1:] *= math.sqrt(2/M)

        return dct_rows @ coeffs @ dct_cols.T

    def _quantization_matrix(self, block_size, device):
        """JPEG-style quantization matrix adapted for weight statistics.

        Low-frequency coefficients get fine quantization (important).
        High-frequency coefficients get coarse quantization (noise).
        """
        # Distance from top-left (DC component)
        r = torch.arange(block_size, device=device).float()
        c = torch.arange(block_size, device=device).float()
        dist = (r.unsqueeze(1) + c.unsqueeze(0)) / (2 * block_size - 2)

        # Quantization step increases with frequency distance
        # quality 100 = minimal quantization, quality 1 = maximum
        if self.quality >= 50:
            scale = (100 - self.quality) / 50
        else:
            scale = 50 / self.quality

        q_matrix = 1 + dist * 50 * scale
        return q_matrix.clamp(min=1)

    def compress_matrix(self, weight: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compress a single weight matrix using DCT codec.

        Returns quantized DCT coefficients and metadata for decompression.
        """
        rows, cols = weight.shape
        bs = self.block_size
        device = weight.device

        # Pad to multiple of block_size
        pad_rows = (bs - rows % bs) % bs
        pad_cols = (bs - cols % bs) % bs
        padded = F.pad(weight, (0, pad_cols, 0, pad_rows))
        pr, pc = padded.shape

        # Store global statistics for denormalization
        w_mean = weight.mean().item()
        w_std = weight.std().item()
        padded_norm = (padded - w_mean) / (w_std + 1e-8)

        # Block-wise DCT
        n_blocks_r = pr // bs
        n_blocks_c = pc // bs
        blocks = padded_norm.reshape(n_blocks_r, bs, n_blocks_c, bs).permute(0, 2, 1, 3)
        # blocks: (n_blocks_r, n_blocks_c, bs, bs)

        q_matrix = self._quantization_matrix(bs, device)

        dct_blocks = torch.zeros_like(blocks)
        for i in range(n_blocks_r):
            for j in range(n_blocks_c):
                dct_blocks[i, j] = self._dct_matrix(blocks[i, j])

        # Quantize
        quantized = torch.round(dct_blocks / q_matrix)

        # Count non-zero coefficients (effective compression)
        n_nonzero = (quantized != 0).sum().item()
        n_total = quantized.numel()

        meta = {
            'rows': rows, 'cols': cols,
            'pad_rows': pad_rows, 'pad_cols': pad_cols,
            'w_mean': w_mean, 'w_std': w_std,
            'block_size': bs,
            'n_nonzero': n_nonzero, 'n_total': n_total,
        }

        return quantized, meta

    def decompress_matrix(self, quantized: torch.Tensor, meta: dict) -> torch.Tensor:
        """Decompress from quantized DCT coefficients."""
        bs = meta['block_size']
        device = quantized.device

        q_matrix = self._quantization_matrix(bs, device)

        # Dequantize
        dct_blocks = quantized * q_matrix

        n_blocks_r, n_blocks_c = dct_blocks.shape[:2]

        # Inverse DCT per block
        blocks = torch.zeros_like(dct_blocks)
        for i in range(n_blocks_r):
            for j in range(n_blocks_c):
                blocks[i, j] = self._idct_matrix(dct_blocks[i, j])

        # Reassemble
        padded = blocks.permute(0, 2, 1, 3).reshape(
            n_blocks_r * bs, n_blocks_c * bs)

        # Denormalize
        padded = padded * (meta['w_std'] + 1e-8) + meta['w_mean']

        # Remove padding
        return padded[:meta['rows'], :meta['cols']]

    def compress_model(self, matrix_stacks, norm_weights):
        """Compress all weight matrices."""
        compressed = {}
        total_original = 0
        total_nonzero = 0

        for mtype, stack in matrix_stacks.items():
            n_layers = stack.shape[0]
            layer_results = []
            for li in range(n_layers):
                q, meta = self.compress_matrix(stack[li])
                layer_results.append((q, meta))
                total_original += stack[li].numel()
                total_nonzero += meta['n_nonzero']
            compressed[mtype] = layer_results

            ratio = stack.numel() / sum(r[1]['n_nonzero'] for r in layer_results)
            print(f"  {mtype}: {ratio:.1f}x (nonzero coeffs)")

        norm_size = sum(v.numel() for v in norm_weights.values())
        effective_size = total_nonzero + norm_size  # Each nonzero needs index + value
        overall = total_original / effective_size
        print(f"  Overall: {overall:.1f}x ({effective_size*4/1e6:.1f} MB effective)")

        return compressed, norm_weights

    def decompress_model(self, compressed, norm_weights, n_layers=28):
        """Decompress all matrices."""
        reconstructed = {}
        for mtype, layer_results in compressed.items():
            for li, (q, meta) in enumerate(layer_results):
                mat = self.decompress_matrix(q, meta)
                reconstructed[f'{mtype}_layer{li}'] = mat

        return reconstructed, norm_weights


# ================================================================
# APPROACH 3: WEIGHT DNA (Pattern Detection & Encoding)
# ================================================================

class WeightDNA:
    """Encode weight matrices as compact pattern descriptions.

    Analyzes weight structure and encodes it as rules:
    - Cross-layer deltas (layer N ≈ layer N-1 + small delta)
    - Low-rank factorization (W ≈ A @ B)
    - Scale relationships (layer N ≈ alpha * layer M)
    - Periodic patterns (oscillating weight values)

    The "DNA" is the set of rules + parameters. Much smaller than raw weights.
    """

    def __init__(self, delta_rank=16, scale_threshold=0.95):
        self.delta_rank = delta_rank
        self.scale_threshold = scale_threshold

    def analyze_cross_layer(self, stack: torch.Tensor) -> dict:
        """Find optimal encoding: keyframe + deltas vs independent."""
        n_layers = stack.shape[0]
        flat = stack.reshape(n_layers, -1)

        # Strategy 1: Store layer 0 as keyframe, rest as deltas from previous
        deltas = flat[1:] - flat[:-1]  # (n_layers-1, rows*cols)

        # Low-rank compress each delta
        delta_compressed = []
        delta_sizes = []
        for i in range(deltas.shape[0]):
            d = deltas[i].reshape(stack.shape[1], stack.shape[2])
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            r = min(self.delta_rank, U.shape[1])
            dU = U[:, :r] * S[:r].unsqueeze(0).sqrt()
            dV = Vh[:r] * S[:r].unsqueeze(1).sqrt()
            delta_compressed.append((dU, dV))
            delta_sizes.append(dU.numel() + dV.numel())

        # Keyframe itself — low-rank
        kf = flat[0].reshape(stack.shape[1], stack.shape[2])
        kU, kS, kVh = torch.linalg.svd(kf, full_matrices=False)
        kf_rank = min(self.delta_rank * 2, kU.shape[1])  # Keyframe gets more rank
        kf_U = kU[:, :kf_rank] * kS[:kf_rank].unsqueeze(0).sqrt()
        kf_V = kVh[:kf_rank] * kS[:kf_rank].unsqueeze(1).sqrt()

        total_size = kf_U.numel() + kf_V.numel() + sum(delta_sizes)
        ratio = stack.numel() / total_size

        # Compute reconstruction error
        recon = torch.zeros_like(flat)
        recon[0] = (kf_U @ kf_V).reshape(-1)
        for i in range(len(delta_compressed)):
            dU, dV = delta_compressed[i]
            recon[i+1] = recon[i] + (dU @ dV).reshape(-1)
        rmse = (flat - recon).pow(2).mean().sqrt().item()

        return {
            'keyframe': (kf_U, kf_V),
            'deltas': delta_compressed,
            'ratio': ratio,
            'rmse': rmse,
            'total_size': total_size,
        }

    def compress(self, matrix_stacks, norm_weights):
        """Compress using keyframe + delta encoding."""
        compressed = {}
        total_original = 0
        total_compressed = 0

        for mtype, stack in matrix_stacks.items():
            result = self.analyze_cross_layer(stack)
            compressed[mtype] = result
            total_original += stack.numel()
            total_compressed += result['total_size']
            print(f"  {mtype}: {result['ratio']:.1f}x, RMSE={result['rmse']:.6f}")

        norm_size = sum(v.numel() for v in norm_weights.values())
        total_compressed += norm_size
        overall = total_original / total_compressed
        print(f"  Overall: {overall:.1f}x ({total_compressed*4/1e6:.1f} MB)")

        return compressed, norm_weights

    def decompress(self, compressed, norm_weights, n_layers=28):
        """Reconstruct from keyframe + deltas."""
        reconstructed = {}
        for mtype, result in compressed.items():
            kf_U, kf_V = result['keyframe']
            prev = (kf_U @ kf_V)
            reconstructed[f'{mtype}_layer0'] = prev

            for i, (dU, dV) in enumerate(result['deltas']):
                current = prev + (dU @ dV)
                reconstructed[f'{mtype}_layer{i+1}'] = current
                prev = current

        return reconstructed, norm_weights


# ================================================================
# APPROACH 4: STACKED PIPELINE (Quantize + SVD + Sparse)
# ================================================================

class StackedPipeline:
    """Multi-stage compression like video codecs.

    Stage 1: Quantize to N bits (proven, preserves most quality)
    Stage 2: SVD across layers on quantized weights (exploit redundancy)
    Stage 3: Sparse code the residual (capture fine detail)

    Each stage compresses what the previous one couldn't.
    """

    def __init__(self, quant_bits=4, svd_rank=8, sparse_ratio=0.005):
        self.quant_bits = quant_bits
        self.svd_rank = svd_rank
        self.sparse_ratio = sparse_ratio

    def _quantize(self, w, bits):
        """Uniform quantization to N bits."""
        n_levels = 2 ** bits
        wmin, wmax = w.min(), w.max()
        scale = (wmax - wmin) / (n_levels - 1)
        if scale == 0:
            return w, wmin, scale
        quantized = torch.round((w - wmin) / scale)
        return quantized, wmin, scale

    def _dequantize(self, q, wmin, scale):
        return q * scale + wmin

    def compress(self, matrix_stacks, norm_weights):
        """Three-stage compression."""
        results = {}
        total_original = 0
        total_compressed = 0

        for mtype, stack in matrix_stacks.items():
            n_layers, rows, cols = stack.shape
            total_original += stack.numel()

            # Stage 1: Quantize each layer
            quantized_layers = []
            quant_params = []
            for li in range(n_layers):
                q, wmin, scale = self._quantize(stack[li], self.quant_bits)
                quantized_layers.append(q)
                quant_params.append((wmin.item(), scale.item()))

            # Dequantize for residual computation
            dequantized = torch.stack([
                self._dequantize(quantized_layers[li],
                                quant_params[li][0], quant_params[li][1])
                for li in range(n_layers)
            ])

            # Stage 1 size: N bits per weight + 2 floats per layer (scale, zero)
            stage1_size = n_layers * rows * cols * self.quant_bits / 32 + n_layers * 2

            # Stage 2: SVD on quantized weights to find cross-layer structure
            flat_q = dequantized.reshape(n_layers, -1)
            U, S, Vh = torch.linalg.svd(flat_q, full_matrices=False)
            r = min(self.svd_rank, U.shape[1])

            # Low-rank approximation of quantized weights
            svd_approx = (U[:, :r] * S[:r]) @ Vh[:r]
            svd_residual = flat_q - svd_approx.reshape(n_layers, -1)

            # Full residual from original
            full_residual = stack.reshape(n_layers, -1) - svd_approx.reshape(n_layers, -1)

            # Stage 2 size: SVD components
            stage2_size = r * (n_layers + rows * cols)

            # Stage 3: Sparse on full residual (original - SVD approximation)
            n_sparse = max(1, int(full_residual.numel() * self.sparse_ratio))
            residual_flat = full_residual.reshape(-1)
            _, top_idx = residual_flat.abs().topk(n_sparse)
            sparse_vals = residual_flat[top_idx]

            stage3_size = n_sparse * 2  # index + value

            total_size = stage1_size + stage2_size + stage3_size
            total_compressed += total_size

            # Compute final reconstruction quality
            final_recon = svd_approx.reshape(n_layers, -1).clone()
            final_recon.reshape(-1)[top_idx] += sparse_vals
            rmse = (stack.reshape(n_layers, -1) - final_recon).pow(2).mean().sqrt().item()
            ratio = stack.numel() / total_size

            results[mtype] = {
                'svd_U': U[:, :r] * S[:r],
                'svd_Vh': Vh[:r],
                'sparse_idx': top_idx,
                'sparse_val': sparse_vals,
                'quant_params': quant_params,
                'shape': (n_layers, rows, cols),
            }
            print(f"  {mtype}: {ratio:.1f}x (Q{self.quant_bits}+SVD{r}+sparse), RMSE={rmse:.6f}")

        norm_size = sum(v.numel() for v in norm_weights.values())
        total_compressed += norm_size
        overall = total_original / total_compressed
        print(f"  Overall: {overall:.1f}x ({total_compressed*4/1e6:.1f} MB)")

        return results, norm_weights

    def decompress(self, results, norm_weights, n_layers=28):
        """Reconstruct from stacked representation."""
        reconstructed = {}
        for mtype, data in results.items():
            n_layers, rows, cols = data['shape']
            flat = (data['svd_U'] @ data['svd_Vh']).reshape(n_layers, -1)

            # Add sparse
            full_flat = flat.reshape(-1).clone()
            full_flat[data['sparse_idx']] += data['sparse_val']
            flat = full_flat.reshape(n_layers, rows * cols)

            for li in range(n_layers):
                reconstructed[f'{mtype}_layer{li}'] = flat[li].reshape(rows, cols)

        return reconstructed, norm_weights
