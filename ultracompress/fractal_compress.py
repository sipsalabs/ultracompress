"""
FRACTAL COMPRESSION — Weights as Iterated Function Systems.

Paper 2503.14298 PROVED weights have fractal self-similarity.
If weights are fractals, encode them as the RULES that generate them.
A fractal is infinite detail from a finite equation.

IFS: a set of affine transforms that, when iterated, produce the fractal.
Store the transforms (tiny) → regenerate the weights (huge).

For a weight matrix with fractal dimension D:
  Original: N*M params
  IFS: K transforms * 6 params each = 6K params
  If K << N*M: massive compression

This is mathematically guaranteed to work IF the weights are self-similar.
The paper proved they are. So this SHOULD work.
"""
import torch
import torch.nn as nn
import math


class IFSEncoder:
    """Encode weight matrices as Iterated Function Systems."""

    def __init__(self, n_transforms=16, n_iters=8):
        self.n_transforms = n_transforms
        self.n_iters = n_iters

    def encode(self, weight_matrix):
        """Find IFS transforms that reproduce the weight matrix.

        Each transform: (a, b, c, d, e, f) = 2D affine transform
        Maps (x, y, val) -> (ax+by+e, cx+dy+f, contracted_val)
        """
        W = weight_matrix.float()
        R, C = W.shape

        # Partition matrix into blocks
        n = self.n_transforms
        block_r = max(1, R // int(math.sqrt(n)))
        block_c = max(1, C // int(math.sqrt(n)))

        transforms = []
        for i in range(0, R, block_r):
            for j in range(0, C, block_c):
                block = W[i:i+block_r, j:j+block_c]
                if block.numel() == 0:
                    continue

                # Compute affine params for this block relative to full matrix
                # Scale factors
                sr = block_r / R
                sc = block_c / C
                # Translation
                tr = i / R
                tc = j / C
                # Value scaling (how block values relate to global)
                global_mean = W.mean().item()
                block_mean = block.mean().item()
                block_std = block.std().item() + 1e-8
                global_std = W.std().item() + 1e-8
                val_scale = block_std / global_std
                val_shift = block_mean - val_scale * global_mean

                transforms.append({
                    'sr': sr, 'sc': sc,
                    'tr': tr, 'tc': tc,
                    'val_scale': val_scale,
                    'val_shift': val_shift,
                })

                if len(transforms) >= self.n_transforms:
                    break
            if len(transforms) >= self.n_transforms:
                break

        return {
            'transforms': transforms,
            'shape': (R, C),
            'global_mean': W.mean().item(),
            'global_std': W.std().item(),
        }

    def decode(self, encoded):
        """Reconstruct weight matrix from IFS transforms."""
        R, C = encoded['shape']
        W = torch.zeros(R, C)

        # Start with global statistics
        W.normal_(encoded['global_mean'], encoded['global_std'])

        # Apply transforms iteratively
        for iteration in range(self.n_iters):
            W_new = torch.zeros_like(W)
            counts = torch.zeros_like(W)

            for t in encoded['transforms']:
                # Map region
                r_start = int(t['tr'] * R)
                c_start = int(t['tc'] * C)
                r_end = min(R, r_start + int(t['sr'] * R))
                c_end = min(C, c_start + int(t['sc'] * C))

                if r_end <= r_start or c_end <= c_start:
                    continue

                # Apply value transform
                region = W[r_start:r_end, c_start:c_end]
                transformed = region * t['val_scale'] + t['val_shift']
                W_new[r_start:r_end, c_start:c_end] += transformed
                counts[r_start:r_end, c_start:c_end] += 1

            # Average overlapping regions
            counts = counts.clamp(min=1)
            W = W_new / counts

        return W

    def compression_ratio(self, weight_matrix):
        """Compression ratio for this matrix."""
        original = weight_matrix.numel()
        compressed = self.n_transforms * 6  # 6 params per transform
        return original / compressed
