"""
Activation-Aware Quantization (AWQ-style)

Key insight from GPTQ/AWQ: use activation statistics to decide which channels
deserve more precision. Protect high-activation channels, compress the rest.
"""
import torch
from dataclasses import dataclass
from typing import Dict

@dataclass
class ActivationProfile:
    """Per-layer activation stats from calibration."""
    mean_mag: torch.Tensor; max_mag: torch.Tensor; variance: torch.Tensor; n_samples: int = 0
    @property
    def importance(self) -> torch.Tensor:
        s = self.mean_mag * (1.0 + self.variance.sqrt()); return s / (s.max() + 1e-10)

class ActivationProfiler:
    """Run calibration data through model, record per-channel activation magnitudes."""
    def __init__(self): self.hooks, self.stats = [], {}

    def profile(self, model, calibration_data) -> Dict[str, ActivationProfile]:
        acc = {}
        def make_hook(name):
            def hook(mod, inp, out):
                x = out.detach().float()
                if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
                d = x.shape[-1]
                if name not in acc:
                    acc[name] = [torch.zeros(d, device=x.device) for _ in range(3)] + [0]
                a = acc[name]; mag = x.abs()
                a[0] += mag.sum(0); a[1] += (x**2).sum(0)
                a[2] = torch.max(a[2], mag.max(0).values); a[3] += x.shape[0]
            return hook
        for n, m in model.named_modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.ndim == 2:
                self.hooks.append(m.register_forward_hook(make_hook(n)))
        with torch.no_grad():
            for b in (calibration_data if isinstance(calibration_data, list) else [calibration_data]):
                model(b)
        for h in self.hooks: h.remove()
        self.hooks.clear()
        for name, (s, sq, mx, n) in acc.items():
            mean = s / n; var = (sq / n - mean**2).clamp(min=0)
            self.stats[name] = ActivationProfile(mean, mx, var, n)
        return self.stats

class AWQQuantizer:
    """Mixed-precision quantization guided by activation importance."""
    def quantize(self, weight: torch.Tensor, importance: torch.Tensor, target_bits=3.0):
        out_f = weight.shape[0]
        imp = importance[:out_f].to(weight.device) if len(importance) >= out_f \
            else importance.repeat((out_f // len(importance)) + 1)[:out_f]
        bit_map = torch.full((out_f,), 2, dtype=torch.int32)
        budget, used = target_bits * out_f, 2.0 * out_f
        for idx in imp.argsort(descending=True):
            if used >= budget: break
            gain = min(6, budget - used); bit_map[idx] = 2 + int(gain); used += int(gain)
        codes, scales = [], []
        for r in range(out_f):
            levels = (1 << bit_map[r].item()) - 1; row = weight[r].float()
            s = row.abs().max() / (levels / 2 + 1e-10)
            codes.append(((row / (s+1e-10)).round().clamp(-levels//2, levels//2)).to(torch.int8))
            scales.append(s)
        return {"codes": torch.stack(codes), "scales": torch.tensor(scales),
                "bit_map": bit_map, "avg_bits": bit_map.float().mean().item()}

class CalibrationDataGenerator:
    """Generate synthetic calibration data via bootstrapping (no real data needed)."""
    @staticmethod
    def generate(model, hidden_size, n_samples=8, seq_len=32, n_rounds=3, device="cpu"):
        x = torch.randn(n_samples, seq_len, hidden_size, device=device) * 0.02
        with torch.no_grad():
            for _ in range(n_rounds):
                try:
                    out = model(inputs_embeds=x) if hasattr(model, 'forward') else model(x)
                    logits = out.logits if hasattr(out, 'logits') else out
                    if logits.shape[-1] != hidden_size: break
                    x = logits.float().detach(); x = x / (x.std() + 1e-6) * 0.02
                except Exception: break
        return x
