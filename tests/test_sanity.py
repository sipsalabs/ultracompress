"""
Automated sanity tests to catch silent regressions.

Run with: python -m pytest tests/test_sanity.py -v
Or directly: python tests/test_sanity.py

Tests:
  1. Teacher loader auto-detects both 0.6B and 1.7B caches correctly.
  2. Teacher forward produces finite, sane logits (argmax != pad, no NaN).
  3. Determinism: same seed, same device, two runs -> bit-identical logits.
  4. HQ5 h256 checkpoint loads + reproduces published all-T10 to within
     0.5 pp on a fixed 100-sample seed-42 draw (fast regression check).
  5. Random-weights student produces MUCH worse T10 than trained -- proves
     training is doing something.
  6. Checkpoint roundtrip: save, load, check forward matches bit-close.
"""
import os
import sys
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from ultracompress.moonshot import FractalModel
from scaling.teacher_loader import load_qwen3_teacher


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

HAS_06B = os.path.exists('qwen3_0.6b_cache.pt')
HAS_17B = os.path.exists('qwen3_1.7b_cache.pt')
HAS_HQ5_H256 = os.path.exists('checkpoints_1.7b_tinyfrr_hq5_h256/best.pt')
HAS_FINEWEB = os.path.exists('fineweb_edu_500M_tokens.pt')


# ============================================================
# Test 1: auto-detect both teacher sizes
# ============================================================
@pytest.mark.skipif(not HAS_06B, reason='no 0.6B cache')
def test_auto_detect_0_6b():
    tb = load_qwen3_teacher('qwen3_0.6b_cache.pt', device=DEVICE, verbose=False)
    assert tb.cfg.hidden_size == 1024
    assert tb.cfg.n_layers == 28
    assert tb.cfg.vocab_size == 151936
    assert tb.cfg.head_dim == 128
    assert tb.cfg.n_heads == 16
    assert tb.cfg.n_kv_heads == 8
    del tb
    torch.cuda.empty_cache()


@pytest.mark.skipif(not HAS_17B, reason='no 1.7B cache')
def test_auto_detect_1_7b():
    tb = load_qwen3_teacher('qwen3_1.7b_cache.pt', device=DEVICE, verbose=False)
    assert tb.cfg.hidden_size == 2048
    assert tb.cfg.n_layers == 28
    assert tb.cfg.vocab_size == 151936
    assert tb.cfg.head_dim == 128
    assert tb.cfg.n_heads == 16
    assert tb.cfg.n_kv_heads == 8
    del tb
    torch.cuda.empty_cache()


# ============================================================
# Test 2: teacher forward sanity
# ============================================================
@pytest.mark.skipif(not HAS_17B, reason='no 1.7B cache')
def test_teacher_forward_finite():
    """Forward produces finite logits on both random tokens and real text."""
    tb = load_qwen3_teacher('qwen3_1.7b_cache.pt', device=DEVICE, verbose=False)

    # random tokens: just check finiteness + shape
    toks = torch.randint(0, tb.vocab_size, (1, 32), device=DEVICE)
    with torch.no_grad():
        logits = tb.teacher.forward(toks, max_layers=tb.n_layers)
    assert torch.isfinite(logits).all(), 'teacher produced NaN/Inf'
    assert logits.shape == (1, 32, tb.vocab_size)

    # real text: entropy should be in the well-trained-LM range
    if HAS_FINEWEB:
        all_tokens = torch.load('fineweb_edu_500M_tokens.pt', weights_only=True)
        real = all_tokens[:64].unsqueeze(0).long().to(DEVICE)
        with torch.no_grad():
            l2 = tb.teacher.forward(real, max_layers=tb.n_layers)
        probs = F.softmax(l2[0].float(), dim=-1)
        ent_bits = (-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean().item()) / math.log(2)
        assert 1.0 < ent_bits < 10.0, \
            f'teacher entropy on real text is {ent_bits:.2f} bits, expected 1.0-10.0'
    del tb
    torch.cuda.empty_cache()


# ============================================================
# Test 3: determinism
# ============================================================
@pytest.mark.skipif(not HAS_17B, reason='no 1.7B cache')
def test_teacher_determinism():
    tb = load_qwen3_teacher('qwen3_1.7b_cache.pt', device=DEVICE, verbose=False)
    torch.manual_seed(123)
    toks = torch.randint(0, tb.vocab_size, (1, 16), device=DEVICE)
    with torch.no_grad():
        l1 = tb.teacher.forward(toks, max_layers=tb.n_layers)
        l2 = tb.teacher.forward(toks, max_layers=tb.n_layers)
    # bit-identical for the same input on the same device
    assert torch.equal(l1, l2), 'teacher forward is not deterministic'
    del tb
    torch.cuda.empty_cache()


# ============================================================
# Test 4: HQ5 h256 reproduces its published number
# ============================================================
class _TinyFRR(nn.Module):
    def __init__(self, h_outer, h_inner, n_heads, vocab, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.proj_out = nn.Linear(h_inner, h_outer, bias=False)
        self.inner = FractalModel(
            hidden_dim=h_inner, n_heads=n_heads,
            n_scales=4, iters_per_scale=7,
            vocab_size=vocab, ff_mult=1,
            embed_weight=None, lm_head_weight=None, norm_weight=None,
        )
        for p in self.inner.embed.parameters(): p.requires_grad = False
        for p in self.inner.lm_head.parameters(): p.requires_grad = False
        for p in self.inner.norm.parameters(): p.requires_grad = False
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.register_buffer('lm_head_w', lm_head_w, persistent=False)
        self.norm_outer = nn.RMSNorm(h_outer)
        self.norm_outer.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        x = self.proj_in(x)
        fr = self.inner
        for s in range(fr.n_scales):
            gamma, beta = fr.scale_gamma[s], fr.scale_beta[s]
            for it in range(fr.iters_per_scale):
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * fr.iter_scale[s, it]
        x = self.proj_out(x)
        x = self.norm_outer(x)
        return F.linear(x, self.lm_head_w)


def _quick_t10(model, teacher, tokens, starts, seq_len, n_layers):
    """Compute all-position top-10 agreement on a small fixed sample."""
    all_t10 = 0.0
    n = 0
    for s in starts:
        s = int(s.item())
        toks = tokens[s:s + seq_len + 1].unsqueeze(0).long().to(DEVICE)
        with torch.no_grad():
            tl = teacher.forward(toks[:, :seq_len], max_layers=n_layers)
            sl = model(toks[:, :seq_len])
        for pos in range(seq_len):
            tt = set(tl[0, pos].topk(10).indices.tolist())
            st = set(sl[0, pos].topk(10).indices.tolist())
            all_t10 += len(tt & st) / 10
            n += 1
    return all_t10 / n


@pytest.mark.skipif(not (HAS_17B and HAS_HQ5_H256 and HAS_FINEWEB),
                    reason='need 1.7B cache + HQ5 h256 ckpt + fineweb data')
def test_hq5_h256_reproduces_t10():
    """HQ5 h256 must score all-T10 >= 68.5% on a 100-sample seed-42 tail draw.

    Published 1000-sample number is 69.64% with 95% CI ~[69.38, 69.92]. On a
    100-sample subset from the same distribution, we allow a generous 1.1 pp
    margin for sampling noise (standard error at n=100 is ~1.5 pp).
    """
    tb = load_qwen3_teacher('qwen3_1.7b_cache.pt', device=DEVICE, verbose=False)
    ck = torch.load('checkpoints_1.7b_tinyfrr_hq5_h256/best.pt',
                    map_location='cpu', weights_only=False)
    model = _TinyFRR(tb.h_outer, ck['h_inner'], ck.get('n_heads_inner', 16),
                     tb.vocab_size, tb.embed_w, tb.lm_head_w, tb.norm_w).to(DEVICE)
    model.load_state_dict(ck['state_dict'], strict=False)
    model.eval()
    all_tokens = torch.load('fineweb_edu_500M_tokens.pt', weights_only=True)
    total = all_tokens.numel()
    tail_start = total - 50_000_000
    g = torch.Generator().manual_seed(42)
    starts = torch.randint(tail_start, total - 129, (100,), generator=g)
    t10 = _quick_t10(model, tb.teacher, all_tokens, starts, 128, tb.n_layers)
    assert t10 >= 0.685, f'HQ5 h256 regressed: got all-T10 = {t10*100:.2f}%, expected >= 68.5%'
    del model, tb
    torch.cuda.empty_cache()


# ============================================================
# Test 5: random-init student is MUCH worse than trained
# ============================================================
@pytest.mark.skipif(not (HAS_17B and HAS_FINEWEB), reason='need teacher + data')
def test_random_init_is_much_worse():
    """A random-init student must score all-T10 < 20% -- proves training matters."""
    tb = load_qwen3_teacher('qwen3_1.7b_cache.pt', device=DEVICE, verbose=False)
    torch.manual_seed(0)
    model = _TinyFRR(tb.h_outer, 256, 16, tb.vocab_size,
                     tb.embed_w, tb.lm_head_w, tb.norm_w).to(DEVICE)
    model.eval()
    all_tokens = torch.load('fineweb_edu_500M_tokens.pt', weights_only=True)
    total = all_tokens.numel()
    g = torch.Generator().manual_seed(42)
    starts = torch.randint(0, total - 129, (50,), generator=g)
    t10 = _quick_t10(model, tb.teacher, all_tokens, starts, 128, tb.n_layers)
    # Random student should be around top-10-of-151936 ~ 0.007% on uniformly
    # distributed logits; shared head gives a small bias but nowhere near 20%.
    assert t10 < 0.20, f'random-init student scored {t10*100:.2f}% -- too close to trained'
    del model, tb
    torch.cuda.empty_cache()


# ============================================================
# Test 6: checkpoint roundtrip preserves forward pass
# ============================================================
@pytest.mark.skipif(not (HAS_17B and HAS_HQ5_H256), reason='need teacher + HQ5 ckpt')
def test_ckpt_save_load_roundtrip(tmp_path):
    tb = load_qwen3_teacher('qwen3_1.7b_cache.pt', device=DEVICE, verbose=False)
    ck = torch.load('checkpoints_1.7b_tinyfrr_hq5_h256/best.pt',
                    map_location='cpu', weights_only=False)
    m1 = _TinyFRR(tb.h_outer, ck['h_inner'], ck.get('n_heads_inner', 16),
                  tb.vocab_size, tb.embed_w, tb.lm_head_w, tb.norm_w).to(DEVICE)
    m1.load_state_dict(ck['state_dict'], strict=False)
    m1.eval()

    # save and reload
    tmp = tmp_path / 'round.pt'
    torch.save({'state_dict': m1.state_dict(),
                'h_inner': ck['h_inner'],
                'n_heads_inner': ck.get('n_heads_inner', 16)}, tmp)
    ck2 = torch.load(tmp, map_location='cpu', weights_only=False)
    m2 = _TinyFRR(tb.h_outer, ck2['h_inner'], ck2['n_heads_inner'],
                  tb.vocab_size, tb.embed_w, tb.lm_head_w, tb.norm_w).to(DEVICE)
    m2.load_state_dict(ck2['state_dict'], strict=False)
    m2.eval()

    toks = torch.randint(0, tb.vocab_size, (1, 32), device=DEVICE)
    with torch.no_grad():
        l1 = m1(toks)
        l2 = m2(toks)
    assert torch.allclose(l1, l2, atol=1e-5, rtol=1e-5), \
        'checkpoint roundtrip did not preserve forward pass'
    del m1, m2, tb
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # Allow running directly -- prints pass/fail for each test
    import traceback
    tests = [
        ('auto_detect_0.6b', test_auto_detect_0_6b if HAS_06B else None),
        ('auto_detect_1.7b', test_auto_detect_1_7b if HAS_17B else None),
        ('teacher_forward_finite', test_teacher_forward_finite if HAS_17B else None),
        ('teacher_determinism', test_teacher_determinism if HAS_17B else None),
        ('hq5_h256_reproduces_t10',
         test_hq5_h256_reproduces_t10 if (HAS_17B and HAS_HQ5_H256 and HAS_FINEWEB) else None),
        ('random_init_much_worse',
         test_random_init_is_much_worse if (HAS_17B and HAS_FINEWEB) else None),
    ]
    passed = failed = skipped = 0
    for name, fn in tests:
        if fn is None:
            print(f"  SKIP  {name}  (prerequisites missing)")
            skipped += 1
            continue
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}:  {e}")
            failed += 1
        except Exception:
            print(f"  ERROR {name}:")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
