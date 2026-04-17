"""
CEILING DIAGNOSTIC — is 68% a data/train ceiling or an architecture ceiling?

Three diagnostics, all on the SAME 1000-sample hires eval set:

(A) TEACHER SELF-CONSISTENCY
    Run teacher twice with different dropout/random seed masks → top-1/top-10
    overlap of teacher-with-itself. This is the irreducible upper bound.
    (In eval mode teacher is deterministic, so this should be 100%. If so,
    we skip to B.)

(B) TEACHER-TRUNCATED BASELINE
    Run teacher with only the first K ∈ {7, 14, 21, 28} layers. This shows
    how much the LAST layers contribute to top-10 overlap. If teacher[21]
    already gets T10=85%, then the last 7 layers only add 15pp — meaning
    our shared-block student at "14 effective layers" might be closer to
    the ceiling than we thought.

(C) STUDENT ERROR ANALYSIS
    For our pure_kl record, on positions where student is WRONG, how wrong
    is it? Check:
      - Is teacher-top-1 in student's top-10? (near-miss)
      - Is teacher-top-10 confidence concentrated or flat? (ambiguous ctx)
      - What's the KL per token? (distributional gap)
    If 90% of misses are "teacher has flat distribution" (entropy>4), then
    we're hitting INPUT AMBIGUITY, not model capacity.

Output: a single decisive number for each, printed to stdout.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock

DEVICE = 'cuda:0'
SEQ_LEN = 64
N_SAMPLES = 500       # 500 samples * 64 positions = 32k evaluation points
N_TEACHER_LAYERS = 28

print("=" * 70)
print("CEILING DIAGNOSTIC")
print("  Goal: determine if T10=68% ceiling is data, KL, or architecture wall")
print("=" * 70)

# ── Load teacher ──
print("\nLoading teacher...")
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight',
    'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight',
    'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight',
    'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(2048)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(N_TEACHER_LAYERS):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()
del wd

hidden = gd['token_embd.weight'].shape[1]
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(
    n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
    hidden_size=hidden, intermediate_size=hidden * 3,
    vocab_size=vocab_size, head_dim=hidden // 16,
)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)

# ── Data ──
print("Loading data...")
tokens_all = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)

torch.manual_seed(42)
starts = torch.randint(0, tokens_all.numel() - SEQ_LEN, (N_SAMPLES,)).tolist()
batches = [tokens_all[s:s+SEQ_LEN].long().unsqueeze(0).to(DEVICE) for s in starts]

# =========================================================
# (B) TEACHER-TRUNCATED BASELINE — upper bound at K layers
# =========================================================
print("\n" + "=" * 70)
print("(B) TEACHER-TRUNCATED: top-10 overlap vs full 28-layer teacher")
print("    Shows diminishing returns of deep layers — gives architectural context")
print("=" * 70)

@torch.no_grad()
def overlap_full_vs_k(K_list):
    """For each K, compute all-pos T1/T10 of teacher[K] against teacher[28]."""
    results = {}
    # Compute full teacher logits once
    full_logits = []
    for b in batches:
        full_logits.append(teacher.forward(b, max_layers=N_TEACHER_LAYERS))
    for K in K_list:
        t1, t10, n = 0, 0, 0
        for i, b in enumerate(batches):
            part = teacher.forward(b, max_layers=K)
            full = full_logits[i]
            for pos in range(SEQ_LEN):
                f_top = full[0, pos].topk(10).indices
                p_top = part[0, pos].topk(10).indices
                t1 += int(p_top[0] == f_top[0])
                t10 += len(set(f_top.tolist()) & set(p_top.tolist())) / 10
                n += 1
        results[K] = (t1/n, t10/n)
        print(f"   teacher[{K:2d}] vs teacher[28]:  T1={t1/n*100:5.1f}%  T10={t10/n*100:5.1f}%")
    return results

trunc_results = overlap_full_vs_k([7, 14, 21, 24, 26, 27])

# =========================================================
# (C) STUDENT ERROR ANALYSIS on pure_kl record
# =========================================================
print("\n" + "=" * 70)
print("(C) STUDENT ERROR ANALYSIS — pure_kl record vs teacher")
print("    For each MISS, is teacher confident or ambiguous?")
print("=" * 70)

# Build FRR-1.7B as pure_kl (no rotations, no LoRA) with per-layer γ/β
class PureFRR(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale, vocab_size,
                 embed_w, lm_head_w, norm_w):
        super().__init__()
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.block = FractalBlock(hidden_dim, n_heads, 1)
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_w, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        self.norm.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        li = 0
        for s in range(self.n_scales):
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[li]
                beta = self.layer_beta[li]
                iter_s = self.iter_scale[s, it]
                block_out = self.block(x, gamma, beta)
                x = x + (block_out - x) * iter_s
                li += 1
        return self.lm_head(self.norm(x))

embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)

# Try to find the pure_kl record checkpoint
candidates = [
    'checkpoints_1.7b_pure_kl/pure_kl_step5000.pt',
    'checkpoints_1.7b_pure_kl/pure_kl_best.pt',
    'checkpoints_1.7b_pure_kl/pure_kl_final.pt',
]
record_ckpt = None
for c in candidates:
    if os.path.exists(c):
        record_ckpt = c
        break

if record_ckpt is None:
    print("   No pure_kl checkpoint found — listing checkpoints_*:")
    for d in os.listdir('.'):
        if d.startswith('checkpoints_'):
            print(f"     {d}: {os.listdir(d)[:5]}")
    print("   Skipping student error analysis")
else:
    print(f"   Loading student record: {record_ckpt}")
    student = PureFRR(hidden, 16, 4, 7, vocab_size, embed_w, lm_head_w, norm_w).to(DEVICE)
    sd = torch.load(record_ckpt, map_location=DEVICE, weights_only=True)
    # Filter and load
    student_sd = student.state_dict()
    loaded, skipped = 0, 0
    for k, v in sd.items():
        if k in student_sd and student_sd[k].shape == v.shape:
            student_sd[k] = v
            loaded += 1
        else:
            skipped += 1
    # Handle scale_gamma/scale_beta -> per-layer expansion if present
    if 'scale_gamma' in sd and 'layer_gamma' not in sd:
        print("   Expanding scale_gamma/beta -> per-layer")
        sg = sd['scale_gamma']; sb = sd['scale_beta']
        for s in range(4):
            for it in range(7):
                student_sd['layer_gamma'][s*7+it] = sg[s]
                student_sd['layer_beta'][s*7+it] = sb[s]
        if 'iter_scale' in sd:
            student_sd['iter_scale'] = sd['iter_scale']
    student.load_state_dict(student_sd)
    student.eval()
    print(f"   Loaded {loaded} params, skipped {skipped}")

    with torch.no_grad():
        n_tot = 0
        n_correct_top1 = 0
        n_correct_top10 = 0
        miss_info = {
            'teacher_in_s10': 0,       # teacher top1 in student top10 (near-miss)
            'teacher_flat': 0,          # teacher entropy > 4 (ambiguous)
            'teacher_sharp_miss': 0,    # teacher confident but student wrong
        }
        ent_hits = []
        ent_misses = []
        kl_per_tok = []

        for b in batches:
            tl = teacher.forward(b, max_layers=N_TEACHER_LAYERS)
            sl = student(b)

            # Per-position
            t_prob = F.softmax(tl[0], dim=-1)  # (T, V)
            s_prob = F.softmax(sl[0], dim=-1)
            # Teacher entropy per position
            t_ent = -(t_prob * (t_prob.clamp_min(1e-12).log())).sum(-1)  # (T,)
            # KL per position
            kl = (t_prob * (t_prob.clamp_min(1e-12).log() - s_prob.clamp_min(1e-12).log())).sum(-1)

            t_top1 = tl[0].argmax(-1)
            s_top1 = sl[0].argmax(-1)
            s_top10 = sl[0].topk(10, dim=-1).indices  # (T, 10)

            for pos in range(SEQ_LEN):
                n_tot += 1
                is_top1 = (s_top1[pos] == t_top1[pos]).item()
                in_top10 = (t_top1[pos].item() in s_top10[pos].tolist())
                if is_top1:
                    n_correct_top1 += 1
                    ent_hits.append(t_ent[pos].item())
                else:
                    ent_misses.append(t_ent[pos].item())
                    if in_top10:
                        miss_info['teacher_in_s10'] += 1
                    if t_ent[pos].item() > 4.0:
                        miss_info['teacher_flat'] += 1
                    elif t_ent[pos].item() < 2.0:
                        miss_info['teacher_sharp_miss'] += 1
                if in_top10:
                    n_correct_top10 += 1
                kl_per_tok.append(kl[pos].item())

        import statistics as stats
        print(f"\n   Top-1 correct:  {n_correct_top1}/{n_tot} = {n_correct_top1/n_tot*100:.2f}%")
        print(f"   Top-10 correct: {n_correct_top10}/{n_tot} = {n_correct_top10/n_tot*100:.2f}%")
        n_miss = n_tot - n_correct_top1
        print(f"\n   Of {n_miss:,} top-1 MISSES:")
        print(f"     teacher's top-1 was in student's top-10:  {miss_info['teacher_in_s10']:>6,}  ({miss_info['teacher_in_s10']/n_miss*100:.1f}%) [near-miss]")
        print(f"     teacher entropy > 4 (ambiguous context):  {miss_info['teacher_flat']:>6,}  ({miss_info['teacher_flat']/n_miss*100:.1f}%) [teacher unsure]")
        print(f"     teacher entropy < 2 (sharp, real miss):   {miss_info['teacher_sharp_miss']:>6,}  ({miss_info['teacher_sharp_miss']/n_miss*100:.1f}%) [hard miss]")
        print(f"\n   Mean teacher entropy on HITS:   {sum(ent_hits)/len(ent_hits):.3f}")
        print(f"   Mean teacher entropy on MISSES: {sum(ent_misses)/len(ent_misses):.3f}")
        print(f"   Mean KL per token:              {sum(kl_per_tok)/len(kl_per_tok):.4f}")

# =========================================================
# SYNTHESIS
# =========================================================
print("\n" + "=" * 70)
print("SYNTHESIS")
print("=" * 70)
print(f"   teacher[14] upper bound (7 iters × 2 scales equivalent): T10 ≈ {trunc_results[14][1]*100:.1f}%")
print(f"   teacher[21] upper bound:                                  T10 ≈ {trunc_results[21][1]*100:.1f}%")
print(f"   teacher[28] (full):                                       T10  = 100%")
print(f"   Our student (28 virtual layers, 1 shared block):          T10  = 68.2%")
print()
print("   Interpretation guide:")
print("     If teacher[21] T10 >> 68%:  we're hitting CAPACITY wall (block too small)")
print("     If teacher[14] T10 ≈ 68%:   we're NOT hitting capacity; we effectively collapse to ~14 layers")
print("     If most misses are 'near-miss': student knows ranking, just miscounts")
print("     If most misses are 'teacher flat':  ambiguous context — nothing to fix")
print("=" * 70)
