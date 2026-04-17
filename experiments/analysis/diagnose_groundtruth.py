"""
GROUND TRUTH DIAGNOSTIC — measure teacher AND student against the real next token.

The 68% "ceiling" assumes teacher is oracle. But:
  - Teacher entropy ~5.9 bits on fineweb (highly uncertain)
  - Teacher disagrees with teacher-minus-1-layer 25% of the time (T10)
  - 83% of student misses are on positions where teacher entropy > 4

What if the teacher is WRONG on many "student misses"? Then:
  - Student hitting 83% vs teacher but 50% vs ground truth = teacher-aligned
  - Student hitting 83% vs teacher AND 50% vs ground truth = matching teacher's errors too
  - Student hitting 60% vs teacher but 55% vs ground truth = student DIVERGES USEFULLY

Metrics (all measured on 500 samples × 64 positions = 32k predictions):
  T_vs_GT_top1:  teacher top-1 == real next token
  S_vs_GT_top1:  student top-1 == real next token
  T_vs_GT_top10: real next token in teacher's top-10
  S_vs_GT_top10: real next token in student's top-10
  S_vs_T_top1:   student top-1 == teacher top-1 (our usual 'accuracy')

Decisive question: when student DISAGREES with teacher, who wins on ground truth?
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
N_SAMPLES = 500
N_TEACHER_LAYERS = 28

print("=" * 72)
print("GROUND-TRUTH DIAGNOSTIC — measure against real next token, not teacher")
print("=" * 72)

# ── Load teacher weights ──
print("\nLoading teacher...")
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
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
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=hidden, intermediate_size=hidden*3,
                  vocab_size=vocab_size, head_dim=hidden//16)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)

# ── Data: need SEQ_LEN+1 so we have the real next token ──
print("Loading data...")
tokens_all = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
torch.manual_seed(42)
starts = torch.randint(0, tokens_all.numel() - SEQ_LEN - 1, (N_SAMPLES,)).tolist()

# ── Student: pure_kl record ──
class PureFRR(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale, vocab_size):
        super().__init__()
        self.n_scales = n_scales; self.iters_per_scale = iters_per_scale
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
                block_out = self.block(x, self.layer_gamma[li], self.layer_beta[li])
                x = x + (block_out - x) * self.iter_scale[s, it]
                li += 1
        return self.lm_head(self.norm(x))

student = PureFRR(hidden, 16, 4, 7, vocab_size).to(DEVICE)
sd = torch.load('checkpoints_1.7b_pure_kl/pure_kl_step5000.pt', map_location=DEVICE, weights_only=True)
ssd = student.state_dict()
for k, v in sd.items():
    if k in ssd and ssd[k].shape == v.shape:
        ssd[k] = v
if 'scale_gamma' in sd:
    sg, sb = sd['scale_gamma'], sd['scale_beta']
    for s in range(4):
        for it in range(7):
            ssd['layer_gamma'][s*7+it] = sg[s]
            ssd['layer_beta'][s*7+it] = sb[s]
    ssd['iter_scale'] = sd['iter_scale']
student.load_state_dict(ssd)
student.eval()

# ── Counters ──
# For each position, we have:
#   GT         = actual next token from corpus
#   t_top1, t_top10 = teacher's prediction
#   s_top1, s_top10 = student's prediction

T_gt_t1 = T_gt_t10 = 0      # teacher vs GT
S_gt_t1 = S_gt_t10 = 0      # student vs GT
S_vs_T_t1 = S_vs_T_t10 = 0  # student vs teacher
n_total = 0

# When S != T on top-1 (they disagree), who wins on GT?
disagree_count = 0
disagree_teacher_wins = 0   # GT == teacher's top-1
disagree_student_wins = 0   # GT == student's top-1
disagree_neither_wins = 0   # GT is neither

# Teacher entropy stratification
low_ent_tot = low_ent_sgt1 = low_ent_tgt1 = 0   # positions where teacher is CONFIDENT
high_ent_tot = high_ent_sgt1 = high_ent_tgt1 = 0

print(f"\nEvaluating {N_SAMPLES} samples × {SEQ_LEN} positions = {N_SAMPLES*SEQ_LEN:,} predictions...")

with torch.no_grad():
    for i, s_idx in enumerate(starts):
        # Input tokens[0:SEQ_LEN], targets tokens[1:SEQ_LEN+1]
        inp = tokens_all[s_idx:s_idx+SEQ_LEN].long().unsqueeze(0).to(DEVICE)
        gt = tokens_all[s_idx+1:s_idx+SEQ_LEN+1].long().to(DEVICE)  # (SEQ_LEN,)

        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)
        sl = student(inp)

        t_prob = F.softmax(tl[0], dim=-1)
        t_ent = -(t_prob * t_prob.clamp_min(1e-12).log()).sum(-1)   # (SEQ_LEN,)

        t_top1 = tl[0].argmax(-1)                            # (SEQ_LEN,)
        s_top1 = sl[0].argmax(-1)
        t_top10 = tl[0].topk(10, dim=-1).indices             # (SEQ_LEN, 10)
        s_top10 = sl[0].topk(10, dim=-1).indices

        for pos in range(SEQ_LEN):
            g = gt[pos].item()
            n_total += 1

            # Teacher vs GT
            if t_top1[pos].item() == g:
                T_gt_t1 += 1
            if g in t_top10[pos].tolist():
                T_gt_t10 += 1

            # Student vs GT
            if s_top1[pos].item() == g:
                S_gt_t1 += 1
            if g in s_top10[pos].tolist():
                S_gt_t10 += 1

            # Student vs Teacher
            if s_top1[pos].item() == t_top1[pos].item():
                S_vs_T_t1 += 1
            s_set = set(s_top10[pos].tolist())
            t_set = set(t_top10[pos].tolist())
            S_vs_T_t10 += len(s_set & t_set) / 10

            # Disagreement analysis (when S_top1 != T_top1)
            if s_top1[pos].item() != t_top1[pos].item():
                disagree_count += 1
                if t_top1[pos].item() == g:
                    disagree_teacher_wins += 1
                elif s_top1[pos].item() == g:
                    disagree_student_wins += 1
                else:
                    disagree_neither_wins += 1

            # Entropy stratification
            e = t_ent[pos].item()
            if e < 2.0:
                low_ent_tot += 1
                if t_top1[pos].item() == g: low_ent_tgt1 += 1
                if s_top1[pos].item() == g: low_ent_sgt1 += 1
            elif e > 5.0:
                high_ent_tot += 1
                if t_top1[pos].item() == g: high_ent_tgt1 += 1
                if s_top1[pos].item() == g: high_ent_sgt1 += 1

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{N_SAMPLES} done")

# ── Report ──
print("\n" + "=" * 72)
print("RESULTS")
print("=" * 72)
print(f"\nAgainst GROUND TRUTH (next real token):")
print(f"  Teacher top-1:  {T_gt_t1/n_total*100:6.2f}%    top-10: {T_gt_t10/n_total*100:6.2f}%")
print(f"  Student top-1:  {S_gt_t1/n_total*100:6.2f}%    top-10: {S_gt_t10/n_total*100:6.2f}%")
print(f"  GAP top-1:  {(T_gt_t1 - S_gt_t1)/n_total*100:+.2f}pp  (teacher - student)")
print(f"  GAP top-10: {(T_gt_t10 - S_gt_t10)/n_total*100:+.2f}pp")
print(f"\nAgainst TEACHER (what we were measuring):")
print(f"  Student top-1:   {S_vs_T_t1/n_total*100:6.2f}%")
print(f"  Student top-10:  {S_vs_T_t10/n_total*100:6.2f}%")

print(f"\n{'─'*72}")
print(f"DISAGREEMENT ANALYSIS: on positions where student's top-1 ≠ teacher's top-1")
print(f"  Disagreements: {disagree_count:,} / {n_total:,} ({disagree_count/n_total*100:.1f}%)")
if disagree_count > 0:
    print(f"  Teacher wins (GT = teacher's pick):   {disagree_teacher_wins:>6,}  ({disagree_teacher_wins/disagree_count*100:.1f}%)")
    print(f"  Student wins (GT = student's pick):   {disagree_student_wins:>6,}  ({disagree_student_wins/disagree_count*100:.1f}%)")
    print(f"  Neither (GT = other token):           {disagree_neither_wins:>6,}  ({disagree_neither_wins/disagree_count*100:.1f}%)")
    ratio = disagree_student_wins / max(disagree_teacher_wins, 1)
    print(f"  → Student:Teacher 'win ratio' when they disagree: {ratio:.2f}x")

print(f"\n{'─'*72}")
print(f"ENTROPY STRATIFICATION (teacher's confidence at each position)")
print(f"  LOW entropy (< 2.0, teacher CONFIDENT)   — {low_ent_tot:,} positions")
if low_ent_tot > 0:
    print(f"    Teacher vs GT:  {low_ent_tgt1/low_ent_tot*100:6.2f}%")
    print(f"    Student vs GT:  {low_ent_sgt1/low_ent_tot*100:6.2f}%")
print(f"  HIGH entropy (> 5.0, teacher UNSURE)     — {high_ent_tot:,} positions")
if high_ent_tot > 0:
    print(f"    Teacher vs GT:  {high_ent_tgt1/high_ent_tot*100:6.2f}%")
    print(f"    Student vs GT:  {high_ent_sgt1/high_ent_tot*100:6.2f}%")

print("\n" + "=" * 72)
print("INTERPRETATION")
print("=" * 72)
t_gt = T_gt_t1 / n_total
s_gt = S_gt_t1 / n_total
if s_gt >= t_gt - 0.005:
    print(f"  🎯 Student matches/exceeds teacher on GROUND TRUTH (Δ = {(s_gt-t_gt)*100:+.2f}pp)")
    print(f"  The 'teacher disagreement' is not a compression loss — student is")
    print(f"  just converging to DIFFERENT-BUT-EQUALLY-VALID predictions.")
    print(f"  The 68% ceiling vs teacher is a PROXY metric issue, not a real wall.")
elif s_gt >= t_gt - 0.02:
    print(f"  Student is {(t_gt-s_gt)*100:.2f}pp behind teacher on ground truth.")
    print(f"  Most of the 'teacher disagreement' is distillation error but small.")
else:
    print(f"  Student is {(t_gt-s_gt)*100:.2f}pp behind teacher on ground truth —")
    print(f"  real capability gap exists.")
print("=" * 72)
