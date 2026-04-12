"""SUPERTRAIN — Test whether training recipe is the bottleneck.

Hypothesis: Current 53% top-10 ceiling is from bad training, not bad architecture.

Key changes vs existing training:
  1. ALL-POSITION LOSS: KL div on all 32 positions, not just last token (32x more gradient)
  2. PROGRESSIVE + JOINT: Per-layer MSE init (all positions) → joint KL fine-tune (all positions)
  3. INTERMEDIATE SUPERVISION: During joint phase, add hidden-state cosine loss at layer boundaries
  4. LONGER TRAINING: 1K steps/layer progressive + 20K joint
  5. WARMUP + COSINE: Proper LR warmup to avoid early divergence

Runs on GPU 0 (separate from overnight on GPU 1).
"""
import torch, sys, os, time, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer, TransformerLayer
from ultracompress.genome_compressor import GenomeModel, MicroTransformerLayer

device = 'cuda'
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

hf_to_gguf = {'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight', 'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight', 'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight', 'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.down_proj.weight': 'ffn_down.weight'}
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(1024)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(28):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd: gd[f'blk.{li}.{g}'] = wd[k].float()

config = ModelConfig(n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
                     intermediate_size=3072, vocab_size=151936, head_dim=128)
teacher = MiniTransformer(config, device)
teacher.load_weights(gd)

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head = gd['output.weight'].to(device)

all_results = {}


def eval_forward(forward_fn, n=100):
    """Eval with top-1 and top-10 token agreement."""
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = forward_fn(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1/n, sum(t10s)/len(t10s)


def all_position_kl(student_logits, teacher_logits, temp=2.0):
    """KL divergence across ALL positions, not just last token."""
    # student_logits: (B, T, V), teacher_logits: (B, T, V)
    B, T, V = student_logits.shape
    s = student_logits.reshape(-1, V)
    t = teacher_logits.reshape(-1, V)
    return F.kl_div(
        F.log_softmax(s / temp, dim=-1),
        F.softmax(t / temp, dim=-1),
        reduction='batchmean',
    ) * (temp ** 2)


def cosine_hidden_loss(student_hidden, teacher_hidden):
    """Cosine similarity loss on hidden states (all positions)."""
    # Both: (B, T, D)
    s = student_hidden.reshape(-1, student_hidden.shape[-1])
    t = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
    cos = F.cosine_similarity(s, t, dim=-1)
    return (1 - cos).mean()


# ============================================================
# EXPERIMENT A: BASELINE (last-token loss, standard recipe)
# Same as existing V1 online for fair comparison
# ============================================================
print("=" * 70)
print("SUPERTRAIN EXPERIMENT A: BASELINE (last-token KL, 10K steps)")
print("=" * 70)
sys.stdout.flush()

genome_a = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4, n_layers=28,
    embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
).to(device)
g_params = genome_a.genome_param_count()
print(f"Genome: {g_params:,} ({g_params*2/1e6:.1f} MB)")
sys.stdout.flush()

opt = torch.optim.AdamW(genome_a.genome_layers.parameters(), lr=0.001, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)

t0 = time.time()
for step in range(10000):
    tokens = torch.randint(100, 100000, (8, 32), device=device)
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]  # LAST TOKEN ONLY
    student_logits = genome_a(tokens, max_layers=28)[:, -1, :]
    loss = F.kl_div(F.log_softmax(student_logits/2, -1), F.softmax(teacher_logits/2, -1), reduction='batchmean') * 4
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(genome_a.genome_layers.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 2000 == 0:
        t1_e, t10_e = eval_forward(lambda t, _m=genome_a: _m(t, max_layers=28))
        print(f"  Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

t1_a, t10_a = eval_forward(lambda t: genome_a(t, max_layers=28))
print(f"  RESULT A (baseline): Top1={t1_a*100:.0f}% Top10={t10_a*100:.0f}% Time={time.time()-t0:.0f}s")
all_results['A_baseline'] = {'top1': t1_a, 'top10': t10_a, 'time': time.time()-t0}
sys.stdout.flush()
del genome_a; torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT B: ALL-POSITION LOSS (the big change)
# Same architecture, but KL loss on ALL 32 positions
# ============================================================
print(f"\n{'='*70}")
print("SUPERTRAIN EXPERIMENT B: ALL-POSITION KL LOSS (32x more gradient)")
print(f"{'='*70}")
sys.stdout.flush()

genome_b = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4, n_layers=28,
    embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
).to(device)
print(f"Genome: {genome_b.genome_param_count():,} params")
sys.stdout.flush()

opt = torch.optim.AdamW(genome_b.genome_layers.parameters(), lr=0.001, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)

t0 = time.time()
for step in range(10000):
    tokens = torch.randint(100, 100000, (8, 32), device=device)
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)  # ALL POSITIONS
    student_logits = genome_b(tokens, max_layers=28)  # ALL POSITIONS
    loss = all_position_kl(student_logits, teacher_logits, temp=2.0)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(genome_b.genome_layers.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 2000 == 0:
        t1_e, t10_e = eval_forward(lambda t, _m=genome_b: _m(t, max_layers=28))
        print(f"  Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

t1_b, t10_b = eval_forward(lambda t: genome_b(t, max_layers=28))
print(f"  RESULT B (all-pos): Top1={t1_b*100:.0f}% Top10={t10_b*100:.0f}% Time={time.time()-t0:.0f}s")
all_results['B_allpos'] = {'top1': t1_b, 'top10': t10_b, 'time': time.time()-t0}
sys.stdout.flush()
del genome_b; torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT C: PROGRESSIVE INIT + ALL-POSITION JOINT FINE-TUNING
# Per-layer MSE → joint all-position KL
# ============================================================
print(f"\n{'='*70}")
print("SUPERTRAIN EXPERIMENT C: PROGRESSIVE + ALL-POSITION JOINT")
print(f"{'='*70}")
sys.stdout.flush()

genome_c = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4, n_layers=28,
    embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
).to(device)
print(f"Genome: {genome_c.genome_param_count():,} params")
sys.stdout.flush()

positions = torch.arange(32, device=device)

# Phase 1: Progressive per-layer distillation (MSE on delta, all positions)
print("\n--- Phase 1: Progressive per-layer distillation ---")
t0 = time.time()
for li in range(28):
    opt = torch.optim.AdamW(genome_c.genome_layers[li].parameters(), lr=0.002, weight_decay=0.005)
    for step in range(1000):
        torch.manual_seed(step + li * 10000)
        tokens = torch.randint(100, 100000, (8, 32), device=device)
        with torch.no_grad():
            x = F.embedding(tokens, embed).float()
            for prev in range(li):
                x = teacher.layers[prev](x, positions)
            x_in = x.clone()
            x_out = teacher.layers[li](x, positions)
            target_delta = x_out - x_in  # (B, T, 1024) — ALL positions
        pred_delta = genome_c.genome_layers[li](x_in)
        loss = F.mse_loss(pred_delta, target_delta)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(genome_c.genome_layers[li].parameters(), 1.0)
        opt.step()
    if li % 7 == 0 or li == 27:
        print(f"  Layer {li}/27: delta_mse={loss.item():.6f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

# Quick eval after progressive phase
t1_prog, t10_prog = eval_forward(lambda t: genome_c(t, max_layers=28))
print(f"  After progressive: Top1={t1_prog*100:.0f}% Top10={t10_prog*100:.0f}%")
sys.stdout.flush()

# Phase 2: Joint fine-tuning with ALL-POSITION KL loss
print("\n--- Phase 2: Joint fine-tuning (all-position KL, 15K steps) ---")
opt = torch.optim.AdamW(genome_c.genome_layers.parameters(), lr=0.0005, weight_decay=0.01)
# Warmup + cosine
warmup_steps = 500
total_joint = 15000

t0_joint = time.time()
for step in range(total_joint):
    # LR warmup then cosine decay
    if step < warmup_steps:
        lr = 0.0005 * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_joint - warmup_steps)
        lr = 0.0005 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    for pg in opt.param_groups: pg['lr'] = lr

    tokens = torch.randint(100, 100000, (8, 32), device=device)
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)  # ALL positions
    student_logits = genome_c(tokens, max_layers=28)
    loss = all_position_kl(student_logits, teacher_logits, temp=2.0)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(genome_c.genome_layers.parameters(), 1.0)
    opt.step()
    if step % 3000 == 0:
        t1_e, t10_e = eval_forward(lambda t, _m=genome_c: _m(t, max_layers=28))
        print(f"  Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% lr={lr:.6f} ({time.time()-t0_joint:.0f}s)")
        sys.stdout.flush()

t1_c, t10_c = eval_forward(lambda t: genome_c(t, max_layers=28))
total_time_c = time.time() - t0
print(f"  RESULT C (prog+allpos): Top1={t1_c*100:.0f}% Top10={t10_c*100:.0f}% Time={total_time_c:.0f}s")
all_results['C_prog_allpos'] = {'top1': t1_c, 'top10': t10_c, 'time': total_time_c}
sys.stdout.flush()
del genome_c; torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT D: PROGRESSIVE + ALL-POSITION + INTERMEDIATE SUPERVISION
# The full kitchen sink: per-layer + joint + hidden state cosine
# ============================================================
print(f"\n{'='*70}")
print("SUPERTRAIN EXPERIMENT D: FULL PIPELINE (prog + allpos + hidden supervision)")
print(f"{'='*70}")
sys.stdout.flush()

genome_d = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4, n_layers=28,
    embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
).to(device)
print(f"Genome: {genome_d.genome_param_count():,} params")
sys.stdout.flush()

# Phase 1: Progressive (same as C)
print("\n--- Phase 1: Progressive per-layer ---")
t0 = time.time()
for li in range(28):
    opt = torch.optim.AdamW(genome_d.genome_layers[li].parameters(), lr=0.002, weight_decay=0.005)
    for step in range(1000):
        torch.manual_seed(step + li * 10000)
        tokens = torch.randint(100, 100000, (8, 32), device=device)
        with torch.no_grad():
            x = F.embedding(tokens, embed).float()
            for prev in range(li):
                x = teacher.layers[prev](x, positions)
            x_in = x.clone()
            x_out = teacher.layers[li](x, positions)
            target_delta = x_out - x_in
        pred_delta = genome_d.genome_layers[li](x_in)
        loss = F.mse_loss(pred_delta, target_delta)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(genome_d.genome_layers[li].parameters(), 1.0)
        opt.step()
    if li % 7 == 0 or li == 27:
        print(f"  Layer {li}/27: delta_mse={loss.item():.6f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

t1_prog, t10_prog = eval_forward(lambda t: genome_d(t, max_layers=28))
print(f"  After progressive: Top1={t1_prog*100:.0f}% Top10={t10_prog*100:.0f}%")
sys.stdout.flush()

# Phase 2: Joint fine-tuning with ALL-POSITION KL + hidden state supervision
print("\n--- Phase 2: Joint (all-position KL + hidden cosine, 15K steps) ---")
opt = torch.optim.AdamW(genome_d.genome_layers.parameters(), lr=0.0005, weight_decay=0.01)
warmup_steps = 500
total_joint = 15000

# Supervision checkpoints: match hidden states at layers 7, 14, 21, 27
supervision_layers = [7, 14, 21, 27]

t0_joint = time.time()
for step in range(total_joint):
    if step < warmup_steps:
        lr = 0.0005 * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_joint - warmup_steps)
        lr = 0.0005 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    for pg in opt.param_groups: pg['lr'] = lr

    tokens = torch.randint(100, 100000, (8, 32), device=device)

    # Teacher forward with hidden state capture
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)
        # Also get teacher hidden states at supervision points
        tx = F.embedding(tokens, embed).float()
        teacher_hiddens = {}
        for li in range(28):
            tx = teacher.layers[li](tx, positions)
            if li in supervision_layers:
                teacher_hiddens[li] = tx.clone()

    # Student forward with hidden state capture
    sx = genome_d.embed(tokens).float()
    student_hiddens = {}
    for li in range(28):
        sx = sx + genome_d.genome_layers[li](sx)
        if li in supervision_layers:
            student_hiddens[li] = sx

    # Final norm + head
    sx_normed = genome_d.norm(sx)
    student_logits = genome_d.lm_head(sx_normed)

    # Loss 1: All-position KL
    kl_loss = all_position_kl(student_logits, teacher_logits, temp=2.0)

    # Loss 2: Hidden state cosine similarity at supervision points
    hidden_loss = 0.0
    for li in supervision_layers:
        hidden_loss = hidden_loss + cosine_hidden_loss(student_hiddens[li], teacher_hiddens[li])
    hidden_loss = hidden_loss / len(supervision_layers)

    # Combined loss — decay hidden supervision over time
    hidden_weight = max(0.1, 1.0 - step / total_joint)
    loss = kl_loss + hidden_weight * hidden_loss

    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(genome_d.genome_layers.parameters(), 1.0)
    opt.step()

    if step % 3000 == 0:
        t1_e, t10_e = eval_forward(lambda t, _m=genome_d: _m(t, max_layers=28))
        print(f"  Step {step}: kl={kl_loss.item():.4f} hid={hidden_loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0_joint:.0f}s)")
        sys.stdout.flush()

t1_d, t10_d = eval_forward(lambda t: genome_d(t, max_layers=28))
total_time_d = time.time() - t0
print(f"  RESULT D (full pipeline): Top1={t1_d*100:.0f}% Top10={t10_d*100:.0f}% Time={total_time_d:.0f}s")
all_results['D_full_pipeline'] = {'top1': t1_d, 'top10': t10_d, 'time': total_time_d}
sys.stdout.flush()

# Save best genome if it beats existing records
best_t10 = max(t10_d, 0)
if best_t10 > 0.53:
    torch.save(genome_d.genome_layers.state_dict(), 'genome_supertrain_best.pt')
    print(f"  NEW RECORD! Saved genome_supertrain_best.pt")
del genome_d; torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT E: ALL-POSITION + LARGER GENOME (sd=256)
# Does all-position loss + more capacity break through?
# ============================================================
print(f"\n{'='*70}")
print("SUPERTRAIN EXPERIMENT E: LARGER GENOME sd=256 + ALL-POSITION")
print(f"{'='*70}")
sys.stdout.flush()

genome_e = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=256, n_heads=8, n_layers=28,
    embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
).to(device)
e_params = genome_e.genome_param_count()
print(f"Genome: {e_params:,} ({e_params*2/1e6:.1f} MB)")
sys.stdout.flush()

# Progressive init
print("\n--- Phase 1: Progressive ---")
t0 = time.time()
for li in range(28):
    opt = torch.optim.AdamW(genome_e.genome_layers[li].parameters(), lr=0.002, weight_decay=0.005)
    for step in range(1000):
        torch.manual_seed(step + li * 10000)
        tokens = torch.randint(100, 100000, (8, 32), device=device)
        with torch.no_grad():
            x = F.embedding(tokens, embed).float()
            for prev in range(li):
                x = teacher.layers[prev](x, positions)
            x_in = x.clone()
            x_out = teacher.layers[li](x, positions)
            target_delta = x_out - x_in
        pred_delta = genome_e.genome_layers[li](x_in)
        loss = F.mse_loss(pred_delta, target_delta)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(genome_e.genome_layers[li].parameters(), 1.0)
        opt.step()
    if li % 7 == 0 or li == 27:
        print(f"  Layer {li}/27: delta_mse={loss.item():.6f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

t1_prog, t10_prog = eval_forward(lambda t: genome_e(t, max_layers=28))
print(f"  After progressive: Top1={t1_prog*100:.0f}% Top10={t10_prog*100:.0f}%")
sys.stdout.flush()

# Joint all-position KL
print("\n--- Phase 2: Joint all-position KL (15K steps) ---")
opt = torch.optim.AdamW(genome_e.genome_layers.parameters(), lr=0.0005, weight_decay=0.01)
t0_joint = time.time()
for step in range(15000):
    if step < 500:
        lr = 0.0005 * step / 500
    else:
        progress = (step - 500) / 14500
        lr = 0.0005 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    for pg in opt.param_groups: pg['lr'] = lr

    tokens = torch.randint(100, 100000, (8, 32), device=device)
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)
    student_logits = genome_e(tokens, max_layers=28)
    loss = all_position_kl(student_logits, teacher_logits, temp=2.0)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(genome_e.genome_layers.parameters(), 1.0)
    opt.step()
    if step % 3000 == 0:
        t1_e, t10_e = eval_forward(lambda t, _m=genome_e: _m(t, max_layers=28))
        print(f"  Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0_joint:.0f}s)")
        sys.stdout.flush()

t1_e_final, t10_e_final = eval_forward(lambda t: genome_e(t, max_layers=28))
total_time_e = time.time() - t0
print(f"  RESULT E (sd=256 allpos): Top1={t1_e_final*100:.0f}% Top10={t10_e_final*100:.0f}% Size={e_params*2/1e6:.1f}MB Time={total_time_e:.0f}s")
all_results['E_sd256_allpos'] = {'top1': t1_e_final, 'top10': t10_e_final, 'size_mb': e_params*2/1e6, 'time': total_time_e}
sys.stdout.flush()

if t10_e_final > 0.53:
    torch.save(genome_e.genome_layers.state_dict(), 'genome_supertrain_sd256_best.pt')
    print(f"  NEW RECORD! Saved genome_supertrain_sd256_best.pt")
del genome_e; torch.cuda.empty_cache()


# ============================================================
# FINAL COMPARISON
# ============================================================
print(f"\n{'='*70}")
print("SUPERTRAIN RESULTS — Training Recipe Ablation")
print(f"{'='*70}")

sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('top10', 0), reverse=True)
for i, (name, r) in enumerate(sorted_results):
    medal = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"
    t1 = r.get('top1', 0)
    t10 = r.get('top10', 0)
    t_sec = r.get('time', 0)
    size = r.get('size_mb', 23.9)
    print(f"  {medal:>3}: {name:<25} Top1={t1*100:>4.0f}% Top10={t10*100:>4.0f}% Size={size:>6.1f}MB Time={t_sec/60:.0f}m")

with open('supertrain_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to supertrain_results.json")

# Diagnosis
print(f"\n{'='*70}")
print("DIAGNOSIS")
print(f"{'='*70}")
baseline_t10 = all_results.get('A_baseline', {}).get('top10', 0)
allpos_t10 = all_results.get('B_allpos', {}).get('top10', 0)
prog_allpos_t10 = all_results.get('C_prog_allpos', {}).get('top10', 0)
full_t10 = all_results.get('D_full_pipeline', {}).get('top10', 0)

if allpos_t10 > baseline_t10 * 1.15:
    print(">>> ALL-POSITION LOSS is a major win! Training was the bottleneck.")
    print(f"    Improvement: {baseline_t10*100:.0f}% -> {allpos_t10*100:.0f}% (+{(allpos_t10-baseline_t10)*100:.0f}%)")
elif allpos_t10 > baseline_t10 * 1.05:
    print(">>> All-position loss helps moderately. Both training and architecture matter.")
else:
    print(">>> All-position loss barely helps. Architecture is the main bottleneck.")
    print("    Need: bigger genome, better attention mechanism, or hybrid approach.")

if full_t10 > prog_allpos_t10 * 1.05:
    print(">>> Hidden supervision adds value on top of progressive + all-position.")
else:
    print(">>> Hidden supervision doesn't help much beyond progressive + all-position.")

best_name, best = sorted_results[0]
print(f"\nBest approach: {best_name} at {best['top10']*100:.0f}% top-10")
if best['top10'] > 0.7:
    print(">>> 70%+ top-10 achieved! Recipe breakthrough — scale this up.")
elif best['top10'] > 0.6:
    print(">>> 60%+ top-10. Good progress. Combine best recipe with larger genome.")
else:
    print(">>> Still under 60%. May need fundamentally different architecture.")
print(f"{'='*70}")
