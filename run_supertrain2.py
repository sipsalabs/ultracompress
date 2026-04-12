"""SUPERTRAIN v2 — Scale up the winning recipe from v1.

v1 findings:
  - D (progressive + all-pos KL + hidden supervision) = 44% top-1, 63% top-10 (NEW RECORD)
  - Hidden supervision prevents catastrophic collapse during joint fine-tuning
  - D was STILL IMPROVING at step 15K — no plateau
  - Progressive phase struggled on later layers (L21: MSE=1.58, L27: MSE=7.13)
  - sd=256 failed WITHOUT hidden supervision — try it WITH

v2 changes:
  1. Adaptive progressive: later layers get MORE training (3K steps for L20-27)
  2. Longer joint fine-tuning: 30K steps (D was still climbing)
  3. Test sd=128 and sd=256 both with full D recipe
  4. All supervision layers (every 4th layer, not just 7/14/21/27)
  5. Temperature annealing: start warm (T=4), cool down to T=1

Runs on GPU 0.
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
positions = torch.arange(32, device=device)

all_results = {}
pipeline_start = time.time()


def eval_forward(forward_fn, n=100):
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
    B, T, V = student_logits.shape
    s = student_logits.reshape(-1, V)
    t = teacher_logits.reshape(-1, V)
    return F.kl_div(F.log_softmax(s / temp, -1), F.softmax(t / temp, -1), reduction='batchmean') * (temp ** 2)


def cosine_hidden_loss(student_hidden, teacher_hidden):
    s = student_hidden.reshape(-1, student_hidden.shape[-1])
    t = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
    return (1 - F.cosine_similarity(s, t, dim=-1)).mean()


def run_d_recipe(small_dim, n_heads, name, prog_steps_early=1000, prog_steps_late=3000,
                 joint_steps=30000, supervision_every=4):
    """Run the full D recipe with configurable parameters."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"  sd={small_dim}, prog_early={prog_steps_early}, prog_late={prog_steps_late}")
    print(f"  joint={joint_steps}, supervision_every={supervision_every} layers")
    print(f"{'='*70}")
    sys.stdout.flush()

    genome = GenomeModel(
        vocab_size=151936, big_dim=1024, small_dim=small_dim, n_heads=n_heads, n_layers=28,
        embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
    ).to(device)
    g_params = genome.genome_param_count()
    print(f"Genome: {g_params:,} ({g_params*2/1e6:.1f} MB)")
    sys.stdout.flush()

    # Supervision layers: every Nth layer
    supervision_layers = list(range(supervision_every - 1, 28, supervision_every))
    if 27 not in supervision_layers:
        supervision_layers.append(27)
    print(f"Supervision at layers: {supervision_layers}")

    # ---- Phase 1: Adaptive progressive per-layer distillation ----
    print(f"\n--- Phase 1: Adaptive progressive per-layer ---")
    t0 = time.time()
    for li in range(28):
        # Later layers are harder — give them more training
        if li >= 20:
            steps = prog_steps_late
        elif li >= 14:
            steps = (prog_steps_early + prog_steps_late) // 2
        else:
            steps = prog_steps_early

        opt = torch.optim.AdamW(genome.genome_layers[li].parameters(), lr=0.002, weight_decay=0.005)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

        best_loss = float('inf')
        for step in range(steps):
            torch.manual_seed(step + li * 10000)
            tokens = torch.randint(100, 100000, (8, 32), device=device)
            with torch.no_grad():
                x = F.embedding(tokens, embed).float()
                for prev in range(li):
                    x = teacher.layers[prev](x, positions)
                x_in = x.clone()
                x_out = teacher.layers[li](x, positions)
                target_delta = x_out - x_in
            pred_delta = genome.genome_layers[li](x_in)
            loss = F.mse_loss(pred_delta, target_delta)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(genome.genome_layers[li].parameters(), 1.0)
            opt.step(); sched.step()
            if loss.item() < best_loss:
                best_loss = loss.item()

        if li % 4 == 0 or li == 27:
            print(f"  Layer {li:>2}/27 ({steps:>4} steps): best_mse={best_loss:.6f} final={loss.item():.6f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    t1_prog, t10_prog = eval_forward(lambda t, _m=genome: _m(t, max_layers=28))
    print(f"  After progressive: Top1={t1_prog*100:.0f}% Top10={t10_prog*100:.0f}%")
    sys.stdout.flush()

    # ---- Phase 2: Joint fine-tuning with hidden supervision ----
    print(f"\n--- Phase 2: Joint ({joint_steps} steps, hidden supervision) ---")
    opt = torch.optim.AdamW(genome.genome_layers.parameters(), lr=0.0005, weight_decay=0.01)
    warmup_steps = 1000

    t0_joint = time.time()
    best_t10 = 0
    best_state = None

    for step in range(joint_steps):
        # LR: warmup then cosine
        if step < warmup_steps:
            lr = 0.0005 * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (joint_steps - warmup_steps)
            lr = 0.0005 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        for pg in opt.param_groups: pg['lr'] = lr

        # Temperature annealing: start warm, cool down
        temp = 4.0 - 2.0 * min(1.0, step / (joint_steps * 0.5))  # 4 -> 2 over first half

        tokens = torch.randint(100, 100000, (8, 32), device=device)

        # Teacher forward with hidden capture
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)
            tx = F.embedding(tokens, embed).float()
            teacher_hiddens = {}
            for li in range(28):
                tx = teacher.layers[li](tx, positions)
                if li in supervision_layers:
                    teacher_hiddens[li] = tx.clone()

        # Student forward with hidden capture
        sx = genome.embed(tokens).float()
        student_hiddens = {}
        for li in range(28):
            sx = sx + genome.genome_layers[li](sx)
            if li in supervision_layers:
                student_hiddens[li] = sx

        sx_normed = genome.norm(sx)
        student_logits = genome.lm_head(sx_normed)

        # All-position KL loss with temperature
        kl_loss = all_position_kl(student_logits, teacher_logits, temp=temp)

        # Hidden supervision (cosine)
        hidden_loss = 0.0
        for li in supervision_layers:
            hidden_loss = hidden_loss + cosine_hidden_loss(student_hiddens[li], teacher_hiddens[li])
        hidden_loss = hidden_loss / len(supervision_layers)

        # Hidden weight: high early, decay slowly
        hidden_weight = max(0.05, 0.5 * (1 - step / joint_steps))
        loss = kl_loss + hidden_weight * hidden_loss

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(genome.genome_layers.parameters(), 1.0)
        opt.step()

        if step % 3000 == 0:
            t1_e, t10_e = eval_forward(lambda t, _m=genome: _m(t, max_layers=28))
            elapsed = time.time() - t0_joint
            print(f"  Step {step:>5}: kl={kl_loss.item():.4f} hid={hidden_loss.item():.4f} "
                  f"Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% T={temp:.1f} lr={lr:.6f} ({elapsed:.0f}s)")
            sys.stdout.flush()

            # Save best checkpoint
            if t10_e > best_t10:
                best_t10 = t10_e
                best_state = {k: v.clone() for k, v in genome.genome_layers.state_dict().items()}
                print(f"    ^ New best! Saving checkpoint...")

    # Restore best checkpoint
    if best_state is not None:
        genome.genome_layers.load_state_dict(best_state)
        print(f"  Restored best checkpoint (top10={best_t10*100:.0f}%)")

    t1_final, t10_final = eval_forward(lambda t, _m=genome: _m(t, max_layers=28))
    total_time = time.time() - t0
    print(f"\n  RESULT {name}: Top1={t1_final*100:.0f}% Top10={t10_final*100:.0f}% "
          f"Size={g_params*2/1e6:.1f}MB Time={total_time/60:.0f}min")
    sys.stdout.flush()

    result = {
        'top1': t1_final, 'top10': t10_final, 'best_t10': best_t10,
        'size_mb': g_params*2/1e6, 'params': g_params, 'time': total_time,
        'prog_top10': t10_prog, 'small_dim': small_dim,
    }
    all_results[name] = result

    # Save genome
    if t10_final > 0.53:
        save_path = f'genome_v2_{name.lower().replace(" ", "_")}.pt'
        torch.save(genome.genome_layers.state_dict(), save_path)
        print(f"  Saved {save_path}")

    del genome; torch.cuda.empty_cache()
    return result


# ============================================================
# RUN EXPERIMENTS
# ============================================================

# Experiment 1: sd=128, longer training (the scale-up of D)
run_d_recipe(
    small_dim=128, n_heads=4, name="D_v2_sd128",
    prog_steps_early=1000, prog_steps_late=3000,
    joint_steps=30000, supervision_every=4,
)

# Experiment 2: sd=256 WITH hidden supervision (E failed without it)
run_d_recipe(
    small_dim=256, n_heads=8, name="D_v2_sd256",
    prog_steps_early=1000, prog_steps_late=3000,
    joint_steps=30000, supervision_every=4,
)

# Experiment 3: sd=128 with DENSE supervision (every 2 layers)
run_d_recipe(
    small_dim=128, n_heads=4, name="D_v2_dense_sup",
    prog_steps_early=1500, prog_steps_late=4000,
    joint_steps=30000, supervision_every=2,
)


# ============================================================
# FINAL COMPARISON
# ============================================================
total_pipeline = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"SUPERTRAIN v2 RESULTS (Total: {total_pipeline/60:.0f} min)")
print(f"{'='*70}")
print(f"  Previous best (v1-D): Top1=44% Top10=63% at 23.9MB")
print()

sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('top10', 0), reverse=True)
for i, (name, r) in enumerate(sorted_results):
    medal = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"
    t1 = r.get('top1', 0)
    t10 = r.get('top10', 0)
    bt10 = r.get('best_t10', 0)
    size = r.get('size_mb', 0)
    print(f"  {medal:>3}: {name:<25} Top1={t1*100:>4.0f}% Top10={t10*100:>4.0f}% "
          f"(best={bt10*100:.0f}%) Size={size:>6.1f}MB Time={r.get('time',0)/60:.0f}m")

with open('supertrain2_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to supertrain2_results.json")

best_name, best = sorted_results[0]
print(f"\n{'='*70}")
print(f"WINNER: {best_name}")
print(f"  Top1={best['top1']*100:.0f}% Top10={best['top10']*100:.0f}% at {best.get('size_mb', 0):.1f} MB")

if best['top10'] > 0.75:
    print("  >>> 75%+ top-10! Major breakthrough. Ready for 8B scaling test.")
elif best['top10'] > 0.65:
    print("  >>> 65%+ top-10. Solid improvement over v1-D (63%). Keep pushing training.")
else:
    print("  >>> Under 65%. Marginal gains. Need architectural innovation.")

# Scaling projections
print(f"\n  Qwen3-8B projection: ~{best.get('size_mb', 0) * (36/28) * (4096/1024)**0.5:.0f} MB genome")
print(f"  70B projection: ~{best.get('size_mb', 0) * (80/28) * (8192/1024)**0.5:.0f} MB genome")
print(f"{'='*70}")
