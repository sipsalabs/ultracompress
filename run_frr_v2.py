"""FRR V2 — Fractal Residual Recursion + Hidden Supervision + Temperature Annealing.

V1 hit 62% top-10 with basic KL loss.
Supertrain showed hidden supervision pushed genome from 53% to 63%.
FRR V2 combines both: shared block recursion + hidden cosine supervision.

Target: 70%+ top-10 at 42x compression.
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
print("Loading teacher...")
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
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(device)

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)
positions = torch.arange(32, device=device)

all_results = {}


def eval_model(forward_fn, n=100):
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


def cosine_hidden_loss(student, teacher_h):
    s = student.reshape(-1, student.shape[-1])
    t = teacher_h.reshape(-1, teacher_h.shape[-1])
    return (1 - F.cosine_similarity(s, t, dim=-1)).mean()


print("=" * 70)
print("FRR V2 — FRACTAL + HIDDEN SUPERVISION + TEMP ANNEALING")
print("Target: 70%+ top-10 at 42x compression")
print("=" * 70)
sys.stdout.flush()

# Best config from V1 was 4s7i. Test that with enhanced training.
for n_scales, iters, n_heads, n_steps, name in [
    (4, 7, 8, 25000, "FRR-v2-4s7i-25K"),
    (7, 4, 8, 25000, "FRR-v2-7s4i-25K"),
]:
    print(f"\n--- {name} ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = FractalModel(
            hidden_dim=1024, n_heads=n_heads, n_scales=n_scales,
            iters_per_scale=iters, vocab_size=151936, ff_mult=2,
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        fractal_p = model.fractal_params()
        print(f"  Params: {fractal_p:,} ({fractal_p*2/1e6:.1f} MB), 42x compression")
        sys.stdout.flush()

        # Supervision at every 7th virtual layer
        total_virtual = n_scales * iters
        supervision_steps = list(range(6, total_virtual, 7))
        if total_virtual - 1 not in supervision_steps:
            supervision_steps.append(total_virtual - 1)
        print(f"  Hidden supervision at virtual layers: {supervision_steps}")

        trainable = list(model.block.parameters()) + [model.scale_gamma, model.scale_beta, model.iter_scale]
        opt = torch.optim.AdamW(trainable, lr=0.0005, weight_decay=0.01)
        warmup = 1000

        best_t10 = 0
        best_state = None

        for step in range(n_steps):
            if step < warmup:
                lr = 0.0005 * step / warmup
            else:
                lr = 0.0005 * 0.5 * (1 + math.cos((step - warmup) / (n_steps - warmup) * math.pi))
            for pg in opt.param_groups: pg['lr'] = lr

            # Temperature annealing: T=4 -> T=2 over first half
            temp = 4.0 - 2.0 * min(1.0, step / (n_steps * 0.5))

            tokens = torch.randint(100, 100000, (8, 32), device=device)

            # Teacher forward + hidden state capture
            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)
                tx = F.embedding(tokens, embed).float()
                teacher_hiddens = {}
                for li in range(28):
                    tx = teacher.layers[li](tx, positions)
                    if li in supervision_steps:
                        teacher_hiddens[li] = tx.clone()

            # Student forward with hidden capture
            x = model.embed(tokens).float()
            student_hiddens = {}
            layer_count = 0
            for scale in range(model.n_scales):
                gamma = model.scale_gamma[scale]
                beta = model.scale_beta[scale]
                for it in range(model.iters_per_scale):
                    iter_s = model.iter_scale[scale, it]
                    x = x + (model.block(x, gamma, beta) - x) * iter_s
                    if layer_count in supervision_steps:
                        student_hiddens[layer_count] = x
                    layer_count += 1
            x = model.norm(x)
            student_logits = model.lm_head(x)

            # Loss 1: All-position KL with temperature
            B, T, V = student_logits.shape
            kl_loss = F.kl_div(
                F.log_softmax(student_logits.reshape(-1, V) / temp, -1),
                F.softmax(teacher_logits.reshape(-1, V) / temp, -1),
                reduction='batchmean') * (temp ** 2)

            # Loss 2: Hidden supervision (cosine at supervision points)
            hidden_loss = 0.0
            n_sup = 0
            for li in supervision_steps:
                if li in student_hiddens and li in teacher_hiddens:
                    hidden_loss = hidden_loss + cosine_hidden_loss(student_hiddens[li], teacher_hiddens[li])
                    n_sup += 1
            if n_sup > 0:
                hidden_loss = hidden_loss / n_sup

            # Combined: decay hidden weight over time
            hidden_weight = max(0.05, 0.5 * (1 - step / n_steps))
            loss = kl_loss + hidden_weight * hidden_loss

            if torch.isnan(loss):
                for pg in opt.param_groups: pg['lr'] *= 0.1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()

            if step % 5000 == 0:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t))
                h_val = hidden_loss.item() if isinstance(hidden_loss, torch.Tensor) else hidden_loss
                print(f"    Step {step}: kl={kl_loss.item():.4f} hid={h_val:.4f} "
                      f"Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% T={temp:.1f} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()
                if t10_e > best_t10:
                    best_t10 = t10_e
                    best_state = {k: v.clone() for k, v in model.state_dict().items()
                                 if 'block' in k or 'scale' in k or 'iter' in k}
                    print(f"      ^ New best!")

        # Restore best
        if best_state:
            model.load_state_dict(best_state, strict=False)

        t1, t10 = eval_model(lambda t, _m=model: _m(t))
        elapsed = time.time() - t0
        print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
              f"Size={fractal_p*2/1e6:.1f}MB 42x Time={elapsed:.0f}s "
              f"(best_during={best_t10*100:.0f}%)")
        all_results[name] = {
            'top1': t1, 'top10': t10, 'best_t10': best_t10,
            'size_mb': fractal_p*2/1e6, 'time': elapsed,
        }
        sys.stdout.flush()

        if t10 > 0.62 or best_t10 > 0.62:
            torch.save(best_state or model.state_dict(), f'frr_v2_best_{name}.pt')
            print(f"  Saved checkpoint!")

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()
    sys.stdout.flush()

# Summary
print(f"\n{'='*70}")
print("FRR V2 RESULTS")
print(f"{'='*70}")
print(f"V1 baseline: FRR-4s7i-h8 = 62% top-10 at 21MB (42x)")
for n, r in sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True):
    print(f"  {n:<25} Top1={r['top1']*100:.0f}% Top10={r['top10']*100:.0f}% "
          f"(best={r['best_t10']*100:.0f}%) {r['size_mb']:.1f}MB")

best_n = max(all_results, key=lambda k: all_results[k]['top10']) if all_results else None
if best_n and all_results[best_n]['top10'] > 0.65:
    print(f"\n>>> 65%+ ACHIEVED. FRR + hidden supervision > genome. Scale to 8B.")
elif best_n and all_results[best_n]['top10'] > 0.62:
    print(f"\n>>> Beats V1. Hidden supervision helps FRR too.")
print(f"{'='*70}")

with open('frr_v2_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
