"""FRR V3 — Fractal + LoRA Adapters + Hidden Supervision.

V1: 62% top-10 (basic KL loss)
V2: running (hidden supervision + temp annealing)
V3: per-layer LoRA adapters — research says this closes the gap

The insight: gamma/beta modulation (~8K params) does the heavy lifting,
but LoRA adapters (~32K per layer = 896K total) let each virtual layer
truly specialize while keeping 39x compression.
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


def cosine_hidden_loss(s, t):
    return (1 - F.cosine_similarity(s.reshape(-1, s.shape[-1]), t.reshape(-1, t.shape[-1]), dim=-1)).mean()


print("=" * 70)
print("FRR V3 — FRACTAL + LORA ADAPTERS + HIDDEN SUPERVISION")
print("Target: 65%+ top-10 (beat genome), still 39x compression")
print("=" * 70)
sys.stdout.flush()

for adapter_rank, n_steps, name in [
    (16, 20000, "FRR-v3-r16"),
    (32, 20000, "FRR-v3-r32"),
]:
    print(f"\n--- {name} ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = FractalModel(
            hidden_dim=1024, n_heads=8, n_scales=4, iters_per_scale=7,
            vocab_size=151936, ff_mult=2,
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        model.enable_adapters(rank=adapter_rank)

        total_p = model.fractal_params()
        teacher_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))
        compression = teacher_params / total_p
        print(f"  Total params: {total_p:,} ({total_p*2/1e6:.1f} MB), {compression:.0f}x compression")

        supervision_layers = [6, 13, 20, 27]

        trainable = (list(model.block.parameters()) +
                    [model.scale_gamma, model.scale_beta, model.iter_scale] +
                    list(model.adapters.parameters()))

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
            temp = 4.0 - 2.0 * min(1.0, step / (n_steps * 0.5))

            tokens = torch.randint(100, 100000, (8, 32), device=device)

            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)
                tx = F.embedding(tokens, embed).float()
                teacher_hiddens = {}
                for li in range(28):
                    tx = teacher.layers[li](tx, positions)
                    if li in supervision_layers:
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
                    if model.adapters is not None:
                        x = model.adapters[layer_count](x)
                    if layer_count in supervision_layers:
                        student_hiddens[layer_count] = x
                    layer_count += 1
            x = model.norm(x)
            student_logits = model.lm_head(x)

            B, T, V = student_logits.shape
            kl_loss = F.kl_div(
                F.log_softmax(student_logits.reshape(-1, V) / temp, -1),
                F.softmax(teacher_logits.reshape(-1, V) / temp, -1),
                reduction='batchmean') * (temp ** 2)

            hidden_loss = sum(cosine_hidden_loss(student_hiddens[li], teacher_hiddens[li])
                            for li in supervision_layers if li in student_hiddens) / len(supervision_layers)

            hidden_weight = max(0.05, 0.5 * (1 - step / n_steps))
            loss = kl_loss + hidden_weight * hidden_loss

            if torch.isnan(loss):
                for pg in opt.param_groups: pg['lr'] *= 0.1
                continue

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()

            if step % 4000 == 0:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t))
                h_val = hidden_loss.item() if isinstance(hidden_loss, torch.Tensor) else hidden_loss
                print(f"    Step {step}: kl={kl_loss.item():.4f} hid={h_val:.4f} "
                      f"Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% T={temp:.1f} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()
                if t10_e > best_t10:
                    best_t10 = t10_e
                    best_state = {k: v.clone() for k, v in model.state_dict().items()
                                 if 'block' in k or 'scale' in k or 'iter' in k or 'adapter' in k}
                    print(f"      ^ New best!")

        if best_state:
            model.load_state_dict(best_state, strict=False)
        t1, t10 = eval_model(lambda t, _m=model: _m(t))
        elapsed = time.time() - t0
        print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
              f"Size={total_p*2/1e6:.1f}MB {compression:.0f}x (best={best_t10*100:.0f}%) Time={elapsed:.0f}s")
        all_results[name] = {'top1': t1, 'top10': t10, 'best_t10': best_t10,
                            'size_mb': total_p*2/1e6, 'compression': compression, 'time': elapsed}
        if best_t10 > 0.62:
            torch.save(best_state or model.state_dict(), f'frr_v3_{name}.pt')
            print(f"  Saved checkpoint!")
        sys.stdout.flush()

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()

print(f"\n{'='*70}")
print("FRR V3 RESULTS")
print(f"{'='*70}")
print(f"V1: 62% at 42x | V2: running | V3:")
for n, r in sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True):
    print(f"  {n:<20} Top10={r['top10']*100:.0f}% (best={r['best_t10']*100:.0f}%) "
          f"{r['size_mb']:.1f}MB {r['compression']:.0f}x")
with open('frr_v3_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"{'='*70}")
