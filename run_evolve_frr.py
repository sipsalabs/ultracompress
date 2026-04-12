"""EVOLUTIONARY FRR ARCHITECTURE SEARCH — Let the algorithm invent.

Instead of hand-designing the FRR block, EVOLVE it.
A genetic algorithm over FRR configurations finds the best architecture.

Genome: {n_scales, iters_per_scale, n_heads, ff_mult, lr, hidden_supervision}
Fitness: top10_accuracy * compression_ratio^0.5

Population of 12, evolve for 20 generations.
Each individual trains for 5K steps (fast eval).
Total: 12 * 20 = 240 evaluations = ~40 hours on one GPU.
Can run overnight.
"""
import torch, sys, os, time, json, math, gc, random, traceback, copy
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
EVAL_STEPS = 5000  # Quick eval per individual
POP_SIZE = 12
N_GENERATIONS = 20

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

teacher_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))


def eval_model(forward_fn, n=50):
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


# ============================================================
# GENOME: FRR architecture configuration
# ============================================================

def random_genome():
    """Generate a random FRR architecture."""
    n_scales = random.choice([2, 3, 4, 5, 7, 14])
    total_layers = 28
    iters = total_layers // n_scales
    return {
        'n_scales': n_scales,
        'iters_per_scale': iters,
        'n_heads': random.choice([4, 8, 16]),
        'ff_mult': random.choice([1, 2, 3]),
        'lr': random.choice([0.0003, 0.0005, 0.001]),
        'use_adapters': random.choice([True, False]),
        'adapter_rank': random.choice([8, 16, 32]),
    }


def mutate(genome):
    """Mutate one random gene."""
    g = copy.deepcopy(genome)
    gene = random.choice(list(g.keys()))
    if gene == 'n_scales':
        options = [2, 3, 4, 5, 7, 14]
        g['n_scales'] = random.choice(options)
        g['iters_per_scale'] = 28 // g['n_scales']
    elif gene == 'n_heads':
        g['n_heads'] = random.choice([4, 8, 16])
    elif gene == 'ff_mult':
        g['ff_mult'] = random.choice([1, 2, 3])
    elif gene == 'lr':
        g['lr'] *= random.choice([0.5, 0.7, 1.5, 2.0])
        g['lr'] = max(0.0001, min(0.01, g['lr']))
    elif gene == 'use_adapters':
        g['use_adapters'] = not g['use_adapters']
    elif gene == 'adapter_rank':
        g['adapter_rank'] = random.choice([8, 16, 32])
    return g


def crossover(g1, g2):
    """Combine two genomes."""
    child = {}
    for key in g1:
        child[key] = g1[key] if random.random() < 0.5 else g2[key]
    # Fix iters to match scales
    child['iters_per_scale'] = 28 // child['n_scales']
    return child


def evaluate_genome(genome, gen_id=""):
    """Build, train, and evaluate an FRR with this genome."""
    try:
        model = FractalModel(
            1024, genome['n_heads'], genome['n_scales'], genome['iters_per_scale'],
            151936, genome['ff_mult'],
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        if genome['use_adapters']:
            model.enable_adapters(rank=genome['adapter_rank'])

        params = model.fractal_params()
        compression = teacher_params / params

        trainable = [p for p in model.parameters() if p.requires_grad
                     and id(p) not in {id(ep) for ep in model.embed.parameters()}
                     and id(p) not in {id(lp) for lp in model.lm_head.parameters()}
                     and id(p) not in {id(np_) for np_ in model.norm.parameters()}]

        opt = torch.optim.AdamW(trainable, lr=genome['lr'], weight_decay=0.01)

        for step in range(EVAL_STEPS):
            lr = genome['lr'] * 0.5 * (1 + math.cos(step / EVAL_STEPS * math.pi))
            for pg in opt.param_groups: pg['lr'] = lr

            tokens = torch.randint(100, 100000, (8, 32), device=device)
            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)
            student_logits = model(tokens)
            B, T, V = student_logits.shape
            loss = F.kl_div(F.log_softmax(student_logits.reshape(-1,V)/2,-1),
                           F.softmax(teacher_logits.reshape(-1,V)/2,-1),
                           reduction='batchmean') * 4
            if torch.isnan(loss): break
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()

        t1, t10 = eval_model(lambda t, _m=model: _m(t))
        fitness = t10 * (compression ** 0.5)  # Balance quality and compression
        return {'top1': t1, 'top10': t10, 'params': params,
                'compression': compression, 'fitness': fitness, 'genome': genome}

    except Exception as e:
        return {'top1': 0, 'top10': 0, 'params': 0,
                'compression': 0, 'fitness': 0, 'genome': genome, 'error': str(e)}
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()


# ============================================================
# EVOLUTION
# ============================================================

print("=" * 70)
print(f"EVOLUTIONARY FRR SEARCH — Pop={POP_SIZE}, Gens={N_GENERATIONS}")
print(f"Each individual: {EVAL_STEPS} training steps")
print("=" * 70)

# Initialize population
population = [random_genome() for _ in range(POP_SIZE)]
best_ever = None
history = []

for gen in range(N_GENERATIONS):
    print(f"\n--- Generation {gen+1}/{N_GENERATIONS} ---")
    sys.stdout.flush()

    # Evaluate all individuals
    results = []
    for i, genome in enumerate(population):
        r = evaluate_genome(genome, f"G{gen+1}I{i+1}")
        results.append(r)
        print(f"  [{i+1}/{POP_SIZE}] scales={genome['n_scales']}x{genome['iters_per_scale']} "
              f"heads={genome['n_heads']} ff={genome['ff_mult']} lr={genome['lr']:.4f} "
              f"adapt={'Y' if genome['use_adapters'] else 'N'} "
              f"-> Top10={r['top10']*100:.0f}% {r['compression']:.0f}x fitness={r['fitness']:.3f}")
        sys.stdout.flush()

    # Sort by fitness
    results.sort(key=lambda x: x['fitness'], reverse=True)

    # Track best
    gen_best = results[0]
    if best_ever is None or gen_best['fitness'] > best_ever['fitness']:
        best_ever = gen_best
        print(f"  NEW BEST: Top10={best_ever['top10']*100:.0f}% "
              f"{best_ever['compression']:.0f}x fitness={best_ever['fitness']:.3f}")

    history.append({
        'generation': gen + 1,
        'best_fitness': gen_best['fitness'],
        'best_top10': gen_best['top10'],
        'best_genome': gen_best['genome'],
        'avg_fitness': sum(r['fitness'] for r in results) / len(results),
    })

    # Selection: keep top 4
    elite = [r['genome'] for r in results[:4]]

    # Create next generation
    new_pop = list(elite)  # Keep elites
    while len(new_pop) < POP_SIZE:
        if random.random() < 0.3:
            # Crossover
            p1, p2 = random.sample(elite, 2)
            child = crossover(p1, p2)
        else:
            # Mutation
            parent = random.choice(elite)
            child = mutate(parent)
        new_pop.append(child)

    population = new_pop

    # Save progress after each generation
    with open('evolution_progress.json', 'w') as f:
        json.dump({'best_ever': best_ever, 'history': history}, f, indent=2)


print(f"\n{'='*70}")
print(f"EVOLUTION COMPLETE — {N_GENERATIONS} generations")
print(f"{'='*70}")
print(f"BEST GENOME: {best_ever['genome']}")
print(f"  Top10={best_ever['top10']*100:.0f}% Compression={best_ever['compression']:.0f}x "
      f"Fitness={best_ever['fitness']:.3f}")
print(f"\nFitness over time:")
for h in history:
    print(f"  Gen {h['generation']:>2}: best={h['best_fitness']:.3f} avg={h['avg_fitness']:.3f} "
          f"top10={h['best_top10']*100:.0f}%")

with open('evolution_results.json', 'w') as f:
    json.dump({'best_ever': best_ever, 'history': history}, f, indent=2)
print(f"{'='*70}")
