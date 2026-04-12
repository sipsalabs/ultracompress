"""THE ARENA — Automated sandbox for testing compression approaches.

Runs multiple compression strategies head-to-head on the same model.
Each "fighter" gets the same training budget, same eval.
Best fighter wins and gets scaled up.

Usage:
  python arena.py              # Run all fighters
  python arena.py --quick      # Quick 2K steps each
  python arena.py --fighter X  # Run specific fighter
"""
import torch, sys, os, time, json, argparse
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer

device = 'cuda'


class ArenaFighter:
    """Base class for compression approaches."""
    name = "base"

    def build(self, big_dim, n_layers, device):
        """Build the model. Return (model, trainable_params_list)."""
        raise NotImplementedError

    def forward(self, model, tokens):
        """Forward pass. Return logits."""
        raise NotImplementedError

    def param_count(self, model):
        """Count trainable parameters."""
        raise NotImplementedError


class GenomeV1Fighter(ArenaFighter):
    name = "Genome-V1-sd128"

    def build(self, big_dim, n_layers, device, embed, head, norm):
        from ultracompress.genome_compressor import GenomeModel
        model = GenomeModel(
            vocab_size=151936, big_dim=big_dim, small_dim=128, n_heads=4,
            n_layers=n_layers,
            embed_weight=embed, lm_head_weight=head, norm_weight=norm,
        ).to(device)
        return model, list(model.genome_layers.parameters())

    def forward(self, model, tokens):
        return model(tokens, max_layers=model.n_layers)

    def param_count(self, model):
        return model.genome_param_count()


class NeuralDNAFighter(ArenaFighter):
    name = "NeuralDNA-512g"

    def build(self, big_dim, n_layers, device, embed, head, norm):
        from sandbox2_neuraldna import NeuralDNAModel
        model = NeuralDNAModel(
            vocab_size=151936, big_dim=big_dim,
            n_genes=512, gene_dim=128, n_active=64,
            n_layers=n_layers,
            embed_weight=embed, lm_head_weight=head, norm_weight=norm,
        ).to(device)
        params = list(model.dna_bank.parameters()) + list(model.genome_layers.parameters())
        return model, params

    def forward(self, model, tokens):
        return model(tokens, max_layers=model.n_layers)

    def param_count(self, model):
        return model.genome_param_count()


class MoEFighter(ArenaFighter):
    name = "MoE-8x32"

    def build(self, big_dim, n_layers, device, embed, head, norm):
        from ultracompress.genome_moe import MoEGenomeModel
        model = MoEGenomeModel(
            vocab_size=151936, big_dim=big_dim,
            expert_dim=32, n_experts=8, top_k=2,
            n_layers=n_layers,
            embed_weight=embed, lm_head_weight=head, norm_weight=norm,
        ).to(device)
        return model, list(model.genome_layers.parameters())

    def forward(self, model, tokens):
        return model(tokens, max_layers=model.n_layers)

    def param_count(self, model):
        return model.genome_param_count()


class MultiViewFighter(ArenaFighter):
    name = "MultiView-V2a"

    def build(self, big_dim, n_layers, device, embed, head, norm):
        from ultracompress.genome_v2 import MultiViewGenomeLayer

        class V2aModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding.from_pretrained(embed, freeze=True)
                self.lm_head = nn.Linear(big_dim, 151936, bias=False)
                self.lm_head.weight = nn.Parameter(head, requires_grad=False)
                self.norm = nn.RMSNorm(big_dim)
                self.norm.weight = nn.Parameter(norm, requires_grad=False)
                self.n_layers = n_layers
                self.genome_layers = nn.ModuleList([
                    MultiViewGenomeLayer(big_dim, small_dim=128, n_views=4)
                    for _ in range(n_layers)
                ])

            def forward(self, token_ids, max_layers=None):
                x = self.embed(token_ids).float()
                n = max_layers or self.n_layers
                for i in range(min(n, len(self.genome_layers))):
                    x = x + self.genome_layers[i](x)
                return self.lm_head(self.norm(x))

            def genome_param_count(self):
                return sum(p.numel() for p in self.genome_layers.parameters())

        model = V2aModel().to(device)
        return model, list(model.genome_layers.parameters())

    def forward(self, model, tokens):
        return model(tokens, max_layers=model.n_layers)

    def param_count(self, model):
        return model.genome_param_count()


def run_arena(fighters, n_steps=5000, batch_size=8, quick=False):
    """Run all fighters head-to-head."""
    if quick:
        n_steps = 2000

    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
    config = ModelConfig(n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
                         intermediate_size=3072, vocab_size=151936, head_dim=128)

    hf_to_gguf = {'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight', 'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight', 'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight', 'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.down_proj.weight': 'ffn_down.weight'}
    gd = {}
    gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
    gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(1024)).float()
    gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
    for li in range(28):
        for h, g in hf_to_gguf.items():
            k = f'model.layers.{li}.{h}'
            if k in wd: gd[f'blk.{li}.{g}'] = wd[k].float()

    teacher = MiniTransformer(config, device)
    teacher.load_weights(gd)

    embed = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    head_w = gd['output.weight'].to(device)

    def eval_fighter(model, fighter, n=100):
        t1, t10s = 0, []
        for trial in range(n):
            torch.manual_seed(trial * 13 + 9999)
            t = torch.randint(100, 50000, (1, 16), device=device)
            with torch.no_grad():
                tl = teacher.forward(t, max_layers=28)
                tp = tl[0, -1].argmax().item()
                tt10 = set(tl[0, -1].topk(10).indices.tolist())
                gl = fighter.forward(model, t)
                gp = gl[0, -1].argmax().item()
                gt10 = set(gl[0, -1].topk(10).indices.tolist())
                if tp == gp: t1 += 1
                t10s.append(len(tt10 & gt10) / 10)
        return t1 / n, sum(t10s) / len(t10s)

    results = []

    print("=" * 70)
    print("THE ARENA — Compression Approach Tournament")
    print(f"Budget: {n_steps} steps each, batch_size={batch_size}, online training")
    print("=" * 70)

    for fighter in fighters:
        print(f"\n{'─'*60}")
        print(f"FIGHTER: {fighter.name}")
        print(f"{'─'*60}")

        try:
            model, params = fighter.build(1024, 28, device, embed, head_w, norm_w)
            n_params = fighter.param_count(model)
            print(f"  Params: {n_params:,} ({n_params*2/1e6:.1f} MB)")

            opt = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

            t0 = time.time()
            best_t10 = 0
            for step in range(n_steps):
                tokens = torch.randint(100, 100000, (batch_size, 32), device=device)
                with torch.no_grad():
                    teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]

                student_logits = fighter.forward(model, tokens)[:, -1, :]
                loss = F.kl_div(
                    F.log_softmax(student_logits / 2, -1),
                    F.softmax(teacher_logits / 2, -1),
                    reduction='batchmean',
                ) * 4

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()

                if step % (n_steps // 5) == 0:
                    t1, t10 = eval_fighter(model, fighter, n=50)
                    elapsed = time.time() - t0
                    speed = (step + 1) / elapsed if elapsed > 0 else 0
                    print(f"  Step {step:>5}: loss={loss.item():.3f} Top1={t1*100:.0f}% Top10={t10*100:.0f}% [{speed:.1f} s/s]")
                    sys.stdout.flush()
                    if t10 > best_t10:
                        best_t10 = t10

            # Final eval
            t1_final, t10_final = eval_fighter(model, fighter, n=100)
            elapsed = time.time() - t0

            result = {
                'name': fighter.name,
                'params': n_params,
                'size_mb': n_params * 2 / 1e6,
                'top1': t1_final,
                'top10': t10_final,
                'best_top10': best_t10,
                'time': elapsed,
                'steps': n_steps,
            }
            results.append(result)

            print(f"\n  >>> {fighter.name}: Top1={t1_final*100:.0f}% Top10={t10_final*100:.0f}% ({n_params*2/1e6:.1f}MB, {elapsed:.0f}s)")

        except Exception as e:
            print(f"  CRASHED: {e}")
            results.append({'name': fighter.name, 'error': str(e)})

        # Cleanup
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()

    # Leaderboard
    print(f"\n{'='*70}")
    print("ARENA LEADERBOARD")
    print(f"{'='*70}")
    valid = [r for r in results if 'error' not in r]
    valid.sort(key=lambda x: x['top10'], reverse=True)
    for i, r in enumerate(valid):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(f"  {medal} {r['name']:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% Size={r['size_mb']:>6.1f}MB")

    # Save results
    with open('arena_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to arena_results.json")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()

    fighters = [
        GenomeV1Fighter(),
        MultiViewFighter(),
        NeuralDNAFighter(),
        MoEFighter(),
    ]

    run_arena(fighters, n_steps=args.steps, quick=args.quick)
