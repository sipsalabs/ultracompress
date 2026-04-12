"""
STANDARD EVAL: Run real benchmarks on FRR models.
Tests: perplexity on WikiText-2, HellaSwag accuracy, basic QA.

This gives us numbers we can compare against published baselines
instead of just random-token top-10 matching.

Requires: a trained FRR model (.pt file)
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'


def load_teacher_and_frr(frr_path=None):
    """Load teacher + optionally load FRR from checkpoint."""
    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
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
    if teacher.lm_head is not None: teacher.lm_head = teacher.lm_head.to(device)

    embed_w = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    lm_head_w = gd['output.weight'].to(device)

    frr = None
    if frr_path and os.path.exists(frr_path):
        frr = FractalModel(
            hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
            vocab_size=151936, ff_mult=1,
            embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
        ).to(device)
        frr.load_state_dict(torch.load(frr_path, map_location=device))
        frr.eval()

    return teacher, frr


def eval_perplexity_wikitext(model_fn, tokenizer, n_samples=100, seq_len=128):
    """Approximate WikiText-2 perplexity using streaming samples."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds['text'] if len(t) > 100][:n_samples]
    except Exception:
        print("  Could not load WikiText-2, using random text approximation")
        return None

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text, max_length=seq_len + 1,
                                     truncation=True, return_tensors='pt')
            if tokens.shape[1] < 10:
                continue
            tokens = tokens.to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits = model_fn(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.numel()

    if total_tokens == 0:
        return None
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


def eval_hellaswag_approx(model_fn, tokenizer, n_samples=200):
    """Approximate HellaSwag by testing next-sentence prediction quality."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation")
        samples = list(ds)[:n_samples]
    except Exception:
        print("  Could not load HellaSwag")
        return None

    correct = 0
    total = 0

    with torch.no_grad():
        for sample in samples:
            ctx = sample['ctx']
            endings = sample['endings']
            label = int(sample['label'])

            scores = []
            for ending in endings:
                text = ctx + " " + ending
                tokens = tokenizer.encode(text, max_length=128, truncation=True,
                                         return_tensors='pt').to(device)
                ctx_len = len(tokenizer.encode(ctx, max_length=128, truncation=True))

                logits = model_fn(tokens[:, :-1])
                log_probs = F.log_softmax(logits[0], dim=-1)

                # Score = sum of log probs for ending tokens
                score = 0
                for i in range(ctx_len - 1, tokens.shape[1] - 1):
                    if i < log_probs.shape[0]:
                        score += log_probs[i, tokens[0, i + 1]].item()
                scores.append(score)

            pred = max(range(len(scores)), key=lambda i: scores[i])
            if pred == label:
                correct += 1
            total += 1

    return correct / total if total > 0 else None


def main():
    print("=" * 70)
    print("STANDARD EVALUATION: Real benchmarks")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    teacher, frr = load_teacher_and_frr("frr_100k_best.pt")

    models = [("Teacher (Qwen3-0.6B)", lambda t: teacher.forward(t, max_layers=28))]
    if frr is not None:
        models.append(("FRR 100K (60x compressed)", lambda t: frr(t)))

    for name, model_fn in models:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # WikiText-2 perplexity
        print("  Computing WikiText-2 perplexity...")
        ppl = eval_perplexity_wikitext(model_fn, tokenizer)
        if ppl:
            print(f"  WikiText-2 perplexity: {ppl:.2f}")
        else:
            print(f"  WikiText-2: could not compute")

        # HellaSwag
        print("  Computing HellaSwag accuracy (200 samples)...")
        acc = eval_hellaswag_approx(model_fn, tokenizer)
        if acc:
            print(f"  HellaSwag accuracy: {acc*100:.1f}%")
        else:
            print(f"  HellaSwag: could not compute")

    print("\nDone!")


if __name__ == '__main__':
    main()
