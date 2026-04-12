"""
PROPER TEXT-BASED EVAL: Measures actual language model quality, not random-token matching.

Three metrics:
1. Perplexity on real text (FineWeb-Edu) — the standard LM metric
2. Next-token accuracy on real text — does the model predict real language?
3. Text generation quality — qualitative samples

This fixes the eval metric mismatch we found overnight: real-text-trained models
score low on random-token eval but may have better actual language quality.
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel


def load_teacher(device='cuda'):
    """Load Qwen3-0.6B teacher."""
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
    return teacher, gd


def get_real_text_batches(tokenizer, n_batches=50, seq_len=64, batch_size=4):
    """Get batches of real tokenized text from FineWeb-Edu."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    ds_iter = iter(ds)

    batches = []
    for _ in range(n_batches):
        tokens_list = []
        for _ in range(batch_size):
            while True:
                try:
                    sample = next(ds_iter)
                    text = sample.get('text', '')
                    if len(text) < 100:
                        continue
                    toks = tokenizer.encode(text, max_length=seq_len + 1,
                                           truncation=True, return_tensors='pt')[0]
                    if len(toks) >= seq_len + 1:
                        tokens_list.append(toks[:seq_len + 1])
                        break
                except StopIteration:
                    ds_iter = iter(ds)
        batches.append(torch.stack(tokens_list))
    return batches


def eval_perplexity(model_fn, batches, device='cuda'):
    """Compute perplexity on real text batches."""
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model_fn(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return perplexity, avg_loss


def eval_next_token_accuracy(model_fn, batches, device='cuda'):
    """What fraction of next tokens does the model predict correctly?"""
    correct = 0
    total = 0
    top5_correct = 0
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model_fn(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            top5 = logits.topk(5, dim=-1).indices
            top5_correct += (top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
            total += targets.numel()
    return correct / total, top5_correct / total


def eval_generation(model, tokenizer, prompts, device='cuda', max_tokens=50):
    """Generate text from prompts and return for qualitative review."""
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = model(generated)
                next_logit = logits[0, -1] / 0.7  # temperature
                probs = F.softmax(next_logit, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        results.append((prompt, text))
    return results


def full_eval(model_fn, model_for_gen, name, batches, tokenizer, device='cuda'):
    """Run all three eval metrics."""
    print(f"\n{'='*60}")
    print(f"  EVAL: {name}")
    print(f"{'='*60}")

    # Perplexity
    ppl, avg_loss = eval_perplexity(model_fn, batches, device)
    print(f"  Perplexity: {ppl:.2f} (avg_loss: {avg_loss:.4f})")

    # Next-token accuracy
    top1_acc, top5_acc = eval_next_token_accuracy(model_fn, batches, device)
    print(f"  Next-token accuracy: top1={top1_acc*100:.1f}% top5={top5_acc*100:.1f}%")

    # Generation
    prompts = [
        "The future of artificial intelligence is",
        "In a distant galaxy, scientists discovered",
        "The most important principle in physics is",
    ]
    if model_for_gen is not None:
        gens = eval_generation(model_for_gen, tokenizer, prompts, device)
        for prompt, text in gens:
            print(f"\n  Prompt: '{prompt}'")
            print(f"  Output: {text[:200]}")

    return {'perplexity': ppl, 'avg_loss': avg_loss, 'top1_acc': top1_acc, 'top5_acc': top5_acc}


if __name__ == '__main__':
    device = 'cuda'
    print("Loading teacher and tokenizer...")
    teacher, gd = load_teacher(device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    print("Loading real text eval batches (50 batches x 4 samples x 64 tokens)...")
    batches = get_real_text_batches(tokenizer, n_batches=50, seq_len=64, batch_size=4)
    print(f"Loaded {len(batches)} batches")

    embed_w = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    lm_head_w = gd['output.weight'].to(device)

    # Eval teacher
    full_eval(lambda t: teacher.forward(t, max_layers=28), None,
              "Teacher (Qwen3-0.6B)", batches, tokenizer, device)

    # Eval FRR if model exists
    if os.path.exists('frr_100k_best.pt'):
        print("\nLoading FRR 100K model...")
        model = FractalModel(
            hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
            vocab_size=151936, ff_mult=1,
            embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
        ).to(device)
        model.load_state_dict(torch.load('frr_100k_best.pt', map_location=device))
        full_eval(lambda t: model(t), model, "FRR 100K", batches, tokenizer, device)
    else:
        print("\nNo FRR model found (frr_100k_best.pt). Run run_100k_train.py first.")

    print("\nDone!")
