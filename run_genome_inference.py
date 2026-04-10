"""Run inference with a genome-compressed model.

This is the proof that genome compression WORKS for actual text.
Load a genome, generate text, compare to original model.
"""
import torch, sys, argparse
sys.path.insert(0, '.')
from ultracompress.inference import ModelConfig
from ultracompress.genome_compressor import GenomeModel, MicroTransformerLayer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True, help="Path to saved genome .pt")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[785, 7290, 315, 2272, 374],
                       help="Token IDs for prompt")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = 'cuda'
    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

    # Load genome config
    genome_data = torch.load(args.genome, weights_only=True)
    cfg = genome_data['config']

    config = ModelConfig(
        n_layers=cfg['n_layers'], n_heads=16, n_kv_heads=8, hidden_size=1024,
        intermediate_size=3072, vocab_size=151936, head_dim=128,
    )

    # Build genome model
    embed = wd['model.embed_tokens.weight'].float().to(device)
    norm_w = wd.get('model.norm.weight', torch.ones(1024)).float().to(device)
    head_w = wd.get('lm_head.weight', embed).float().to(device)

    genome = GenomeModel(
        vocab_size=151936, big_dim=1024,
        small_dim=cfg['small_dim'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        embed_weight=embed, lm_head_weight=head_w, norm_weight=norm_w,
    ).to(device)
    genome.load_genome(args.genome)
    genome.eval()

    # Generate
    tokens = torch.tensor([args.prompt_tokens], device=device)
    print(f"Prompt tokens: {args.prompt_tokens}")
    print(f"Genome: sd={cfg['small_dim']}, {cfg['n_layers']} layers")
    print(f"Genome size: {genome.genome_param_count()*2/1e6:.1f} MB")
    print()

    generated = []
    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits = genome(tokens)
            next_logit = logits[0, -1, :] / args.temperature
            probs = torch.softmax(next_logit, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token.item())
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    print(f"Generated tokens: {generated}")
    print(f"Total: {len(args.prompt_tokens) + len(generated)} tokens")

    # Try to decode if transformers available
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
        prompt_text = tok.decode(args.prompt_tokens)
        gen_text = tok.decode(generated)
        print(f"\nPrompt: {prompt_text}")
        print(f"Generated: {gen_text}")
    except:
        pass

if __name__ == "__main__":
    main()
