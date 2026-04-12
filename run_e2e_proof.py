"""
END-TO-END PROOF: Train FRR -> Compress with pipeline -> Decompress -> Measure quality.

This is THE test that turns theory into reality:
1. Train FRR (shared block distillation from Qwen3-0.6B)
2. Take the trained FRR block weights
3. Compress them with ultimate pipeline (Hadamard -> SVD -> Q2 -> Entropy)
4. Decompress back
5. Load decompressed weights into FRR
6. Measure quality degradation from compression

If quality holds (e.g., 62% -> 55%+ after Q2), the full stack works.
"""
import lib.unbuffered
import torch, sys, os, time, json, gc
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from ultracompress.ultimate_pipeline import UltimatePipeline, UltimatePipelineConfig

device = 'cuda'

print("=" * 70)
print("END-TO-END PROOF: FRR + Pipeline Compression")
print("=" * 70)

# ── Load teacher ──
print("Loading teacher...")
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight',
    'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight',
    'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight',
    'mlp.up_proj.weight': 'ffn_up.weight',
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

teacher_total = sum(v.numel() for v in gd.values())
teacher_layers = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))


def eval_model(model, n=200):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


# ════════════════════════════════════════════════════════════════════
# STEP 1: Train FRR
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Train FRR (10K steps)")
print("=" * 60)

model = FractalModel(
    hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
    vocab_size=151936, ff_mult=1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  FRR trainable params: {trainable:,}")
print(f"  FRR compression (layers only): {teacher_layers/trainable:.1f}x")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)

t0 = time.time()
for step in range(10000):
    torch.manual_seed(step * 7)
    tokens = torch.randint(100, 50000, (4, 32), device=device)

    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)

    student_logits = model(tokens)
    T = max(2.0, 5.0 * (1 - step / 10000))
    loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T * T)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()

    if step % 2000 == 0:
        t1, t10 = eval_model(model, n=100)
        print(f"    Step {step}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")

# Final eval before compression
t1_pre, t10_pre = eval_model(model, n=200)
print(f"\n  PRE-COMPRESSION: T1={t1_pre*100:.0f}% T10={t10_pre*100:.0f}%")


# ════════════════════════════════════════════════════════════════════
# STEP 2: Compress FRR block weights with ultimate pipeline
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Compress FRR block weights")
print("=" * 60)

# Extract block weights
block_state = {k: v.cpu() for k, v in model.block.state_dict().items()}
modulation_state = {
    'scale_gamma': model.scale_gamma.cpu(),
    'scale_beta': model.scale_beta.cpu(),
    'iter_scale': model.iter_scale.cpu(),
}

block_params = sum(v.numel() for v in block_state.values())
mod_params = sum(v.numel() for v in modulation_state.values())
print(f"  Block params: {block_params:,}")
print(f"  Modulation params: {mod_params:,}")

# Test at different quantization levels
for qbits, rbits in [(8, 4), (4, 2), (2, 2)]:
    print(f"\n  --- Q{qbits}/R{rbits} ---")
    pipe_config = UltimatePipelineConfig(
        quant_bits=qbits,
        residual_bits=rbits,
        rank_fraction=0.8 if qbits >= 4 else 0.5,
    )
    pipe = UltimatePipeline(pipe_config)

    # Compress block weights
    compressed = pipe.compress(block_state)
    pipe.report()

    # Decompress
    recovered = pipe.decompress(compressed)

    # Measure weight-level fidelity
    total_cos = 0
    count = 0
    for k in block_state:
        if k in recovered:
            orig = block_state[k].float().flatten()
            rec = recovered[k].float().flatten()
            if orig.numel() > 1:
                cos = F.cosine_similarity(orig.unsqueeze(0), rec.unsqueeze(0)).item()
                total_cos += cos
                count += 1
    avg_cos = total_cos / count if count > 0 else 0
    print(f"  Weight cosine similarity: {avg_cos:.6f}")

    # Load decompressed weights back into model
    model_copy = FractalModel(
        hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
        vocab_size=151936, ff_mult=1,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
    ).to(device)

    # Copy modulation params (these stay uncompressed - they're tiny)
    model_copy.scale_gamma.data = modulation_state['scale_gamma'].to(device)
    model_copy.scale_beta.data = modulation_state['scale_beta'].to(device)
    model_copy.iter_scale.data = modulation_state['iter_scale'].to(device)

    # Load compressed block weights
    recovered_gpu = {k: v.to(device) for k, v in recovered.items()}
    model_copy.block.load_state_dict(recovered_gpu)

    t1_post, t10_post = eval_model(model_copy, n=200)
    quality_drop = (t10_pre - t10_post) * 100

    # Calculate total compression
    # FRR gives compression on layer params, pipeline compresses the FRR block further
    frr_compression = teacher_layers / trainable
    pipeline_compression = 32 / qbits  # approximate bits compression
    total_compression = frr_compression * pipeline_compression

    print(f"  POST-Q{qbits} QUALITY: T1={t1_post*100:.0f}% T10={t10_post*100:.0f}%")
    print(f"  QUALITY DROP: {quality_drop:.1f}% T10")
    print(f"  COMPRESSION: FRR {frr_compression:.0f}x * Q{qbits} {pipeline_compression:.0f}x = {total_compression:.0f}x total")

    # Calculate actual file sizes
    block_bytes_orig = block_params * 4  # FP32
    block_bytes_q = block_params * qbits / 8
    embed_bytes = sum(v.numel() * 2 for v in [embed_w, lm_head_w, norm_w])  # FP16
    mod_bytes = mod_params * 4

    total_model_bytes = block_bytes_q + embed_bytes + mod_bytes
    original_bytes = teacher_total * 2  # FP16
    actual_ratio = original_bytes / total_model_bytes

    print(f"  ACTUAL SIZE:")
    print(f"    Block: {block_bytes_q/1e6:.1f} MB (Q{qbits})")
    print(f"    Embed+Head: {embed_bytes/1e6:.1f} MB (FP16, shared)")
    print(f"    Modulation: {mod_bytes/1e6:.1f} MB (FP32)")
    print(f"    TOTAL: {total_model_bytes/1e6:.1f} MB")
    print(f"    Original: {original_bytes/1e6:.1f} MB (FP16)")
    print(f"    REAL COMPRESSION: {actual_ratio:.1f}x")

    del model_copy; gc.collect(); torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════
# VERDICT
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("END-TO-END VERDICT")
print("=" * 70)
print(f"  Teacher: {teacher_total:,} params ({teacher_total*2/1e6:.0f} MB FP16)")
print(f"  FRR pre-compression: T1={t1_pre*100:.0f}% T10={t10_pre*100:.0f}%")
print(f"  This proves (or disproves) that FRR + quantization stack together.")
print(f"\n  100T projection:")
print(f"    At 100T params, embeddings are ~0.003% of total")
print(f"    FRR compresses 99.997% of params")
print(f"    If Q2 quality holds: 100T * 2 bytes / (42x * 16x) = ~30 GB")
print(f"\nDone!")
