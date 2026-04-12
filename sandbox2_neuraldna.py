"""SANDBOX 2: Neural DNA Expression — Neurons that change behavior by context.

THE PARADIGM SHIFT:
In biology, every cell has the same DNA but expresses different genes
based on environment. A liver cell and neuron have identical DNA but
behave completely differently.

Apply this to neural networks:
- Every "neuron" has a DNA sequence (small learned vector)
- An "expression function" reads the context and activates different
  parts of the DNA for different inputs
- One neuron can act like THOUSANDS of different neurons
- A 1M neuron model behaves like a 1B neuron model

This is NOT:
- MoE (fixed experts with routing)
- Adaptive computation (skipping layers)
- HyperNetworks (generating weights from scratch)

This IS:
- Neurons with internal state that modulates per-input
- Like biological gene expression: same DNA, different phenotype
- The "DNA" is shared across layers (cross-layer parameter sharing)
- The "expression" is input-dependent (context-aware)

For 1000T target:
- DNA bank: 10M parameters (shared across ALL layers)
- Expression function: 1M params per layer * 200 layers = 200M
- Total: ~210M params = ~420 MB at FP16
- 1000T model in 420MB would be 0.0000003 BPW

Runs on GPU 0 (cuda:0).
"""
import torch, sys, os, time
import torch.nn as nn
import torch.nn.functional as F
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer

device = 'cuda'


class DNABank(nn.Module):
    """Shared DNA bank — the "genome" that all layers read from.

    Contains N DNA sequences, each a small vector.
    Layers select and combine DNA sequences based on context.
    """
    def __init__(self, n_genes, gene_dim):
        super().__init__()
        self.n_genes = n_genes
        self.gene_dim = gene_dim
        # The DNA: a bank of gene vectors
        self.genes = nn.Parameter(torch.randn(n_genes, gene_dim) * 0.02)

    def express(self, expression_weights):
        """Given expression weights (B, T, n_genes), produce expressed phenotype.

        Like gene expression: weights determine how much of each gene
        is "turned on" for this specific input context.
        Returns: (B, T, gene_dim)
        """
        # expression_weights: (B, T, n_genes) — soft selection
        # genes: (n_genes, gene_dim)
        return torch.matmul(expression_weights, self.genes)  # (B, T, gene_dim)


class ExpressionFunction(nn.Module):
    """Determines WHICH genes to express based on input context.

    Like transcription factors in biology — reads the input and
    decides which genes to activate.
    """
    def __init__(self, input_dim, n_genes, n_active=32):
        super().__init__()
        self.n_genes = n_genes
        self.n_active = n_active  # Top-K genes to express

        # Context encoder — reads input, produces gene activation pattern
        self.context_proj = nn.Linear(input_dim, n_genes, bias=False)
        # Temperature for gene selection sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """x: (B, T, input_dim) -> expression_weights: (B, T, n_genes)"""
        logits = self.context_proj(x) / self.temperature.clamp(min=0.1)

        # Sparse activation: only top-K genes expressed (like real biology)
        if self.n_active < self.n_genes:
            top_vals, top_idx = logits.topk(self.n_active, dim=-1)
            sparse_logits = torch.full_like(logits, float('-inf'))
            sparse_logits.scatter_(-1, top_idx, top_vals)
            return F.softmax(sparse_logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class NeuralDNALayer(nn.Module):
    """A single layer that uses DNA expression instead of fixed weights.

    Process:
    1. Read input context
    2. Express relevant genes from DNA bank
    3. Use expressed genes as the "weights" for this input
    4. Apply transformation
    5. Project back to model space
    """
    def __init__(self, big_dim, dna_bank, n_active=32, ff_mult=2):
        super().__init__()
        self.big_dim = big_dim
        self.dna_bank = dna_bank
        gene_dim = dna_bank.gene_dim

        # Expression function — decides which genes to activate
        self.expression = ExpressionFunction(big_dim, dna_bank.n_genes, n_active)

        # Transform expressed genes into a weight-like transformation
        # The expressed genes become a dynamic "weight matrix"
        self.gene_to_down = nn.Linear(gene_dim, big_dim, bias=False)  # gene -> projection weights
        self.gene_to_up = nn.Linear(gene_dim, big_dim, bias=False)

        # Small fixed components
        self.norm = nn.RMSNorm(big_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Initialize small
        nn.init.zeros_(self.gene_to_up.weight)

    def forward(self, x):
        """x: (B, T, big_dim) -> delta: (B, T, big_dim)"""
        xn = self.norm(x)

        # Step 1: Express genes based on context
        expr_weights = self.expression(xn)  # (B, T, n_genes)
        expressed = self.dna_bank.express(expr_weights)  # (B, T, gene_dim)

        # Step 2: Use expressed genes as dynamic weights
        # The expressed genes modulate how the input is transformed
        # This is the key: different inputs activate different genes,
        # creating different effective weight matrices
        down = torch.tanh(self.gene_to_down(expressed))  # (B, T, big_dim)
        gate = down * xn  # element-wise gating based on expressed genes

        # Step 3: Project through gene-dependent transformation
        up = self.gene_to_up(expressed)  # (B, T, big_dim)
        delta = gate * up  # (B, T, big_dim)

        return delta * self.scale


class NeuralDNAModel(nn.Module):
    """Complete model with Neural DNA expression layers."""

    def __init__(self, vocab_size, big_dim, n_genes, gene_dim, n_active,
                 n_layers, embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.big_dim = big_dim
        self.n_layers = n_layers

        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, big_dim)

        if lm_head_weight is not None:
            self.lm_head = nn.Linear(big_dim, vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        else:
            self.lm_head = nn.Linear(big_dim, vocab_size, bias=False)

        if norm_weight is not None:
            self.norm = nn.RMSNorm(big_dim)
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)
        else:
            self.norm = nn.RMSNorm(big_dim)

        # SHARED DNA bank — same genes used by ALL layers
        # This is the key to compression: one DNA, many expressions
        self.dna_bank = DNABank(n_genes, gene_dim)

        # Per-layer expression functions (tiny)
        self.genome_layers = nn.ModuleList([
            NeuralDNALayer(big_dim, self.dna_bank, n_active)
            for _ in range(n_layers)
        ])

    def forward(self, token_ids, max_layers=None):
        x = self.embed(token_ids).float()
        n = max_layers or self.n_layers
        for i in range(min(n, len(self.genome_layers))):
            x = x + self.genome_layers[i](x)
        x = self.norm(x)
        return self.lm_head(x)

    def genome_param_count(self):
        # DNA bank + all expression layers
        dna_params = sum(p.numel() for p in self.dna_bank.parameters())
        layer_params = sum(p.numel() for p in self.genome_layers.parameters())
        return dna_params + layer_params


# ============================================================
# RUN THE EXPERIMENT
# ============================================================

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

embed = gd['token_embd.weight']
norm_w = gd['output_norm.weight']
head_w = gd['output.weight']

print("=" * 60)
print("SANDBOX 2: NEURAL DNA EXPRESSION")
print("Same DNA, different expression per input — like biology")
print("=" * 60)


def eval_model(model, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model(t, max_layers=28)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


# Test different DNA configurations
configs = [
    # (n_genes, gene_dim, n_active, name)
    (256, 64, 32, "256 genes, dim=64, top-32"),
    (512, 128, 64, "512 genes, dim=128, top-64"),
    (1024, 256, 128, "1024 genes, dim=256, top-128"),
]

for n_genes, gene_dim, n_active, name in configs:
    print(f"\n--- {name} ---")

    model = NeuralDNAModel(
        vocab_size=151936, big_dim=1024,
        n_genes=n_genes, gene_dim=gene_dim, n_active=n_active,
        n_layers=28,
        embed_weight=embed.to(device),
        lm_head_weight=head_w.to(device),
        norm_weight=norm_w.to(device),
    ).to(device)

    total_params = model.genome_param_count()
    dna_params = sum(p.numel() for p in model.dna_bank.parameters())
    print(f"  DNA bank: {dna_params:,} params (SHARED across all 28 layers)")
    print(f"  Total genome: {total_params:,} params ({total_params*2/1e6:.1f} MB)")

    # Online training — deduplicate params (DNA bank is shared inside layers)
    seen = set()
    trainable = []
    for p in list(model.dna_bank.parameters()) + list(model.genome_layers.parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)

    t0 = time.time()
    best_t10 = 0
    for step in range(10000):
        tokens = torch.randint(100, 100000, (8, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]

        student_logits = model(tokens, max_layers=28)[:, -1, :]
        loss = F.kl_div(
            F.log_softmax(student_logits / 2, -1),
            F.softmax(teacher_logits / 2, -1),
            reduction='batchmean',
        ) * 4

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        if step % 2000 == 0:
            t1, t10 = eval_model(model, n=50)
            elapsed = time.time() - t0
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"    Step {step:>5}: loss={loss.item():.4f} Top1={t1*100:.0f}% Top10={t10*100:.0f}% [{speed:.1f} s/s]")
            sys.stdout.flush()
            if t10 > best_t10:
                best_t10 = t10

    t1_final, t10_final = eval_model(model, n=100)
    elapsed = time.time() - t0
    print(f"  RESULT: Top1={t1_final*100:.0f}% Top10={t10_final*100:.0f}%")
    print(f"  Size: {total_params*2/1e6:.1f} MB | DNA shared: {dna_params*2/1e6:.1f} MB")
    print(f"  Best Top10: {best_t10*100:.0f}% | Time: {elapsed:.0f}s")

    # Scaling projection
    if total_params > 0:
        bpw = total_params * 16 / (440_000_000)  # vs 0.6B original params
        print(f"  Effective BPW: {bpw:.4f}")
        # Project to 1000T
        # DNA bank scales sub-linearly (shared), expression scales linearly with layers
        projected_1000T_gb = (dna_params * 2 + (total_params - dna_params) * 2 * (200/28)) / 1e9
        print(f"  Projected 1000T size: {projected_1000T_gb:.1f} GB")

    del model
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("SANDBOX 2 COMPLETE — Neural DNA Expression")
print(f"{'='*60}")
