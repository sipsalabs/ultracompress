"""API-Based Compression — Compress models you can't download.

For models that only exist behind APIs (GPT-4 class, future 10T+ models),
you can't download weights. But you CAN query them.

This module builds a genome by querying the model's API:
1. Send diverse prompts to the API
2. Record the logit distributions (or top-k tokens + probs)
3. Train a local genome to match those distributions
4. The genome runs locally forever — no more API calls needed

Cost: ~$50-500 in API calls to build the genome (one-time)
Result: Run the model locally forever at zero ongoing cost

This is how Athena absorbs the intelligence of any model she can talk to.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import Callable, Dict, List, Tuple


class APICompressor:
    """Compress any API-accessible model into a local genome."""

    def __init__(self, api_fn: Callable = None, device='cuda'):
        """
        Args:
            api_fn: Function that takes a list of token IDs and returns
                    logits or top-k (token_id, probability) pairs.
                    Signature: api_fn(token_ids: List[int]) -> Dict
                    Returns: {'top_tokens': [...], 'top_probs': [...]}
        """
        self.api_fn = api_fn
        self.device = device
        self.cache_dir = "api_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def build_cache_from_api(
        self,
        n_queries: int = 10000,
        seq_len: int = 32,
        top_k: int = 100,
        vocab_size: int = 151936,
        save_path: str = None,
    ) -> dict:
        """Query the API and build a local cache of responses.

        Each query: send random tokens, get back top-k logits.
        Total API calls: n_queries
        Estimated cost: n_queries * $0.001-0.01 per call = $10-100

        The cache is saved to disk so you never need to re-query.
        """
        if save_path is None:
            save_path = os.path.join(self.cache_dir, "api_cache.pt")

        # Check for existing cache
        if os.path.exists(save_path):
            print(f"Loading existing cache from {save_path}")
            return torch.load(save_path, weights_only=True)

        print(f"Querying API for {n_queries} samples...")
        all_tokens = []
        all_top_tokens = []
        all_top_probs = []

        for i in range(n_queries):
            # Generate random input
            torch.manual_seed(i)
            tokens = torch.randint(100, vocab_size, (seq_len,)).tolist()

            # Query API
            response = self.api_fn(tokens)

            all_tokens.append(tokens)
            all_top_tokens.append(response['top_tokens'][:top_k])
            all_top_probs.append(response['top_probs'][:top_k])

            if i % 100 == 0:
                print(f"  {i}/{n_queries}")

        cache = {
            'tokens': torch.tensor(all_tokens),
            'top_tokens': torch.tensor(all_top_tokens),
            'top_probs': torch.tensor(all_top_probs),
            'n_queries': n_queries,
            'top_k': top_k,
            'vocab_size': vocab_size,
        }

        torch.save(cache, save_path)
        print(f"Cache saved to {save_path}")
        return cache

    def simulate_api_from_local(self, model, tokenizer=None):
        """Create an API function from a local model (for testing).

        This lets us test the API compression pipeline using
        a local model as the "API endpoint."
        """
        def api_fn(token_ids):
            tokens = torch.tensor([token_ids], device=self.device)
            with torch.no_grad():
                logits = model.forward(tokens)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                top_probs, top_tokens = probs.topk(100)
            return {
                'top_tokens': top_tokens.cpu().tolist(),
                'top_probs': top_probs.cpu().tolist(),
            }
        return api_fn

    def train_genome_from_cache(
        self,
        cache: dict,
        genome_model,
        n_steps: int = 20000,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True,
    ):
        """Train a genome model from cached API responses.

        Uses a sparse cross-entropy loss on the top-k tokens
        (we don't have full logits from the API, just top-k).
        """
        tokens_data = cache['tokens'].to(self.device)
        top_tokens_data = cache['top_tokens'].to(self.device)
        top_probs_data = cache['top_probs'].to(self.device)
        n_samples = tokens_data.shape[0]

        opt = torch.optim.AdamW(
            genome_model.genome_layers.parameters(),
            lr=lr, weight_decay=0.005
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

        for step in range(n_steps):
            idx = torch.randint(0, n_samples, (batch_size,))
            tokens = tokens_data[idx]
            target_tokens = top_tokens_data[idx]  # (B, top_k)
            target_probs = top_probs_data[idx]    # (B, top_k)

            student_logits = genome_model(tokens)[:, -1, :]  # (B, vocab)

            # Sparse KL: only compute loss on the top-k tokens
            # Gather student logits at target token positions
            student_at_targets = student_logits.gather(1, target_tokens.long())
            student_probs = F.softmax(student_at_targets, dim=1)

            # KL divergence on the top-k distribution
            loss = F.kl_div(
                student_probs.log().clamp(min=-100),
                target_probs,
                reduction='batchmean',
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(genome_model.genome_layers.parameters(), 1.0)
            opt.step()
            sched.step()

            if verbose and step % 5000 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

        return genome_model
