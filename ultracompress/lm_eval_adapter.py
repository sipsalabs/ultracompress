"""
lm-eval-harness adapter for FRR models.

Enables evaluation on standard benchmarks:
  MMLU, HellaSwag, ARC-Challenge, WinoGrande, TruthfulQA, GSM8K

Usage:
  lm_eval --model frr --model_args frr_path=frr_100k_best.pt --tasks mmlu,hellaswag

Or programmatically:
  from ultracompress.lm_eval_adapter import FRRModelAdapter
  model = FRRModelAdapter(frr_path="frr_100k_best.pt")
"""
import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    HAS_LM_EVAL = True
except ImportError:
    # Stub for when lm-eval isn't installed
    HAS_LM_EVAL = False
    class LM:
        pass
    def register_model(name):
        def decorator(cls):
            return cls
        return decorator


@register_model("frr")
class FRRModelAdapter(LM):
    """lm-eval-harness adapter for FRR compressed models."""

    def __init__(self, frr_path="frr_100k_best.pt", base_model="Qwen/Qwen3-0.6B",
                 device="cuda", batch_size=1, **kwargs):
        super().__init__()
        self._device = device
        self._batch_size = batch_size

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        # Load FRR model
        from ultracompress.hf_model import FRRForCausalLM
        self.model = FRRForCausalLM.from_frr(frr_path, base_model, device)
        self.model.eval()

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """Run model on input tensor, return logits."""
        with torch.no_grad():
            output = self.model(inps.to(self._device))
            return output.logits

    def _model_generate(self, context, max_length, eos_token_id):
        """Generate tokens from context."""
        with torch.no_grad():
            return self.model.generate(
                context.to(self._device),
                max_new_tokens=max_length,
                do_sample=False,
            )

    def loglikelihood(self, requests):
        """Compute log-likelihood for each request."""
        results = []
        for context, continuation in [req.args for req in requests]:
            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)
            all_tokens = torch.tensor([ctx_tokens + cont_tokens], device=self._device)

            with torch.no_grad():
                logits = self._model_call(all_tokens)

            # Get log probs for continuation tokens
            log_probs = F.log_softmax(logits[0], dim=-1)
            cont_start = len(ctx_tokens)
            ll = 0.0
            is_greedy = True
            for i, tok in enumerate(cont_tokens):
                pos = cont_start + i - 1  # -1 because logits predict next token
                if pos >= 0 and pos < log_probs.shape[0]:
                    ll += log_probs[pos, tok].item()
                    if logits[0, pos].argmax().item() != tok:
                        is_greedy = False

            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood (for perplexity)."""
        results = []
        for (string,) in [req.args for req in requests]:
            tokens = self.tok_encode(string)
            all_tokens = torch.tensor([tokens], device=self._device)

            with torch.no_grad():
                logits = self._model_call(all_tokens)

            log_probs = F.log_softmax(logits[0], dim=-1)
            ll = 0.0
            for i in range(1, len(tokens)):
                ll += log_probs[i - 1, tokens[i]].item()

            results.append((ll,))
        return results

    def generate_until(self, requests):
        """Generate text until stop sequence."""
        results = []
        for request in requests:
            context = request.args[0]
            until = request.args[1] if len(request.args) > 1 else {"until": ["\n"]}

            ctx_tokens = torch.tensor([self.tok_encode(context)], device=self._device)
            generated = self._model_generate(
                ctx_tokens,
                max_length=self.max_gen_toks,
                eos_token_id=self.eot_token_id
            )
            text = self.tok_decode(generated[0][len(ctx_tokens[0]):].tolist())

            # Truncate at stop sequences
            stop_seqs = until.get("until", ["\n"]) if isinstance(until, dict) else ["\n"]
            for stop in stop_seqs:
                if stop in text:
                    text = text[:text.index(stop)]

            results.append(text)
        return results
