"""
UNIVERSAL SEED FORMAT — The invention.

Not a model. Not an architecture. A FORMAT for intelligence.

Works BOTH directions:
  EXTRACT: Take ANY existing model → produce a seed
  GROW: Train a seed from scratch → it becomes intelligent

The seed has two parts:
  1. CORE: a tiny computational kernel (discovered, not designed)
  2. PROGRAM: how to run the kernel to produce intelligence

The core is architecture-agnostic. It could be:
  - A set of rotation angles
  - A tiny neural net
  - A set of rules
  - Something we haven't invented yet
  The FORMAT doesn't care. It's universal.

For existing models (compress):
  model + calibration_data → extract(model) → UniversalSeed
  UniversalSeed.run(input) ≈ model(input)

For new models (grow):
  training_data → grow(seed) → UniversalSeed
  UniversalSeed.run(input) = intelligent output

Same format. Same inference. Two ways in.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class UniversalSeed:
    """The universal intelligence format.

    A seed contains:
    - core_type: what kind of computation ('rotation', 'frr', 'organism', 'custom')
    - core_state: the learned parameters of the core
    - program: execution schedule (how many cycles, modulation per cycle)
    - metadata: model info, compression ratio, quality metrics
    - embeddings: shared vocabulary mapping (from teacher or learned)

    The seed can be:
    - Extracted from any model via extract()
    - Grown from scratch via grow()
    - Loaded and run via run()
    - Converted between core types via convert()
    """

    def __init__(self):
        self.core_type = None
        self.core_model = None
        self.program = None
        self.metadata = {}
        self.embed = None
        self.lm_head = None
        self.norm = None

    @classmethod
    def extract_from_model(cls, model_name_or_path, core_type='frr',
                           steps=50000, device='cuda'):
        """Extract a seed from ANY existing model.

        This is the COMPRESS direction.
        Takes a trained model, distills it into a universal seed.
        """
        seed = cls()
        seed.core_type = core_type
        seed.metadata['source'] = model_name_or_path
        seed.metadata['direction'] = 'extracted'

        # Load the model
        from transformers import AutoModelForCausalLM, AutoConfig
        print(f"Loading {model_name_or_path}...")
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        hidden = config.hidden_size
        n_heads = config.num_attention_heads
        n_layers = config.num_hidden_layers
        vocab = config.vocab_size

        seed.metadata['hidden_size'] = hidden
        seed.metadata['n_layers'] = n_layers
        seed.metadata['vocab_size'] = vocab

        # Build core based on type
        if core_type == 'frr':
            from .moonshot import FractalModel
            seed.core_model = FractalModel(
                hidden_dim=hidden, n_heads=n_heads,
                n_scales=4, iters_per_scale=n_layers // 4,
                vocab_size=vocab, ff_mult=1)
        elif core_type == 'rotation':
            from .rotation_engine import RotationEngine
            seed.core_model = RotationEngine(
                hidden_dim=hidden, n_planes=64,
                n_cycles=n_layers, vocab_size=vocab)
        else:
            raise ValueError(f"Unknown core type: {core_type}")

        seed.metadata['core_params'] = sum(
            p.numel() for p in seed.core_model.parameters())

        # TODO: distill from model into core
        # (uses existing distillation pipeline)

        return seed

    @classmethod
    def grow_from_scratch(cls, core_type='rotation', hidden_dim=1024,
                          n_cycles=28, vocab_size=151936,
                          training_data='fineweb-edu', steps=100000,
                          device='cuda'):
        """Grow a seed from scratch on training data.

        This is the BUILD NEW direction.
        No teacher. No existing model. Pure learning.
        """
        seed = cls()
        seed.core_type = core_type
        seed.metadata['direction'] = 'grown'
        seed.metadata['training_data'] = training_data

        if core_type == 'rotation':
            from .rotation_engine import RotationEngine
            seed.core_model = RotationEngine(
                hidden_dim=hidden_dim, n_planes=64,
                n_cycles=n_cycles, vocab_size=vocab_size)
        elif core_type == 'frr':
            from .moonshot import FractalModel
            seed.core_model = FractalModel(
                hidden_dim=hidden_dim, n_heads=16,
                n_scales=4, iters_per_scale=n_cycles // 4,
                vocab_size=vocab_size, ff_mult=1)
        else:
            raise ValueError(f"Unknown core type: {core_type}")

        # All params trainable (no frozen teacher weights)
        for p in seed.core_model.parameters():
            p.requires_grad = True

        seed.metadata['core_params'] = sum(
            p.numel() for p in seed.core_model.parameters())

        # TODO: train on real text data
        # (uses from-scratch training pipeline)

        return seed

    def run(self, tokens, device='cuda'):
        """Run the seed on input tokens. Works regardless of how seed was created."""
        if self.core_model is None:
            raise ValueError("Seed has no core model. Extract or grow first.")
        self.core_model.to(device)
        self.core_model.eval()
        with torch.no_grad():
            return self.core_model(tokens.to(device))

    def save(self, path):
        """Save seed to disk."""
        os.makedirs(path, exist_ok=True)
        # Save metadata
        with open(os.path.join(path, 'seed.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        # Save core
        if self.core_model is not None:
            torch.save(self.core_model.state_dict(),
                       os.path.join(path, 'core.pt'))

    @classmethod
    def load(cls, path, device='cuda'):
        """Load seed from disk."""
        seed = cls()
        with open(os.path.join(path, 'seed.json')) as f:
            seed.metadata = json.load(f)

        core_type = seed.metadata.get('core_type', 'frr')
        hidden = seed.metadata.get('hidden_size', 1024)
        n_layers = seed.metadata.get('n_layers', 28)
        vocab = seed.metadata.get('vocab_size', 151936)

        if core_type == 'frr':
            from .moonshot import FractalModel
            seed.core_model = FractalModel(
                hidden_dim=hidden, n_heads=16,
                n_scales=4, iters_per_scale=n_layers // 4,
                vocab_size=vocab, ff_mult=1)
        elif core_type == 'rotation':
            from .rotation_engine import RotationEngine
            seed.core_model = RotationEngine(
                hidden_dim=hidden, n_planes=64,
                n_cycles=n_layers, vocab_size=vocab)

        core_path = os.path.join(path, 'core.pt')
        if os.path.exists(core_path):
            seed.core_model.load_state_dict(
                torch.load(core_path, map_location=device))

        return seed

    def convert(self, new_core_type):
        """Convert seed between core types (e.g., FRR → rotation).

        This enables: extract as FRR (high quality) → convert to rotation
        (smaller, faster). Or: grow as rotation → convert to FRR (higher quality).
        """
        # TODO: cross-architecture distillation
        # The universal format makes this possible because both cores
        # have the same input/output interface (tokens → logits).
        raise NotImplementedError(
            "Cross-core conversion coming. The universal format makes it "
            "possible because all cores share the same interface.")

    def info(self):
        """Print seed info."""
        print(f"Universal Seed")
        print(f"  Core: {self.metadata.get('core_type', 'unknown')}")
        print(f"  Direction: {self.metadata.get('direction', 'unknown')}")
        print(f"  Params: {self.metadata.get('core_params', 'unknown'):,}")
        print(f"  Source: {self.metadata.get('source', 'from scratch')}")
        if 'hidden_size' in self.metadata:
            print(f"  Hidden: {self.metadata['hidden_size']}")
            print(f"  Layers: {self.metadata['n_layers']}")
