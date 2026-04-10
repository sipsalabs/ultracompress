"""
WeightDNA — A Programming Language for Neural Network Weights

DNA doesn't store organisms. It stores INSTRUCTIONS for building them.
750MB of DNA → 100 trillion specialized cells.

WeightDNA doesn't store weights. It stores INSTRUCTIONS for generating them.
The program IS the compressed model.

The language:
  - Registers: hold vectors/matrices
  - Ops: mathematical operations on registers
  - Generators: produce weight values from parameters
  - Control: loops over layers, weight types, positions

A program is a sequence of instructions. When executed with
(layer_id, weight_type, row, col), it produces a weight value.

The search: evolutionary algorithm finds short programs that
produce weights matching a target model's behavior.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple
import math


# ================================================================
# THE LANGUAGE: Instruction Set
# ================================================================

class WeightDNAInterpreter:
    """Executes WeightDNA programs to generate weight matrices."""

    # Register file: 16 vector registers of size `dim`
    N_REGS = 16

    # Instruction opcodes
    OPS = [
        'SEED',        # reg = hash(constant, seed) — deterministic pseudo-random
        'SCALE',       # reg = reg * scalar
        'ADD',         # reg = reg1 + reg2
        'MUL',         # reg = reg1 * reg2 (element-wise)
        'ROTATE',      # reg = circular_shift(reg, amount)
        'SINE',        # reg = sin(reg * freq + phase)
        'MIX',         # reg = lerp(reg1, reg2, alpha)
        'MODULATE',    # reg = reg1 * (1 + reg2 * strength)
        'FOLD',        # reg = reg[:half] + reg[half:] (dimensionality mixing)
        'SPREAD',      # reg = repeat_interleave(reg[:half], 2) (expand)
        'NORM',        # reg = reg / ||reg|| * target_norm
        'CUMSUM',      # reg = cumulative_sum(reg)
        'DIFF',        # reg = reg[1:] - reg[:-1] (differences)
        'CONTEXT',     # reg = encode(layer_id, weight_type, position)
    ]

    def __init__(self, dim=1024):
        self.dim = dim
        self.registers = None

    def reset(self):
        self.registers = [torch.zeros(self.dim) for _ in range(self.N_REGS)]

    def execute_program(self, program, layer_id, weight_type, n_rows, n_cols, device='cuda'):
        """Execute a WeightDNA program to generate a weight matrix.

        Args:
            program: list of (opcode, *args) instructions
            layer_id: which transformer layer (0-indexed)
            weight_type: 0-6 for q/k/v/o/gate/up/down
            n_rows, n_cols: output matrix dimensions

        Returns:
            weight matrix (n_rows, n_cols)
        """
        self.dim = max(n_rows, n_cols)
        self.reset()
        self.registers = [torch.zeros(self.dim, device=device) for _ in range(self.N_REGS)]

        # Execute instructions
        for instr in program:
            self._execute_instruction(instr, layer_id, weight_type, device)

        # Generate matrix: each row is a variation of register 0
        rows = []
        for i in range(n_rows):
            row_seed = self.registers[0].clone()
            # Modulate by row position
            row_phase = torch.tensor(i / n_rows * 2 * math.pi, device=device)
            modulation = self.registers[1] * torch.sin(row_phase + self.registers[2])
            row = row_seed + modulation
            rows.append(row[:n_cols])

        return torch.stack(rows)

    def _execute_instruction(self, instr, layer_id, weight_type, device):
        op = instr[0]
        args = instr[1:]

        if op == 'SEED':
            dst, seed_val = args[0], args[1]
            gen = torch.Generator(device='cpu')
            gen.manual_seed(int(seed_val * 10000))
            self.registers[dst] = torch.randn(self.dim, generator=gen).to(device) * 0.02

        elif op == 'SCALE':
            dst, src, scalar = args
            self.registers[dst] = self.registers[src] * scalar

        elif op == 'ADD':
            dst, src1, src2 = args
            self.registers[dst] = self.registers[src1] + self.registers[src2]

        elif op == 'MUL':
            dst, src1, src2 = args
            self.registers[dst] = self.registers[src1] * self.registers[src2]

        elif op == 'ROTATE':
            dst, src, amount = args
            shift = int(amount * self.dim) % self.dim
            self.registers[dst] = torch.roll(self.registers[src], shift)

        elif op == 'SINE':
            dst, src, freq, phase = args
            self.registers[dst] = torch.sin(self.registers[src] * freq + phase)

        elif op == 'MIX':
            dst, src1, src2, alpha = args
            self.registers[dst] = self.registers[src1] * (1 - alpha) + self.registers[src2] * alpha

        elif op == 'MODULATE':
            dst, src1, src2, strength = args
            self.registers[dst] = self.registers[src1] * (1 + self.registers[src2] * strength)

        elif op == 'FOLD':
            dst, src = args[0], args[1]
            half = self.dim // 2
            r = self.registers[src]
            self.registers[dst] = torch.zeros_like(r)
            self.registers[dst][:half] = r[:half] + r[half:]
            self.registers[dst][half:] = r[:half] - r[half:]

        elif op == 'NORM':
            dst, src, target = args
            r = self.registers[src]
            norm = r.norm().clamp(min=1e-10)
            self.registers[dst] = r / norm * target

        elif op == 'CONTEXT':
            dst = args[0]
            # Encode layer_id and weight_type into the register
            ctx = torch.zeros(self.dim, device=self.registers[0].device)
            # Fourier encoding of layer and type
            for f in range(min(self.dim // 4, 64)):
                freq = (f + 1) * math.pi
                ctx[f * 4] = math.sin(layer_id * freq / 100)
                ctx[f * 4 + 1] = math.cos(layer_id * freq / 100)
                ctx[f * 4 + 2] = math.sin(weight_type * freq / 7)
                ctx[f * 4 + 3] = math.cos(weight_type * freq / 7)
            self.registers[dst] = ctx

        elif op == 'CUMSUM':
            dst, src = args[0], args[1]
            self.registers[dst] = torch.cumsum(self.registers[src], dim=0)

        elif op == 'DIFF':
            dst, src = args[0], args[1]
            r = self.registers[src]
            self.registers[dst] = torch.zeros_like(r)
            self.registers[dst][1:] = r[1:] - r[:-1]
            self.registers[dst][0] = r[0]


# ================================================================
# THE SEARCH: Evolutionary Program Synthesis
# ================================================================

def random_instruction(n_regs=16):
    """Generate a random valid instruction."""
    ops = WeightDNAInterpreter.OPS
    op = random.choice(ops)

    r = lambda: random.randint(0, n_regs - 1)
    f = lambda: random.uniform(-2, 2)

    if op == 'SEED':
        return (op, r(), f())
    elif op == 'SCALE':
        return (op, r(), r(), f())
    elif op == 'ADD':
        return (op, r(), r(), r())
    elif op == 'MUL':
        return (op, r(), r(), r())
    elif op == 'ROTATE':
        return (op, r(), r(), f())
    elif op == 'SINE':
        return (op, r(), r(), f(), f())
    elif op == 'MIX':
        return (op, r(), r(), r(), random.uniform(0, 1))
    elif op == 'MODULATE':
        return (op, r(), r(), r(), f())
    elif op == 'FOLD':
        return (op, r(), r())
    elif op == 'NORM':
        return (op, r(), r(), random.uniform(0.01, 0.1))
    elif op == 'CONTEXT':
        return (op, r())
    elif op == 'CUMSUM':
        return (op, r(), r())
    elif op == 'DIFF':
        return (op, r(), r())
    else:
        return (op, r())


def random_program(min_len=5, max_len=20):
    """Generate a random WeightDNA program."""
    length = random.randint(min_len, max_len)
    return [random_instruction() for _ in range(length)]


def mutate_program(program, mutation_rate=0.3):
    """Mutate a program by changing/adding/removing instructions."""
    prog = list(program)

    if random.random() < mutation_rate and len(prog) > 3:
        # Delete random instruction
        idx = random.randint(0, len(prog) - 1)
        prog.pop(idx)

    if random.random() < mutation_rate and len(prog) < 30:
        # Add random instruction
        idx = random.randint(0, len(prog))
        prog.insert(idx, random_instruction())

    if random.random() < mutation_rate and prog:
        # Replace random instruction
        idx = random.randint(0, len(prog) - 1)
        prog[idx] = random_instruction()

    return prog


def crossover(prog1, prog2):
    """Crossover two programs."""
    if len(prog1) < 2 or len(prog2) < 2:
        return list(prog1)
    cut1 = random.randint(1, len(prog1) - 1)
    cut2 = random.randint(1, len(prog2) - 1)
    return prog1[:cut1] + prog2[cut2:]


def evaluate_program(interp, program, target_W, layer_id=0, weight_type=0, device='cuda'):
    """Evaluate how well a program generates the target weight matrix."""
    m, n = target_W.shape
    try:
        generated = interp.execute_program(program, layer_id, weight_type, m, n, device)
        # Match scale
        gen_std = generated.std()
        tgt_std = target_W.std()
        if gen_std > 1e-10:
            generated = generated * (tgt_std / gen_std)

        cos = F.cosine_similarity(
            target_W.reshape(1, -1), generated.reshape(1, -1)
        ).item()

        if math.isnan(cos):
            return -1.0
        return cos
    except Exception:
        return -1.0


def evolve(target_W, n_generations=100, population_size=200,
           layer_id=0, weight_type=0, device='cuda', verbose=True):
    """Evolutionary search for a WeightDNA program."""
    interp = WeightDNAInterpreter()

    # Initialize population
    population = [random_program(5, 15) for _ in range(population_size)]

    best_ever_cos = -1
    best_ever_prog = None

    for gen in range(n_generations):
        # Evaluate
        scores = []
        for prog in population:
            score = evaluate_program(interp, prog, target_W, layer_id, weight_type, device)
            scores.append(score)

        # Sort by fitness
        ranked = sorted(zip(scores, population), key=lambda x: -x[0])
        best_cos = ranked[0][0]
        best_prog = ranked[0][1]

        if best_cos > best_ever_cos:
            best_ever_cos = best_cos
            best_ever_prog = list(best_prog)

        if verbose and (gen % 20 == 0 or gen == n_generations - 1):
            avg = sum(scores) / len(scores)
            print(f"  Gen {gen:>3}: best={best_cos:.6f} avg={avg:.6f} prog_len={len(best_prog)}")

        # Selection: top 20%
        survivors = [prog for _, prog in ranked[:population_size // 5]]

        # Generate new population
        new_pop = list(survivors)
        while len(new_pop) < population_size:
            if random.random() < 0.5 and len(survivors) >= 2:
                # Crossover
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                child = crossover(p1, p2)
            else:
                # Mutate survivor
                parent = random.choice(survivors)
                child = mutate_program(parent)
            new_pop.append(child)

        population = new_pop

    return best_ever_prog, best_ever_cos


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    device = 'cuda'
    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

    W = wd['model.layers.0.self_attn.q_proj.weight'].float().to(device)
    # Use smaller submatrix for speed
    W_small = W[:256, :256]

    print("=== WeightDNA: Evolving Programs That Generate Weights ===")
    print(f"Target: {W_small.shape} submatrix of q_proj layer 0")
    print(f"Target std: {W_small.std():.6f}")
    print()

    print("Evolving (200 population, 200 generations)...")
    best_prog, best_cos = evolve(
        W_small, n_generations=200, population_size=200, device=device,
    )

    print(f"\nBest program cosine: {best_cos:.6f}")
    print(f"Program length: {len(best_prog)} instructions")
    print(f"Program size: ~{len(best_prog) * 20} bytes")
    print(f"Weight matrix size: {W_small.numel() * 2} bytes")
    print(f"Compression: {W_small.numel() * 2 / (len(best_prog) * 20):.0f}x")
    print()

    # Show the program
    print("Best program:")
    for i, instr in enumerate(best_prog):
        print(f"  {i:>2}: {instr}")

    # Test: does this program generalize to OTHER weight matrices?
    print("\nGeneralization test (same program, different layers/types):")
    interp = WeightDNAInterpreter()
    for li in range(4):
        for wt, wname in enumerate(['q_proj', 'k_proj', 'gate_proj']):
            key = f'model.layers.{li}.self_attn.{wname}.weight' if wt < 2 else f'model.layers.{li}.mlp.gate_proj.weight'
            if key in wd:
                W_test = wd[key].float().to(device)[:256, :256]
                cos = evaluate_program(interp, best_prog, W_test, li, wt, device)
                print(f"  Layer {li} {wname}: cos={cos:.6f}")
