#!/usr/bin/env python
"""Queue runner: run v7 configs on GPU1 sequentially after each completes."""
import subprocess, sys, os, time
os.chdir(r'C:\Users\scamd\ultracompress')

# GPU1 queue
SWEEPS = [
    # (tag, K_global, D, steps, lr)
    ('K2048_D16', 2048, 16, 2000, 3e-4),    # ~0.69 bits/w  target ~1400x, T1>=60%
    ('K4096_D16', 4096, 16, 2000, 3e-4),    # ~0.75 bits/w  target ~1300x
    ('K512_D8',    512,  8,  1500, 3e-4),   # ~1.125 bits/w, should hold T1 well
    ('K2048_D8',  2048,  8,  2000, 3e-4),   # ~1.375 bits/w, richer
]
env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = '1'

for tag, K, D, steps, lr in SWEEPS:
    out = f'qwen3_1.7b_sb7_{tag}.pt'
    log = f'v7_{tag}.log'
    print(f'\n===== v7 {tag}: K={K} D={D} steps={steps} =====', flush=True)
    cmd = [
        sys.executable, 'compress_vocab_v7.py',
        '--sb4_ckpt', 'qwen3_1.7b_sb4_xtreme.pt',
        '--teacher_cache', 'qwen3_1.7b_cache.pt',
        '--global_K', str(K), '--subvec', str(D),
        '--qat_steps', str(steps), '--batch', '4', '--seq', '128', '--lr', str(lr),
        '--eval_every', '100', '--eval_seqs', '24',
        '--out', out, '--device', 'cuda:0',
    ]
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f'  {tag} done rc={p.returncode} in {(time.time()-t0)/60:.1f} min', flush=True)
