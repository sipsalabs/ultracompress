#!/usr/bin/env python
"""Launcher: sweep v6 PQ configs on GPU0 sequentially with best-state tracking."""
import subprocess, sys, os, time
os.chdir(os.environ.get('UC_REPO_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SWEEPS = [
    # (tag, codebook_size, subvec, qat_steps, lr)
    ('k256_d4_best',  256,  4, 2000, 3e-4),   # 585x target, with best-state -> T1 should beat 74%
    ('k4096_d4_best', 4096, 4, 2000, 3e-4),   # 3 bit/w ~ 488x, aim T1>=77%
    ('k256_d8_best',  256,  8, 2500, 3e-4),   # 1 bit/w, aim beat K=16 D=4 (60%)
    ('k64_d4_best',   64,   4, 2000, 3e-4),   # 1.5 bit/w, between d4/d8
]
for tag, K, D, steps, lr in SWEEPS:
    out = f'qwen3_1.7b_sb6_{tag}.pt'
    log = f'compress_vocab_v6_{tag}.log'
    print(f'\n===== {tag}: K={K} D={D} steps={steps} lr={lr} =====', flush=True)
    cmd = [
        sys.executable, 'compress_vocab_v6.py',
        '--sb4_ckpt', 'qwen3_1.7b_sb4_xtreme.pt',
        '--teacher_cache', 'qwen3_1.7b_cache.pt',
        '--codebook_size', str(K), '--subvec', str(D),
        '--qat_steps', str(steps), '--batch', '4', '--seq', '128', '--lr', str(lr),
        '--eval_every', '100', '--eval_seqs', '24',
        '--out', out, '--device', 'cuda:0',
    ]
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f'  {tag} done rc={p.returncode} in {(time.time()-t0)/60:.1f} min', flush=True)
