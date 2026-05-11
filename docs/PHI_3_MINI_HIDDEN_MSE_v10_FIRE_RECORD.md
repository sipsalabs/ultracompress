# Phi-3-mini-4k-instruct v10 Hidden-State Distillation -- Fire Record

## Run Info

- **Script**: `scripts/overlay/hidden_mse_phi_3_mini_v10.py`
- **Command**: `python scripts/overlay/hidden_mse_phi_3_mini_v10.py --device cuda:0 --seed 42 --bpw 5 --rank 48 --train_steps 300 --out_json docs/HIDDEN_MSE_PHI_3_MINI_4K_v10_RESULTS.json`
- **PID**: 399619
- **Log**: `phi_3_mini_v10_hidden_mse.log`
- **Fired**: 2026-05-10 ~21:55 MDT
- **Device**: cuda:0 (RTX 5090 32GB)

## Hyperparameters (identical to Mistral v10)

| Param | Value |
|-------|-------|
| bpw | 5 |
| block_size | 64 |
| rank | 48 |
| train_steps | 300/layer |
| train_lr | 5e-4 |
| train_bs | 4 |
| n_calib | 128 |
| n_eval | 50 |
| seq_len (calib) | 512 |
| eval_seq_len | 1024 |
| n_kl_probe | 8 |
| seed | 42 |

## Expected Outputs

- **Result JSON**: `docs/HIDDEN_MSE_PHI_3_MINI_4K_v10_RESULTS.json`
- **Partial JSON**: `docs/HIDDEN_MSE_PHI_3_MINI_4K_v10_PARTIAL.json` (every 5 layers)
- **Per-layer artifacts**: `scripts/overlay/_e2e_phi_3_mini_4k_v10_hidden_mse/layer_XXX.pt`

## Baseline

- **Teacher PPL (bf16)**: 7.8500 (eval_seq_len=1024, n=50, seed=42)
- **VRAM at load**: 7.6 GB (3.8B params, bf16)
- **Existing published record**: 1.00262x PPL ratio (but at seq_len=128, v0.5.x)

## Success Criteria

- **Target**: PPL ratio < 1.005x (would match or beat Mistral v10's 1.0055x on a different architecture)
- **Stretch**: PPL ratio < 1.003x (would beat the existing 1.00262x record even at harder eval_seq_len=1024)
- **Minimum**: PPL ratio < 1.01x (confirms cross-architecture generalization of the objective)

## ETA

- Mistral-7B v10 took 3892s (~65 min) for 32 layers at ~120s/layer average
- Phi-3-mini is ~half the params (3.8B vs 7.2B) with same 32 layers
- Expected: ~40-55 min total (faster per-layer forward, smaller hidden dim 3072 vs 4096)
- **ETA**: ~22:40-22:50 MDT

## Fallback

If this run fails or produces PPL ratio > 1.02x:
- Next test architecture: Llama-3.1-8B (closest to Mistral in structure)
- Debug: check if Phi3RotaryEmbedding position_embeddings interface differs from Mistral
