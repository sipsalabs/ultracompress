"""HF model card Charter-clean refresh v0.6.0.

Downloads each card, scrubs Charter-prohibited terms, uploads back.
Idempotent — safe to re-run.
"""
import re, sys, time
from huggingface_hub import HfApi, hf_hub_download, upload_file

# Same scrub map as README v0.6.0
SCRUBS = [
    # Recipe equation (paren variants of (U @ V), W_full, W_base)
    (r'W_full = grid\[codes\] \* absmax \+ alpha \* \(?U @ V\)?',
     'W_reconstructed = scalar_dequantize(codes, scale) + low_rank_overlay'),
    (r'W_base = grid\[codes\] \* absmax \+ alpha \* \(?U @ V\)?',
     'W_reconstructed = scalar_dequantize(codes, scale) + low_rank_overlay'),
    (r'W_base = absmax × grid\[codes\]', 'W_reconstructed = scalar_dequantize(codes, scale)'),
    (r'W_base = absmax \* grid\[codes\]', 'W_reconstructed = scalar_dequantize(codes, scale)'),
    # Bare grid[codes] in code blocks
    (r'grid\[codes\.long\(\)\]', 'scalar_dequantize(codes)'),
    (r'grid\[codes\]', 'scalar_dequantize(codes)'),
    (r'\(U @ V\)', 'low_rank(U, V)'),
    (r' U @ V', ' low_rank(U, V)'),
    # Datestamps with hours (ISO 8601)
    (r'\b2026-05-07T[0-9:]+Z?', '2026-05'),
    (r'\b2026-05-07\b', '2026-05'),
    (r'\b2026-05-08T[0-9:]+Z?', '2026-05'),
    (r'\b2026-05-09T[0-9:]+Z?', '2026-05'),
    # Codenames
    (r'V18-C correction matrices', 'low-rank correction matrices'),
    (r'V18-C correction', 'correction overlay'),
    (r'V18-C overlay', 'correction overlay'),
    (r'V18-C subspace allocation', 'subspace allocation'),
    (r'V18-C SVD warm-start', 'SVD warm-start'),
    (r'V18-C depth-adaptive train_steps', 'depth-adaptive variant'),
    (r'V18-C against the cache', 'the correction overlay against the cache'),
    (r'V18-C value', 'correction overlay value'),
    (r'V18-C', 'correction overlay'),
    (r'GSQ-only', 'scalar-only'),
    (r'GSQ \+ V18-C', 'scalar + correction overlay'),
    (r'GSQ ', 'scalar quantization '),
    (r'GSQ\.', 'scalar quantization.'),
    (r'GSQ,', 'scalar quantization,'),
    (r'GSQ$', 'scalar quantization'),
    (r'GSQ\)', 'scalar quantization)'),
    # Recipe specifics
    (r'rank-32 V18-C correction', 'low-rank correction'),
    (r'rank-32', 'low-rank'),
    (r'rank=32\b', 'low-rank (production-tuned)'),
    (r'rank=48\b', 'low-rank (production-tuned)'),
    (r'rank=64\b', 'low-rank (production-tuned)'),
    (r'train_steps=400', 'production training schedule'),
    (r'train_steps=1500', 'production training schedule'),
    (r'200-step KL distillation', 'KL distillation pass'),
    (r'200 KL-distillation steps', 'production KL distillation'),
    # Trainer paths
    (r'`?scripts/overlay/streaming_compression_runner\.py`?',
     '(production trainer, patent-protected)'),
    (r'`?scripts/overlay/eval_compressed_only\.py`?', '`uc verify`'),
    (r'streaming_compression_runner', 'production-trainer'),
    # Datestamps that leak research timeline
    (r'\b2026-05-07\b', '2026-05'),
    # Bare grid[codes] (markdown code spans)
    (r'`grid\[codes\][^`]*`', '`scalar dequantize`'),
    # Stray U_factor / V_factor
    (r'`?U_factor`?', '`U`'),
    (r'`?V_factor`?', '`V`'),
    # Patent-protection footer (will dedupe)
    # (added separately at end)
]

PATENT_FOOTER = '\n\nCodec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).\n'

REPOS = [
    'SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5',
    'SipsaLabs/qwen3-8b-uc-v3-bpw5',
    'SipsaLabs/qwen3-14b-uc-v3-bpw5',
    'SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5',
    'SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5',
    'SipsaLabs/qwen3-1.7b-streaming-bpw5',
    'SipsaLabs/qwen3-8b-streaming-bpw5',
    'SipsaLabs/qwen3-14b-streaming-bpw5',
    'SipsaLabs/hermes-3-llama-3.1-405b-streaming-bpw5',
    'SipsaLabs/mistral-7b-v0.3-streaming-bpw5',
]

LEAK_PATTERNS = [
    r'GSQ\b', r'V18-C', r'V18C', r'grid\[codes', r'W_base = grid', r'W_base = absmax',
    r'Cure A4', r'gsq_codecs', r'_gsq_inverse', r'streaming_compression_runner',
    r'2026-05-07', r'symmetric range', r'grid \{-15', r'U_factor', r'V_factor',
    r'rank=32\b', r'rank=48\b', r'rank=64\b', r'train_steps=400', r'train_steps=1500',
    r'200-step KL', r'200 KL-distillation', r'U @ V', r'production-trainer',
]

def scrub(text: str) -> str:
    for pat, rep in SCRUBS:
        text = re.sub(pat, rep, text)
    if 'patent-protected (USPTO' not in text:
        text = text.rstrip() + PATENT_FOOTER
    return text

def count_leaks(text: str) -> int:
    n = 0
    for pat in LEAK_PATTERNS:
        n += len(list(re.finditer(pat, text, re.MULTILINE)))
    return n

api = HfApi()
results = []
for repo in REPOS:
    try:
        path = hf_hub_download(repo, 'README.md', repo_type='model')
        with open(path, 'r', encoding='utf-8') as f:
            orig = f.read()
        before = count_leaks(orig)
        cleaned = scrub(orig)
        after = count_leaks(cleaned)
        if cleaned != orig:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo='README.md',
                repo_id=repo,
                repo_type='model',
                commit_message='docs: Charter v0.6.0 cleanup (no codenames or recipe internals)',
            )
            print(f'  UPDATED  {repo}  leaks: {before} -> {after}')
            results.append((repo, 'updated', before, after))
        else:
            print(f'  unchanged {repo}  leaks: {before}')
            results.append((repo, 'unchanged', before, after))
        time.sleep(0.5)
    except Exception as e:
        em = str(e)[:140]
        print(f'  ERROR  {repo}: {em}')
        results.append((repo, 'error', -1, -1))

print('\n=== SUMMARY ===')
for repo, status, before, after in results:
    print(f'  {status:10s}  {repo}  ({before} -> {after})')
total_before = sum(b for _, _, b, _ in results if b >= 0)
total_after = sum(a for _, _, _, a in results if a >= 0)
print(f'\nTOTAL leak count {total_before} -> {total_after}')
