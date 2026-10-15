import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'runs/qwen-hn1g/last4096-shared-8192'
log = [json.loads(line) for line in (out / 'native.log').read_text(encoding='utf-8-sig').splitlines() if line.startswith('{')]
assert log[0]['gpu'] == 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
assert log[-1]['evaluation_complete'] is True and log[-1]['optimizer_updates'] == 0
N, CONTEXT = 8192, 4096
names = ['base', 'regular', 'sequential']
nll = np.column_stack([np.fromfile(out / f'{name}.f32', dtype='<f4').astype(np.float64) for name in names])
assert nll.shape == (N, 3) and np.all(np.isfinite(nll))
byte_count = np.fromfile(out / 'target_bytes.u32', dtype='<u4').astype(np.float64)
starts = np.fromfile(out / 'starts.u64', dtype='<u8')
assert len(byte_count) == len(starts) == N and np.all(byte_count > 0)
assert starts.max() + CONTEXT < 284024229
windows = np.memmap(out / 'inputs.bin', dtype='<u4', mode='r', offset=16, shape=(N, CONTEXT + 1))
with (out / 'inputs.bin').open('rb') as stream:
    assert struct.unpack('<QII', stream.read(16)) == (0x514e494e50555431, N, CONTEXT)
tokens = np.memmap(ROOT / '.cache/qwen-hn1g/full.bin', dtype='<u4', mode='r')
lengths = np.fromfile(ROOT / '.cache/native/token_bytes.bin', dtype='<u4')
for batch in range(0, N, 128):
    expected = tokens[starts[batch:batch+128, None] + np.arange(CONTEXT + 1, dtype=np.uint64)[None, :]]
    assert np.array_equal(windows[batch:batch+128], expected)
assert np.array_equal(byte_count, lengths[windows[:, -1]])
point = nll.sum(axis=0) / (math.log(2) * byte_count.sum())

# Resample the same window indices jointly for all three models. All ratios
# therefore use identical bootstrap target-byte denominators.
repetitions = 10000
rng = np.random.default_rng(5811)
bootstrap = np.empty((repetitions, 3))
for offset in range(0, repetitions, 100):
    count = min(100, repetitions - offset)
    indices = rng.integers(0, N, size=(count, N))
    denominator = math.log(2) * byte_count[indices].sum(axis=1)
    for model in range(3):
        bootstrap[offset:offset+count, model] = nll[indices, model].sum(axis=1) / denominator

before = json.loads((out / 'checkpoint-hashes-before.json').read_text(encoding='utf-8-sig'))
after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in before}
assert before == after, 'Read-only evaluation changed a checkpoint'
(out / 'checkpoint-hashes-after.json').write_text(json.dumps(after, indent=2) + '\n', encoding='utf-8')
result = {'context': CONTEXT, 'samples': N, 'input_tokens_per_model': N * CONTEXT,
          'sampling_seed': 20261010, 'dataset': 'Complete hn1g training corpus, no held-out split',
          'native_cuda': True, 'precision': 'BF16 frozen weights and GEMMs, FP32 loss accumulation',
          'gpu': log[0]['gpu'], 'inference_physical_batch': log[0]['batch'], 'optimizer_updates': 0,
          'estimator': 'Pooled last-token NLL / (ln(2) * original UTF-8 last-target bytes)',
          'confidence_method': 'Paired window bootstrap, 10,000 resamples, percentile 95% intervals, seed 5811',
          'total_last_target_bytes': int(byte_count.sum()), 'models': [], 'comparisons': [],
          'validation': {'all_8192_complete_windows_match_original_corpus': True,
                         'all_target_byte_lengths_match': True, 'checkpoint_hashes_unchanged': True,
                         'peak_arena_gib': log[-1]['peak_arena_gib']}}
for i, name in enumerate(names):
    native = next(entry for entry in log if entry.get('model') == name and entry.get('complete'))
    audit = next(entry for entry in log if entry.get('model') == name and 'trainer_loss_mean_abs_error' in entry)
    assert abs(point[i] - native['last_bpb']) < 1e-12 and audit['trainer_loss_mean_abs_error'] <= 1e-4
    result['models'].append({'model': name, 'checkpoint_step': None if i == 0 else 1240,
                            'last_bpb': float(point[i]), 'ci95': np.quantile(bootstrap[:, i], [.025, .975]).tolist(),
                            'mean_last_loss_nats': float(nll[:, i].mean()),
                            'mean_group16_bpb': float((nll[:, i].reshape(-1, 16).sum(axis=1) / math.log(2) /
                                                       byte_count.reshape(-1, 16).sum(axis=1)).mean()),
                            'seconds': native['seconds'], 'trainer_head_mean_loss_error': audit['trainer_loss_mean_abs_error']})
for left, right in [(1, 2), (1, 0), (2, 0)]:
    ci = np.quantile(bootstrap[:, left] - bootstrap[:, right], [.025, .975]).tolist()
    result['comparisons'].append({'difference': f'{names[left]} minus {names[right]}',
                                  'delta_bpb': float(point[left] - point[right]), 'paired_ci95': ci,
                                  'ci_excludes_zero': bool(ci[0] > 0 or ci[1] < 0)})
result['provenance'] = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        'checkpoint_sha256': after,
                        'input_sha256': hashlib.sha256((out / 'inputs.bin').read_bytes()).hexdigest(),
                        'binary_sha256': hashlib.sha256((out / 'large-last-bpb.exe').read_bytes()).hexdigest(),
                        'source_sha256': {str(path.relative_to(out / 'source')): hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in (out / 'source').rglob('*') if path.is_file()}}
(out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
with (out / 'scores.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['sample', 'token_start', 'target_bytes', 'base_last_nll', 'regular_last_nll', 'sequential_last_nll'])
    for i in range(N):
        writer.writerow([i, int(starts[i]), int(byte_count[i]), *nll[i].tolist()])
print(json.dumps({key: result[key] for key in ['models', 'comparisons', 'validation']}, indent=2))
