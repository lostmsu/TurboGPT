import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'runs/qwen-hn1g/original-near256'
manifest = json.loads((out / 'inputs.json').read_text())
log = [json.loads(line) for line in (out / 'native.log').read_text(encoding='utf-8-sig').splitlines() if line.startswith('{')]
assert log[-1]['complete'] is True
result = {key: value for key, value in manifest.items() if key != 'jobs'}
result['measurement'] = 'Native CUDA BF16 base weights, adapters disabled, original native forward and cross entropy; no optimization or held-out data.'
result['validation'] = {'all_input_hashes_verified': True, 'all_target_byte_counts_verified': True,
                        'zero_lora_hidden_and_loss_exact': True, 'last_only_head_matches_full_trainer_loss': True,
                        'peak_arena_gib': log[-1]['peak_arena_gib']}
result['runs'] = []
for job in manifest['jobs']:
    directory = out / job['run']
    assert hashlib.sha256((directory / 'inputs.bin').read_bytes()).hexdigest() == job['input_sha256']
    with (directory / 'original.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 64 and [int(r['step']) for r in rows] == list(range(1177, 1241))
    byte_counts = np.array([int(r['target_bytes']) for r in rows], dtype=np.uint64)
    assert byte_counts.tolist() == job['target_bytes_per_update']
    nll = np.array([float(r['last_nll']) for r in rows])
    bpb = np.array([float(r['bpb']) for r in rows])
    assert np.all(np.isfinite(bpb)) and np.allclose(bpb, nll / math.log(2) / byte_counts, atol=1e-12, rtol=0)
    audit = next(entry for entry in log if entry.get('run') == job['run'] and 'zero_lora_exact' in entry)
    assert audit['zero_lora_exact'] is True and audit['trainer_loss_mean_abs_error'] <= 1e-4
    original = float(bpb.mean())
    result['runs'].append({'run': job['run'], 'training_context': job['training_context'],
                           'input_tokens_seen': job['input_tokens_seen'], 'sampling': job['sampling'],
                           'original_avg_bpb': original, 'tuned_avg_bpb': job['tuned_avg_bpb'],
                           'improvement_bpb': original - job['tuned_avg_bpb'],
                           'improvement_percent': 100 * (original - job['tuned_avg_bpb']) / original,
                           'original_pooled_bpb': float(nll.sum() / math.log(2) / byte_counts.sum()),
                           'target_observations': job['input_samples'], 'target_bytes': int(byte_counts.sum()),
                           'trainer_head_audit_mean_loss_error': audit['trainer_loss_mean_abs_error']})
source_files = ['tools/native-position-audit/original-baseline.cu', 'tools/native-position-audit/build-baseline.ps1',
                'tools/native-position-audit/prepare-original.py', 'tools/native-position-audit/summarize-original.py',
                'native/qwen.cu', 'native/kernels.cuh', 'native/delta.cuh', 'native/optimizers.cuh',
                'native/experiment.h', 'build/tensorboard.obj', 'build/revision.obj',
                '.cache/native-position-audit/original-baseline.exe']
result['provenance'] = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        'command': '.cache/native-position-audit/original-baseline.exe',
                        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}}
(out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(result['runs'][0]))
    writer.writeheader()
    writer.writerows(result['runs'])
print(json.dumps(result['runs'], indent=2))
