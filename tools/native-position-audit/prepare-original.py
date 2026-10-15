import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
suite = ROOT / 'runs/qwen-hn1g'
cache = ROOT / '.cache/native-position-audit'
out = suite / 'original-near256'
out.mkdir(exist_ok=True)
plan = json.loads((suite / 'plan.json').read_text())
near = json.loads((suite / 'near256-bpb.json').read_text())
tokens = np.memmap(ROOT / '.cache/qwen-hn1g/full.bin', dtype='<u4', mode='r')
lengths = np.fromfile(ROOT / '.cache/native/token_bytes.bin', dtype='<u4')
manifest = {'model': 'Qwen/Qwen3.5-0.8B-Base',
            'model_revision': 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
            'precision': 'native BF16, matching the six training runs',
            'training_window_tokens': plan['window_tokens'], 'steps': [1177, 1240],
            'averaging': 'Arithmetic mean of 64 per-update position BPBs',
            'jobs': []}
for record in near['runs']:
    name, context = record['run'], record['context']
    sampler = json.loads((suite / name / 'sampler.json').read_text())
    effective, seen = sampler['effective_batch'], record['input_tokens_seen']
    if sampler['sampling'] == 'random':
        starts = np.fromfile(cache / f'starts-{context}.u64', dtype='<u8').reshape(1240, effective)[-64:]
    else:
        starts = np.asarray(sampler['starts'], dtype=np.uint64)[None, :] + np.arange(1176, 1240, dtype=np.uint64)[:, None] * context
    windows = np.asarray(tokens[starts[:, :, None] + np.arange(seen + 1, dtype=np.uint64)[None, None, :]], dtype='<u4')
    target_bytes = lengths[windows[:, :, -1]].sum(axis=1, dtype=np.uint64)
    assert int(target_bytes.sum()) == record['target_bytes']
    directory = out / name
    directory.mkdir(exist_ok=True)
    inputs = directory / 'inputs.bin'
    with inputs.open('wb') as stream:
        stream.write(struct.pack('<QII', 0x514e494e50555431, 64 * effective, seen))
        stream.write(windows.tobytes())
    starts.astype('<u8').tofile(directory / 'starts.u64')
    manifest['jobs'].append({'run': name, 'training_context': context, 'input_tokens_seen': seen,
                             'sampling': sampler['sampling'], 'effective_batch': effective,
                             'input_samples': 64 * effective, 'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
                             'target_bytes_per_update': target_bytes.tolist(), 'tuned_avg_bpb': record['avg_bpb']})
(out / 'inputs.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(out), 'samples': sum(j['input_samples'] for j in manifest['jobs']),
                  'input_tokens': sum(j['input_samples'] * j['input_tokens_seen'] for j in manifest['jobs'])}))
