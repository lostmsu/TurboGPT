import hashlib
import json
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
prefix = ROOT / 'runs/qwen-hn1g/original-near256'
out = ROOT / 'runs/qwen-hn1g/original-near256-fullcontext'
out.mkdir(exist_ok=True)
manifest = json.loads((prefix / 'inputs.json').read_text())
manifest['evaluation_windows'] = 'Complete original training windows and physical batch sizes; only the selected position is scored.'
tokens = np.memmap(ROOT / '.cache/qwen-hn1g/full.bin', dtype='<u4', mode='r')
for job in manifest['jobs']:
    name, context, effective = job['run'], job['training_context'], job['effective_batch']
    starts = np.fromfile(prefix / name / 'starts.u64', dtype='<u8').reshape(64, effective)
    windows = np.asarray(tokens[starts[:, :, None] + np.arange(context + 1, dtype=np.uint64)[None, None, :]], dtype='<u4')
    directory = out / name
    directory.mkdir(exist_ok=True)
    path = directory / 'inputs.bin'
    with path.open('wb') as stream:
        stream.write(struct.pack('<QII', 0x514e494e50555431, 64 * effective, context))
        stream.write(windows.tobytes())
    prefix_ids = np.fromfile(prefix / name / 'inputs.bin', dtype='<u4', offset=16).reshape(64, effective, job['input_tokens_seen'] + 1)
    assert np.array_equal(windows[:, :, :job['input_tokens_seen'] + 1], prefix_ids)
    job['prefix_input_sha256'] = job['input_sha256']
    job['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    starts.astype('<u8').tofile(directory / 'starts.u64')
(out / 'inputs.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(out), 'samples': sum(j['input_samples'] for j in manifest['jobs']), 'context_tokens': 6 * 4194304}))
