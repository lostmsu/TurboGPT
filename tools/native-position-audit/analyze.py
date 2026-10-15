import csv
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qwen_tuning.experiments import canonical_training_rows
from tools.run_hn1g_suite import scalar_value
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

suite = ROOT / 'runs/qwen-hn1g'
cache = ROOT / '.cache/native-position-audit'
tokens = np.memmap(ROOT / '.cache/qwen-hn1g/full.bin', dtype='<u4', mode='r')
lengths = np.fromfile(ROOT / '.cache/native/token_bytes.bin', dtype='<u4')
plan = json.loads((suite / 'plan.json').read_text())
assert len(tokens) == 284024229
assert plan['steps'] == 1240 and plan['seed'] == 3407
window_updates = plan['window_tokens'] // plan['tokens_per_update']
result = {'window_tokens': plan['window_tokens'], 'window_updates': window_updates,
          'position_convention': 'Input tokens seen before the prediction: loss_257 is zero-based prediction offset 257 and therefore sees 258 input tokens.',
          'normalization': 'Each update: position NLL / (ln(2) * original UTF-8 target bytes); arithmetic mean over the final equal-token updates.',
          'validation': {}, 'runs': []}
for name in plan['order']:
    directory = suite / name
    summary = json.loads((directory / 'summary.json').read_text())
    sampler = json.loads((directory / 'sampler.json').read_text())
    rows = canonical_training_rows(directory / 'metrics.jsonl')
    context, effective = summary['context'], summary['effective_batch']
    if summary['sampling'] == 'random':
        starts = np.fromfile(cache / f'starts-{context}.u64', dtype='<u8').reshape(plan['steps'], effective)
        with (directory / 'checkpoint.bin').open('rb') as stream:
            header = struct.unpack('<QQiiiifiiQ', stream.read(52))
            stored_rng = stream.read(header[-1])
        assert header[0] == 0x5157454e4c4f5233 and header[2:4] == (1240, context)
        assert stored_rng == (cache / f'state-{context}.txt').read_bytes(), f'RNG replay mismatch: {name}'
        rng_exact = True
    else:
        lanes = np.array(sampler['starts'], dtype=np.uint64)
        assert len(lanes) == effective and sampler['restored_step'] == 0
        starts = lanes[None, :] + np.arange(plan['steps'], dtype=np.uint64)[:, None] * context
        rng_exact = None
    # Every update's logged ending-byte sum validates the reconstructed windows.
    ending_bytes = lengths[tokens[starts + context]].sum(axis=1, dtype=np.uint64)
    assert np.array_equal(ending_bytes, [r['last_bytes'] for r in rows]), f'Ending-byte replay mismatch: {name}'
    # Check all 65,536 targets for the first and final updates as well.
    for index in [0, len(rows) - 1]:
        ids = tokens[starts[index, :, None] + np.arange(1, context + 1, dtype=np.uint64)[None, :]]
        assert int(lengths[ids].sum(dtype=np.uint64)) == rows[index]['predicted_bytes'], f'Full-batch replay mismatch: {name}'
    seen = min(258, context)
    if context > 258:
        events = EventAccumulator(str(directory), size_guidance={'tensors': 0}, purge_orphaned_data=False)
        events.Reload()
        logged = events.Tensors('loss_257')
        assert [e.step for e in logged] == [r['processed_tokens'] for r in rows]
        mean_nll = np.array([scalar_value(e.tensor_proto) for e in logged], dtype=np.float64)
        byte_count = lengths[tokens[starts + seen]].sum(axis=1, dtype=np.uint64)
        nll = mean_nll * effective
        bpb = nll / (math.log(2) * byte_count)
        source = 'TensorBoard loss_257 and reconstructed exact target-byte counts'
    else:
        nll = np.array([r['last_nll'] for r in rows])
        mean_nll = nll / effective
        byte_count = ending_bytes
        bpb = np.array([r['last_bpb'] for r in rows])
        source = 'Recorded last-token BPB'
    tail = slice(-window_updates, None)
    record = {'run': name, 'context': context, 'sampling': summary['sampling'],
              'input_tokens_seen': seen, 'avg_bpb': float(np.mean(bpb[tail])),
              'pooled_bpb': float(nll[tail].sum() / (math.log(2) * byte_count[tail].sum())),
              'avg_loss_nats': float(mean_nll[tail].mean()),
              'target_observations': effective * window_updates,
              'target_bytes': int(byte_count[tail].sum()), 'source': source}
    result['runs'].append(record)
    result['validation'][name] = {'rng_checkpoint_bit_exact': rng_exact,
                                 'all_1240_ending_byte_sums_match': True,
                                 'first_and_final_full_batch_byte_sums_match': True}
    with (directory / 'near256-bpb.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'processed_tokens', 'input_tokens_seen', 'mean_loss_nats', 'target_bytes', 'bpb'])
        for index, row in enumerate(rows):
            writer.writerow([row['step'], row['processed_tokens'], seen, mean_nll[index], int(byte_count[index]), bpb[index]])
(suite / 'near256-bpb.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
