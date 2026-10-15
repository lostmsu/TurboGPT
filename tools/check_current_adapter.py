import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

root = Path(__file__).resolve().parents[1]
current = root / 'runs/qwen-hn1g/v-4096/checkpoint.bin'
source_inputs = root / 'runs/context4096-accuracy-bf16/inputs.bin'
data = source_inputs.read_bytes()
assert hashlib.sha256(data).hexdigest() == '66869cf269d768412f56d8df187361250fd5de9042f35f9c54bb7566800a8c69'
magic, samples, context = struct.unpack('<QII', data[:16])
assert (magic, samples, context) == (0x514e494e50555431, 64, 4096)
assert len(data) == 16 + samples * (context + 1) * 4
scratch = root / 'runs/qwen-hn1g-diagnostics/current-old-subset'
scratch.mkdir(parents=True, exist_ok=False)
snapshot = scratch / 'input-checkpoint.bin'
shutil.copyfile(current, snapshot)
with snapshot.open('rb') as stream:
    magic, count, completed = struct.unpack('<QQi', stream.read(20))
assert magic == 0x5157454E4C4F5233 and count == 10822656
windows = scratch / 'eval-windows.bin'
windows.write_bytes(data[16:])
assets = json.loads((root / '.cache/qwen-hn1g/assets.json').read_text())
assets['paths']['validation'] = str(windows)
assets['diagnostic'] = {'purpose': 'Compare saved adapter against the earlier fixed 64-window evaluation', 'input_sha256': hashlib.sha256(data).hexdigest(), 'checkpoint_step': completed, 'reserved_held_out': False, 'note': 'These windows are part of the full training corpus used by the six experiments.'}
(scratch / 'assets.json').write_text(json.dumps(assets, indent=2), encoding='utf-8')
(scratch / 'paths.txt').write_text('\n'.join(assets['paths'][key] for key in ('weights', 'manifest', 'train', 'validation', 'token_bytes')) + '\n', encoding='utf-8')
header = root / 'native/experiment.h'
original_header = header.read_bytes()
text = original_header.decode('utf-8')
values = {'Context': 4096, 'Batch': 4, 'EffectiveBatch': 16, 'Steps': completed, 'VramGiB': 20, 'Evaluate': 'true', 'Sequential': 'false', 'EvalSamples': 64, 'SequentialStarts': '""', 'Assets': json.dumps((scratch / 'paths.txt').relative_to(root).as_posix()), 'Output': json.dumps((scratch / 'evaluation').relative_to(root).as_posix())}
for name, value in values.items():
    text, n = re.subn(rf'(\b{re.escape(name)}\s*=\s*)[^,;\n]+', lambda m: m[1] + str(value), text)
    assert n == 1, name
try:
    header.write_text(text, encoding='utf-8')
    print(json.dumps({'state': 'building_diagnostic', 'checkpoint_step': completed, 'extra_training_updates': 0}), flush=True)
    with (scratch / 'build.log').open('w', encoding='utf-8') as log:
        subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'build.ps1', '-CudaArch', '120'], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in ('qwen-tune.exe', 'revision.txt', 'revision.diff'):
        shutil.copyfile(root / 'build' / name, scratch / name)
    (scratch / 'experiment.h').write_text(text, encoding='utf-8')
finally:
    header.write_bytes(original_header)
print(json.dumps({'state': 'evaluating_native', 'checkpoint_step': completed, 'header_restored': header.read_bytes() == original_header}), flush=True)
with (scratch / 'console.log').open('w', encoding='utf-8') as log:
    subprocess.run([str(scratch / 'qwen-tune.exe'), '--resume', str(snapshot)], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
report = json.loads((scratch / 'evaluation/report.json').read_text())
assert report['steps'] == completed and report['training_tokens'] == 0
assert report['baseline_eval_last_bpb'] == report['final_eval_last_bpb']
assert hashlib.sha256(snapshot.read_bytes()).digest() == hashlib.sha256((scratch / 'evaluation/checkpoint.bin').read_bytes()).digest()
old = json.loads((root / 'runs/native-bf16-4096-hn1g/report.json').read_text())
summary = {'checkpoint_step': completed, 'processed_tokens': completed * 65536, 'original_last_bpb': old['baseline_eval_last_bpb'], 'after_60_updates_last_bpb': old['final_eval_last_bpb'], 'current_last_bpb': report['final_eval_last_bpb'], 'delta_from_60_updates': report['final_eval_last_bpb'] - old['final_eval_last_bpb'], 'current_all_token_bpb': report['final_eval_bpb'], 'original_all_token_bpb': old['baseline_eval_bpb'], 'after_60_updates_all_token_bpb': old['final_eval_bpb'], 'current_last_loss': report['final_eval_last_loss'], 'input_sha256': hashlib.sha256(data).hexdigest(), 'checkpoint_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(), 'training_updates_during_diagnostic': 0, 'restored_checkpoint_bit_exact_after_evaluation': True, 'reserved_held_out': False, 'native_cpp_cuda': True}
(scratch / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary), flush=True)
