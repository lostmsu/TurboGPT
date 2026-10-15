from pathlib import Path
import json, math
from qwen_tuning.experiments import canonical_training_rows
root=Path('runs/qwen-hn1g/lr-sweep-4096')
reference=canonical_training_rows('runs/qwen-hn1g/v-4096/metrics.jsonl')
status=json.loads((root/'status.json').read_text())
name=status.get('run')
rows=canonical_training_rows(root/name/'metrics.jsonl') if name else []
if rows:
    n=len(rows); window=min(64,n); lo=n-window
    def stats(a):
        return {'mean_last_bpb':sum(x['last_bpb'] for x in a)/len(a), 'pooled_last_bpb':sum(x['last_nll'] for x in a)/math.log(2)/sum(x['last_bytes'] for x in a), 'mean_all_bpb':sum(x['bpb'] for x in a)/len(a)}
    print(json.dumps({'state':status['state'],'run':name,'updates':n,'tokens':rows[-1]['processed_tokens'],'window_updates':window,'current':stats(rows[lo:n]),'control_same_window':stats(reference[lo:n]),'latest_grad_norm':rows[-1]['grad_norm'],'tokens_per_second':rows[-1]['predicted_tokens']/rows[-1]['update_seconds'],'hours_training_remaining':(1240-n)*sum(x['update_seconds'] for x in rows[-min(n,16):])/min(n,16)/3600},indent=2))
else: print(status)
