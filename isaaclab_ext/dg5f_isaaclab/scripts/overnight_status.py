"""Read campaign progress without importing Torch or starting a simulator."""
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
state = json.loads((root / 'logs/overnight_ab/results.json').read_text())
for name, record in state['experiments'].items():
    if record['status'] == 'completed':
        m = record['metrics']
        print(f'{name}: completed held={m["held_success_rate"]:.3f} drop={m["drop_rate"]:.3f} '
              f'goals={m["goals_completed_per_episode"]:.3f} min={m["min_error_deg"]:.2f}')
    else:
        logs = list((root / 'logs/overnight_ab' / name).glob('*train.log'))
        log = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
        content = log.read_text(errors='replace') if log else ''
        iterations = re.findall(r'Learning iteration\s+(\d+/\d+)', content)
        eta = re.findall(r'ETA:\s+(\S+)', content)
        noise = re.findall(r'Mean action noise std:\s+(\S+)', content)
        print(f'{name}: {record["status"]}; iteration={iterations[-1:]}; ETA={eta[-1:]}; std={noise[-1:]}')
        if record.get('error'):
            print(record['error'])
repose = root / 'logs/overnight_repose/results.json'
if repose.exists():
    state = json.loads(repose.read_text())
    print('repose:', state['status'], 'capacity:', state.get('max_tested_safe_num_envs'),
          'probes:', {key: value['passed'] for key, value in state.get('probes', {}).items()})
    if state.get('error'):
        print(state['error'])
print(subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
                     capture_output=True, text=True, check=True).stdout.strip())
