"""Wait for the A/B GPU lease, then validate, smoke and measure the DG5F port."""
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

from overnight_process import run_checked, assert_gpu_free

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/overnight_repose'


def write(state):
    temporary = OUT / 'results.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(OUT / 'results.json')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    lease = (ROOT / 'logs/overnight_ab/driver.lock').open('a')
    print('[REPOSE] Waiting for A/B driver GPU lease', flush=True)
    fcntl.flock(lease, fcntl.LOCK_EX)
    assert_gpu_free()
    path = OUT / 'results.json'
    state = json.loads(path.read_text()) if path.exists() else {'status': 'running', 'probes': {}}
    state['status'] = 'running'
    state.pop('error', None)
    write(state)
    try:
        if not (OUT / 'layout.json').exists():
            run_checked([sys.executable, 'scripts/probe_repose.py', '--headless', '--num_envs', '32',
                         '--output', str(OUT / 'layout.json')], OUT / 'layout.log', ROOT, timeout=900)
        state['layout'] = json.loads((OUT / 'layout.json').read_text())
        write(state)
        def smoke(count, iterations):
            key = str(count)
            if key in state['probes']:
                return state['probes'][key]['passed']
            tag = f'repose_smoke_{count}'
            command = [sys.executable, 'scripts/rsl_rl/train.py', '--task', 'Isaac-Repose-Cube-DG5F-Direct-v0',
                       '--headless', '--num_envs', str(count), '--max_iterations', str(iterations),
                       f'agent.run_name={tag}']
            try:
                metrics = run_checked(command, OUT / f'{tag}.log', ROOT, timeout=1800)
                content = (OUT / f'{tag}.log').read_text(errors='replace')
                # PhysX capacity errors silently invalidate contacts: no successful capacity claim.
                for bad in ('Patch buffer overflow', 'Contact buffer overflow', 'increase its size',
                            'GPU contact', 'nan', 'CUDA out of memory'):
                    if bad in content:
                        raise RuntimeError(f'Invalid physics/numerics: {bad}')
                run = max(p for p in (ROOT / 'logs/rsl_rl/dg5f_repose').iterdir() if p.name.endswith(tag))
                checkpoint = run / f'model_{iterations-1}.pt'
                if not checkpoint.exists():
                    raise RuntimeError('Final checkpoint missing')
                # Leave 256 MiB of measured board-memory headroom for a practical long run.
                passed = metrics['peak_total_MiB'] <= 6144 - 256
                record = {'passed': passed, 'iterations': iterations, 'checkpoint': str(checkpoint), **metrics}
            except Exception:
                record = {'passed': False, 'iterations': iterations, 'error': traceback.format_exc()}
                metadata = OUT / f'{tag}.process.json'
                if metadata.exists():
                    info = json.loads(metadata.read_text())
                    record.update({k: info[k] for k in ('peak_compute_MiB', 'peak_total_MiB') if k in info})
            state['probes'][key] = record
            write(state)
            print(f'[REPOSE] {count}: {record}', flush=True)
            assert_gpu_free()
            return record['passed']
        # Coarse-to-fine search in multiples of 256; never run two simulations together.
        low, high = 0, None
        if smoke(1024, 25):
            low = 1024
            for count in (2048, 4096, 8192):
                if smoke(count, 20):
                    low = count
                else:
                    high = count
                    break
        else:
            high = 1024
        if high is not None:
            while high - low > 256:
                middle = ((low + high) // 512) * 256
                if smoke(middle, 20):
                    low = middle
                else:
                    high = middle
        state['max_tested_safe_num_envs'] = low
        state['first_unsafe_num_envs'] = high
        state['capacity_resolution'] = 256
        write(state)
        reference = OUT / 'shadow_reference.log'
        if not state.get('reference'):
            metrics = run_checked([sys.executable, 'scripts/rsl_rl/play.py', '--headless', '--task',
                                   'Isaac-Repose-Cube-Shadow-Direct-v0', '--num_envs', '32',
                                   '--use_pretrained_checkpoint', '--max_steps', '1200'],
                                  reference, ROOT, timeout=1200)
            content = reference.read_text(errors='replace')
            if '[PLAY] completed 1200 steps' not in content or 'Loading model checkpoint from:' not in content:
                raise RuntimeError('Pretrained Shadow reference did not actually run')
            state['reference'] = metrics
        state['status'] = 'completed' if low else 'failed'
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        print(state['error'], flush=True)
    write(state)
    print('[REPOSE] Finished', flush=True)


if __name__ == '__main__':
    main()
