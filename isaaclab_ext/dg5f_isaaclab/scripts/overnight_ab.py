"""Fixed, resumable sequential E2/E1/E5/E4/E3/E6 experiment ladder (one GPU)."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from overnight_process import ResumeError, assert_gpu_free, run_checked

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'logs/rsl_rl/dg5f_cube_direct'
WARM = RUNS / '2026-09-29_00-00-00_warm_start_v4/model_0.pt'
STD1 = RUNS / '2026-09-29_00-00-01_warm_start_std1/model_0.pt'
EXPERIMENTS = [
    {'name': 'E2_exploration', 'std1': True, 'overrides': ['agent.algorithm.entropy_coef=0.005']},
    {'name': 'E1_horizon', 'overrides': ['agent.algorithm.gamma=0.998']},
    {'name': 'E5_goal10', 'bootstrap': True, 'overrides': []},
    {'name': 'E4_reward_shape', 'overrides': ['env.orientation_baseline=false']},
    {'name': 'E3_batch', 'overrides': ['agent.num_steps_per_env=64']},
    {'name': 'E6_combination', 'std1': True, 'overrides': [
        'agent.algorithm.gamma=0.998', 'agent.algorithm.entropy_coef=0.005']},
]
REFERENCES = [
    ('warm start 20deg', .039, .078, .039, 13.1),
    ('old reward @250', .109, .500, .117, 12.4),
    ('old reward @500', .070, .570, .070, 13.4),
    ('grasp defence @100', .047, .289, .050, 12.2),
    ('warm start 10deg', .250, .156, .300, 6.4),
    ('zero policy', .000, .031, .000, 18.6),
]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def summary(state):
    lines = ['| experiment | held | drop | goals/ep | min err deg | reward | action_std |',
             '|---|---:|---:|---:|---:|---:|---:|']
    reference_path = ROOT / 'reports/overnight_references.json'
    references = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    for name, held, drop, goals, error in REFERENCES:
        reward, std = '—', '—'
        if name in references:
            m = references[name]['metrics']
            held, drop, goals, error = (m[key] for key in (
                'held_success_rate', 'drop_rate', 'goals_completed_per_episode', 'min_error_deg'))
            reward = f'{m["episode_reward"]:.2f}'
            std = f'{m["action_std"]:.3f}' if m.get('action_std') is not None else '—'
        label = name + (' @5' if name == 'warm start 10deg' else '')
        lines.append(f'| {label} (reference) | {held:.3f} | {drop:.3f} | {goals:.3f} | {error:.1f} | {reward} | {std} |')
    for name, record in state['experiments'].items():
        if record['status'] != 'completed':
            lines.append(f'| {name}: {record["status"]} | | | | | | |')
            continue
        m = record['metrics']
        lines.append(f'| {name} | {m["held_success_rate"]:.3f} | {m["drop_rate"]:.3f} | '
                     f'{m["goals_completed_per_episode"]:.3f} | {m["min_error_deg"]:.2f} | '
                     f'{m["episode_reward"]:.2f} | {m["action_std"]:.3f} |')
    return '\n'.join(lines) + '\n'


def newest(tag):
    return max(p for p in RUNS.iterdir() if p.is_dir() and p.name.endswith('_' + tag))


def latest(run):
    return max(run.glob('model_*.pt'), key=lambda p: int(p.stem[6:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/overnight_ab')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    path = args.output / 'results.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'created': datetime.now().isoformat(), 'iterations': 600, 'num_envs': 1024,
        'eval_num_envs': 128, 'eval_seed': 1234, 'plan': EXPERIMENTS,
        'warm_sha256': hashlib.sha256(WARM.read_bytes()).hexdigest(),
        'gate': {'held': .70, 'drop': .15, 'goals_per_episode': 1.5}, 'experiments': {}}
    if state['plan'] != EXPERIMENTS or state['warm_sha256'] != hashlib.sha256(WARM.read_bytes()).hexdigest():
        raise RuntimeError('Plan or warm start changed; refusing to mix results')
    save(path, state)
    assert_gpu_free()
    if not STD1.exists():
        run_checked([sys.executable, 'scripts/make_warm_start.py', '--source', str(WARM),
                     '--out', str(STD1), '--init_noise_std', '1.0', '--learning_rate', '1.0e-3'],
                    args.output / 'make_warm.log', ROOT, timeout=120)
    # CPU-only verification: no accidental actor/critic/normalization confound.
    import torch
    base = torch.load(WARM, map_location='cpu', weights_only=False)
    high = torch.load(STD1, map_location='cpu', weights_only=False)
    different = [k for k, v in base['model_state_dict'].items()
                 if not torch.equal(v, high['model_state_dict'][k])]
    if different != ['std'] or not torch.all(high['model_state_dict']['std'] == 1):
        raise RuntimeError(f'Warm-start regeneration changed tensors other than std: {different}')
    for spec in EXPERIMENTS:
        name = spec['name']
        record = state['experiments'].get(name, {})
        if record.get('status') == 'completed':
            continue
        warm = STD1 if spec.get('std1') else WARM
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        tag = 'overnight_' + name
        try:
            if record.get('status') != 'trained':
                record = {'status': 'training', 'spec': spec, 'warm_start': str(warm)}
                state['experiments'][name] = record
                save(path, state)
                if spec.get('bootstrap'):
                    old = ROOT / 'logs/stream/state.json'
                    if old.exists():
                        old.rename(old.with_name(f'state.before_overnight_{datetime.now():%Y%m%d_%H%M%S}.json'))
                    cmd = [sys.executable, 'scripts/train_stream.py', '--bootstrap', '--warm_start',
                           str(warm.parent), '--num_envs', '1024', '--max_iterations', '600',
                           '--eval_every_iterations', '600', '--state', str(folder / 'state.json'),
                           '--log_dir', str(folder), '--run_prefix', tag]
                    run_checked(cmd, folder / 'train.log', ROOT, timeout=7200)
                    stream = json.loads((folder / 'state.json').read_text())
                    if stream['total_iterations'] != 600:
                        raise RuntimeError('E5 did not complete 600 iterations')
                    checkpoint = RUNS / stream['load_run'] / stream['load_checkpoint']
                else:
                    cmd = [sys.executable, 'scripts/rsl_rl/train.py', '--task', 'DG5F-Cube-Stream-Direct-v0',
                           '--headless', '--num_envs', '1024', '--max_iterations', '600', '--resume',
                           '--load_run', warm.parent.name, '--checkpoint', warm.name,
                           'env.goal_stream_stage=A', f'agent.run_name={tag}', *spec['overrides']]
                    run_checked(cmd, folder / 'train.log', ROOT, expected=warm)
                    checkpoint = latest(newest(tag))
                if int(checkpoint.stem[6:]) != 599:
                    raise RuntimeError(f'Incomplete training: {checkpoint}')
                record.update(status='trained', checkpoint=str(checkpoint))
                save(path, state)
            checkpoint = Path(record['checkpoint'])
            cmd = [sys.executable, 'scripts/eval_checkpoints.py', '--task', 'DG5F-Cube-Stream-Direct-v0',
                   '--headless', '--num_envs', '128', '--seed', '1234', '--baselines', '', '--stage', 'A',
                   '--run', f'{name}={checkpoint.parent}:{checkpoint.stem[6:]}',
                   '--output', str(folder / 'eval.json')]
            if spec.get('bootstrap'):
                cmd += ['--goal_angle_deg', '10.0']
            if name == 'E4_reward_shape':
                cmd += ['--orientation_baseline', 'false']
            run_checked(cmd, folder / 'eval.log', ROOT, timeout=1200)
            metrics = next(iter(json.loads((folder / 'eval.json').read_text()).values()))
            if metrics['episodes'] != 128:
                raise RuntimeError('Incomplete deterministic evaluation')
            record.update(status='completed', metrics=metrics,
                          gate_passed=(metrics['held_success_rate'] >= .70 and metrics['drop_rate'] <= .15
                                       and metrics['goals_completed_per_episode'] >= 1.5))
        except ResumeError:
            record.update(status='failed', error=traceback.format_exc())
            save(path, state)
            raise  # A broken resume invalidates the ladder, not just an experiment.
        except Exception:
            record.update(status='trained' if record.get('checkpoint') else 'failed', error=traceback.format_exc())
            print(record['error'], flush=True)
        save(path, state)
        table = summary(state)
        (args.output / 'summary.md').write_text(table)
        print(table, flush=True)
        assert_gpu_free()
    print('[AB] Ladder finished', flush=True)


if __name__ == '__main__':
    main()
