"""Continue E2 unchanged to 3600 total iterations, then test 10 degrees + exploration.

One uninterrupted training process preserves E2's trajectory after the initial resume.
Intermediate checkpoints are evaluated sequentially AFTER training to avoid GPU overlap
and repeated simulator/optimizer restarts. No curriculum/stage/std-plateau watchdog is
used for the E2 budget. The separate 10-degree experiment uses a per-run threshold 3.0.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from overnight_ab import ROOT, RUNS, STD1, save
from overnight_process import assert_gpu_free, run_checked

OUT = ROOT / 'logs/e2_continuation'
TASK = 'DG5F-Cube-Stream-Direct-v0'
TOTAL = 3600


def latest_run(tag):
    return max(p for p in RUNS.iterdir() if p.name.endswith('_' + tag))


def report(state):
    lines = ['# Продолжение E2', '', f'Статус: {state["status"]}.', '',
             'E2: ещё 3000 итераций после исходных 600, gamma=0.99, entropy_coef=0.005, '
             '1024 среды, rollout=32, 20°, стадия A. Актор, критик, std, нормализаторы, '
             'Adam и его адаптивный LR продолжаются из model_599.pt. '
             'Физическое состояние симулятора и RNG в checkpoint не сохраняются: '
             'при первом возобновлении среды сбрасываются, затем обучение непрерывное.', '',
             'Оценки сохранённых промежуточных checkpoint выполняются после обучения, '
             'последовательно на той же GPU: 128 эпизодов, seed=1234, mean-action policy. '
             'Гейт: held >=0.70, drop <=0.15, целей/эп >=1.5.', '',
             '| Вариант / итерации | held | drop | целей/эп | min ошибка | reward | std |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, m in state['evaluations'].items():
        lines.append(f'| {label} | {m["held_success_rate"]:.3f} | {m["drop_rate"]:.3f} | '
                     f'{m["goals_completed_per_episode"]:.3f} | {m["min_error_deg"]:.2f} | '
                     f'{m["episode_reward"]:.2f} | {m["action_std"]:.3f} |')
    lines += ['', 'Следующий отдельный опыт: 10° + исследование, 600 итераций от '
              'warm_start_std1/model_0.pt, entropy_coef=0.005, gamma=0.99. '
              'Порог watchdog для этого режима 3.0 (он не ограничивает sigma и не меняет PPO); '
              'гейты не меняются. E2 вообще не использует низкошумовой watchdog 0.35.', '']
    if state.get('error'):
        lines += ['Ошибка:', '```', state['error'], '```']
    (ROOT / 'reports/e2_continuation.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lease = (ROOT / 'logs/overnight_ab/driver.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    baseline = json.loads((ROOT / 'logs/overnight_ab/results.json').read_text())['experiments']['E2_exploration']
    original = Path(baseline['checkpoint'])
    path = OUT / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'status': 'queued', 'source': str(original),
        'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'total_iterations': TOTAL, 'additional_iterations': 3000,
        'training_complete': False, 'ten_degree_complete': False,
        'evaluations': {'E2/600': baseline['metrics']}}
    if state['source_sha256'] != hashlib.sha256(original.read_bytes()).hexdigest():
        raise RuntimeError('Source E2 checkpoint changed')
    import torch
    def checkpoint_iteration(checkpoint):
        return int(torch.load(checkpoint, weights_only=False, map_location='cpu')['iter'])
    def persist(status):
        state['status'] = status
        state.pop('error', None)
        save(path, state)
        report(state)
    def evaluate(label, checkpoint, angle):
        if label in state['evaluations']:
            return
        dest = OUT / (label.replace('/', '_') + '_eval.json')
        command = [sys.executable, 'scripts/eval_checkpoints.py', '--task', TASK, '--headless',
                   '--num_envs', '128', '--seed', '1234', '--baselines', '', '--stage', 'A',
                   '--goal_angle_deg', str(angle), '--run',
                   f'cur={checkpoint.parent}:{checkpoint.stem[6:]}', '--output', str(dest)]
        run_checked(command, dest.with_suffix('.log'), ROOT, timeout=1200)
        metrics = next(iter(json.loads(dest.read_text()).values()))
        if metrics['episodes'] != 128:
            raise RuntimeError('Incomplete evaluation')
        state['evaluations'][label] = metrics
        persist('evaluating')
    assert_gpu_free()
    try:
        if not state['training_complete']:
            # Recover a interrupted continuation from its most recent saved update.
            tag = 'e2_continue_3600'
            candidates = [p for run in RUNS.glob('*_' + tag) for p in run.glob('model_*.pt')]
            checkpoint = max(candidates, key=checkpoint_iteration) if candidates else original
            completed = checkpoint_iteration(checkpoint) + 1
            if completed < TOTAL:
                persist('training_E2')
                command = [sys.executable, 'scripts/rsl_rl/train.py', '--task', TASK, '--headless',
                           '--num_envs', '1024', '--max_iterations', str(TOTAL - completed),
                           '--resume', '--load_run', checkpoint.parent.name, '--checkpoint', checkpoint.name,
                           '--restore_continuation_state', 'env.goal_stream_stage=A',
                           f'agent.run_name={tag}', 'agent.algorithm.entropy_coef=0.005']
                log = OUT / f'train_from_{completed}.log'
                run_checked(command, log, ROOT, expected=checkpoint, timeout=30000)
                checkpoint = latest_run(tag) / f'model_{TOTAL-1}.pt'
            if not checkpoint.exists() or checkpoint_iteration(checkpoint) != TOTAL - 1:
                raise RuntimeError('E2 did not reach the complete 3600-iteration budget')
            state['training_complete'] = True
            state['checkpoint'] = str(checkpoint)
            persist('trained_E2')
        # Include every retained half-thousand update, including earlier recovered segments.
        choices = {}
        for run in sorted(RUNS.glob('*_e2_continue_3600')):
            for checkpoint in run.glob('model_*.pt'):
                index = int(checkpoint.stem[6:])
                if index in (1000, 1500, 2000, 2500, 3000, 3500, 3599):
                    choices[index] = checkpoint
        for index, checkpoint in sorted(choices.items()):
            evaluate(f'E2/{index+1}', checkpoint, 20.0)
        if f'E2/{TOTAL}' not in state['evaluations']:
            raise RuntimeError('Final E2 evaluation missing')
        if not state['ten_degree_complete']:
            persist('training_10deg_exploration')
            folder = OUT / 'goal10_exploration'
            folder.mkdir(exist_ok=True)
            stream_state = folder / 'state.json'
            command = [sys.executable, 'scripts/train_stream.py', '--bootstrap', '--warm_start', str(STD1.parent),
                       '--num_envs', '1024', '--max_iterations', '600', '--eval_every_iterations', '600',
                       '--state', str(stream_state), '--log_dir', str(folder), '--run_prefix', 'goal10_exploration',
                       '--entropy_coef', '0.005', '--action_std_watchdog', '3.0']
            # If evaluation had already completed before an interruption, avoid training twice.
            prior = json.loads(stream_state.read_text()) if stream_state.exists() else {}
            if prior.get('total_iterations') != 600 or not prior.get('evaluations'):
                run_checked(command, folder / 'driver.log', ROOT, timeout=7200)
            result = json.loads(stream_state.read_text())
            if result['total_iterations'] != 600 or result['evaluations'][-1]['angle_deg'] != 10.0:
                raise RuntimeError('Wrong bootstrap budget or angle')
            state['evaluations']['10deg_exploration/600'] = result['evaluations'][-1]['metrics']
            state['ten_degree_checkpoint'] = result['load_run'] + '/' + result['load_checkpoint']
            state['ten_degree_complete'] = True
        persist('completed')
        journal = ROOT / 'JOURNAL.md'
        old = journal.read_text()
        from datetime import datetime
        entry = (f'## {datetime.now():%Y-%m-%d} — продолжение E2 и 10° с исследованием завершены\n\n'
                 'Измеренные оценки всех checkpoint: reports/e2_continuation.md и '
                 'logs/e2_continuation/state.json. E2 получил полные 3600 итераций, '
                 'отдельный 10°-опыт — 600 от warm_start_std1. Параметры по ходу не подбирались.\n\n')
        position = old.index('## ')
        journal.write_text(old[:position] + entry + old[position:])
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        save(path, state)
        report(state)
        raise


if __name__ == '__main__':
    main()
