"""Render stochastic training trends separately from final deterministic evaluation."""
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'std': 'Mean action noise std', 'reward': 'Mean reward',
          'drop': 'Mean episode episode_drop_rate', 'goals': 'Mean episode episode_goals_completed',
          'episode_length': 'Mean episode length'}


def parse(path):
    blocks = re.split(r'Learning iteration\s+(\d+)/\d+', path.read_text())
    rows = []
    for offset in range(1, len(blocks), 2):
        row = {'iteration': int(blocks[offset]) + 1}
        for name, label in LABELS.items():
            match = re.search(re.escape(label) + r':\s+([\d.e+-]+)', blocks[offset + 1])
            row[name] = float(match.group(1)) if match else None
        rows.append(row)
    return rows


def moving_mean(rows, key, width=100):
    result = []
    for stop in range(1, len(rows) + 1):
        values = [r[key] for r in rows[max(0, stop-width):stop] if r[key] is not None and math.isfinite(r[key])]
        result.append(sum(values) / len(values) if values else float('nan'))
    return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    experiments = ['E2_exploration', 'E6_combination', 'E1_horizon']
    data = {name: parse(ROOT / 'logs/overnight_ab' / name / 'train.log') for name in experiments}
    (ROOT / 'reports/exploration_training_curves.json').write_text(json.dumps(data, indent=2))
    (ROOT / 'reports/e2_training_curve.json').write_text(json.dumps(data['E2_exploration'], indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for axis, key in zip(axes.flat, ['reward', 'drop', 'goals', 'std']):
        for name, rows in data.items():
            axis.plot([r['iteration'] for r in rows], moving_mean(rows, key), label=name.split('_')[0])
        axis.set_ylabel(key)
        axis.grid(alpha=.2)
        axis.legend()
    axes[1, 0].set_xlabel('Completed PPO iterations')
    axes[1, 1].set_xlabel('Completed PPO iterations')
    fig.suptitle('Stochastic training: trailing mean of 100 iteration logs (not deterministic evaluation)')
    fig.savefig(ROOT / 'reports/exploration_training_curves.png', dpi=160)
    fig.savefig(ROOT / 'reports/exploration_training_curves.svg')
    rows = data['E2_exploration']
    lines = ['# Уточнение интерпретации ночной лестницы', '',
             '600 итераций были бюджетом отбора гипотез, а не проверкой сходимости. '
             'E2 завершился на улучшающемся позднем участке; оснований объявлять его '
             'неперспективным или достигшим плато нет. Он остановлен по заданному бюджету.', '',
             '![Training curves](exploration_training_curves.png)', '',
             '## E2: средние по непересекающимся окнам из 100 итераций', '',
             '| Итерации | std | reward | training drop | training целей/эп | длина эпизода |',
             '|---|---:|---:|---:|---:|---:|']
    for start in range(0, 600, 100):
        window = rows[start:start+100]
        means = {key: sum(row[key] for row in window) / len(window) for key in LABELS}
        lines.append(f'| {start+1}–{start+100} | {means["std"]:.3f} | {means["reward"]:.2f} | '
                     f'{means["drop"]:.3f} | {means["goals"]:.3f} | {means["episode_length"]:.1f} |')
    lines += ['', 'Это усреднение строк лога, без взвешивания по числу завершённых эпизодов. '
              'Между первыми двумя окнами reward/drop ухудшались; поздний участок улучшается. '
              'Точки отдельных итераций заметно шумят: 0.579 drop на последней итерации '
              'не означает 0.579 в среднем на последней сотне (там 0.790).', '',
              'Training drop и детерминированный drop — разные измерения. Конечный '
              'детерминированный drop E2 0.641 против 0.078 у исходной политики остаётся фактом, '
              'но не описывает направление обучения и не служит причиной остановки. '
              'Исходная политика также завершала 0.039 целей/эпизод, поэтому буквально '
              '«ничего не делает» — неточное описание.', '',
              'E5 проверил 10° только в режиме std0.2 / entropy0: вариант 10° + исследование '
              'оставался неизмеренным. Результат E5 не опровергает этот вариант.', '',
              'E3 тоже имел held выше 0.28 (0.289), хотя std уменьшался. Поэтому утверждение '
              '«только E2 и E6 выше 0.28» неверно. Исследование — наиболее перспективный '
              'рычаг этой лестницы, но его единственность и универсальное превосходство не доказаны.', '',
              'gamma=0.998 исключён из следующего плана. Итоги E1/E6 не дают оснований '
              'предпочесть его E2; они не доказывают универсальную вредность gamma=0.998 '
              'в других бюджетах или режимах.', '',
              'Продолжение: reports/e2_continuation.md; logs/e2_continuation/state.json. '
              'E2 получает ещё 3000 итераций без изменения PPO/задачи. Сохранён adaptive LR '
              '0.00011390625, а не конфиговый 0.001: upstream load() восстанавливает Adam, '
              'но не поле PPO.learning_rate. Исправление выполнено в локальном train.py, '
              'без правки RSL-RL. Затем отдельный 600-итерационный опыт 10° + std1 + entropy0.005.', '']
    (ROOT / 'reports/e2_trend_interpretation.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
