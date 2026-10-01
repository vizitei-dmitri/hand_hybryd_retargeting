"""Render the measured A/B campaign and append a final local journal entry."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from overnight_ab import summary

ROOT = Path(__file__).resolve().parents[1]


def build(final=False):
    ab_path = ROOT / 'logs/overnight_ab/results.json'
    ab = json.loads(ab_path.read_text())
    repose_path = ROOT / 'logs/overnight_repose/results.json'
    repose = json.loads(repose_path.read_text()) if repose_path.exists() else {'status': 'queued'}
    done = {name: record for name, record in ab['experiments'].items() if record['status'] == 'completed'}
    lines = ['# Ночная диагностика DG5F', '',
             '**Уточнение после анализа кривых:** E2 закончил 600 итераций на улучшающемся '
             'позднем участке и продолжен до бюджета 3600. Отсутствие прохода гейта '
             'при 600 итерациях не означает провал идеи исследования или плато. '
             'Разбор с графиками: [e2_trend_interpretation.md](e2_trend_interpretation.md).', '',
             f'Срез: {datetime.now().isoformat(timespec="seconds")}. Завершено A/B: {len(done)}/6.', '',
             'Исходный коммит ec922e8; ветка experiment/dg5f-overnight-ab-repose.', '',
             'Все E* стартуют независимо от одного актора, 600 итераций, 1024 среды, стадия A. '
             'E2/E6 отличаются в исходном checkpoint только std=1.0. Оценки детерминированные, '
             '128 полных эпизодов, seed 1234. E5 обучается/оценивается на 10°, остальные на 20°.', '',
             summary(ab),
             'Гейт неизменен: held ≥ 0.70, drop ≤ 0.15, целей/эпизод ≥ 1.5.', '',
             '## Сдвиги относительно исходной политики', '',
             '| Эксперимент | Δ held | Δ drop | Δ целей/эп | Δ min ошибки | Гейт |',
             '|---|---:|---:|---:|---:|---|']
    for name, record in done.items():
        m = record['metrics']
        baseline = (.25, .15625, .296875, 6.389602731644787) if name == 'E5_goal10' else (.0390625, .078125, .0390625, 13.113211245285544)
        lines.append(f'| {name} | {m["held_success_rate"]-baseline[0]:+.3f} | '
                     f'{m["drop_rate"]-baseline[1]:+.3f} | {m["goals_completed_per_episode"]-baseline[2]:+.3f} | '
                     f'{m["min_error_deg"]-baseline[3]:+.2f}° | {"да" if record["gate_passed"] else "нет"} |')
    lines += ['', 'Отрицательный Δ drop/ошибки означает улучшение. Для E5 сравнение идёт со своей '
              'опорной точкой 10°: сравнивать его преимущество с 20° как эффект обучения нельзя. '
              'Опорные числа уточнены по сохранённым JSON (reports/overnight_references.json). '
              'Один seed и 128 эпизодов дают диагностический срез, а не проверку статистической устойчивости. '
              'Опора 10° фактически снята после 5 итераций (b/model_4), а не при нулевом обучении. '
              'Старые episode_reward измерены до последующих reward-правок и приведены как история, '
              'их величины нельзя напрямую сравнивать с сегодняшней наградой.', '']
    passes = [name for name, record in done.items() if record['gate_passed']]
    lines.append('Гейт прошли: ' + ', '.join(passes) + '.' if passes else 'Ни один завершённый прогон гейт не прошёл.')
    improvements = []
    for name, record in done.items():
        m = record['metrics']
        b = (.25, .15625, .296875) if name == 'E5_goal10' else (.0390625, .078125, .0390625)
        if m['held_success_rate'] > b[0] and m['drop_rate'] <= b[1] and m['goals_completed_per_episode'] > b[2]:
            improvements.append(name)
    lines += ['Одновременное улучшение held и целей/эп без роста drop: ' + ', '.join(improvements) + '.'
              if improvements else 'Одновременного улучшения held и целей/эп без роста drop в завершённых прогонах нет.', '',
              'E4 выполняет ровно env.orientation_baseline=false: существующая функция — '
              'scale·exp(-(err/sigma)²), sigma=10°, а не exp(-err/sigma) из текстового обоснования. '
              'Награда в таблице измерена с той же формой, что при обучении; её уровень между '
              'E4 и другими экспериментами напрямую несопоставим.', '',
              '## Порт исходной задачи', '',
              'Задача Isaac-Repose-Cube-DG5F-Direct-v0 использует исходную InHandManipulationEnv. '
              'EnvCfg наследует награду, допуск, падение, сбросы и частоту Shadow. '
              'Фактические размерности: 19 действий, 146 наблюдений, 0 отдельных состояний критика.', '',
              'rj_dg_5_1 остаётся неподвижным в нуле: в отдельном импортируемом ассете ось fixed, '
              'поскольку нулевая ширина PhysX-лимитов дала бы деление на ноль в upstream unscale. '
              'Сохранены все тела/массы/коллизии и параметры 19 подвижных осей. Исходный URDF не изменён. '
              'Куб 60 мм, 50 г. Фиксированная ось — явное отличие представления от численной блокировки '
              'лимитами в stream.', '',
              'Порт использует штатные абсолютные targets upstream. Он проверяет эту задачу на DG5F, '
              'но не проверяет трёхшаговую задержку и интегратор delta-команд stream. '
              'Смок 20–25 итераций доказывает работоспособность и расход памяти; отсутствие обучения '
              'за такой бюджет не доказывает дефект кисти.', '',
              f'Статус порта: {repose["status"]}.', '',
              '| Сред | Итераций | Пик compute, МиБ | Пик всей GPU, МиБ | Прошёл |',
              '|---:|---:|---:|---:|---|']
    for count, record in sorted(repose.get('probes', {}).items(), key=lambda pair: int(pair[0])):
        lines.append(f'| {count} | {record["iterations"]} | {record.get("peak_compute_MiB", "—")} | '
                     f'{record.get("peak_total_MiB", "—")} | {"да" if record["passed"] else "нет"} |')
    if 'max_tested_safe_num_envs' in repose:
        lines += ['', f'Максимум с измеренным запасом ≥256 МиБ: {repose["max_tested_safe_num_envs"]} сред; '
                  f'следующая неуспешная точка: {repose["first_unsafe_num_envs"]}. Шаг поиска 256 сред. '
                  'Пики семплируются nvidia-smi каждые 5 с; более короткие пики могли не попасть в выборку.']
    if repose.get('reference'):
        lines += ['', 'Эталон Shadow с опубликованным checkpoint проверен headless при 32 средах, '
                  '1200 шагов, без камер/GUI: logs/overnight_repose/shadow_reference.log.']
    if len(done) == 6:
        lines += ['', '## Вывод по шести прогонам', '',
                  'E2 дал наибольший сдвиг к достижению цели: held 0.039 → 0.461 (+42.2 п.п.), '
                  'целей/эпизод 0.039 → 0.633, min ошибка 13.11° → 4.02°. Цена — drop 0.078 → 0.641 '
                  '(+56.3 п.п.). Достижение цели улучшилось, надёжность хвата ухудшилась.', '',
                  'E3 улучшил held до 0.289 и целей/эпизод до 0.391, при drop 0.508. '
                  'Но 600 итераций E3 — это 39 321 600 переходов, вдвое больше, чем 19 660 800 '
                  'в остальных прогонах: эффект размера батча не отделён от дополнительного опыта.', '',
                  'E6 относительно E2 снизил drop на 24.2 п.п. (0.641 → 0.398), но held снизился '
                  'на 16.4 п.п. (0.461 → 0.297), а целей/эпизод — с 0.633 до 0.297. '
                  'Это обмен частоты достижения целей на надёжность, а не прохождение гейта.', '',
                  'E1 отдельно ухудшил held до нуля. E5 ухудшил свою опору 10°: held 0.250 → 0.094, '
                  'drop 0.156 → 0.617, целей/эпизод 0.297 → 0.102. '
                  'E4 дал небольшой рост held до 0.078 при большом росте drop до 0.555. '
                  'Ни один вариант не решил одновременно переориентацию и удержание.', '']
    if repose.get('reference'):
        lines += ['Опубликованный Shadow checkpoint потребовал адаптации старых отдельных '
                  'obs_norm_state_dict/critic_obs_norm_state_dict к RSL-RL 3.1.2. '
                  'Локальный checkpoint_compat.py сохраняет веса и mean/var/std без изменения; '
                  'копия пригодна только для inference (исходные counts не сохранены). '
                  'RSL-RL не правился. После адаптации: 1200 шагов, consecutive_successes=8.3243.', '']
    lines += ['', '## Проверки и артефакты', '',
              '- CPU-проверки: logs/overnight_ab/tests_all.log.',
              '- Полные A/B метрики и checkpoint: logs/overnight_ab/results.json.',
              '- Команды, подтверждения resume, логи и GPU-сэмплы: logs/overnight_ab/E*/.',
              '- Проверки порта и измерения: logs/overnight_repose/results.json и layout.json.',
              '- Повторный запуск лестницы: python scripts/overnight_ab.py (завершённые оценки пропускаются).',
              '- Повторный запуск порта: python scripts/overnight_repose.py (ждёт освобождения GPU лестницей).', '']
    for name, record in ab['experiments'].items():
        if record['status'] != 'completed':
            lines += [f'Незавершённый {name}: {record["status"]}. Ошибка: {record.get("error", "ещё выполняется")}', '']
    if repose.get('error'):
        lines += ['Ошибка порта:', '```', repose['error'], '```', '']
    report = ROOT / 'reports/overnight_ab_repose.md'
    report.parent.mkdir(exist_ok=True)
    report.write_text('\n'.join(lines))
    if final:
        journal = ROOT / 'JOURNAL.md'
        old = journal.read_text()
        marker = 'ночная лестница — измеренный итог'
        if marker not in old:
            entry = (f'## {datetime.now():%Y-%m-%d} — {marker}\n\n'
                     f'Завершено {len(done)}/6 A/B, прошли гейт: {", ".join(passes) or "никто"}. '
                     f'Порт: {repose["status"]}. Числа, сдвиги и ограничения: '
                     'reports/overnight_ab_repose.md; машинные результаты — '
                     'logs/overnight_ab/results.json и logs/overnight_repose/results.json.\n\n'
                     'Вывод нельзя делать по training drop_rate: таблица построена по отдельным '
                     'детерминированным оценкам. E5 сравнивается с 10°, E4 — с правильной формой награды. '
                     'Короткий смок порта не отделяет нехватку бюджета от дефекта управления.\n\n')
            position = old.index('## ')
            journal.write_text(old[:position] + entry + old[position:])
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    if args.watch:
        while True:
            path = ROOT / 'logs/overnight_repose/results.json'
            if path.exists() and json.loads(path.read_text()).get('status') in ('completed', 'failed'):
                break
            time.sleep(30)
    build(final=args.final)


if __name__ == '__main__':
    main()
