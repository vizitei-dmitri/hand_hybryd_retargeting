"""Render measured results even if the bounded campaign terminates early."""
import json
from pathlib import Path
import numpy as np


def write_report(state, out):
    evaluations = state.get("evaluations", {})
    lines = ["# Ночной эксперимент: сглаживание действий", "",
             f"Начало: {state['started_at']}; предел обучения: {state['hard_deadline']}; крайний срок завершения: {state['cleanup_deadline']}.",
             f"Фактическое завершение: {state.get('completed_at', state.get('current_timestamp'))}.",
             f"Git HEAD: `{state['git_HEAD']}`. Коммиты не выполнялись.", "",
             "Гипотеза: увеличение стоимости величины команды уменьшит детерминированное насыщение, "
             "сохранив манипуляцию и, возможно, уменьшив падения. Это не гипотеза о шуме исследования.", "",
             "Физика, SysID, куб60 мм/50 г, cap0.4 Нм, delta1°, задержка, частота, кэш хватов и Stage A/20° неизменны. "
             "Принятый аудит скоростей не блокировал обучение; p99/π≈95.4% остаётся риском переноса на реальную кисть.", "",
             "μ — сырой выход актора до клипа. Команда a=clamp(μ,-1,1), и именно mean(a²) входит в штраф. "
             "Все новые подробные статистики относятся к первым эпизодам 128 сред, включая последний шаг; "
             "переходы через reset исключены. Исторический near-limit≈0.8693 усреднял также перезапущенные среды; "
             "здесь рядом сохранён исторический показатель, но для решений использован единый first-episode показатель≈0.8671. "
             "Скорость FD измерена на физических подшагах и в таблице исключает первую секунду эпизода.", "",
             "## Preflight", "",
             "task_abs = |orientation_state| + |orientation_progress| + |goal_dwell| + нормированный за шаг success_bonus. "
             "Кандидаты пересчитаны офлайн по одной исходной траектории, без повторной физики для каждого веса.", "",
             "| action scale | abs penalty/шаг | % task_abs | прогноз reward/эп | zero reward |",
             "|---:|---:|---:|---:|---:|"]
    for r in state.get("preflight", {}).get("candidates", []):
        lines.append(f'| {r["scale"]:.7g} | {r["mean_abs_action_penalty"]:.5f} | {100*r["fraction_of_task_abs"]:.2f}% | {r["projected_return"]:.3f} | {r["zero_return"]:.3f} |')
    lines += ["", f"Выбран action scale={state.get('selected_action_penalty_scale')}: минимальный доступный кандидат в целевом диапазоне15–35%, "
              "при котором компетентная политика выгоднее нулевой (если диапазон недостижим — явно меньший доступный кандидат). "
              "Первичный action-rate scale=0.001. Полные перцентили каждого вклада находятся в `BASELINE_3599.json` → smoothing.reward_terms.", "",
              f"Фиксированный LR={state.get('fixed_lr')}: точное значение Adam в обоих model_3599 до адаптивного обвала. "
              "Поэтому restore_continuation_state не конфликтует с выбором: восстановленный LR равен выбранному. "
              "Задан schedule=fixed; фактические PPO/Adam LR проверены до обучения, на каждой итерации и в каждом конечном checkpoint. "
              "Проверяются все тензоры актора/критика/нормализаторов и все состояния Adam; разрешены только явно заданные изменения двух action-штрафов. "
              "Entropy=0.001 одинаков для H/L. H сохраняет std5.12175; L начинает с0.5. Иные отличия между smoke не вводились.", "",
              "Дополнительное основание LR: lr_evidence.json содержит последние10 значений TensorBoard перед model_3599 "
              "(6 из10 около2.56289e-4, остальные от1e-5 до5.7665e-4), исходный default1e-3 и первые/последние "
              "значения прошлого low-std adaptive smoke. У него падение до1e-5 произошло уже на первом обновлении. "
              "В этой кампании выбран точный Adam LR исходника и устранён adaptive schedule, а не взят LR model_3699.", "",
              "Исходные checkpoint не содержат состояния физики/RNG/среды; после каждой границы100/500 обновлений "
              "симулятор перезапускается. Модель, критик, нормализаторы, Adam и индекс продолжаются; Stage A фиксирован. "
              "В обеих smoke границы возобновления одинаковы. Исходники подтверждены в checkpoint_inspection.json.", "",
              "## Полная детерминированная траектория", "",
              "| checkpoint / seed | held | completion | целей/эп | drop | near-limit | mean|μ| | p95|μ| | std | FD qdot p99 | эпизод,с |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    # Avoid literal pipes in column labels breaking Markdown tables.
    lines[-2] = "| checkpoint / seed | held | completion | целей/эп | drop | near-limit | mean abs μ | p95 abs μ | std | FD qdot p99 | эпизод,с |"
    for label, m in evaluations.items():
        lines.append(f'| {label} / {m.get("seed",1234)} | {m["held_success_rate"]:.3f} | {m["target_completion_rate"]:.3f} | '
                     f'{m["goals_completed_per_episode"]:.3f} | {m["drop_rate"]:.3f} | {m["near_limit_fraction"]:.3f} | '
                     f'{m["mean_abs_mu"]:.3f} | {m["p95_abs_mu"]:.3f} | {m["action_std"] if m["action_std"] is not None else "—"} | '
                     f'{m["qdot_fd_p99"]} | {m["episode_length_s"]:.2f} |')
    lines += ["", "Число завершённых целей, умноженное на20°, — сумма выданных целевых поворотов, а не интеграл фактической угловой траектории. "
              "held означает хотя бы одну завершённую цель, а не удержание без падения весь эпизод.", "",
              "## Решение по smoke и первичному прогону", "", str(state.get("smoke_decision", "Smoke не завершены.")), "",
              f"Выбрана ветка: {state.get('selected_branch')}. Причина завершения: {state.get('stop_reason','достигнут предел эксперимента')}.", "",
              "Критерии объявлены до длинного прогона: useful smoke held≥0.70, цели≥70% baseline, drop не хуже baseline+0.10. "
              "Два последовательных плохих eval останавливают ветку; после1500 дополнительных итераций отсутствие снижения насыщения хотя бы на1п.п. "
              "также останавливает её. Материальное сглаживание для продолжения сверх4000 — минимум5п.п. при сохранённой компетенции. "
              "Выбор не использует stochastic training reward.", "",
              "## Action-rate и разбор падений", "",
              f"Измеримое частое переключение выявлено: {state.get('action_rate_justified')}. Отдельный опыт выполнен: {state.get('action_rate_ran',False)}.",
              "```json", json.dumps(state.get("action_rate_preflight", {}), indent=2), "```", "",
              "Для каждого падения в eval JSON сохранено до60 предшествующих управляющих шагов: контакты каждого пальца, "
              "grasp_quality, palm force, ускорение объекта и насыщение каждого сустава. Эти данные не доказывают причинность контактов. "
              "Отдельный rate smoke никогда не меняет исходный magnitude-эксперимент.", ""]
    for label in filter(None, ("BASELINE_3599", state.get("best_primary_label"), state.get("final_primary_label"))):
        if label not in evaluations:
            continue
        raw = next(iter(json.loads(Path(evaluations[label]["json"]).read_text()).values()))
        drops = raw["smoothing"]["drop_windows"]
        lost = [0] * 5
        for event in drops:
            frames = event["last_60_steps"]
            for prev, curr in zip(frames, frames[1:]):
                changes = [i for i, (a,b) in enumerate(zip(prev["tips"],curr["tips"])) if a and not b]
                if changes:
                    for i in changes:
                        lost[i] += 1
                    break
        lines += [f"{label}: {len(drops)} падений; первый исчезнувший контакт в последнем1с окне, пальцы1..5: {lost}. "
                  "Одновременные потери учитываются для каждого пальца.", ""]
        lines += ["| До падения, с | эпизодов | mean tip count | mean quality | mean palm force,N | p99 accel,m/s² | mean saturation |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for offset in (60, 30, 6, 1):
            frames = [e["last_60_steps"][-offset] for e in drops if len(e["last_60_steps"]) >= offset]
            if frames:
                lines.append(f'| {(offset-1)/60:.3f} | {len(frames)} | {np.mean([sum(f["tips"]) for f in frames]):.3f} | '
                             f'{np.mean([f["quality"] for f in frames]):.5f} | {np.mean([f["palm_force"] for f in frames]):.3f} | '
                             f'{np.quantile([f["acceleration"] for f in frames],.99):.3f} | {np.mean([f["near_limit"] for f in frames]):.3f} |')
        per_joint = raw["smoothing"]["command"]["per_joint_near_limit"]
        lines += ["", "Максимальное насыщение по суставам: " + ", ".join(
                  f"{k}={v:.3f}" for k,v in sorted(per_joint.items(), key=lambda kv: -kv[1])[:5]), ""]
    lines += ["## Исторический гейт и лучший checkpoint", "", "Гейт неизменен: held≥0.70, drop≤0.15, целей/эп≥1.5.", ""]
    for label, m in evaluations.items():
        if label.startswith("FINAL_") or label == "RATE_200":
            passed = m["held_success_rate"] >= .70 and m["drop_rate"] <= .15 and m["goals_completed_per_episode"] >= 1.5
            lines.append(f"- {label}: {'PASS' if passed else 'FAIL'}; `{m['checkpoint']}`")
    best = evaluations.get(state.get("best_primary_label"), {})
    reason = state.get("stop_reason", "")
    interpretation = {
        "saturation_did_not_move_after_1500": "Гипотеза primary не сработала: насыщение не уменьшилось хотя бы на1п.п. после1500 итераций. Продолжать этот режим без изменения гипотезы не стали.",
        "action_smoothing_succeeded_grasp_survival_did_not": "Action smoothing succeeded, grasp survival did not. В этом опыте снижение насыщения не дало ожидаемого улучшения выживания; дополнительно повышать magnitude penalty не стали.",
        "goals_below_70_percent_for_two_evaluations": "Регуляризация не сохранила компетенцию: цели ниже70% baseline в двух последовательных оценках.",
        "drop_materially_worse_for_two_evaluations": "Падения устойчиво ухудшились более чем на10п.п.; primary остановлен.",
    }.get(reason, "Остаточное ограничение следует из таблицы: совместное сохранение достижения целей, уменьшение насыщения и снижение drop оцениваются отдельно; отсутствие прохождения гейта не обнуляет манипуляционную компетенцию.")
    transfer = evaluations.get("FINAL_TRANSFER", evaluations.get("TRANSFER_10_TO_20"))
    if transfer:
        tm = next(iter(json.loads(Path(transfer["json"]).read_text()).values()))
        lines += ["", f"10°→20° без обучения: first-target held={tm['first_goal_held_success_rate']:.4f}, "
                  f"held={tm['held_success_rate']:.4f}, целей/эп={tm['goals_completed_per_episode']:.4f}, "
                  f"drop={tm['drop_rate']:.4f}, final error={tm['final_error_deg']:.3f}°, min error={tm['min_error_deg']:.3f}°, "
                  f"near-limit={transfer['near_limit_fraction']:.4f}."]
    lines += ["", f"Лучший primary checkpoint: `{best.get('checkpoint','primary не запускался/не завершил оценку')}`.", "",
              "LR/std на каждой обучающей итерации и точные команды/PID сохранены в night_state.json → jobs; "
              "фактический LR каждого primary segment проверен на равенство выбранному. "
              "Полные вклады награды, Δμ, sign flips, per-joint saturation и временные сводки доступны в каждом eval JSON.", "",
              "## Оставшееся ограничение", "", interpretation, "", "## Один следующий эксперимент", ""]
    baseline = evaluations.get("BASELINE_3599", {})
    useful = any(m["held_success_rate"] >= .70 and m["goals_completed_per_episode"] >= 6.79
                 and m["drop_rate"] <= baseline.get("drop_rate", 0) + .10
                 for k,m in evaluations.items() if k.startswith(("PRIMARY_", "RATE_200")))
    if reason in ("saturation_did_not_move_after_1500", "drop_materially_worse_for_two_evaluations") and not useful:
        recommendation = ("Отдельный контролируемый smoke с регуляризацией сырого среднего актора μ до клипа, при прежней "
                          "физике и fixed LR. Размер нового терма предварительно подобрать по измеренной μ²; одновременно "
                          "другие изменения не вводить. Это новая гипотеза, а не продолжение неудавшегося усиления стоимости уже обрезанной команды.")
    elif useful:
        recommendation = ("Повторить лучший из компетентных измеренных режимов на трёх независимых training seeds с тем же бюджетом и "
                          "парной детерминированной оценкой. Один текущий training seed не устанавливает устойчивость небольших различий drop.")
    else:
        recommendation = "Следующий отдельный smoke: меньший magnitude penalty при том же fixed LR; одновременно другие параметры не менять."
    lines.append(recommendation)
    if state.get("error"):
        lines += ["", "## Ошибка / ограничение выполнения", "```", state["error"], "```"]
    (out / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    journal = out.parents[1] / "JOURNAL.md"
    if journal.exists():
        old = journal.read_text()
        entry = (f"## Ночное сглаживание — {state['started_at']}\n\n"
                 f"Отчёт: logs/night_smoothing/FINAL_REPORT.md. Ветка: {state.get('selected_branch')}; "
                 f"завершение: {state.get('stop_reason')}. Физика и delta1° неизменны. Коммитов нет.\n\n")
        pos = old.index("## ")
        if entry not in old:
            journal.write_text(old[:pos] + entry + old[pos:])
