# Index FREE-motion slowdown: hardware bundle 18:05:11

Источник: `debug_runs/dg5f_debug_2026-09-28_18-05-11_146230.tar.gz`.
Работа поверх ветки `feature/dg5f-servo-control`, HEAD `9490086`, с сохранением
предыдущих незакоммиченных servo/runtime/ARM исправлений.

## Точная причина 0.128854

В `servo_ticks.jsonl` на recorder t=84.426414910 s, joint index 6 = `rj_dg_2_3`:

| Величина | Значение |
|---|---:|
| Operator target | +66.079181° |
| previous_q_cmd | −1.988766° |
| measured | −3.700000° |
| Raw servo step | +2.000000° |
| post_compliance_cmd | +0.011234° |
| post_contact_cmd / post_guard_cmd / q_cmd | −1.731057° |
| Current | 8 mA |
| joint_current_scale | 0.9999999999999996 |
| joint_slope_scale / joint_contact_scale | 1 / 1 |
| joint_lead_budget_deg | 8° |
| object_contact_state / reason | FREE / OPERATOR_OPENING |
| contact_limited_pairs | [] |
| limited_fingers | [index] |
| physical_limit_reason | NONE |

Ограничение появляется в FREE-ветке `PerFingerObjectContact.update`, когда
сохраняется `finger.resume_offset` после CONTACT_RELEASED:

```python
target = min(soft_target, operator_target - resume_offset)
target = min(target, effective + resume_rate_deg_s * dt)
# resume_rate_deg_s = 15, dt = min(elapsed, 0.05)
```

При dt=0.017180593000375666 s второй cap даёт шаг
`15 × dt = 0.257708895005635°`. Диагностический `joint_tracking_scale` учитывает
фактический command gain после contact:

```text
min(combined_scale,
    abs(post_guard - effective) / min(abs(proposal - effective), nominal_step))
= min(≈1, 0.257708895 / 2)
= 0.1288544475028175
```

Это **legacy contact resume ramp на 15°/с поверх servo 120°/с**. Именно stage
`post_compliance → post_contact` уменьшил шаг. Ток, slope, геометрический pair
contact и физический lead cap в этом sample не ограничивали движение.
`post_contact_gain=1` в прежней диагностике обозначал коэффициент mapping при
открытии/FREE, а не фактический gain FREE resume ramp; по нему этот cap не виден.

## Почему ограничение сохранялось после release

CONTACT_RELEASED переводил state в FREE, но создавал `resume_offset`.
При OPERATOR_OPENING offset уменьшался только если ни один joint цепочки не
открывался и хотя бы один закрывался; при неподвижном operator target он мог
не уменьшаться. Даже нулевой offset очищался лишь когда **вся цепочка** достигала
desired. До этого второй cap 15°/с продолжал действовать. Меняющиеся targets и
сохранение offset через tracking gaps продлевали это состояние.

В данном sample предыдущий index RELEASED был на t=77.993240 s, то есть примерно
6.43 s назад. Сохранённые anchors сами не используются FREE ramp для lead;
ограничение определяется оставшимся resume state. Точный resume_offset старый
bundle не записывал. Для численной реконструкции взят допустимый неблокирующий
offset 50°: исходная версия даёт **точно** записанные target/scale; после патча
тот же вход даёт target=+0.011234115711098447°, scale=1, step=2°.

## Проверка error, lead и отличий пальцев

Полный путь: operator target → ServoController.propose (±120×min(dt,1/60)) →
current/slope/pair compliance → object-contact mapping/resume → hard-current
freeze + direction projection → ServoController trajectory limit → physical
rate limit + существующий measured±lead envelope → backend latest mailbox.

- `current_error = effective_q_cmd - measured`, `delta = proposal - effective`.
  Lead не вычисляется от большого operator-target error. В sample current_error
  до шага равен +1.711234°, свободный бюджет 8° позволяет полный proposal.
- `combined_scale` в servo — minimum smoothed joint current и adaptive scale.
  Adaptive scale — существующее attack/release сглаживание minimum joint slope
  и pair contact scales. `lead_budget = 1 + 7 * combined_scale` при default config.
  Additional step cap при нагрузке и окончательный physical envelope сохранены.
- Отдельных `error_scale` и `progress_scale` нет. Windowed progress ratio и
  command/measured error входят в contact evidence вместе с current thresholds.
  Огромный operator target не становится сам по себе stall/contact: fallback
  requested motion ограничен `error_deg`, а latch требует ток и недостаточный
  progress. Stall diagnostic также требует ток, command error и низкую скорость.
- У index/middle/ring одинаковые thresholds, положительное направление flexion,
  цепочки [5,6,7], [9,10,11], [13,14,15]. Little использует [18,19]; broken joint
  16 по-прежнему disabled. Mapping и signs не менялись.
- Pair geometry может связывать пальцы, но в проблемном sample pair list пуст,
  contact scale=1, физический lead не активен. Причина здесь не coupling.

Из 6696 submitted ticks (всего 7553 servo records):

| Палец | CONTACT_LATCHED | scale<0.2, abs(raw_step)>1° | Из них FREE и current<20 mA |
|---|---:|---:|---:|
| index | 4 | 1335 | 864 |
| middle | 1 | 0 | 0 |
| ring | 2 | 72 | 35 |
| little | 0 | 0 | 0 |

Все 864 index и 35 ring FREE/low-current slow ticks отмечены object-contact
limited. Index первым вошёл в контакт (t≈26.16 s) и чаще повторял его, поэтому
чаще получал и дольше сохранял FREE resume state. Это наблюдаемая история
нагрузки/targets, а не специальный index threshold. Причину физического различия
нагрузки пальцев этот software trace сам по себе не устанавливает.

## Минимальное изменение поведения

`AdaptiveCurrentGuard` передаёт `apply_resume_ramp=False`, когда задан отдельный
`operator_target_deg` (существующий servo путь). Default True сохраняет legacy.
При servo FREE старый resume_offset очищается; при release новый не создаётся.
Release tick сохраняет прежний bounded output; следующий tick идёт к live target
через существующий ServoController. q_cmd не переинициализируется.

CONTACT_PENDING/HOLD detection, preload/yield/anchors, direction projection,
hard-current logic, lead budget/envelope не изменены. Никаких новых filters,
acceleration/jerk или порогов. ARM/TRACKING_HOLD/resume, SDK, retarget, MANO,
servo controller, 60 Hz / 120°/с и recorder не изменены.

## Разложение для следующего debug-record-teleop

Диагностические поля добавлены в existing guard result; bridge уже переносит
их в `servo_ticks.jsonl`, recorder сохраняет их без изменения своей реализации.
Проверять `[6]` для rj_dg_2_3 (а также [5]/[7] для остальных flexion index):

| Стадия | Поля |
|---|---|
| Ток и его release smoothing | joint_raw_current_scale, joint_current_scale |
| dI/dt | joint_current_slope_ma_s, joint_slope_scale |
| Pair geometry/load | pair_contact_weights, pair_load_risk, pair_tracking_scale, joint_contact_scale, contact_limited_pairs |
| Adaptive attack/release | joint_raw_adaptive_scale, joint_adaptive_scale, joint_combined_scale |
| Error/lead | joint_command_error_deg, joint_requested_step_deg, joint_load_evidence, joint_lead_budget_deg, joint_lead_limited, joint_lead_step_scale |
| Contact evidence | object_contact_window_ready, joint_requested_motion_deg, joint_measured_motion_deg, joint_progress_ratio, joint_object_contact_evidence |
| Остаточный resume | object_contact_resume_ramp_enabled, object_contact_resume_offset_deg, object_contact_resume_limited |
| Фактические gains по стадиям | joint_compliance_gain, joint_object_contact_gain, joint_guard_gain, joint_tracking_scale |
| Причины | joint_limiting_reason, joint_hard_freeze, command_direction_reason, physical_limit_reason |

Stage gains — signed отношение output-step к input-step соответствующей стадии,
считая от effective q_cmd; при нулевом input-step условно 1. Это наблюдения,
не новые limiters. `joint_tracking_scale` сохраняет прежнюю формулу diagnostic
minimum, поэтому не является произведением всех перечисленных scalars.
`joint_lead_step_scale` относится к раннему compliance step cap; окончательный
physical cap виден отдельно в `physical_limit_reason` и `low_level_submitted`.
Reasons могут перечислять несколько действующих механизмов через `|`.

Ожидаемый FREE sample с низким током, правильным progress и lag около 3°:
resume_ramp_enabled=false, resume_offset=0, resume_limited=false,
joint_tracking_scale≈1, joint_limiting_reason=NONE; шаг около 2° при полном tick,
если target ещё далеко. При реальной нагрузке остаются конкретные причины
JOINT_CURRENT / JOINT_CURRENT_SLOPE / PAIR_CONTACT / ADAPTIVE_RELEASE /
LEAD_BUDGET / OBJECT_CONTACT_HOLD / CURRENT_HARD_FREEZE и physical envelope reason.
HOLD records не являются новыми guard evaluations: учитывать compliance_age_ms.

## Проверки

- Focused suite: **100 passed**, включая 30 новых тестов free-motion, численную
  реконструкцию 0.128854, четыре пальца × три низких тока × наличие/отсутствие
  старого offset, реальные contact→release→FREE и сохранение diagnostics recorder.
- Полный LeRobot suite: **227 passed, 1 known baseline failure**:
  `test_contact_and_rising_current_decrease_convergence_before_soft_threshold`,
  прежние 2.5970459272235527 < 4.88195919791379 / 2 не выполняются. Unrelated
  assertion/алгоритм не менялся. Прежние current_guard RuntimeWarning сохранены.
- Legacy comparison с сохранённой до патча реализацией: **600 ticks**, точное
  равенство targets/limited masks/trip/stall через contact/release/current cycles.
- Hybrid replay последнего bundle: **533×20 outputs побитно равны HEAD**,
  max difference **0 rad** (первые 240 frames и интервал 82–86 s).
- SHA-256 retarget/config/MANO/ServoController/constants совпадают с до-патча;
  `git diff --check` и Python compilation PASS. Timer regression входит в suite.
- Физический ARM/recover/motion для проверки не запускался; аппаратная проверка
  скорости index после патча ещё предстоит.

Артефакты: `debug_runs/index_slowdown_validation/` (не отслеживаются Git):
`trace_summary.json`, `logged_sample.json`, `guard_comparison.json`,
`compare_guard.py`, baseline copies, `replay_retarget.py`, `retarget_*.npy`,
`retarget_equivalence.json`, `unchanged.sha256`, `lerobot.xml`.
