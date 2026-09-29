# Acceleration-limited servo trajectory

## Stable checkpoint

Перед изменением trajectory создан commit **41b9b7e**
`fix(dg5f): stabilize hybrid retargeting servo teleop` и локальный annotated tag
**dg5f-servo-stable-pre-accel**. В нём все DG5F/Quest/servo/contact/recorder
исправления, включая ARM без tracking и index FREE fix. IsaacLab изменения
не включены. Push не выполнялся.

Hardware evidence checkpoint: `dg5f_debug_2026-09-28_18-30-01_157769.tar.gz`,
74.816 s, 0 CURRENT_GUARD_TRIP/STALL_GUARD_TRIP/SDK_ERROR/DISARM/disconnect,
max current420mA. Есть7 диагностических CURRENT_LOAD_THRESHOLD, без DISARM.
Перед commit проверены LeRobot+retarget+Unity:250 passed и3 known baseline
failures; fake SDK PASS. Это аппаратно проверенная версия без acceleration.

## Что добавлено

Только servo trajectory: persistent `command_pose_deg` (q_cmd) и
`velocity_deg_s` (v_cmd), default **720 deg/s²**, прежние **60 Hz / 120 deg/s**.
Measured feedback не используется как начало обычного tick. ARM/recovery
по-прежнему делают одноразовый measured reset, который теперь также даёт v=0.

```text
error = clipped_target - q_cmd
dv_max = a_max * dt
v_stop = sqrt(dv_max² + 2*a_max*abs(error)) - dv_max
v_desired = sign(error) * min(120, v_stop)
v_next = v_cmd + clip(v_desired-v_cmd, -dv_max, +dv_max)
q_proposed = q_cmd + v_next*dt
```

Формула stopping speed — дискретный консервативный вариант sqrt(2*a*distance):
резервирует travel текущего tick плюс v²/(2a). Она предотвращает обычный
overshoot при торможении. Пересечение цели дополнительно clamp-ится до target;
terminal velocity обнуляется. Если target скачком переместился внутрь уже
имеющейся stopping distance, одновременно выдержать acceleration bound и
запрет overshoot физически невозможно: приоритет у target clamp, событие видно
по `target_clamped`. Обычные fixed-target tests проходят без нарушения a_max.

При reverse velocity сначала доходит до нуля, затем меняет знак. Несколько
тиков торможения могут продолжать движение в прежнюю сторону. Это явно
помечено `trajectory_braking`. Существующая direction projection в guard
допускает только величину запланированного proposal в этой ситуации, иначе
она превратила бы плавный reverse в мгновенный HOLD. Real operator target
по-прежнему отдельно передаётся в object-contact. Незапланированный отход
дальше proposal остаётся заблокирован и диагностируется.

Используется реальный monotonic dt при обычном jitter (до50ms). При outage
больше50ms сохраняется защита от большого catch-up: интеграция за один
номинальный tick, оба значения видны в `dt` и `limiter_dt`. Первый tick после
ARM/HOLD также использует номинальные1/60s. При jitter шаг может отличаться
от2°; ограничение скорости остаётся120deg/s относительно limiter_dt.

## Downstream ограничения и tracking

Порядок: acceleration-limited proposal → прежние current/contact/lead rules →
physical output → SDK. Acceleration **не ограничивает реакцию safety**.
Принятая guard-поза остаётся persistent q_cmd. При guard correction velocity
согласуется с принятым trajectory step; при physical envelope correction
обнуляется, чтобы после снятия ограничения не сохранилась скрытая скорость.
Состояния q_cmd/v_cmd фиксируются только после успешного backend submission.

В TRACKING_HOLD pose frozen, v=0, накопленного движения нет. Automatic recovery
стартует от удерживаемого q_cmd и v=0. Runtime policy не менялась.
SDK fault/manual disarm также проходят через прежний hold/reset lifecycle.

Contact thresholds, current thresholds, lead-budget formulas, object-contact
и index resume fix, mapping/limits, MANO, Dex Hybrid, SDK и debug recorder
не изменены. В current_guard единственная поведенческая правка — допуск
bounded intentional reversal braking в direction projection, без изменения
расчётов нагрузки. Legacy trajectory path не менялся. Jerk/S-curve/filters нет.

## Diagnostics

Existing servo_state и `servo_ticks.jsonl` теперь содержат:

- `desired_velocity_deg_s`: braking-aware desired velocity;
- `trajectory_velocity_deg_s`: velocity proposal до downstream guard;
- `commanded_velocity_deg_s`: принятый persistent v_cmd;
- `acceleration_deg_s2`: изменение принятого v_cmd / limiter_dt;
- `acceleration_limited`: desired velocity ограничена dv_max;
- `max_acceleration_deg_s2`: фактическое значение настройки;
- `trajectory_braking`, `target_clamped`, `safety_velocity_override`: позволяют
  отличить intentional braking, terminal clamp и немедленный safety correction.

При safety override/terminal clamp acceleration может превышать trajectory
setting — это намеренный приоритет downstream safety и запрета overshoot.
HOLD records показывают нулевые velocity/acceleration; сам вход в HOLD —
немедленный stop через существующий lifecycle. Физическая скорость setpoint
считается отдельно из `physical_step_deg / limiter_dt`; SDK low_level_cmd
по-прежнему асинхронный snapshot, не ACK. Recorder код не менялся: произвольные
servo fields уже сохраняются полностью.

## Проверки

Полный LeRobot+retarget+Unity suite после patch: **273 passed,3 known baseline
failures**, ровно те же failures, что перед checkpoint:

1. compliance convergence assertion:2.597045927 <4.881959198/2 не выполняется;
2. две параметризации retarget velocity profile: неполный SimpleNamespace без
   `_publish_contact`/`_warn_throttled`.

Новых failures нет. Отдельно **22/22 acceleration unit tests PASS**, плюс
новый ROS bridge integration test reverse→HOLD→recovery PASS. Проверены большой
step, обе стороны, малая цель, braking без oscillation/overshoot, actual dt
jitter включая торможение, scheduler gap, hard/physical safety override,
failed-send state, reset, HOLD, recorder fields, invalid configuration.
Прежние reversal/contact/index tests сохранены и проверяют теперь planned
braking вместо требования мгновенно изменить направление/скорость.
Полный test_timer_streams сохраняет независимые60Hz и ограничение скорости
с учётом измеренного dt.

29 сентября перед финальным commit повторены acceleration+bridge tests:
**54 passed**, только прежние current_guard RuntimeWarning.

Hybrid replay stable checkpoint vs acceleration: **524frames×20 joints
побитно равны**, max difference **0rad**, вход из последнего18:30 bundle.
Нулевой diff retarget/MANO/mapping/models/object_contact/recorder/vendor.
Launch `--show-args`: default720 подтверждён; bash syntax/Python compilation/
`git diff --check` PASS. Hardware motion не запускался.

Offline 0→70°, без downstream нагрузки, при60Hz:

| Acceleration | До120deg/s | До ошибки<0.01° | Max velocity |
|---|---:|---:|---:|
| 480deg/s² | 0.250s | 0.850s | 120deg/s |
| 720deg/s² | 0.167s | 0.767s | 120deg/s |
| 960deg/s² | 0.133s | 0.733s | 120deg/s |

Это software trajectory, не измеренная динамика физического мотора. Скорость
cruise сохранена, startup/stop стали плавными; короткие движения неизбежно
занимают больше времени, чем при мгновенном выходе на120deg/s.
Артефакты в `debug_runs/acceleration_validation/`: checkpoint/after JUnit XML,
replay script/outputs, metrics.json, ramps_480/720/960.csv, launch_args.txt.

## Следующий hardware test

Из корня проекта, после завершения предыдущего hardware процесса:

```bash
# Терминал1: новая trajectory, прежние60Hz/120deg/s
DG5F_CONTROL_MODE=servo DG5F_SERVO_MAX_ACCEL_DEG_S2=720 \
  bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72

# Терминал2
bash scripts/stack.sh debug-record-teleop

# Терминал3: отсутствие tracking не мешает ARM в TRACKING_HOLD
bash scripts/stack.sh arm
```

Для сравнения заменить720 в первой команде на480 или960 и перезапустить
hardware процесс. Переменная читается при запуске; изменение env в другом
терминале не меняет уже работающий controller. ROS parameter:
`servo_max_acceleration_deg_s2`; Dg5fConfig — одноимённое поле.

Сначала проверить разгон0→120, reverse через0, отсутствие overshoot у неподвижной
цели, затем contact/release и tracking loss/recovery. При FREE после первых
десяти тиков на720 и далёкой цели ожидается120deg/s, joint_tracking_scale≈1,
object_contact_resume_limited=false. При contact/HOLD реакция должна оставаться
немедленной. Не менять скорость120deg/s при сравнении acceleration.

Commit acceleration отдельно от stable checkpoint; tag остаётся на41b9b7e.
