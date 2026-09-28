# Исправление servo после аппаратного теста 2026-09-28

Ветка `feature/dg5f-servo-control`. Источник:
`debug_runs/dg5f_debug_2026-09-28_12-18-04.tar.gz` и распакованный одноимённый каталог.
Дополнительно прочитаны ROS-логи того же запуска в контейнере
`/home/hand/.ros/log/2026-09-28-12-17-26-082170-yoba-Legion-5-15IAH7H-1496/`
и файлы процессов `python3_1518…`, `python3_1520…`, `python3_1522…`, `python3_1526…`.

## 1. Root cause движения index назад

Ошибка интеграции двух различных сигналов в
`ros_bridge_node.py::_send_servo_tick()` → `_guard_current_target()` →
`current_guard.py::AdaptiveCurrentGuard.update()`:
**ограниченная servo-предложенная команда передавалась в object-contact как
полная желаемая поза оператора** (`desired_deg`).

Object-contact использует последовательность `desired_deg` как человеческое
намерение, вычисляет `human_step`, определяет открытие, сохраняет contact anchors
и offset после отпускания. Его формула в
`object_contact.py::PerFingerObjectContact.update()`, ветка FREE/resume:

```python
target = min(soft_target, desired - resume_offset)
```

Когда `desired` на самом деле `previous_q_cmd + 0.5`, оставшийся offset вычитается
из относительного servo-шага заново каждый тик. Если offset больше шага,
результат оказывается ниже предыдущего command. Servo ограничивал **модуль**
этого обратного движения и принимал его как следующую исходную команду.
Уменьшение следующего `desired` ошибочно считалось продолжающимся открытием
оператора; при reason `OPERATOR_OPENING` offset переставал убывать. Так образовалось
самоподдерживающееся движение к нижнему limit.

Данные bundle непосредственно подтверждают подмену intent:

| t, s | Retarget q6, ° | Contact anchor desired q6, ° | Contact anchor effective q6, ° |
|---|---:|---:|---:|
| 12.631345 | ≈90.047 | 88.250532 | 87.753112 |
| 54.582563 | ≈71.440 | 60.995538 | 60.495538 |
| 69.519466 | ≈63.191 | 49.634720 | 49.134720 |

Contact desired равен effective + servo step, а не retarget target.
Первый latch: 12.631345 s; release `OPERATOR_OPENING`: 12.764846 s.
В 17.550 s target≈+64.10°, command≈−57.95°, low-level≈−56.46°,
measured≈−55.50°. Ошибочное направление уже присутствует в Python command;
SDK лишь исполняет его. Асинхронный low-level snapshot несколько старше command.

Детерминированное воспроизведение старой интеграции с достижимым FREE/resume
состоянием (offset=5°), даже при **нулевом токе**:

```text
real target  previous command  raw servo  post-contact  submitted
+70          -50.0             -49.5      -54.5         -50.5
+70          -50.5             -50.0      -55.0         -51.0
+70          -51.0             -50.5      -55.5         -51.5
```

Это не ошибка degrees/radians, joint index, второго sign mapping или SDK.
`trajectory_to_degrees()` упорядочивает позиции по имени и конвертирует rad→deg
один раз; controller и backend работают в deg. В limiter нет сохранённой velocity
и он правильно предложил −49.5°. Первая неверная команда появляется **после
object-contact**, в вычитании resume offset. Target callback и timer имеют
обычную mutually-exclusive ROS callback group; входная поза теперь дополнительно
снимается копией для всего тика. MANO/retargeting/mapping/limits не изменены.

Bundle не содержал каждого servo tick и внутренних offsets: это не побитовое
воспроизведение полного аппаратного запуска. Проверены сохранённые anchors,
последовательность событий и детерминированный механизм на исходном коде.

## 2. Минимальный patch

- В guard добавлен отдельный optional `operator_target_deg`. Servo передаёт туда
  полную текущую цель, а `desired_deg` остаётся bounded proposal для токовой защиты.
  `PerFingerObjectContact` теперь получает настоящий operator intent.
- В servo-ветке guard выход ограничен интервалом между предыдущим command и
  operator target. Старый contact offset не может отправить сустав от цели.
  Блокировка имеет reason `BLOCKED_CONTACT_OFFSET_REVERSAL`.
- Исключение — явный `CONTACT_PRELOAD_YIELD` при подтверждённом CONTACT_HOLD:
  он ограничен прежними 15°/s, текущей measured position + preload **и фиксированным
  contact anchor + preload**. Обратное движение измеренной позиции не может
  постепенно утянуть эту нижнюю границу к −90°.
- Если нежелательный retreat заменён HOLD, он больше не считается relief для
  stall guard. Токовые пороги, emergency disarm и watchdog сохранены.
- При tracking auto-resume сбрасывается старое contact-состояние одновременно с
  переинициализацией servo из свежей measured pose. ARM/recovery уже делали reset.
- `legacy` передаёт `operator_target_deg=None` и использует исходные правила.
  Алгоритм `object_contact.py` не переписывался; Dex Hybrid, MANO и C++ не менялись.

## 3. Скорость и частоты

Новый default: **120°/s**, **60 Hz**, максимум **2°/tick**.
При коротком тике шаг меньше; после пропущенного тика нет catch-up скачка.
Acceleration/jerk limiter не добавлены.

Основная переменная: `DG5F_SERVO_MAX_SPEED_DEG_S`.
Старое имя `DG5F_SERVO_MAX_VELOCITY_DEG_S` поддерживается как fallback;
если заданы оба, новое имя имеет приоритет. ROS/launch parameter остался
`servo_max_velocity_deg_s`. Legacy `max_speed_deg_s=30` не менялся.

SDK keepalive остаётся номинально 200 Hz; новый q_cmd формируется на 60 Hz.

## 4. Диагностика каждого servo tick

`/dg5f/lerobot/servo_state` — JSON в `std_msgs/String`, градусы:

```text
target, previous_q_cmd, raw_servo_step, q_proposed, new_q_cmd
post_compliance_cmd, post_contact_cmd, post_guard_cmd
low_level_cmd, low_level_valid, low_level_sample_monotonic_s, measured
q_target, q_cmd, q_measured, tracking_error, dt, actual_rate_hz, max_step_deg
direction_violation, post_contact_direction_violation
command_moved_away_from_target, distance_before, distance_after
command_direction_reason
```

`post_compliance_cmd` — после soft current/compliance, `post_contact_cmd` — после
object-contact до контроля направления; `post_guard_cmd` — после hard guard и
проверки направления; `new_q_cmd` — принятая backend команда с последним rate cap.
`low_level_cmd` — асинхронный SDK snapshot, **не подтверждение именно этого тика**.
Все массивы имеют порядок `JOINT_NAMES`, q6 = `rj_dg_2_3`.

При движении от target — ERROR `COMMAND_MOVED_AWAY_FROM_TARGET`; при заблокированной
инверсии — ERROR `COMMAND_DIRECTION_VIOLATION_BLOCKED`. Сообщение и bridge event
содержат joint name, reason и все промежуточные значения. Epsilon: 1e-8.
Разрешённый contact yield тоже логируется с причиной.

`debug-record` теперь сохраняет **каждый полученный tick** в `servo_ticks.jsonl`,
независимо от 30 Hz timeline. Исправлен разбор contact-state/anchor массивов:
в исходном bundle contact-state колонки timeline были пустыми из-за recorder parser.

## 5. Tracking около 74.883 s

Установлено по имеющимся данным:

1. Последняя успешная команда/контактные данные были около **74.447 s**:
   `last_command_age_ms` и `proximity_signal_age_ms` после этого только растут.
2. Retarget-процесс продолжал работать. В его ROS-логе в
   **1790597959.679742420 UTC unix time** (t≈74.875 s) записан
   `Hand tracking timeout: holding the last safe joint target`.
   Это `RetargetNode._watchdog()` в `src/dg5f_teleop/dg5f_teleop/retarget_node.py`:
   `self._now_seconds() - self._last_frame_time > watchdog_timeout` (0.35 s).
   Именно `_last_frame_time` перестал обновляться. Он меняется в конце успешного
   `_on_landmarks()`; там же публикуются tracking=true и очередная joint pose.
   Источник: `/quest/hand_pose` → `unity_mano_adapter` → `/hands/right/landmarks`
   → `dg5f_retarget` → `/dg5f/tracking_ok` (`std_msgs/Bool`).
   Возраст успешной команды в bridge фиксирует тот же разрыв. Heartbeat Bool
   не прекратился: watchdog продолжал посылать **false**, поэтому его малый
   `last_tracking_age_ms` не означает живое hand tracking.
3. В 74.883 s recorder увидел `tracking_ok=false`; false продолжал публиковаться
   watchdog примерно каждые 0.1 s. Это не зависший bridge или Tesollo disconnect.
4. В 89.813 s истекло существующее tracking grace 15 s и произошёл DISARM.
   Команда ARM в 85.459 s не создаёт новые Quest кадры и не сбрасывает эту причину.
5. Нет SDK disconnect/reconnect; SDK telemetry остаётся свежей.
   До пользовательского Ctrl-C (1790597981.503) ни retarget, ни adapter не завершились.

**Атрибуция внешней причины: `not fully attributable without Quest/Unity logs`.**

**Что не установлено:** bundle не записывал `/quest/hand_pose`,
`/hands/right/landmarks`, их source timestamps, Unity client log или TCP capture.
В adapter нет ошибок invalid length; в retarget нет сообщения rejected frame;
в endpoint нет зарегистрированного disconnect в момент потери. Это согласуется
с прекращением hand frames до adapter/retarget, но не доказывает, продолжал ли
Unity TCP передавать другие сообщения, завис ли отправитель или Quest потерял руку.
Факт «рука вернулась в поле зрения» не подтверждается входными сообщениями в
имеющейся записи. Поэтому конкретная причина на стороне Quest/Unity остаётся
неизвестной; watchdog не отключён и предположение не выдано за факт.

Для следующего теста bridge пассивно наблюдает raw Quest и normalized landmarks
и добавляет в diagnostics/timeline:

```text
last_quest_age_ms, last_landmarks_age_ms, last_retarget_age_ms
last_quest_received_monotonic_s, last_landmarks_received_monotonic_s
last_successful_retarget_monotonic_s
quest_header_stamp_s, landmarks_header_stamp_s, landmarks_stamp_age_ms
quest_message_count, landmarks_message_count
tracking_ok, tracking_loss_reason, tracking_loss_detail
```

Причины: `NO_LANDMARKS`, `STALE_LANDMARKS`, `INVALID_HAND`, `RETARGET_STALE`,
`TRACKING_STATUS_STALE`, `UPSTREAM_TRACKING_FALSE`.
Detail различает `QUEST_INPUT_STALE`, `LANDMARK_ADAPTER_STALE`,
`SOURCE_TIMESTAMP_STALLED` и invalid raw/normalized hand.
Времена получения — monotonic clock bridge; successful retarget отмечается
при получении валидного `/dg5f/joint_command` (это наблюдение результата,
не внутренний timestamp оптимизатора).
Наблюдатель не меняет conversion, output retargeting или решения watchdog.
`unity_connection_status=UNKNOWN`: отсутствие hand frames не выдаётся за
доказанный TCP disconnect.

## 6. Новый аппаратный тест

Пакеты собраны через `bash scripts/stack.sh compile`. Завершить предыдущий
hardware launch перед новым. Терминал 1 — выбрать **одну** скорость:

```bash
DG5F_CONTROL_MODE=servo DG5F_SERVO_MAX_SPEED_DEG_S=60 \
  bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72

DG5F_CONTROL_MODE=servo DG5F_SERVO_MAX_SPEED_DEG_S=120 \
  bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72

DG5F_CONTROL_MODE=servo DG5F_SERVO_MAX_SPEED_DEG_S=180 \
  bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
```

Терминал 2 — начать запись, ещё один терминал — ARM после свежего tracking:

```bash
bash scripts/stack.sh debug-record-teleop
# В другом терминале:
bash scripts/stack.sh arm
```

Пассивная проверка из ROS-окружения контейнера:

```bash
ros2 topic hz /dg5f/lerobot/servo_state
ros2 topic echo /dg5f/lerobot/servo_state --once --field data
ros2 topic echo /dg5f/lerobot/diagnostics --once
```

`debug-record-teleop` — одна команда для timestamped bundle. Она автоматически
записывает:

- `quest_hand.jsonl`: raw `/quest/hand_pose`, original header stamp, все points,
  время получения monotonic + unix wall time;
- `landmarks.jsonl`: `/hands/right/landmarks` с теми же временными полями;
- `rosout.jsonl`: ROS-сообщения Unity endpoint, adapter, retarget, bridge;
- `quest_logcat.log`: доступные Unity/AndroidRuntime/OVRPlugin/OpenXR/VrApi логи
  Quest через host `adb logcat -v epoch`, без очистки logcat;
- `servo_ticks.jsonl`, `timeline.csv`, events, manifest и network status.

Остановить **терминал recorder** через Ctrl+C: процессы записи завершаются,
после чего создаётся `debug_runs/dg5f_debug_<timestamp>_<pid>.tar.gz`.
Recorder не запускает и не останавливает управление рукой. Все файлы этого
запуска попадают в один архив. Запись лучше начать до ARM.

Для Quest logcat нужен подключённый и авторизованный ADB; при нескольких устройствах
можно указать `ANDROID_SERIAL=<Quest serial> bash scripts/stack.sh debug-record-teleop`.
Если ADB/Quest недоступен, остальные логи продолжаются; причина и exit status
записаны в `network_status.json`/manifest и `quest_logcat.log`. Отсутствующая
запись явно отмечается, а не считается успешной. `--no-adb` — для offline smoke.
Без подключения/прав ADB захват внутренних Quest логов гарантировать невозможно.

При желании дополнительно записать стандартный rosbag:

```bash
ros2 bag record -o /workspace/bags/servo_input_check \
  /quest/hand_pose /hands/right/landmarks /dg5f/joint_command \
  /dg5f/tracking_ok /dg5f/lerobot/servo_state /dg5f/lerobot/diagnostics \
  /dg5f/lerobot/events /rosout
```

Отключение: `bash scripts/stack.sh disarm`.
Legacy: `DG5F_CONTROL_MODE=legacy bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72`.

## 7. Проверки после patch

- Все 5 ROS packages собраны (`bash scripts/stack.sh compile`).
- LeRobot: **174 passed, 1 существующее падение**. Включает **54** servo/tracking/
  recording проверки (38 добавлены после первого аппаратного теста).
- Retargeting: **17 passed, 2 существующих падения**.
- Unity adapter: **6 passed**; ROS TCP endpoint: **3 passed**.
- IsaacLab CPU-only contracts: **20 passed**; файлы IsaacLab не менялись этой работой.
- Fake SDK: PASS (никакого подключения к физической руке).
- `git diff --check`, `bash -n`, Python compilation: passed.

Три прежних падения уже воспроизводились на исходном `HEAD=9490086`:
`test_contact_and_rising_current_decrease_convergence_before_soft_threshold`
(те же 2.5970459272 vs 4.8819591979/2) и два случая `test_velocity_profile`
(в test double отсутствуют `_publish_contact` / `_warn_throttled`).
Порогов guard и retargeting ради этих тестов не менял.

Regression coverage: знак/уменьшение расстояния на каждом joint, −50→+70,
+40→−30, crossing zero, быстро меняющийся target, сохранённый contact offset,
активный CONTACT_HOLD с дрейфующим measured, ограниченный yield и его reason,
необход stall safety при блокировке неправильного retreat, tracking lost/recovered,
explicit recovery/manual rearm, 60/120/180°/s, запись промежуточных команд.

End-to-end проверка команды `debug-record-teleop --no-adb` в ROS domain 99:
71 raw Quest frame + 71 normalized frame + 71 ROS log, servo ticks и manifest
успешно сохранены и упакованы после SIGINT в один timestamped архив.
Это синтетический offline запуск; реальный Quest logcat и повторное движение
физической рукой в ходе этой работы не выполнялись.

Сохранён пример результата offline recorder:
`debug_runs/dg5f_debug_2026-09-28_15-33-51_86802.tar.gz`.
