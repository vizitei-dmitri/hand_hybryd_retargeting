# Servo runtime policy после hardware test 15:39:45

Дополнение после regression 17:23:37: [servo ARM без tracking](DG5F_SERVO_ARM_WITHOUT_TRACKING.md).
Servo ARM теперь инициализирует measured HOLD без Quest/target; свежий input
нужен только для ACTIVE. Legacy сохраняет прежние ARM prerequisites.

Изменения сделаны поверх текущей рабочей ветки `feature/dg5f-servo-control`, без отката предыдущего hardware-fix. Эта версия заменяет описанную в прежних отчётах servo policy отключения/возобновления. `legacy` сохраняет прежнее поведение, включая его защитные отключения.

## Что установлено по последнему bundle

Источник: `debug_runs/dg5f_debug_2026-09-28_15-39-45_90896.tar.gz`.

- На `t=39.911375 s`: `CURRENT_GUARD_TRIP`, Imax=1207 mA, Itotal=1461 mA, продолжительность превышения 50.51 ms.
- Через 0.119 ms записан `DISARMED`, reason=`OVERCURRENT_GUARD`.
- Ближайший snapshot: transport/thread/telemetry/temperature/system/motion ready=true; `motion_ready_reason=READY`, `last_motion_result=0`. Tracking=true; Quest age≈8.1 ms, landmarks≈7.0 ms, retarget≈25.9 ms. Причина этого отключения — software policy, а не доказанная ошибка SDK/device или потеря tracking.
- Последний servo tick: q_target[6]=71.496°, q_cmd[6]=70.984°, q_measured[6]=58.100°, gap=12.884°.
- В CSV на t≈39.864 s budget[6]=1.235°, current[6]=550 mA, tracking_scale[6]=0. CSV target/command/measured хранятся в радианах; budget — в градусах.

Причина обхода budget в `AdaptiveCurrentGuard.update`: он ограничивал **прибавку** к effective command только при `limited & load_evidence>0.01 & delta*current_error>=0`. При слабом evidence gap мог накапливаться; при уменьшении budget или отставании measured существующий gap не устранялся. `max(0, budget-abs(error))` давал нулевую прибавку, а hard-current freeze оставлял старую команду. Object-contact и последующий rate limiter также не обеспечивали абсолютную границу physical-command/measured.

В предыдущем bundle 12:18 отсутствие обновления успешного retarget и его watchdog установлены; происхождение прекращения кадров остаётся `not fully attributable without Quest/Unity logs`. Это не относится к отключению в последнем bundle: здесь tracking был свежим.

## Архитектура и ключевой diff

До: target → persistent servo proposal → current/contact guard → rate limit → один q_cmd/SDK setpoint; trip/stall → global DISARM.

После: target → persistent servo proposal → локальный current/contact guard → ограниченный по скорости q_cmd → **тот же lead budget как граница физической команды** → существующий backend/SDK.

```diff
- if current_trip or stalled: disarm()
+ if current_trip or stalled: emit("CURRENT_LOAD_THRESHOLD")
+ # Только локальное ограничение; total current остаётся диагностикой.

- if tracking_timeout > grace: disarm()
+ if tracking_stale: hold_last_physical_command_indefinitely()
+ if fresh_tracking_and_new_target: resume_persistent_servo()

  q_cmd = rate_limit(guarded_proposal, previous_q_cmd)
- backend.send_positions(q_cmd)
+ physical = rate_limit(q_cmd, previous_physical_command)
+ physical = clip(physical, measured - existing_lead_budget,
+                           measured + existing_lead_budget)
+ backend.send_positions(physical)
+ commit(q_cmd, physical)  # только после успешного submission
```

Нового параллельного lead limiter нет: используется существующий `joint_lead_budget_deg` (по умолчанию 1–8°), но теперь его envelope применяется после contact/hard-freeze/trajectory limiting. Предел действует в servo при включённых current guard и compliance. Disabled joint и исходные joint limits имеют приоритет; measured вне допустимого диапазона может сделать точное выполнение обоих ограничений невозможным.

`q_cmd` не присваивается measured каждый tick и не reseed-ится при tracking recovery. `previous_q_cmd`, proposal, guarded q_cmd и физический submission различимы. Последняя физическая команда сохраняется отдельно. Обычное возобновление ограничивает и trajectory, и physical step. **При сужении lead envelope локальный physical retreat имеет приоритет над velocity limit**: иначе прежний большой gap невозможно убрать сразу. Такое движение отдельно помечается `LEAD_BUDGET_YIELD_OR_HOLD`, а его шаг пишется в `physical_step_deg`. Это не изменение направления persistent trajectory через старый resume_offset bug. `PHYSICAL_RATE_LIMIT` отличает ограничение physical скорости от lead relief.

Настоящий operator target по-прежнему отдельно передаётся в object-contact; direction/distance invariants и fixed contact-anchor floor сохранены. Общие total-current/total-slope коэффициенты больше не замораживают всю руку в servo; joint current/slope и геометрически затронутые contact joints ограничиваются локально. Threshold/stall events не выключают руку.

## Все прежние пути DISARM

| Место | Прежнее условие | Теперь в servo |
|---|---|---|
| `_guard_current_target` | current trip → OVERCURRENT_GUARD | локальный guard/envelope, CURRENT_LOAD_THRESHOLD |
| `_guard_current_target` | stall trip → STALL_GUARD | локальная адаптация + диагностический CURRENT_LOAD_THRESHOLD со stall_threshold=true |
| `_expire_tracking_grace`, вызываемый send/tracking/resume | TRACKING_GRACE_EXPIRED после 15 s | expiry не применяется; TRACKING_HOLD бессрочный |
| `_send_latest` | stale command/tracking → COMMAND_TIMEOUT/TRACKING_TIMEOUT | TRACKING_HOLD, backend keepalive последней позы; targets при невалидном tracking не принимаются |
| `_send_servo_tick` | ошибка send/guard → SDK_COMMAND_REJECTED/INTERNAL_ERROR | latch backend inhibition, SDK_ERROR с причиной и exception, без изменения ARM authorization |
| `_publish_state` | health watchdog | тот же backend latch + точные status fields |
| `_on_enable(False)` | USER_REQUEST | явный manual DISARM сохранён |
| `_on_recover` | RECOVERY_REQUESTED | явное пользовательское recovery сохранено; после успеха нужен fresh target и manual ARM |

`tracking_error >10°` в течение 300 ms и раньше только логировался; automatic DISARM от него не добавлен. Contact сам по себе не disarm-ит. Ошибка первоначального ARM по-прежнему оставляет output неразрешённым. В `legacy` все строки таблицы сохраняют прежнюю реализацию.

Servo runtime: `ACTIVE` / `TRACKING_HOLD`. До initial ARM — `DISARMED`. Для реальной невозможности работать с backend есть отдельное диагностическое `SDK_ERROR`: это не transient tracking/load state. `armed=true` означает сохранённую первоначальную авторизацию, но не обещает работающий SDK; смотреть также `runtime_state`, `motion_ready`, `sdk_block_reason`.

Во время HOLD не отправляются новые команды; C++ keepalive повторяет последнюю принятую. Timer продолжает публиковать диагностику. Fresh true tracking и новый валидный target автоматически переводят в ACTIVE без measured reseed, reset или нового ARM. Если `require_tracking=false`, достаточно нового валидного target после command timeout.

## Какие ошибки действительно доступны в SDK

Проверены локальные `vendor/tesollo_control/include/DGSDK.h`, `DGDataTypes.h`, `dg_control/control.cpp` и Python backend; SDK-код не менялся.

- `DG_RESULT`: 0=NONE, 1=SYSTEM_SETTING_NOT_PERFORMED, 100+ ошибки аргументов/режима/модели, 200+ состояния motion, 500/501 socket errors, 2000+ port errors, 2009=DIAGNOSING_SYSTEM. Сам numeric result не всегда означает физический motor fault.
- Реальный результат `MoveServoJoint` сохраняется как `last_motion_result`; connected/disconnected callbacks показывают transport, diagnosis callback сохраняет process/step/jointId/period/joint/temperature.
- Readiness C++ wrapper различает DISCONNECTED, CONTROL_THREAD_STOPPED, TELEMETRY_STALE, TEMPERATURE_UNSAFE, SYSTEM_NOT_STARTED, MOTION_RESULT_ERROR, RECOVERY_REQUIRED. Telemetry timeout и thermal gate — **проверки wrapper по SDK feedback**, а не отдельные device fault codes.
- Существующий wrapper уже не посылает MoveServoJoint при not-ready, выбрасывает pending targets и требует explicit recovery. Это сохранено, как и номинальный keepalive 200 Hz (5 ms).
- Python логирует SDK_ERROR с точным result, readiness, diagnosis, telemetry, transport и exception; блокирует дальнейшие submissions. Healthy snapshot и ARM не снимают latch. Только успешное explicit recovery; automatic reset/reconnect/rearm не добавлен.
- Исключение Python control pipeline помечается COMMAND_PIPELINE_ERROR; оно не выдаётся за подтверждённый hardware fault. Software значения 850/1050 mA, 10° и 350 ms не названы fault Tesollo.

## Recorder и просмотр

`bash scripts/stack.sh debug-record-teleop` сохранён. Bundle содержит `servo_ticks.jsonl`, `timeline.csv`, `events.jsonl`, `quest_hand.jsonl`, `landmarks.jsonl`, `rosout.jsonl`, `quest_logcat.log`, manifest/status и host network snapshots. Для Quest logcat нужен подключённый и авторизованный ADB; если он недоступен, manifest фиксирует причину и ROS capture продолжается.

Каждый servo timer tick, включая HOLD, содержит q_target/previous_q_cmd/q_cmd/q_measured, raw step, post-contact/post-guard, low_level_submitted и наблюдаемый low_level_cmd, currents/total, budget/actual и submitted lead, contact state/reason, tracking ages/reason и runtime_state. `command_submitted=false` отмечает HOLD; contact snapshot во время HOLD имеет `compliance_age_ms` и не выдаётся за новую guard evaluation. Наблюдаемый SDK command асинхронный, не ACK именно этого tick; учитывать `low_level_valid`. Telemetry arrays в servo JSON — градусы, current — mA.

Новые события не теряются в whitelist recorder: TRACKING_LOST, TRACKING_RECOVERED, CONTACT_PENDING/LATCHED/RELEASED, CURRENT_LOAD_THRESHOLD, SDK_ERROR. Legacy OBJECT_CONTACT_* тоже поддерживаются. В summary добавлены current-load/SDK-error/recovery counters. Старые CURRENT_GUARD_TRIP/DISARMED поддерживаются для legacy и ручных действий, но servo software threshold больше их не генерирует.

Live в другом терминале:

```bash
docker compose exec lerobot_hand bash -lc 'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; ros2 topic hz /dg5f/lerobot/servo_state'
docker compose exec lerobot_hand bash -lc 'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; ros2 topic echo /dg5f/lerobot/servo_state'
```

`actual_rate_hz` и `dt` дают частоту самого timer; topic hz также включает доставку DDS. В HOLD эти 60 Hz — диагностика, удержание SDK работает отдельно. Номинальные 200 Hz SDK не следует путать с asynchronous communication callback rate.

## Проверки

- Overcurrent выше старых thresholds: нет DISARM, нагруженный joint ограничен, другой joint продолжает движение.
- Tracking loss с elapsed >20 s: бессрочный HOLD, target при tracking=false игнорируется, physical command сохраняется.
- Recovery: без ARM/feedback reseed, шаг q_cmd ограничен; отдельно проверен command-timeout при require_tracking=false.
- Stalled index: measured=58°, target=71.5°, ток растёт до 1200 mA; physical lead не превышает budget. Проверен уже накопленный q_cmd=71° при measured=58°: physical≤59° при budget=1°, без присваивания q_cmd=measured.
- Все прежние reversal/cross-zero, contact-offset и fixed-anchor tests сохранены.
- SDK result=500: точный SDK_ERROR, отправка блокируется, armed не сбрасывается, свежий healthy snapshot и ARM latch не снимают.
- Recorder проверяется на сохранение новых событий, physical lead/runtime fields и raw Quest/landmarks/ROS logs.
- Финальный набор servo/tracking/recorder regression: **58 passed**. `git diff --check`, Python compile и shell syntax — PASS.
- Dex Hybrid replay: **673 frames ×20 joints, побитное равенство, max abs difference=0 rad** относительно baseline 9490086. Вход — первые 240 кадров и interval 35–41 s последнего bundle. Артефакты/скрипт: `debug_runs/servo_policy_validation/`; исходники retarget/MANO/mapping/config/limits имеют нулевой diff.
- Полный LeRobot suite: **178 passed, 1 прежний failure** `test_contact_and_rising_current_decrease_convergence_before_soft_threshold` (2.597045927 < 4.881959198/2 не выполняется; воспроизводился на baseline).
- dg5f_teleop: **17 passed, 2 прежних failures** `test_velocity_profile`: SimpleNamespace не содержит `_publish_contact`/`_warn_throttled`. Retarget ради этих заглушек не менялся.
- Fake SDK: PASS (telemetry conversion, persistent thread, stale-target gate, explicit recovery, thermal semantics).
- Hardware motion в ходе этого патча не запускался; результат физического прогона предстоит проверить.

## Изменённые файлы именно этой итерации

В `src/lerobot_robot_dg5f/lerobot_robot_dg5f/`: `ros_bridge_node.py`, `current_guard.py`, `servo_controller.py`, `dg5f.py`, `debug_recorder.py`, `debug_recording.py`.
Tests: `test_servo_bridge.py`, `test_servo_recording.py`. Документация: этот отчёт, `README.md` пакета, комментарии `config/bridge.params.yaml`. Изменения предыдущего hardware-fix в launch/config/recorder scripts сохранены. Unrelated рабочие изменения IsaacLab/object-contact tests не тронуты.

## Следующий hardware test

Сначала запустить hardware stack, затем recorder в другом терминале. Servo ARM
разрешён без tracking: hardware инициализируется в TRACKING_HOLD; fresh input
автоматически включает ACTIVE. Команды выполняются из корня проекта.

```bash
# Терминал 1: 60 Hz, 120 deg/s по умолчанию
DG5F_CONTROL_MODE=servo bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72

# Терминал 2: единый timestamped bundle, включая ADB logcat
bash scripts/stack.sh debug-record-teleop

# Терминал 3: единственный initial ARM
bash scripts/stack.sh arm
```

Ctrl+C в recorder завершает архив `debug_runs/dg5f_debug_<timestamp>_<pid>.tar.gz`; сам hardware stack этим не останавливается. После tracking recovery повторный ARM не нужен. Ручное отключение: `bash scripts/stack.sh disarm`.

Для сравнения legacy (в отдельном запуске stack):

```bash
DG5F_CONTROL_MODE=legacy bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
```

Простой `bash scripts/stack.sh hardware ...` по-прежнему выбирает legacy, если DG5F_CONTROL_MODE не задан. ARM до запуска hardware не активирует будущий процесс.
