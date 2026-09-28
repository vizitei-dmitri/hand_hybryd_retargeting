# Handoff для следующей сессии Codex

Дата: 2026-09-28, timezone Europe/Moscow. Проект: `/home/yoba/Documents/work/hand_hybryd_retargeting`.
Пользователь общается по-русски. Последний запрос — узко исправить замедление index
в FREE после contact по bundle 18:05:11. Аппаратная проверка этого патча предстоит.

## Обновление: index FREE slowdown (2026-09-28)

Checkpoint перед acceleration: hardware bundle
`dg5f_debug_2026-09-28_18-30-01_157769.tar.gz` подтвердил 0 DISARM,
0 CURRENT_GUARD_TRIP/STALL_GUARD_TRIP/SDK_ERROR/disconnect; max current420mA.
Перед checkpoint полный LeRobot+retarget+Unity suite:250 passed,3 известных
baseline failures (compliance assertion и2 retarget SimpleNamespace tests).
Fake SDK PASS. Локальный tag `dg5f-servo-stable-pre-accel` сохраняет эту версию;
последующие acceleration changes должны быть отдельным коммитом. Изменения
IsaacLab исключены. Исторические указания ниже о незакоммиченных servo files
после создания checkpoint уже не актуальны.

Сначала читать [отчёт index](DG5F_INDEX_FREE_MOTION.md). Новый bundle:
`dg5f_debug_2026-09-28_18-05-11_146230.tar.gz`. ARM/tracking/retarget работают.
На t=84.426 s joint6: raw step2°, current8mA, FREE, current/slope/contact scales≈1,
lead8°. Шаг падал до0.257709° из-за **legacy object-contact resume ramp15°/с**,
остающегося после RELEASED при resume_offset!=None; tracking_scale=0.128854.
Это не ошибка расчёта lead от operator target. Index чаще latched (4 раза),
864 FREE/low-current slow ticks против35 ring и0 middle/little.

Минимальная правка: current_guard передаёт object_contact.update
`apply_resume_ramp=False` для servo (отдельный operator_target_deg). FREE offset
очищается, при release не создаётся; следующий шаг ограничивает существующий
ServoController. Legacy defaultTrue, CONTACT_HOLD/yield/current/lead сохранены.
Добавлены диагностические scales/stage gains/limiting reasons/progress/resume
offset в guard result; existing bridge/recorder сохраняют их без правок.
Production changes этой итерации только current_guard.py и object_contact.py.
Новый test_servo_free_motion.py; два предыдущих servo tests обновлены: FREE
offset больше не влияет, а защитная projection проверяется injected bad output.
Чужие test_object_contact*.py не менялись.

Проверки:100 focused passed; полный suite227 passed +1 known baseline compliance
failure с прежними числами. Legacy600 ticks точное равенство before/after;
Hybrid533×20 outputs побитно равны HEAD, maxdiff0rad. Servo60Hz/120deg/s unchanged.
Артефакты debug_runs/index_slowdown_validation/. Без hardware ARM/recover/motion.
Все прежние незакоммиченные изменения сохранены; HEAD9490086 по-прежнему.

## Обновление: ARM без tracking (2026-09-28)

Последний запрос и исправление: [ARM без Quest](DG5F_SERVO_ARM_WITHOUT_TRACKING.md).
Архив `dg5f_debug_2026-09-28_17-23-37_124718.tar.gz` финализирован: пять
ARM_REQUESTED/пять TRACKING_TIMEOUT при исправном hardware и нулевом Quest input.
В servo `_on_enable()` теперь не требует tracking/target: measured seed controller
и однократный backend submission measured HOLD, затем успешный TRACKING_HOLD при
отсутствии fresh input. Свежий tracking + новый target автоматически дают ACTIVE.
Legacy и прочие control algorithms не менялись. Старые указания ниже «дождаться
tracking, затем ARM» для servo отменены; они описывают предыдущую реализацию.
Servo ARM больше не strictly passive: hardware preflight passive, но hold setpoint
инициализируется через backend. SDK recovery/restart не вызываются.
Правка production-кода только в ros_bridge_node.py; новые regressions добавлены
в test_servo_bridge.py. Все прежние незакоммиченные изменения сохранены.
При read-only проверке этого сеанса hardware stack уже остановлен; не запускали
физический ARM/recover. Причина отсутствия Quest input отдельно описана в отчёте;
точный Quest-side дефект по архиву без TCP/ADB capture не устанавливается.
Проверки этой итерации: **94 passed** (servo/tracking/recording/recovery/backend),
прежние current_guard RuntimeWarning; diff check и Python compilation PASS.

## В первую очередь

1. Рабочая ветка **`feature/dg5f-servo-control`**. Все описанные изменения находятся в working tree, **не закоммичены**. HEAD — `9490086d11f362f939d8f0ee87324aa82b540dca`. НЕ делать reset/checkout файлов из HEAD: это уничтожит текущую реализацию и предыдущие исправления.
2. Сначала читать [актуальный отчёт](DG5F_SERVO_RUNTIME_POLICY.md). [Hardware fix](DG5F_SERVO_HARDWARE_FIX.md) и [первый servo audit](DG5F_SERVO_CONTROL.md) — история; их описание отключения/возобновления servo уже частично устарело.
3. Не изменять Dex Hybrid / Vector / DexPilot / MANO / coordinate transforms / joint mapping / limits. Пользователь это несколько раз явно запретил. Изменяем только слой после готового q_target[20].
4. Сохранять legacy path, предыдущие contact-offset corrections, telemetry и `debug-record-teleop`.
5. Не запускать ARM/recover или двигать физическую руку ради теста без соответствующего запроса. В предыдущей работе тестировались mock/fake SDK и пассивно читались live status/logs. В системе может продолжать работать пользовательский hardware stack и recorder: проверить процессы, не перезапускать их без причины.
6. Не использовать subagents: актуальная инструкция этого сеанса запрещала делегирование без явного запроса. AGENTS.md в репозитории/родителях ранее не найден.

## Последний live-инцидент: ARM FAILED: TRACKING_TIMEOUT

Пользователь после запуска hardware выполнил `bash scripts/stack.sh arm` и получил отказ. Проверено read-only непосредственно в работающей системе:

- Hardware launch действительно в **servo**, 60 Hz, 120 deg/s, backend tesollo, hybrid, IP руки 169.254.186.72.
- Все ROS процессы живы: unity_endpoint, unity_mano_adapter, dg5f_retarget, dg5f_mujoco, dg5f_lerobot_bridge.
- Активный recorder: `debug_runs/2026-09-28_17-23-37_124718/` (на момент проверки ещё не финализированный bundle).
- Последний полный timeline snapshot около t=68.768 s:
  - armed=0, tracking_ok=0, runtime_state=DISARMED, disarm_reason=TRACKING_TIMEOUT;
  - motion_ready_reason=READY, last_motion_result=0;
  - quest_message_count=0, landmarks_message_count=0;
  - last_quest/landmarks/retarget/tracking age = NaN (ни одного сообщения);
  - tracking_loss_reason=NO_LANDMARKS, tracking_loss_detail=NO_QUEST_HAND_INPUT.
- `quest_hand.jsonl` и `landmarks.jsonl` пустые. Events только два ARM_REQUESTED; rosout содержит два отказа ARM.
- TCP сервер слушает `0.0.0.0:10000`, **установленных соединений на 10000 нет** (`ss -tnp`).
- `adb devices` пуст; `quest_logcat.log` содержит `- waiting for device -`.
- IP компьютера на момент проверки: Wi-Fi **10.91.94.128**, Ethernet к руке **169.254.186.70**. IP могут измениться; перед повторным советом перепроверить.

Пользователю уже объяснено: сейчас данные Quest отсутствуют вообще; нужно запустить Unity-приложение в Quest, подключить его к **10.91.94.128:10000**, показать руку камерам, дождаться движения виртуальной руки и затем повторить ARM. Это отказ первоначальной авторизации без tracking, а не automatic runtime DISARM. ADB нужен отдельно для logcat, ROS capture работает и без него. Код из-за этого инцидента не менялся; ARM сами не вызывали.

## Итерации и требования пользователя

### Начальный servo

Сделать новый control_mode legacy|servo без замены legacy. Retarget обновляет latest target, независимый 60 Hz servo timer, persistent q_cmd, velocity step limiter. Инициализация measured один раз; никаких measured-based starts на каждом tick. Полная поза 20 суставов streaming, без очереди/ожидания достижения. Сохранить current/safety guard. Логировать target/command/measured/error/dt/rate/max step, warn >10° >=300 ms. Не добавлять PID, acceleration/jerk, planner, фильтрацию measured.

Изначально скорость была 30 deg/s, после hardware fix пользователь принял **120 deg/s default**. Сейчас 2°/tick при 60 Hz; speed CLI/env настраиваемый. Legacy default max_speed_deg_s=30 не менялся.

### Hardware fix, первый bundle

`debug_runs/2026-09-28_12-18-04/`: target положительный около +64…70°, command уходил к -58° и дальше. Причина: object-contact принимал servo proposal (`q_cmd+0.5`) за operator intent. После release `resume_offset` вычитался из следующего proposal, сам создавал OPERATOR_OPENING и повторное отступление.

Исправлено: separate real operator_target_deg для object-contact; direction/distance projection; разрешённый CONTACT_PRELOAD_YIELD ограничен measured+preload И фиксированным contact-anchor floor. Reversal tests сохранены.

В старом bundle retarget heartbeat перестал обновляться ~t74.447, watchdog через >350 ms выставил false (log ~t74.875). False tracking heartbeat сам оставался свежим. Raw Quest/landmarks тогда не писались, точная причина исчезновения input **not fully attributable without Quest/Unity logs**. Пользователь прямо сказал не искать старые Quest logs любой ценой и не блокировать servo fix. Добавлен единый recorder и наблюдательная tracking diagnostics.

### Последняя runtime policy, второй hardware bundle

Запрос целиком в attachment:
`/home/yoba/.codex/attachments/02c15c0a-08e6-49d0-be30-9ad2a646f155/Pasted text.txt`.
Предыдущий запрос hardware fix:
`/home/yoba/.codex/attachments/da14a029-ea0b-4a2e-9b9a-4c6168130683/Pasted text.txt`.

Источник анализа: `debug_runs/dg5f_debug_2026-09-28_15-39-45_90896.tar.gz`, уже распакован в одноимённую папку без `dg5f_debug_` и `.tar.gz`.

Подтверждено:
- t=39.911375: CURRENT_GUARD_TRIP, Imax1207 mA, Itotal1461 mA, duration50.51 ms;
- через0.119 ms DISARMED reason OVERCURRENT_GUARD;
- SDK READY/result0, transport/telemetry/temperature/system хорошие; tracking свежий (Quest age8.1 ms, landmarks7 ms, retarget25.9 ms).
- q_target[6]=71.496°, q_cmd[6]=70.984°, q_measured[6]=58.100°, gap12.884°;
- budget[6] уже≈1.235°, scale0, но gap оставался.

Root cause lead budget: старый guard ограничивал только **увеличение** ошибки и только если limited & load_evidence>0.01 & delta*current_error>=0. При отсутствующем evidence gap накапливался; при сужении budget `max(0,budget-abs(error))` давал HOLD старой завышенной команды. Hard freeze удерживал её же. Абсолютного ограничения physical command относительно measured не было.

Пользователь явно потребовал в servo убрать automatic global DISARM от tracking/current/stall/contact, оставить initial ARM и manual disarm, реальные SDK errors не скрывать и автоматически не сбрасывать. Legacy сохранён с прежними правилами.

## Текущая реализация

Основной Python пакет: `src/lerobot_robot_dg5f/lerobot_robot_dg5f/`.

### servo_controller.py (новый)

`ServoController(PositionCommandShaper)` использует inherited validation/lifecycle/limits, но не legacy smoothing/accel/deadband/blend.
- `propose`: q_cmd + clip(target-q_cmd, ±speed * min(actual_dt, 1/rate)); первая итерация dt = 1/rate. Target clipped/fixed без изменения исходных limits.
- `hold`: сохраняет persistent q_cmd И последнюю физическую команду, очищает tick/warning times; measured не подставляет.
- `GuardedServoCommand`: trajectory_deg + physical lower/upper envelope.
- `constrain_guarded_output`: ограничивает скорость persistent guarded q_cmd.
- `physical_output`: ограничивает скорость относительно прошлого physical submission, затем применяет lead envelope из **существующего current guard**. Joint limits/disabled имеют приоритет.
- Важно: при резком сужении envelope / скачке measured physical retreat может превысить velocity step, чтобы не сохранить прежний большой gap. Причина отдельно `LEAD_BUDGET_YIELD_OR_HOLD`; обычный физический slew `PHYSICAL_RATE_LIMIT`. Это явно описано пользователю/в отчёте.
- `accept_output(command,physical_deg=...)` только после успешного backend send: q_cmd=guarded persistent trajectory, last_sent_pose_deg=physical. Не путать `effective_command()` (physical) и command_pose_deg (q_cmd).
- telemetry: previous_q_cmd, raw_servo_step, q_proposed, q_cmd, direction_violation, distance_before/after, physical_step_deg, low_level_submitted, previous_low_level_submitted, physical_limit_reason; observe raw measured + tracking error.

### dg5f.py

`servo_tick`: propose → bridge guard → constrain q_cmd → physical_output(envelope) → backend.send_positions → commit оба состояния. Legacy send_action не переписан.

`pause_trajectory` только hold software; `hold_position` ещё backend.suspend_motion. Initial ARM/recovery могут reseed measured один раз.

### current_guard.py

- `operator_target_deg` optional: None = прежняя legacy семантика; real target = servo.
- Servo использует per-joint current/slope scales, total current/slope не замораживают все joints. Геометрический contact продолжает влиять на участвующие joints.
- Сохранены object-contact correction и direction projection. CONTACT_PRELOAD_YIELD bounded fixed anchor.
- Trip/stall остаются вычисляемыми диагностическими flags, но bridge servo их не превращает в global DISARM.
- Существующий `joint_lead_budget_deg` (default1–8° при compliance=true) возвращает physical_lower/upper=measured±budget. Envelope применяется **после** contact/hard freeze/trajectory limiting, а не параллельным новым budget.
- При compliance=false budget=inf (нет physical lead cap); при current_guard_enabled=false guard bypass как ранее.
- Guard servo effective — command_pose_deg, legacy effective — last accepted physical command.
- Существует прежний RuntimeWarning в оценке velocity из-за np.where, вычисляющего inf arithmetic. Не исправляли unrelated.

### ros_bridge_node.py

- latest-target mailbox `/dg5f/joint_command`, отдельный steady-clock servo timer60 Hz, state timer30 Hz.
- `_last_retarget_time` — время любого валидного target receipt для диагностики; `_last_command_time` — принятого target. Targets при tracking stale/false (после ARM) не заменяют активный target.
- Servo `_send_latest`: вход stale → `_enter_tracking_grace` (теперь бессрочный HOLD), не вызывает timeout disarm. `_expire_tracking_grace` noop дляservo, прежняя15s policy дляlegacy.
- `_on_tracking` true + следующий fresh target → `_try_auto_resume`; servo сохраняет q_cmd/physical/contact anchors, без measured reseed/reset/ARM. Legacy сохраняет прежний re-anchor+blend.
- `_guard_current_target`: servo emits CURRENT_LOAD_THRESHOLD once per threshold episode, returns GuardedServoCommand; legacy trip/stall → прежние DISARM.
- Contact transition names дляservo CONTACT_PENDING/LATCHED/RELEASED; legacy OBJECT_CONTACT_*.
- `_send_servo_tick`: логи всех stages + raw measured/current + tracking diagnostic snapshot. SDK low_level_cmd — асинхронный snapshot, **не ACK данного tick**; `low_level_valid` обязателен при трактовке actual lead.
- `_publish_servo_hold`: telemetry каждый timer tick, command_submitted=false, pose frozen; SDK keepalive продолжает удержание. Contact snapshot может быть старым — `compliance_age_ms`.
- `_runtime_state`: DISARMED доmanualARM; ACTIVE/TRACKING_HOLD; SDK_ERROR при backend latch.
- `_block_servo_backend`: SDK_ERROR exact numeric result/readiness/diagnosis/error; latch `_sdk_block_reason`, `hold_position`/suspend backend. **Не сбрасывает `_armed`**, не reset/reconnect. ARM при latch отказывает; только успешный explicit recovery снимает latch.
- `_publish_state` servo health faults идут в backend latch, legacy — `_disarm` как ранее.
- `_on_enable(False)` explicit USER_REQUEST disarm; `_on_recover` explicit service disarms/recover/staysdisarmed, затемfreshcommand+manualARM.
- `_armed=true` теперь означает сохранённую initial authorization, не гарантию работающего backend. Смотрим runtime_state/motion_ready/sdk_block_reason.
- >10° >=300ms tracking warning не выключает руку, как и раньше.

### SDK/backend (код C++ не меняли)

`vendor/tesollo_control/dg_control/control.cpp`: единственный production MoveServoJoint ~строка362; loop5ms/номинально200Hz, полная поза 20 суставов, joint 16 = 0. Drains latest target mailbox; keepalive repeats accepted target, не ждёт motor arrival.
- isMotionReady: READY иначе DISCONNECTED/CONTROL_THREAD_STOPPED/TELEMETRY_STALE/TEMPERATURE_UNSAFE/SYSTEM_NOT_STARTED/MOTION_RESULT_ERROR/RECOVERY_REQUIRED.
- wrapper сам не отправляет при not-ready и выкидывает stale pending targets; recovery explicit.
- DG_RESULT в DGDataTypes.h:0NONE,1settings,100+args/mode/model,200+motionstates,500/501socket,2000+port,2009diagnosing. Не каждое значение означает motor fault.
- SDK callbacks transport/diagnosis и result доступны; telemetry/thermal readiness — gates **wrapper**, не выдуманные device fault codes.
- Python pipeline exception помечается COMMAND_PIPELINE_ERROR в SDK_ERROR event, без утверждения о подтверждённом hardware fault.

### TrackingDiagnostics и recorder

`tracking_diagnostics.py` — passive observer, retarget не менялся. Слушает `/quest/hand_pose` ManoLandmarks и `/hands/right/landmarks` PoseArray, сохраняет receipt monotonic / original header stamp / finite21x3 validity / count. Reasons NO_LANDMARKS, STALE_LANDMARKS (QUEST_INPUT_STALE/LANDMARK_ADAPTER_STALE/SOURCE_TIMESTAMP_STALLED), INVALID_HAND, RETARGET_STALE, TRACKING_STATUS_STALE, UPSTREAM_TRACKING_FALSE, NONE. Unity connection UNKNOWN без доказательства. `last_successful_retarget_monotonic_s` означает receipt target в bridge, не внутреннее время solver.

`debug_recording.py`/`debug_recorder.py`:
- timeline.csv30 Hz; servo_ticks.jsonl каждый received tick60 Hz, включая HOLD; events.jsonl;
- --teleop: quest_hand.jsonl, landmarks.jsonl с исходными stamp+points, rosout.jsonl filtered to project/Unity endpoints;
- Raw invalid frames тоже сохраняются;
- runtime_state/sdk_block_reason/actual_command_lead_deg добавлены в CSV; per-tick current/budget/contact/tracking и все command stages в servo JSON;
- Event whitelist расширен новыми runtime/contact/direction events (раньше новые events могли теряться!); test проверяет сохранение;
- summary добавлены current-load/SDK error/tracking recovered counters;
- CSV основные pose arrays в **radians**, servo JSON и lead budgets в **degrees**, currents mA.

`scripts/dg5f_network_capture.py` + `scripts/stack.sh debug-record-teleop`:
- timestamp_pid folder; ROS recorder с deferred archive;
- host adb logcat epoch -T1, Unity/AndroidRuntime/OVRPlugin/OpenXR/VrApi; не очищаетlogcat; ANDROID_SERIAL поддерживается;
- missing adb / process failure записываются в manifest/network_status, ROS capture продолжается;
- optional ping/tcpdump; --no-adb дляoffline;
- ROS_DOMAIN_ID явно переданный host env передаётся в Docker для isolated tests;
- Ctrl+C останавливает принадлежащие recorder процессы, stop-file, finalize, один tar.gz. **Не останавливает hardware stack**.

## Файлы в working tree

Наши изменённые/новые:
- scripts/stack.sh, scripts/dg5f_network_capture.py;
- src/dg5f_unity_teleop/launch/unity_dg5f.launch.py (bridge params);
- src/lerobot_robot_dg5f/README.md, config/bridge.params.yaml, package.xml;
- пакет: config_dg5f.py, dg5f.py, ros_bridge_node.py, current_guard.py, debug_recorder.py, debug_recording.py;
- новые servo_controller.py, tracking_diagnostics.py;
- новые tests test_servo_controller.py, test_servo_bridge.py, test_servo_regressions.py, test_servo_recording.py, test_tracking_diagnostics.py;
- docs/DG5F_SERVO_CONTROL.md, DG5F_SERVO_HARDWARE_FIX.md, DG5F_SERVO_RUNTIME_POLICY.md и этот handoff.

**Чужие/предсуществующие изменения, не трогать:**
- isaaclab_ext/.../dg5f_cube/{dg5f_cube_env.py,dg5f_cube_env_cfg.py,success.py};
- isaaclab_ext/dg5f_isaaclab/.vscode/ и CLAUDE_HANDOFF.md;
- src/lerobot_robot_dg5f/test/test_object_contact.py и новый test_object_contact_temporal.py.

Retarget source/config, MANO adapter source, models/limits и vendor SDK имеют нулевой diff от baseline.

## Проверки и результаты

Финальный focused набор servo/tracking/recording: **58 passed** (после последнего telemetry reason patch).
Полный LeRobot suite: **178 passed, 1 known baseline failure**.
Retarget suite: **17 passed, 2 known baseline failures**.
Fake SDK: PASS, telemetry conversion/persistent thread/stale-target gate/explicit recovery/thermal semantics.
`git diff --check`, Python compile, bash syntax PASS.

Known failures, ранее воспроизведены и на pristine baseline 9490086 (не вызваны servo changes):
1. test_compliance.py::test_contact_and_rising_current_decrease_convergence_before_soft_threshold: 2.597045927 не меньше4.881959198/2.
2. Две параметризации test_velocity_profile: SimpleNamespace missing `_publish_contact`, затем `_warn_throttled`. Retarget source не менять ради этих test doubles.

В предыдущей итерации также прошли Unity: 6, ros_tcp_endpoint: 3, IsaacLab CPU: 20 тестов и build 5 ROS packages. В последней runtime iteration их повторно не запускали, так как эти подсистемы не менялись.

Новые/обновлённые regressions:
- ток выше старого порога локально ограничивает index, остальные суставы движутся, global DISARM отсутствует;
- tracking loss длительностью >20 s (моделируется без wallclock sleep): HOLD; игнорирование targets при tracking=false;
- автоматическое возобновление без measured reseed / manual ARM, с проверкой max step;
- stalled measured[6]=58 target71.5 currents up to1200, ограниченный physical lead; другие mock joints следуют командам;
- уже накопленный q_cmd71/measured58 при budget=1 даёт physical≤59; q_cmd не присваивается measured;
- SDK result500 попадает в лог без искажения; healthy snapshot и ARM не снимают latch;
- прежние проверки reversal -50→+70, +40→-30, crossing zero и rapid targets для всех 20 joints;
- recorder сохраняет canonical events, lead/runtime, исходные Quest stamps, invalid frames и ROS logs;
- timer работает 1.5 s при input 10 Hz; проверяются 50–70 Hz и 70–105 full-pose submissions; input callback ничего не отправляет.

**Dex Hybrid equivalence не просто assertion о git diff**: replay 673 записанных landmark frames (первые 240 и интервал 35–41 s) через baseline 9490086 и current RetargetNode с одинаковой config. Все 673×20 outputs побитно равны; max abs diff = 0 rad.
Артефакты: `debug_runs/servo_policy_validation/{replay_retarget.py,before.npy,after.npy,retarget_equivalence.json,lerobot.xml}`. Baseline внутри контейнера: `/tmp/dg5f-servo-baseline.50noUn` (может исчезнуть при пересоздании контейнера). debug_runs исключён из Git; отчёт содержит вывод.

## Команды и окружение

Docker compose service **lerobot_hand**, ROS Humble, mounted `/workspace`, symlink install. Shell команды к ROS запускать после source. Обычный host Python не имеет полного ROS окружения.

Hardware (пользователь запускает, не запускать автоматически по этому handoff):

```bash
# Терминал 1
DG5F_CONTROL_MODE=servo bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
# Терминал 2
bash scripts/stack.sh debug-record-teleop
# Терминал 3: servo ARM разрешён и без tracking (останется TRACKING_HOLD)
bash scripts/stack.sh arm
```

- DG5F_SERVO_MAX_SPEED_DEG_S=120 default; прежний DG5F_SERVO_MAX_VELOCITY_DEG_S поддержан как fallback; DG5F_SERVO_RATE_HZ=60 default.
- Без DG5F_CONTROL_MODE заданного env default **legacy**.
- `bash scripts/stack.sh disarm` — manualstop.
- После servo tracking recovery ARM не нужен.
- Initial ARM до старта hardware не авторизует будущий процесс.

Read-only live diagnostics:

```bash
docker compose ps
docker compose top
ss -ltn '( sport = :10000 )'
ss -tnp '( sport = :10000 or dport = :10000 )'
ip -brief -4 address show up
adb devices
docker compose exec lerobot_hand bash -lc 'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; ros2 topic hz /dg5f/lerobot/servo_state'
docker compose exec lerobot_hand bash -lc 'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; ros2 topic echo /dg5f/lerobot/servo_state'
```

При активном recorder удобнее пассивно читать последнюю полную CSV-строку, events и число raw frames. Не создавать дополнительное SDK connection: аппаратный backend уже использует SDK.

Tests (изолированные ROS domains, чтобы не вмешиваться в hardware domain 0):

```bash
docker compose exec -T -e ROS_DOMAIN_ID=93 lerobot_hand bash -lc 'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; cd /workspace/src/lerobot_robot_dg5f; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --tb=short test/test_servo_controller.py test/test_servo_bridge.py test/test_servo_regressions.py test/test_servo_recording.py test/test_tracking_diagnostics.py'
# Bridge fixture сама использует domain 94.
docker compose exec -T lerobot_hand bash -lc 'bash /workspace/scripts/test_sdk_fake.sh'
```

Не нужно повторять все успешные проверки без новых изменений. Аппаратное поведение последней runtime policy ещё не подтверждено: последняя попытка остановилась на отсутствии Quest connection, а не на выполнении servo-кода.
