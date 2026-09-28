# Servo ARM без Quest / tracking

Источник regression: `debug_runs/dg5f_debug_2026-09-28_17-23-37_124718.tar.gz`.
Патч от 2026-09-28 поверх runtime policy; legacy не изменён.

## Control-policy regression и исправление

В архиве 9098 samples за 304.601 s: transport_connected=1, motion_ready=1,
sdk_block_reason=NONE во всех samples. Пять ARM_REQUESTED на t≈9.466,
13.946, 117.409, 121.072, 132.431 s; каждому соответствует
`ARM FAILED: TRACKING_TIMEOUT` в rosout. ARMED нет.

Причина: `_on_enable()` вызывал `_input_failure()` до и после hardware preflight,
а также требовал `_latest_command_deg`. Тем самым servo initial ARM оставался
зависимым от Quest, хотя runtime tracking loss уже стал бессрочным HOLD.

Теперь в `_on_enable()`:

- Input prerequisites сохранены только для legacy. В servo tracking/command timeout
  выбирают HOLD, но не отказывают в ARM.
- Сохранены passive `prepare_arm()`, hardware health check и проверка свежего
  measured feedback. Controller один раз инициализируется measured pose.
- Servo ARM один раз передаёт эту же позу в backend: `q_cmd = low_level_submitted =
  q_measured` с прежними joint limits/disabled joint constraints. Это необходимая
  инициализация физического HOLD; обычные servo ticks в HOLD ничего не отправляют.
  Наблюдаемый `low_level_cmd` обновляется асинхронно после обработки SDK mailbox.
- Если свежего tracking/target нет, старый target и его accepted timestamp
  очищаются; успешный ARM сразу устанавливает TRACKING_HOLD. HOLD telemetry
  работает и до первого target (q_target отсутствует: NaN/null, не поддельная поза).
- Свежий tracking status от retarget и следующий валидный target автоматически
  включают ACTIVE. Существующий ServoController ограничивает шаг до 2° при
  60 Hz / 120°/с. Tracking recovery не reseed-ит command и не вызывает ARM.
- ACTIVE → TRACKING_HOLD при потере tracking/command, без DISARM; возвращение
  fresh tracking + нового target снова даёт ACTIVE.

`prepare_arm()` по-прежнему не перезапускает hardware. Сам **servo enable теперь
инициализирует backend setpoint**, поэтому прежнее описание всего `_on_enable()`
как strictly passive больше неверно. Legacy enable остаётся passive.

Реальные причины отказа: transport disconnected, остановленный control thread,
невалидная/stale telemetry, unsafe temperature, system not started,
recovery_required / motion_ready=false / ненулевой SDK motion result,
stale/невалидная measured pose или отказ backend принять measured HOLD.
Отдельно сохранены SDK error latch (снимается explicit recovery), уже выполняемый
recovery/preflight и исключения control pipeline. Tracking timeout не является
причиной отказа servo ARM. Legacy timeout policy сохранена по требованию.

Dex Hybrid, MANO, mapping/limits, servo controller/rate/speed, object-contact,
lead/current logic, SDK и debug-record-teleop этим патчем не изменены.

## Отдельная upstream проблема

Весь архив: quest_message_count=0, landmarks_message_count=0,
tracking_loss_reason=NO_LANDMARKS, detail=NO_QUEST_HAND_INPUT.
`quest_hand.jsonl` и `landmarks.jsonl` пусты. И bridge, и recorder независимо
не получили raw input — это не только фильтрация retarget или запись после ARM.
В ROS graph есть unity_endpoint и unity_mano_adapter; наличие имён topics
не доказывает поступление сообщений. MANO adapter публикует landmarks только
в callback входящего `/quest/hand_pose`; отсутствие Quest input объясняет и
отсутствие landmarks. ARM не управляет этими publishers/subscriptions.

Сохранённый предыдущей сессией live snapshot в CODEX_HANDOFF: TCP listener
0.0.0.0:10000 работал, established connections на этом порту не было. Это
свидетельство отсутствия Unity TCP connection в тот момент, а не за всю запись.
В самом архиве нет TCP capture (tcpdump=false); network counters относятся к
Ethernet руки enp49s0, а не к Quest Wi-Fi. `quest_logcat.log` содержит только
`- waiting for device -`; ADB не был доступен. Поэтому отличить незапущенное
Quest приложение, неправильный endpoint или сетевую недоступность по этому
bundle нельзя. Отсутствие ADB само по себе не мешает Wi-Fi ROS stream.

При текущем read-only осмотре stack уже остановлен, listener отсутствует,
ADB devices пуст. Wi-Fi IP компьютера сейчас 10.91.94.128; Ethernet руки
169.254.186.70. При следующем запуске проверить Unity endpoint
`10.91.94.128:10000` с учётом возможной смены IP.

В manifest `configuration.control_mode=legacy` — recorder default;
`runtime_configuration.control_mode=servo`, rate=60, velocity=120 — фактические
параметры bridge. Причина regression не в запуске legacy. Recorder не менялся.

## Проверка следующего запуска

Запустить hardware с `DG5F_CONTROL_MODE=servo`, затем recorder. В ROS shell
одновременно проверить оба потока в течение примерно 10 секунд:

```bash
timeout --signal=INT 10s ros2 topic hz /quest/hand_pose &
timeout --signal=INT 10s ros2 topic hz /hands/right/landmarks &
wait
```

Для Docker shell: `docker compose exec lerobot_hand bash`, затем
`source /opt/ros/humble/setup.bash` и `source /workspace/install/setup.bash`.
Завершение `timeout` с кодом 124 ожидаемо; отсутствие вывода hz означает,
что частота не была измерена, а не точную оценку 0 Hz.

`bash scripts/stack.sh arm` должен успешно пройти и с полностью выключенным
Quest: armed=true, runtime_state=TRACKING_HOLD, measured-initialized HOLD.
При появлении fresh tracking/valid retarget → ACTIVE без второго arm.
После потери и восстановления — тот же цикл без DISARM/re-ARM.

## Проверки патча

Regression tests в `test_servo_bridge.py`: отсутствующий input, stale tracking,
stale target, tracking=false перед ARM; однократный measured backend submission;
HOLD telemetry до первого target; автоматический ACTIVE после позднего target;
повторные loss/recovery по false tracking, tracking timeout и command timeout;
нет DISARM/второго ARM/reseed, скорость ограничена; hardware readiness failures,
невалидная pose и отклонённый backend submission; прежний legacy ARM timeout.

Аппаратный ARM/recover и движение в ходе патча не запускались.

Результат: **94 passed** (servo controller/bridge/regressions/recording, tracking
diagnostics, recovery, backend/feedback), 6.98 s в Docker, изолированные ROS domains.
32 предупреждения — прежний RuntimeWarning в current_guard.py:267, алгоритм не
менялся. `git diff --check` и Python compilation изменённых Python файлов — PASS.
