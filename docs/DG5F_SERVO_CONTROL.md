# DG5F: legacy / servo

Ветка: `feature/dg5f-servo-control`. База аудита: `9490086`.

Это исторический отчёт первой версии (30°/s). После аппаратного теста исправлена
передача intent в contact layer, добавлены invariants/диагностика и default speed
повышен до 120°/s. Актуальные команды и результаты:
[исправление после аппаратного теста](DG5F_SERVO_HARDWARE_FIX.md).

Изменён только слой физического управления после готовой позы retargeting.
Dex Hybrid, DexPilot/Vector, MANO, преобразования координат, mapping, таблицы
joint limits и fixed joint `rj_dg_5_1=0°` сохранены. `legacy` остаётся режимом
по умолчанию. Переключение режима — при запуске процесса.

Ориентир: [tesollo_control](https://github.com/VAlikV/tesollo_control),
[его streaming loop](https://github.com/VAlikV/tesollo_control/blob/main/dg_control/control.cpp).
Использован принцип передачи последнего setpoint без ожидания достижения позы;
существующий локальный SDK-адаптер и его protections не заменялись upstream-кодом.

## Архитектура

До / `legacy`:

```text
Quest → landmarks → retargeting → latest target (20 joints)
    → bridge timer 50 Hz → existing current/contact guard
    → PositionCommandShaper (direct 4°/update + ARM blend в текущем YAML)
    → backend.set_target_position → C++ adapter / safety / 200 Hz keepalive
    → MoveServoJoint → DG5F
```

После / `servo`:

```text
Quest → landmarks → тот же retargeting → latest target (20 joints)
    → steady-clock bridge timer 60 Hz → ServoController, persistent q_cmd
    → existing current/contact guard → bounded guarded output
    → backend.set_target_position → тот же C++ safety / 200 Hz keepalive
    → MoveServoJoint → DG5F
```

Retargeting callback только заменяет последнюю цель. ROS subscription depth=1;
новой очереди траекторий нет. Низкоуровневый существующий ring buffer вычитывается
целиком: адаптер использует только самый новый setpoint.

60 Hz — частота **формирования и передачи полной позы в существующий backend**.
Сам `MoveServoJoint` продолжает вызываться адаптером примерно на 200 Hz,
повторяя последний безопасный setpoint. Это сохранённый keepalive, не новый
планировщик. Даже неподвижная поза передаётся servo backend на каждом тике.
Ожидания физического завершения движения нет.

Timer работает в отдельном от retargeting ROS-процессе и использует steady clock.
Это обычный ROS/Linux timer, без гарантии hard real-time. Чтение состояния и
проверки безопасности в том же bridge могут задержать тик; задержки видны в `dt`.

## Ключевая логика

Концептуальный diff (полный diff — `git diff` по файлам ниже):

```diff
- guarded_target = current_guard(latest_target)
- legacy_shaper.step(guarded_target)
+ dt = monotonic_now - previous_tick
+ limiter_dt = min(dt, 1 / rate_hz)
+ max_step = max_velocity_deg_s * limiter_dt
+ proposed = q_cmd + clip(limit(q_target) - q_cmd, -max_step, max_step)
+ guarded = current_guard(proposed)
+ if guarded is not None:  # existing trip may disarm
+     output = bound_guarded_step(guarded, q_cmd, max_step)
+     backend.send_positions(output)  # all 20, also when unchanged
+     q_cmd = output                 # only after backend acceptance
+ tracking_error = abs(q_cmd - q_measured)
```

По умолчанию: `servo_rate_hz=60.0`, `servo_max_velocity_deg_s=30.0`.
При нормальном тике шаг ≤0.5°. Короткий тик даёт меньший шаг. После задержки
таймера шаг также ≤0.5°: пропущенное время не компенсируется скачком. В логе
`dt` — реальный интервал, `limiter_dt` — использованный ограничителем.

`q_cmd` хранит последнюю принятую backend команду **после safety**. Ограниченная
защитой команда становится началом следующего шага, поэтому скрытая траектория
не убегает вперёд при контакте. Ограничитель применяется и к safety relief;
направление разгрузки сохраняется. Порог аварийного отключения не меняется.
Номинальный шаг current guard в servo соответствует скорости/rate (0.5°),
в legacy остаётся прежним (4°). Код guard и его токовые пороги не менялись.

Начальное состояние берётся из measured position с существующими joint limits
и fixed joint. Сохранены существующие безопасные переходы: ручной ARM,
explicit recovery и auto-resume после tracking loss один раз переинициализируют
состояние из свежей физической позы. На обычных тиках и при замене target
измеренная позиция **никогда** не задаёт начало следующего шага.
В servo после ARM действует только ограничение скорости; legacy startup blend,
фильтр, deadband и acceleration limiter не используются.

Tracking warning срабатывает, когда **один и тот же сустав** имеет ошибку >10°
непрерывно ≥300 ms. Он выдаётся один раз за эпизод и снова разрешается после
снижения ошибки. Warning не останавливает руку. Существующие current/stall,
telemetry, temperature, command timeout, tracking grace, ARM/recovery защиты
продолжают действовать. Существующая contact logic сохранена; новая не добавлена.

## Аудит исходного пути

| Вопрос | Поведение исходного кода / legacy |
|---|---|
| Где `MoveServoJoint`? | Единственный production-вызов: `vendor/tesollo_control/dg_control/control.cpp:362`, `DGControl::_loop()`. Python `backends.py::send_positions()` вызывает `DGApi.set_target_position()`. |
| Частота? | Bridge timer: 50 Hz по YAML. C++ loop: период 5 ms, номинально 200 Hz, с повтором последней команды при keepalive. DISARMED/unsafe: 0 вызовов. Фактическая частота на физическом SDK в этой работе не измерена; константа 200 и `communication_rate_hz` не являются таким измерением. |
| Возможен скачок на много градусов? | Текущий YAML: direct-профиль ограничивает обычный шаг до 4° на обновление (~200°/s при 50 Hz), то есть до servo он в 8 раз больше. При `max_direct_step_deg=0` этот предел отключается. Низкоуровневый адаптер сам не интерполирует и принимает допустимую позу сразу. |
| Есть ли интерполяция? | В direct-профиле — ограничение шага и software blend 0.70 s при ARM, 1 s при auto-resume. Опциональный smoothing-профиль содержит фильтр, deadband, speed/acceleration limits. C++ `_updatePos()` напрямую копирует допустимые setpoints. |
| Measured как старт? | Не каждый тик: `PositionCommandShaper` уже хранит persistent command. Измерение используется при connect, ARM, recovery и tracking auto-resume; guard также использует feedback для безопасности. |
| Ожидание предыдущего движения? | Нет проверки достижения позы перед отправкой. В Python есть только retry после 2 ms при отказе mailbox; это не ожидание двигателя. |
| Что при недостижении command? | SDK продолжает получать setpoint, feedback не перематывает траекторию. Existing current/contact guard ограничивает рост ошибки/нагрузки; hard current замораживает нагружающее движение; sustained overcurrent/stall может disarm. При низком токе без иных faults само недостижение не означает остановку. Tracking error уже был в агрегированной диагностике, но отдельного предупреждения >10°/300 ms не было. |

## Запуск

Команды с хоста из корня репозитория. Перед новым hardware launch завершить
предыдущий hardware launch: один процесс должен владеть SDK-сессией.
В ходе разработки существующий аппаратный запуск не перезапускался.

```bash
bash scripts/stack.sh compile

# Старое поведение; это также значение по умолчанию.
DG5F_CONTROL_MODE=legacy bash scripts/stack.sh hardware-headless

# Новый режим: 60 Hz / 30°/s.
DG5F_CONTROL_MODE=servo bash scripts/stack.sh hardware-headless

# После запуска, при свежем Quest tracking и успешном preflight:
bash scripts/stack.sh arm
# Отключить выход:
bash scripts/stack.sh disarm
```

Дополнительные переменные: `DG5F_SERVO_RATE_HZ`,
`DG5F_SERVO_MAX_VELOCITY_DEG_S`. Retargeting-профиль выбирается по-прежнему;
`DG5F_CONTROL_MODE` не переключает Hybrid/Vector/DexPilot.

Mock без физической руки:

```bash
DG5F_CONTROL_MODE=servo bash scripts/stack.sh headless
```

Прямой launch в ROS-окружении контейнера:

```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
ros2 launch dg5f_unity_teleop unity_dg5f.launch.py \
  lerobot_backend:=tesollo lerobot_auto_enable:=false \
  control_mode:=servo servo_rate_hz:=60.0 servo_max_velocity_deg_s:=30.0
```

Для legacy заменить `control_mode:=servo` на `control_mode:=legacy`.
Также доступны одноимённые ROS parameters/config в `bridge.params.yaml`.
Отдельный `Dg5f.servo_tick()` требует callback существующей защиты;
`send_action()` остаётся legacy API и не используется servo bridge.

## Частота и логи

В ROS-окружении контейнера, при ARMED и свежем входе:

```bash
ros2 topic hz /dg5f/lerobot/servo_state
ros2 topic echo /dg5f/lerobot/servo_state --once --field data
ros2 bag record -o /workspace/bags/servo_check \
  /dg5f/lerobot/servo_state /dg5f/lerobot/diagnostics \
  /dg5f/lerobot/commanded_joint_states /dg5f/lerobot/joint_states \
  /dg5f/lerobot/events
```

`servo_state`: `std_msgs/String`, JSON на каждом успешном control tick:

| Поле | Значение |
|---|---|
| `q_target` | Последняя исходная цель retargeting, градусы, 20 значений |
| `q_proposed` | Предложение limiter до current guard, градусы |
| `q_cmd` | Полная принятая backend команда после защиты, градусы |
| `q_measured` | Последний доступный raw measured snapshot без фильтрации, градусы |
| `tracking_error` | `abs(q_cmd-q_measured)`, градусы |
| `source_monotonic_s` | Время начала тика по monotonic clock |
| `dt`, `actual_rate_hz` | Интервал между тиками и `1/dt` |
| `limiter_dt` | Интервал, использованный rate limiter |
| `max_step_deg` | Максимум `abs(q_cmd[k]-q_cmd[k-1])` по 20 суставам |
| `first_tick` | Первый тик после initialization/hold: dt принят равным 1/rate |

Для средней реальной частоты использовать `(N-1)/(t_last-t_first)` по
`source_monotonic_s`, исключая паузы/disarm и первые тики. `ros2 topic hz`
измеряет частоту доставки ROS сообщений; интервалы из JSON точнее отражают
control loop. В DISARMED, tracking grace и после timeout servo_state не выходит.
Measured snapshot может быть асинхронным; его возраст виден в diagnostics.

Эти логи подтверждают передачу в backend mailbox, а не физическое достижение
позы и не отдельный успешный вызов `MoveServoJoint`. Частоту именно SDK-вызовов
нужно измерять в C++-адаптере/профилировщике; `communication_rate_hz` описывает
SDK communication callbacks, не Python servo loop и не количество MoveServoJoint.

## Изменённые файлы

- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/servo_controller.py` — новый limiter/state/telemetry/warning.
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/dg5f.py` — выбор controller и guarded servo submission.
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/config_dg5f.py` — параметры режима и валидация.
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/ros_bridge_node.py` — steady timer, downstream guard, публикация telemetry.
- `src/lerobot_robot_dg5f/config/bridge.params.yaml` — новые параметры, default legacy.
- `src/dg5f_unity_teleop/launch/unity_dg5f.launch.py` — прокидывание control parameters только в bridge.
- `scripts/stack.sh` — переменные окружения для обоих launch paths.
- `src/lerobot_robot_dg5f/test/test_servo_controller.py`, `test_servo_bridge.py` — новые проверки.
- `src/lerobot_robot_dg5f/README.md`, этот отчёт — документация.

Изначальные незакоммиченные изменения IsaacLab и object-contact тестов не входят
в эту работу и оставлены на месте.

## Проверки

Новые тесты: 16 passed. Реальный ROS timer с mock backend при target 10 Hz:
91 полная отправка за ~1.5 s, **59.967 Hz**, max step **0.500000°**.
Проверены persistent state при неподвижном feedback, смена направления,
малые шаги, limits/fixed joint, scheduler pause, invalid input, отказ backend,
замена target без очереди, остановка по timeout, current freeze/trip, warning
без disarm и отсутствие отправки в DISARMED.

Fake SDK: PASS (без libDGSDK и сетевого соединения). Launch `--show-args`:
новые параметры доступны, default legacy. `git diff --check`, `bash -n` пройдены.

Общий набор LeRobot: **136 passed, 1 failed**. Он содержит существующее падение
`test_contact_and_rising_current_decrease_convergence_before_soft_threshold`:
`2.5970459272 < 4.8819591979/2` не выполняется. Оно воспроизведено отдельно
на архиве чистого исходного `HEAD=9490086` с теми же числами. Guard и этот тест
не менялись. Также guard выдаёт существующий RuntimeWarning в расчёте velocity.
Физическая рука в новом servo-режиме не запускалась; измерение 59.967 Hz относится
только к mock backend.


Unity adapter: **6 passed**. Retargeting: **17 passed, 2 failed** — оба
параметризованных случая `test_retarget_velocity_cap_can_be_disabled_without_removing_fault_map`.
Тестовый `SimpleNamespace` не содержит `_publish_contact` и `_warn_throttled`.
Оба падения также воспроизведены на чистом исходном `HEAD=9490086`.
Файлы retargeting не менялись.
