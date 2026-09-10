# DG5F LeRobot Retargeting

Docker-версия V2 для цепочки:

```text
Meta Quest 3 / RSL
        |
        v
ROS-TCP-Endpoint -> 21 landmark -> Hybrid retargeting
                                      |
                                      v
                            /dg5f/joint_command
                              raw target, radians
                                      |
                         +------------+-------------+
                         |                          |
                         v                          v
                 MuJoCo visualization       ROS -> LeRobot adapter
                                                    |
                                                    v
                                           Current guard v2
                                                    |
                                                    v
                                          PositionCommandShaper
                                                    |
                                                    v
                                      dg5f_python -> DGSDK
                                                    |
                                                    v
                                    MoveServoJoint -> physical DG5F
```

Проект использует ROS 2 Humble, приложение
[`unity_ros_teleoperation`](https://github.com/leggedrobotics/unity_ros_teleoperation)
на Quest, ROS-TCP-Endpoint, Hybrid/Vector/DexPilot, MuJoCo и LeRobot `0.4.4`.
Плагин `lerobot_robot_dg5f` реализует стандартный интерфейс LeRobot `Robot` и
добавляет backend для Tesollo SDK. Локальный ROS на хосте не нужен.

ROS здесь является message bus для Quest, ретаргетинга, MuJoCo и telemetry.
Моторы не управляются через `ros2_control`: физическая кисть получает position
setpoint напрямую через Python API `dg5f_python.set_target_position()`.

Активный профиль — `direct_guarded + current_guard_v2`: без сглаживания обычного
tracking, с шагом до 5° и токовым ограничением движения в сторону нагрузки.
Экспериментальные compliance/contact-ограничения отключены от управления;
contact/FK остаются только в диагностике. ARM blend и tracking grace/resume сохранены.
Точные параметры и границы отката:
[docs/CURRENT_GUARD_V2_BASELINE.md](docs/CURRENT_GUARD_V2_BASELINE.md).
Reference восстановлен из ZIP + `dg5f_current_guard_v2.patch` для запуска
08.09 12:49 и закреплён тегом `baseline/current-guard-v2-2026-09-08`.

## Неисправный сустав мизинца

На физическом прототипе не работает первый сустав мизинца. Поэтому
`rj_dg_5_1` (индекс `16`, семнадцатый элемент команды) всегда зафиксирован в
нулевом положении. Остальные суставы мизинца — `rj_dg_5_2`, `rj_dg_5_3` и
`rj_dg_5_4` — продолжают двигаться.

Защита применяется на трёх уровнях:

1. Ретаргетер принудительно записывает `0.0 rad` в ROS-команду. Поэтому та же
   неподвижность видна в MuJoCo.
2. LeRobot `send_action()` повторно фиксирует сустав в `0 deg` перед отправкой.
3. Низкоуровневая обёртка DGSDK также фиксирует элемент `16`.

Настройки находятся в:

- `src/dg5f_teleop/config/retarget.params.yaml` — радианы для ROS/MuJoCo;
- `src/lerobot_robot_dg5f/config/bridge.params.yaml` — градусы для LeRobot;
- `vendor/tesollo_control/dg_control/control.cpp` — последний защитный уровень.

Если физическое безопасное положение этого сустава не ноль, нужно согласованно
изменить значение во всех трёх местах: ROS использует радианы, DGSDK — градусы.

## Структура

```text
hand_hybryd_retargeting/
├── compose.yaml
├── docker/                         # автономный ROS 2 + LeRobot образ
├── models/dg5f/                    # модель и mesh-файлы DG5F
├── scripts/stack.sh                # сборка, запуск, тесты и управление
├── src/
│   ├── dg5f_teleop/                # Hybrid/Vector/DexPilot и MuJoCo
│   ├── dg5f_unity_teleop/          # RSL adapter и общий launch
│   ├── lerobot_robot_dg5f/         # Robot, command shaper и ROS bridge
│   ├── ros_tcp_endpoint/
│   └── vr_haptic_msgs/
└── vendor/tesollo_control/         # DGSDK и Python-обёртка
```

## Требования

- Linux, Docker Engine и Docker Compose v2;
- X11 для окна MuJoCo;
- Meta Quest 3 с Developer Mode и установленным RSL;
- `adb` для USB reverse и показа экрана Quest;
- Ethernet-соединение с DG5F для запуска настоящей кисти.

На Ubuntu хостовые утилиты можно установить командой:

```bash
sudo apt update
sudo apt install adb curl netcat-openbsd x11-xserver-utils
```

## Первая сборка и автоматическая проверка

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
bash scripts/stack.sh setup
bash scripts/stack.sh lerobot-check
```

`setup` собирает Docker-образ и пять ROS-пакетов, запускает unit-тесты и
сквозной smoke-тест без Quest и физической кисти. В результате проверки плагина
должно появиться:

```text
dg5f registered: True
actions: 20 observations: 80
```

## Безопасный запуск с Quest и MuJoCo

Не запускайте одновременно этот и старые проекты: они используют одинаковые
имена ROS-узлов и стандартный TCP-порт `10000`.

Подключите Quest по USB и убедитесь, что `adb devices -l` показывает `device`:

```bash
bash scripts/stack.sh usb
bash scripts/stack.sh gui-on
bash scripts/stack.sh launch 10000 hybrid
```

В RSL укажите:

```text
Protocol: ROS 2 / TCP
IP:       127.0.0.1
Port:     10000
```

Затем нажмите `Start Connection`. Команда `launch` всегда использует безопасный
mock-backend LeRobot: физическая кисть не подключается. Для запуска без окна:

```bash
bash scripts/stack.sh headless 10000 hybrid
```

Доступные режимы ретаргетинга:

```bash
bash scripts/stack.sh launch 10000 hybrid    # рекомендуемый режим
bash scripts/stack.sh launch 10000 vector
bash scripts/stack.sh launch 10000 dexpilot
```

Для Wi-Fi вместо `usb` выполните `bash scripts/stack.sh host-ip`, укажите
выведенный IP компьютера в RSL и оставьте порт `10000`.

## Запуск физической DG5F

Сначала закройте DGManager, старые драйверы и другие процессы, подключённые к
кисти. Затем проверьте сеть, не отправляя команд движения:

```bash
ping -c 3 169.254.186.72
nc -vz -w 3 169.254.186.72 502
ip -brief address
```

Если Ethernet-интерфейс не получил адрес в той же link-local сети, назначьте
ему свободный адрес, например:

```bash
sudo ip address add 169.254.186.73/16 dev ИМЯ_ETHERNET_ИНТЕРФЕЙСА
```

Можно отдельно проверить импорт SDK и прочитать 20 суставов. `sdk-check` не
вызывает `set_target_position`; дополнительно он выводит температуры всех
приводов и проверяет аппаратный предел `65 °C`:

```bash
bash scripts/stack.sh sdk-check 169.254.186.72
```

Если хотя бы один привод имеет температуру `65 °C` или выше, не запускайте
движение: DGSDK при этом блокирует обработку position-команд. Дайте кисти
остыть и проверьте указанный привод.

Положите кисть в безопасное положение, освободите рабочую область и запустите:

```bash
bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
```

Запускайте только одну из команд `launch`, `headless`, `hardware` или
`hardware-headless`. Скрипт теперь откажется запускать второй pipeline, чтобы
не было двух одинаковых ROS-сервисов и конфликта TCP-порта.

Этот режим подключает настоящий backend, ждёт свежую начальную позицию, но
оставляет передачу движения `DISARMED`. Если позицию получить не удалось за
`2 s`, hardware backend завершится и включить движение будет невозможно. В
другом терминале проверьте состояние:

Низкоуровневая обёртка **не вызывает `MoveServoJoint()` вообще до первой
реальной position-команды после `arm`**. Pre-SystemStart callback старого DGSDK
может содержать нулевую структуру, поэтому он больше не считается физической
телеметрией. Startup ждёт model/communication discovery, выполняет `SystemStart`,
а затем принимает только post-SystemStart telemetry и по ней инициализирует
LeRobot. Keepalive разрешается только после первой реально принятой servo-команды
и прекращается при `disarm`/tracking timeout. Внутренний adapter loop работает
стабильно на `200 Hz`, а входные setpoint формируются LeRobot с частотой `50 Hz`.

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
bash scripts/stack.sh shell
source /workspace/install/setup.bash
ros2 topic echo /dg5f/lerobot/connected --once
ros2 topic echo /dg5f/lerobot/armed --once
ros2 topic echo /dg5f/tracking_ok --once
ros2 topic echo /dg5f/joint_command --once
ros2 topic echo /dg5f/lerobot/commanded_joint_states --once
ros2 topic echo /dg5f/lerobot/joint_states --once
```

Перед включением ожидается:

- `connected: true`;
- `armed: false`;
- `tracking_ok: true` при видимой правой руке;
- позиция `rj_dg_5_1` в `/dg5f/joint_command` равна `0.0`.

Только после проверки включите отправку команд с хоста:

```bash
bash scripts/stack.sh arm
```

Успешный ответ сервиса означает только, что проверка перед включением прошла.
Если SDK затем обнаружит неисправность, bridge автоматически снова выставит
`armed: false`. Поэтому после `arm` проверьте состояние и температуры:

```bash
ros2 topic echo /dg5f/lerobot/armed --once
ros2 topic echo /dg5f/lerobot/temperatures --once
```

Для немедленного прекращения новых команд:

```bash
bash scripts/stack.sh disarm
```

При потере tracking или отсутствии свежей команды более `0.35 s` bridge
автоматически переходит в `DISARMED`. Версия без окна MuJoCo запускается так:

```bash
bash scripts/stack.sh hardware-headless 10000 hybrid 169.254.186.72
```

> `disarm` останавливает отправку новых целевых положений, но не заменяет
> аппаратную аварийную остановку и отключение питания.

## Основные ROS-топики и сервис

| Имя | Назначение |
|---|---|
| `/quest/hand_pose` | 21 точка руки из RSL |
| `/hands/right/landmarks` | нормализованные landmarks для ретаргетера |
| `/dg5f/joint_command` | 20 целевых углов в радианах, `rj_dg_5_1=0` |
| `/dg5f/joint_states` | фактическое состояние MuJoCo |
| `/dg5f/tracking_ok` | состояние watchdog трекинга |
| `/dg5f/lerobot/commanded_joint_states` | команда LeRobot после safety-обработки |
| `/dg5f/lerobot/joint_states` | состояние настоящего или mock-backend |
| `/dg5f/lerobot/temperatures` | температура 20 приводов |
| `/dg5f/lerobot/connected` | соединение с backend |
| `/dg5f/lerobot/armed` | разрешена ли передача движения |
| `/dg5f/lerobot/diagnostics` | единый snapshot состояния DGSDK (`DiagnosticArray`) |
| `/dg5f/debug_marker` | пользовательская метка только для debug recorder |
| `/dg5f/lerobot/enable` | сервис `std_srvs/SetBool` для пассивного arm/disarm |
| `/dg5f/lerobot/recover` | явный `std_srvs/Trigger` recovery DGSDK; после него остаётся DISARMED |

`/dg5f/joint_command` показывает raw-результат ретаргетинга, MuJoCo следует
ему напрямую. `/dg5f/lerobot/commanded_joint_states` показывает сглаженный
setpoint, реально принятый physical/mock backend. `/dg5f/lerobot/joint_states`
содержит измеренную позицию и скорость. Поле `JointState.effort` содержит
telemetry тока мотора, а не рассчитанный torque.

Схема телеметрии v2 (исправлена 07.09.2026): SDK `joint: float[20]` — градусы,
`current: int[20]` — **мА**, `velocity: int[20]` — **rpm**,
`temperature: float[20]` — °C. Это единицы из `DGDataTypes.h`.
Раньше `memcpy(int[], float[])` искажал ток и скорость: значения порядка
`4e-43` в старых архивах нельзя использовать как физические измерения.
Теперь преобразование поэлементное; LeRobot `.vel` = rpm × 6 (deg/s),
ROS `JointState.velocity` = rpm × 2π / 60 (rad/s). `JointState.effort`
содержит мА, **не Н·м**. В CSV `raw_current_*` — мА, `raw_velocity_*` — rpm,
`current_*` — мА, `velocity_*` — rad/s, положения и `error_*` — rad.

В diagnostics доступны причины `disarm_reason` и `motion_ready_reason`,
а также возраст команды, tracking, телеметрии и communication callback в мс.
`last_sdk_packet_age_ms` — возраст `ReceivedGripperData` callback: SDK
не предоставляет время каждого TCP-пакета, поэтому это явно помеченный proxy.
Настоящие времена сетевых пакетов сохраняются в pcap. `temperature_safe`
описывает только известную измеренную температуру <65 °C; потеря связи
не превращает последнюю температуру 42.5 °C в перегрев.

Причины disarm: `USER_REQUEST`, `COMMAND_TIMEOUT`, `TRACKING_TIMEOUT`,
`SDK_COMMAND_REJECTED`, `SDK_DISCONNECTED`, `SDK_NOT_MOTION_READY`,
`TELEMETRY_STALE`, `TEMPERATURE_LIMIT`, `SYSTEM_NOT_STARTED`,
`RECOVERY_FAILED`, `INTERNAL_ERROR`; нормальное состояние — `NONE`.
Причины неготовности SDK: `DISCONNECTED`, `CONTROL_THREAD_STOPPED`,
`TELEMETRY_STALE`, `TEMPERATURE_UNSAFE`, `SYSTEM_NOT_STARTED`,
`MOTION_RESULT_ERROR`, `RECOVERY_REQUIRED`; готовность — `READY`.

При disconnect control thread остаётся жив до `stop()`, а вывод блокируется.
После reconnect старые цели и старый keepalive не отправляются. **`arm` теперь
строго пассивный:** он только проверяет свежие tracking/target, transport,
control thread, telemetry, температуру, `system_started` и `motion_ready`.
`arm` не вызывает `SystemStart`, `SystemStop`, `MoveServoJoint`, reconnect и не
сбрасывает command shaper. Если SDK-сеанс требует восстановления, `arm`
завершится ошибкой и физически ничего делать не должен.

Восстановление вынесено в отдельную явную команду:

```bash
bash scripts/stack.sh recover
```

`recover` всегда оставляет high-level output в `DISARMED` и сам не вызывает
`MoveServoJoint`. Он ждёт живой transport/model-discovery, при необходимости
перезапускает только `SystemStart`, затем принимает только новую
post-SystemStart telemetry и возвращает измеренную pose. После успешного
recovery shaper синхронизируется с этой pose, старые команды и low-level
keepalive очищены, и для начала teleop требуется отдельный
`bash scripts/stack.sh arm`. Обычный teleop никогда не переинициализирует
shaper из feedback.

## Использование как LeRobot-плагина

Имя пакета следует соглашению сторонних плагинов LeRobot —
`lerobot_robot_dg5f`, тип конфигурации — `dg5f`. Значения позиции в API
LeRobot выражены в градусах, поскольку так работает DGSDK.

Без физической кисти API можно проверить mock-backend:

```python
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()

from lerobot_robot_dg5f import Dg5f, Dg5fConfig

robot = Dg5f(Dg5fConfig(id="dg5f_test", backend="mock"))
robot.connect()
observation = robot.get_observation()
action = {f"rj_dg_{finger}_{joint}.pos": 0.0
          for finger in range(1, 6) for joint in range(1, 5)}
effective_action = robot.send_action(action)
robot.disconnect()
```

Для прямого доступа к реальной кисти создайте ту же реализацию с hardware
backend. `connect()` только запускает SDK и получает initial feedback; первое
движение начинается при первом `send_action()`, поэтому прямой API должен иметь
внешний safety gate пользователя:

```python
robot = Dg5f(Dg5fConfig(
    id="dg5f_real",
    backend="tesollo",
    ip="169.254.186.72",
    port=502,
    slave_id=1,
))
robot.connect()
# После ручной проверки рабочей области:
effective_action = robot.send_action(action)
observation = robot.get_observation()
robot.disconnect()
```

`send_action()` возвращает effective action после joint limits, fault map,
фильтрации и динамических ограничений. Это именно setpoint, принятый backend,
а не исходный target. Observation содержит измеренные position, velocity,
motor current и temperature; feedback не перезаписывает внутреннюю command
trajectory. Такое разделение подходит для будущего LeRobot dataset:
`action=effective command`, `observation=measured state`.

Пакет готов как hardware plugin и ROS action bridge. Для полноценного запуска
`lerobot-record` отдельно задаются камеры и источник действий: teleoperator,
policy или адаптер ROS-команды.

## Position command shaper

По запросу пользователя ROS pipeline теперь стартует в профиле `direct`:
`control_smoothing=false`, `min_send_step_deg=0`, `max_joint_velocity=0`.
Это убирает прежние программные ограничения `30 deg/s`, `60 deg/s²`,
фильтр shaper и ограничитель `3 rad/s` на выходе ретаргетера.
Новая цель передаётся на ближайшем цикле bridge (50 Hz).
Ни суставные пределы, ни прошивка, P/D gains или температурный предел не меняются.
Физическая скорость всё ещё зависит от приводов, нагрузки и собственных
ограничений DGSDK; совпадение с быстрым движением VR нельзя гарантировать.
Внутренние фильтры Hybrid/DexPilot оставлены прежними.

Для возврата программного сглаживания при запуске:

```bash
DG5F_CONTROL_SMOOTHING=true DG5F_MAX_JOINT_VELOCITY=3.0 \
  bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
```

Параметры скорости ниже действуют только в профиле `smoothed`.
У прямого LeRobot API `Dg5fConfig` прежние defaults сохранены; для direct
нужно явно передать `control_smoothing=False, min_send_step_deg=0.0`.

Основная realtime-логика находится в
`src/lerobot_robot_dg5f/lerobot_robot_dg5f/command_shaper.py`, а не в ROS
bridge. Поэтому одинаковое поведение получают ROS, прямой LeRobot API,
teleoperator и будущая policy.

После соединения shaper один раз инициализируется свежей фактической позицией.
Дальше authoritative state — его собственные `command_pose_deg`,
`velocity_deg_s` и `last_sent_pose_deg`. Асинхронный measured feedback
используется только для observation и диагностики.

| Параметр | По умолчанию |
|---|---:|
| `command_rate_hz` | `50 Hz` |
| `max_speed_deg_s` | `30 deg/s` |
| `max_accel_deg_s2` | `60 deg/s²` |
| `response_time_s` | `0.15 s` |
| `filter_tau_s` | `0.05 s` |
| `target_deadband_deg` | `0.20 deg` |
| `min_send_step_deg` | `0.0 deg` (ROS direct profile) |
| `max_dt_s` | `0.05 s` |
| `initial_feedback_timeout_s` | `2.0 s` |
| `telemetry_drain_limit` | `16 samples` |

Параметры находятся в
`src/lerobot_robot_dg5f/config/bridge.params.yaml`. Старый
`max_relative_target_deg` оставлен только для совместимости конфигурации и
больше не является механизмом сглаживания. Mock проходит через тот же shaper,
что и реальная кисть.

## Настройка Hybrid

Основные параметры находятся в
`src/dg5f_teleop/config/retarget.params.yaml`. Изменить контактную коррекцию
можно одной командой:

```bash
# blend start full max_corr lateral correction_alpha release_alpha
bash scripts/set_hybrid_params.sh 0.80 0.055 0.025 0.45 0.12 0.35 0.18
```

После изменения перезапустите launch. При `--symlink-install` пересобирать
проект для YAML не требуется.

## Диагностика и тесты

Запись демонстраций в официальный **LeRobotDataset** (отдельный пассивный узел,
без второго соединения с кистью):
[`docs/LEROBOT_DATASET_RECORDING.md`](docs/LEROBOT_DATASET_RECORDING.md).
Гайд содержит workflow `dataset-record/start/finish/discard/check`, камеры,
обработку tracking loss и offline mock-тест. Текущая логика управления не меняется.

```bash
bash scripts/stack.sh topics
bash scripts/stack.sh logs
bash scripts/stack.sh test
bash scripts/stack.sh smoke
bash scripts/stack.sh lerobot-check
```

Если запуск сообщает `Address already in use`, уже остался другой endpoint на
том же порту. Завершите старый launch через `Ctrl+C`; не запускайте mock
`launch` перед `hardware`. Если после включения появляется сообщение
`rejected the position target twice`, bridge безопасно разоружает кисть и
выводит состояние низкоуровневого control loop. Сначала
запустите `sdk-check`: температура `>= 65 °C` объясняет блокировку; при
нормальной температуре нужно проверять Ethernet и внутренний control loop SDK.

В smoke-тесте автоматически проверяется весь путь от фальшивого RSL TCP-пакета
до MuJoCo и mock LeRobot, включая нулевое значение `rj_dg_5_1` во всех выходах.

### Пассивная запись аппаратного сбоя

`debug-record` не создаёт второй экземпляр `DGApi`, не подключается к порту
кисти `502` и не вызывает arm/disarm или команды движения. Это отдельный
ROS-процесс: он только слушает топики уже работающего pipeline и читает
счётчики Linux из `/sys/class/net/<interface>/statistics`.

Эксперимент запускается в трёх терминалах.

Терминал 1 — pipeline (он стартует в состоянии `DISARMED`):

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
```

Терминал 2 — рекордер до начала движения, чтобы сохранить 5–10 секунд перед
возможным fault:

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
bash scripts/stack.sh debug-record
```

Терминал 3 — после проверки MuJoCo включить настоящую кисть:

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
bash scripts/stack.sh arm
```

В момент касания или предполагаемой коллизии можно поставить метку, не влияющую
на управление:

```bash
bash scripts/stack.sh debug-mark collision
```

После эксперимента нажмите `Ctrl+C` только в терминале рекордера. Он корректно
закроет файлы и напечатает два пути. Результат находится в:

```text
debug_runs/YYYY-mm-dd_HH-MM-SS/
debug_runs/dg5f_debug_YYYY-mm-dd_HH-MM-SS.tar.gz
```

В каталоге будут `manifest.json`, `timeline.csv`, `events.jsonl`,
`ros_topics.txt`, `system.txt`, `network.log`, `README.txt` и `summary.txt`.
CSV пишется с частотой `30 Hz` по монотонным часам и содержит для всех 20
суставов raw target, effective LeRobot command, последнюю low-level команду
DGSDK, измеренные position/velocity/current/temperature и tracking error.
Отдельно сохраняются `transport_connected`, `control_thread_alive`,
`motion_ready`, `system_started`, `telemetry_valid`, `temperature_safe`, частота
связи, коды DGSDK, `DiagnosisSystem` и счётчики disconnect/reconnect. Временно
отсутствующие потоки записываются как `NaN`/пустое значение/`false`, запись не
останавливается.

Опциональный постоянный ping включается `debug-record --ping`. Захват TCP
выключен по умолчанию. Для следующего эксперимента рекомендуется:

```bash
# Установить на HOST один раз, если утилит нет:
sudo apt install iputils-ping tcpdump
# Вместо обычного debug-record:
bash scripts/stack.sh debug-record-network
```

То же самое: `bash scripts/stack.sh debug-record --ping --tcpdump`.
Host helper `scripts/dg5f_network_capture.sh` запрашивает sudo только для
пассивного tcpdump. Он запускает ping `-D -i 0.1`, tcpdump с фильтром
`host 169.254.186.72 and tcp port 502` и записывает sysfs counters, carrier,
operstate и изменения link на host. Если tcpdump/права недоступны, запись
ROS продолжается: результат будет явно указан в `network_status.json` и
`tcpdump.log`. Проверяйте `tcpdump_enabled: true`, если нужен анализ TCP.
Для другого интерфейса: `DG5F_NETWORK_INTERFACE=... bash scripts/stack.sh debug-record-network`.
Все файлы попадают в один эксперимент и один tar.gz **после закрытия pcap**.

При fault сначала сохраните состояние, не перезапуская pipeline:

```bash
bash scripts/stack.sh dg-status
bash scripts/stack.sh debug-mark after_fault
```

Для отдельной проверки восстановления дождитесь возврата транспорта и
освободите кисть от контакта. Сначала выполните **явный recovery**:

```bash
bash scripts/stack.sh recover
```

Он может перезапустить только Tesollo servo session, поэтому выполняйте его
только когда кисть свободна и рядом нет рук/объектов. Recovery обязан сначала
получить свежую measured pose и после завершения всё равно оставить bridge
`DISARMED`. Затем дождитесь нового target от Quest и только после этого:

```bash
bash scripts/stack.sh arm
```

Оставьте recorder работающим: он запишет
`RECOVERY_STARTED/SUCCEEDED/FAILED`, а затем отдельный `ARM_REQUESTED/ARMED`.
Если recovery сейчас не проверяете, просто завершите recorder через Ctrl+C
после `dg-status`.
Не запускайте `sdk-check` одновременно с physical pipeline: он создаёт SDK-клиент.

CSV v2 сохраняет старые колонки и добавляет причины, ages, raw ток/скорость,
wall time. `events.jsonl` содержит snapshot при disconnect/reconnect, включая
частоту связи 100/500/1000 мс назад. Snapshot — ближайший принятый ROS-снимок;
для точных времен TCP используйте pcap. Manifest хранит пары wall/monotonic
времени, единицы, конфигурацию из файла и полученные runtime-параметры bridge.
`diagnostics_age_ms` позволяет отличить остановку ROS-потока от неизменных
значений SDK. Summary группирует disarm по причинам и показывает фактическую
доступность сетевого захвата. Старые hardware-архивы сохраняются без изменений.

Ни один из режимов записи не меняет адрес,
маршрут, NetworkManager или состояние сетевого интерфейса.

Показ экрана Quest и запись rosbag:

```bash
bash scripts/stack.sh quest-view 30
bash scripts/stack.sh record lerobot_test
```

Остановка контейнера:

```bash
bash scripts/stack.sh stop
```

Происхождение и лицензии зависимостей перечислены в
[`THIRD_PARTY.md`](THIRD_PARTY.md).

### Guarded direct tracking

Physical Tesollo mode uses direct tracking without the old 30 deg/s acceleration ramp, but it no longer permits a one-frame 50-90 degree jump. On every ARM the software command starts from the last backend-accepted pose and blends toward the live VR target for 0.70 s. After that, direct mode limits each 50 Hz command step to 5 degrees (about 250 deg/s equivalent). These are software setpoint guards; joint limits, the fixed broken pinky joint, and Tesollo thermal protection remain active.

Tune without rebuilding the image:

```bash
DG5F_STARTUP_BLEND_S=0.70 DG5F_MAX_DIRECT_STEP_DEG=5.0 \
  bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
```

Do not set a very large direct step until the physical fault mechanism is understood.
