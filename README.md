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

Низкоуровневая обёртка не вызывает `MoveServoJoint()` до получения первой
реальной position-команды после `arm`, кроме безопасного keepalive фактической
стартовой позы: Developer Mode Tesollo отключает control session без servo
traffic. Keepalive не использует target Quest и предотвращает расслабление
кисти до `arm`; в `sdk-check` он полностью отключён. Частота, сообщаемая DGSDK,
трактуется как Hz, а внутренний adapter loop ограничен безопасными `200 Hz`;
входные setpoint по-прежнему формируются LeRobot с частотой `50 Hz`.

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
| `/dg5f/lerobot/enable` | сервис `std_srvs/SetBool` для arm/disarm |

`/dg5f/joint_command` показывает raw-результат ретаргетинга, MuJoCo следует
ему напрямую. `/dg5f/lerobot/commanded_joint_states` показывает сглаженный
setpoint, реально принятый physical/mock backend. `/dg5f/lerobot/joint_states`
содержит измеренную позицию и скорость. Поле `JointState.effort` содержит
telemetry тока мотора, а не рассчитанный torque.

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
| `min_send_step_deg` | `0.20 deg` |
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
