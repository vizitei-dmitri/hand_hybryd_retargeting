# DG5F Hybrid Retargeting — Quest/RSL → ROS 2 → MuJoCo

Самостоятельный Docker-проект для телоуправления правой пятипалой кистью
Tesollo DG5F по 21 landmark руки с Meta Quest 3.

Проект рассчитан на уже установленное на Quest приложение
[`leggedrobotics/unity_ros_teleoperation`](https://github.com/leggedrobotics/unity_ros_teleoperation)
(RSL) и использует ROS 2 Humble, ROS-TCP-Endpoint, `dex-retargeting` и MuJoCo.

Hybrid — режим по умолчанию. Он сохраняет форму пальцев с помощью Vector и
добавляет ограниченную DexPilot-поправку только около thumb-to-finger контакта.

## Что входит в проект

```text
hand_hybryd_retargeting/
├── compose.yaml
├── docker/                       # автономный ROS 2 Humble образ
├── models/dg5f/                  # только модель и mesh-файлы DG5F
├── scripts/
│   ├── stack.sh                  # все команды управления
│   ├── set_hybrid_params.sh      # изменение коэффициентов Hybrid
│   ├── unity_smoke_inside.sh     # end-to-end тест без Quest
│   └── fake_unity_tcp_client.py
└── src/
    ├── dg5f_teleop/              # Hybrid, Vector, DexPilot, MuJoCo bridge
    ├── dg5f_unity_teleop/        # ManoLandmarks[21] adapter и launch
    ├── ros_tcp_endpoint/         # ROS-TCP-Endpoint main-ros2 + исправления
    └── vr_haptic_msgs/           # сообщения RSL
```

Здесь нет V1 Hand Tracking Streamer, Unity-проекта, APK, старых bag-файлов,
`build/install/log` и моделей манипуляторов. RSL остаётся на Quest.

## Как работает Hybrid

```text
Quest ManoLandmarks[21]
          │
          ▼
  нормализация кисти
          │
          ├──────────────► Vector ───────► q_vector
          │                 20 костей          │
          │                                   ├──► q_DG5F
          └──────────────► DexPilot ─► Δq ─EMA┘
                         только возле pinch
```

Это два последовательных оптимизатора, а не единая функция потерь:

1. Vector сопоставляет все 20 направлений костей из 21 landmark и задаёт
   анатомическую форму.
2. Расстояние от большого пальца до остальных кончиков включает плавный вес
   контакта.
3. DexPilot стартует из текущего Vector-решения.
4. `q_dexpilot - q_vector` ограничивается по суставам и сглаживается отдельно.
5. Основной Vector-сигнал не проходит через дополнительный contact-фильтр,
   поэтому обычное движение остаётся отзывчивым.

Ограничение текущей версии: Hybrid смешивает решения в пространстве суставов;
в нём пока нет единой collision-aware функции потерь.

## Требования

- Linux с Docker Engine и Docker Compose v2;
- X11 для окна MuJoCo;
- Meta Quest 3 с включённым Developer Mode;
- RSL `unity_ros_teleoperation` на Quest;
- `adb`, если используется USB reverse или зеркало Quest;
- `curl`, `tar` и `sha256sum` для автоматической загрузки `scrcpy`.

Для Ubuntu хостовые утилиты можно установить так:

```bash
sudo apt update
sudo apt install adb curl x11-xserver-utils
```

Python, ROS 2, MuJoCo, Pinocchio и `dex-retargeting` устанавливаются внутри
Docker. Локальный ROS на хосте не нужен.

## Первый запуск

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
bash scripts/stack.sh setup
```

`setup` последовательно собирает образ, запускает контейнер, собирает четыре
ROS-пакета, запускает тесты и выполняет end-to-end smoke-тест без Quest.
Smoke-тест использует отдельные `ROS_DOMAIN_ID=91` и TCP-порт `10001`.

## Подключение Quest

Не запускайте одновременно старый проект `hand` и этот проект: у них одинаковые
имена ROS-узлов и стандартный TCP-порт `10000`.

### Вариант A: USB — проще для первого теста

Подключите Quest кабелем и убедитесь, что `adb devices -l` показывает состояние
`device`, затем:

```bash
bash scripts/stack.sh usb
bash scripts/stack.sh gui-on
bash scripts/stack.sh launch 10000 hybrid
```

В RSL на Quest укажите:

```text
Protocol: ROS 2 / TCP
IP:       127.0.0.1
Port:     10000
```

Нажмите `Start Connection`. `adb reverse` перенаправит Quest
`localhost:10000` на Docker endpoint хоста.

### Вариант B: Wi‑Fi

```bash
bash scripts/stack.sh host-ip
bash scripts/stack.sh gui-on
bash scripts/stack.sh launch 10000 hybrid
```

В RSL задайте IP компьютера в сети Quest и порт `10000`. Для текущей сети это
может быть, например, `10.100.20.192`, но адрес необходимо проверять снова после
переподключения.

## Проверка потока

Оставьте launch работающим, откройте второй терминал и выполните:

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
bash scripts/stack.sh topics
```

| Топик | Тип | Назначение |
|---|---|---|
| `/quest/hand_pose` | `vr_haptic_msgs/msg/ManoLandmarks` | 21 точка из RSL |
| `/hands/right/landmarks` | `geometry_msgs/msg/PoseArray` | вход ретаргетера |
| `/dg5f/joint_command` | `trajectory_msgs/msg/JointTrajectory` | 20 целевых углов DG5F |
| `/dg5f/joint_states` | `sensor_msgs/msg/JointState` | состояние MuJoCo |
| `/dg5f/tracking_ok` | `std_msgs/msg/Bool` | watchdog трекинга |

Проверка частоты и состояния:

```bash
bash scripts/stack.sh shell
source /workspace/install/setup.bash
ros2 topic hz /quest/hand_pose
ros2 topic echo /dg5f/tracking_ok --once
ros2 topic echo /dg5f/joint_command
```

Ожидаемая частота Quest 3 — примерно 60–72 Гц, а `tracking_ok` при видимой
правой руке должен быть `true`.

## Режимы ретаргетинга

```bash
# Рекомендуемый: Vector-форма + DexPilot contact correction
bash scripts/stack.sh launch 10000 hybrid

# Только анатомическая форма по направлениям фаланг
bash scripts/stack.sh launch 10000 vector

# Только fingertip-ориентированный DexPilot
bash scripts/stack.sh launch 10000 dexpilot
```

Одновременно запускайте только один launch. Для смены режима остановите текущий
`Ctrl+C`, затем запустите другой.

## Текущие параметры Hybrid

Файл: `src/dg5f_teleop/config/retarget.params.yaml`.

| Параметр | Значение | Смысл |
|---|---:|---|
| `scaling_factor` | `1.2` | масштаб human-векторов для DG5F |
| `hybrid_contact_start` | `0.055` м | начало DexPilot-поправки |
| `hybrid_contact_full` | `0.025` м | полный вес контакта |
| `hybrid_max_blend` | `0.80` | максимальный вес DexPilot |
| `hybrid_max_joint_correction` | `0.45` рад | предел поправки flexion/thumb |
| `hybrid_lateral_max_correction` | `0.12` рад | предел боковой поправки обычных пальцев |
| `hybrid_correction_alpha` | `0.35` | EMA при входе/удержании контакта |
| `hybrid_release_alpha` | `0.18` | плавное затухание после контакта |
| `max_joint_velocity` | `3.0` рад/с | финальный ограничитель скорости |

Изменить семь основных коэффициентов одной командой:

```bash
# blend start full max_corr lateral correction_alpha release_alpha
bash scripts/set_hybrid_params.sh 0.80 0.055 0.025 0.45 0.12 0.35 0.18
```

После изменения остановите текущий launch и запустите его снова. При
`--symlink-install` повторная сборка для YAML не требуется.

- pinch не доходит: немного увеличить `hybrid_max_blend` или
  `hybrid_max_joint_correction`;
- палец уходит вбок: уменьшить `hybrid_lateral_max_correction`;
- correction дрожит: уменьшить `hybrid_correction_alpha`;
- контакт включается поздно: увеличить `hybrid_contact_start`;
- рука в целом запаздывает: увеличить `max_joint_velocity`, проверяя рывки.

## MuJoCo

По умолчанию используются сервоприводы `Kp=40`, `Kd=0.5` и режим self-collision
`tip_only`: посторонние фаланги не блокируют pinch, но контакты между кончиками
и с внешними объектами сохраняются.

```bash
DG5F_MUJOCO_KP=30 \
DG5F_MUJOCO_KD=0.4 \
DG5F_MUJOCO_SELF_COLLISION=full \
bash scripts/stack.sh launch 10000 hybrid
```

- `tip_only` — рекомендуемый для телеприсутствия;
- `full` — все физические столкновения кисти;
- `off` — self-collision кисти выключен, контакты с окружением сохраняются.

## Запись и зеркало Quest

```bash
bash scripts/stack.sh record hybrid_test
bash scripts/stack.sh quest-view 30
```

Bag появится в `bags/hybrid_test`. `quest-view` показывает изображение, но не
управляет Quest.

## Частые проблемы

### `Mono IO Layer Error 111` или `Connection refused`

```bash
ss -lntp | grep 10000
bash scripts/stack.sh logs
```

Для USB повторите `bash scripts/stack.sh usb`. Для Wi‑Fi проверьте IP через
`host-ip` и firewall.

### Топик существует, но сообщений нет

Endpoint создаёт publisher при регистрации RSL, даже если landmarks не
поступают. Наденьте Quest, восстановите room tracking и покажите правую руку
камерам. Системное окно «Поиск местоположения в комнате» блокирует hand tracking;
не выбирайте работу без отслеживания.

### MuJoCo не открывает окно

```bash
bash scripts/stack.sh gui-on
echo "$DISPLAY"
ls -l /dev/dri
```

Для проверки без окна:

```bash
bash scripts/stack.sh headless 10000 hybrid
```

### Остановка контейнера

```bash
bash scripts/stack.sh stop
```

Исходники, Docker-образ и bag-файлы при этом не удаляются.

## Сторонние компоненты

Происхождение, ревизии и локальные исправления перечислены в
[`THIRD_PARTY.md`](THIRD_PARTY.md). Лицензии ROS-TCP-Endpoint и
`vr_haptic_msgs` сохранены внутри соответствующих каталогов `src/`.
