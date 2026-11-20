# Запуск: онлайн, офлайн и калибровка

Все команды ниже выполняются **из корня репозитория**. Онлайн здесь означает
живой ввод с камеры или Quest, офлайн — запуск без устройств с синтетическим
вводом. Работа без интернета описана отдельно.

| Что нужно запустить | Инструкция | Что потребуется |
|---|---|---|
| RGB-камера → браузер → mock DG5F | [Онлайн с RGB-камерой](#онлайн-с-rgb-камерой) | USB-камера, Docker |
| RGB-камера → физическая DG5F | [Физическая кисть](#rgb-камера-и-физическая-dg5f) | Камера, DG5F и Ethernet |
| Quest → MuJoCo или DG5F | [Онлайн с Quest 3](#онлайн-с-quest-3) | Quest с Unity-приложением, Docker |
| Браузер без камеры и робота | [Офлайн RGB-демо](#офлайн-без-камеры-и-робота) | Собранный образ RGB-демо |
| Проверка ROS-пайплайна без Quest | [Офлайн ROS](#офлайн-проверка-ros-пайплайна) | Собранный ROS-стек |
| Настройка раскрытой ладони | [Калибровка](#калибровка) | Запущенное RGB-демо, видимая правая рука |
| Два Kinect → параметрическая MANO | [Запуск и калибровка Kinect](../kinect_mano/README.md) | Локальная MANO, калибровка и RGB-D данные |

### Требования

- Linux x86_64, Docker Engine и Docker Compose v2.
- Интернет для первой сборки образов и загрузки зависимостей/моделей.
- Для RGB-демо: доступная V4L2-камера, по умолчанию `/dev/video0`.
  CPU достаточно; ROS, Quest и GPU не требуются.
- Для Quest/ROS: ROS 2 Humble находится в контейнере; для окна MuJoCo нужны
  X11 и `/dev/dri`. Headless отключает окно, но текущий Compose всё равно
  содержит проброс `/dev/dri`.

Проверка установленных инструментов и доступных камер:

```bash
docker --version
docker compose version
ls -l /dev/video*
```

### Онлайн с RGB-камерой

Сначала собрать образ и запустить с виртуальным backend:

```bash
docker compose -f dg5f_camera_demo/compose.yaml build
DG5F_BACKEND=mock docker compose -f dg5f_camera_demo/compose.yaml up -d
```

Открыть [http://127.0.0.1:8080](http://127.0.0.1:8080), показать правую руку
целиком. Должны появиться изображение, скелет и target суставов. Backend
`mock` исполняет команды в памяти. После калибровки нажать **ARM**, чтобы
проверить следование mock-кисти.

Если камера находится на другом устройстве:

```bash
DG5F_BACKEND=mock DG5F_CAMERA_DEVICE=/dev/video2 \
  docker compose -f dg5f_camera_demo/compose.yaml up -d
```

Выбранное устройство внутри контейнера будет `/dev/video0`. Для `nearest`
по умолчанию выбирается самая крупная допустимая правая ладонь; левые руки
игнорируются. Ползунок «Отсечь дальние руки» регулирует минимальный видимый
размер ладони. Значение по умолчанию — 12% короткой стороны кадра.

Проверка состояния и логов:

```bash
curl --fail http://127.0.0.1:8080/api/state
docker compose -f dg5f_camera_demo/compose.yaml logs -f --tail 60
```

Для остановки нажать **DISARM**, затем:

```bash
docker compose -f dg5f_camera_demo/compose.yaml down
```

### RGB-камера и физическая DG5F

После проверки mock-режима подключить DG5F по Ethernet. В примере используется
IP `169.254.186.72`; для своего стенда заменить адрес:

```bash
ip route get 169.254.186.72
DG5F_BACKEND=tesollo DG5F_IP=169.254.186.72 \
  docker compose -f dg5f_camera_demo/compose.yaml up -d
```

В браузере проверить backend `TESOLLO` и состояние hardware `READY`.
Выполнить [калибровку ладони](#калибровка), затем нажать **ARM**.
Сервер запускается в `DISARMED`. Без tracking после ARM кисть остаётся в
`TRACKING_HOLD`; свежая допустимая поза переводит её в `ACTIVE`.

**DISARM** останавливает управление. **RECOVER** используется при ошибке SDK
и оставляет систему в `DISARMED`. Закрытие вкладки не останавливает сервер:
завершать работу через DISARM и `docker compose ... down`.

Если IP доступен с хоста, но недоступен из Docker bridge, использовать запуск
с host networking из [инструкции установки RGB-демо](../dg5f_camera_demo/INSTALL.md).

### Онлайн с Quest 3

Этот режим использует отдельный ROS-контейнер. Первая подготовка собирает образ,
компилирует пакеты, запускает тесты и синтетический smoke-тест:

```bash
bash scripts/stack.sh setup
bash scripts/stack.sh lerobot-check
```

На следующих запусках достаточно:

```bash
bash scripts/stack.sh up
```

Для Quest по USB разрешить ADB-подключение в гарнитуре и выполнить:

```bash
bash scripts/stack.sh usb
bash scripts/stack.sh gui-on
bash scripts/stack.sh launch 10000 hybrid
```

В Unity-приложении Quest указать адрес `127.0.0.1`, порт `10000` и начать
передачу hand tracking. Для Wi-Fi вместо `usb` выполнить `host-ip` и указать
в Unity доступный IP компьютера; порт тот же. Само Unity-приложение должно быть
заранее установлено на Quest.

Ожидаемый результат — движение виртуальной DG5F в MuJoCo. Последний аргумент
выбирает `hybrid`, `vector` или `dexpilot`. Для запуска без окна использовать
`headless` вместо `launch`; источник Quest всё равно необходим.

Для физической DG5F остановить симуляционный запуск через Ctrl+C, затем:

```bash
DG5F_CONTROL_MODE=servo bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
```

В другом терминале, также из корня репозитория:

```bash
bash scripts/stack.sh dg-status
bash scripts/stack.sh arm
```

Переменная `DG5F_CONTROL_MODE=servo` явно выбирает ServoController; без неё
скрипт использует `legacy`. Hardware запускается disarmed.
Для завершения выполнить `disarm`, остановить launch через Ctrl+C и контейнер:

```bash
bash scripts/stack.sh disarm
bash scripts/stack.sh stop
bash scripts/stack.sh gui-off
```

### Офлайн без камеры и робота

Для браузерного RGB-демо уже есть синтетический источник. Сначала собрать образ
командой из раздела RGB, затем остановить предыдущий сервер на порту 8080:

```bash
docker compose -f dg5f_camera_demo/compose.yaml down
docker run --rm --init -p 127.0.0.1:8080:8080 \
  dg5f-camera-demo:local --synthetic --backend mock --no-background-blur
```

Открыть [http://127.0.0.1:8080](http://127.0.0.1:8080) и нажать **ARM**.
Синтетический скелет меняется автоматически; камера и физическая кисть
не подключаются. Остановить сервер через Ctrl+C.

Это демонстрация интерфейса и ретаргетинга на искусственном вводе. Чтение
видеофайлов, replay JSONL и реконструкция MANO из записей двух Kinect в этом
CLI сейчас не реализованы. Параметр `--record` сохраняет control-телеметрию,
но не добавляет режим её воспроизведения.

### Офлайн-проверка ROS-пайплайна

После `setup`, либо при наличии собранного образа и скомпилированных пакетов:

```bash
bash scripts/stack.sh up
bash scripts/stack.sh smoke
```

Тест сам подаёт синтетический Unity TCP-ввод и проверяет цепочку
`21 landmarks → Hybrid → MuJoCo + LeRobot mock`. Quest, камера и DG5F не нужны.
Окно не открывается; тест завершает запущенные им процессы автоматически.
По умолчанию используются ROS domain `91` и TCP-порт `10001`.

Для непрерывного просмотра синтетического движения в MuJoCo:

```bash
bash scripts/stack.sh gui-on
bash scripts/stack.sh shell
```

В открывшейся оболочке контейнера:

```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
export ROS_DOMAIN_ID=92
ros2 launch dg5f_unity_teleop unity_dg5f.launch.py \
  start_endpoint:=false fake_mano:=true \
  lerobot_backend:=mock mujoco_viewer:=true
```

Для headless заменить `mujoco_viewer:=true` на `false`; `gui-on` тогда не нужен.
Остановка — Ctrl+C, затем `exit`. Это также синтетический ввод, а не replay записи.

### Калибровка

В RGB-демо реализована **калибровка нулевого сгиба раскрытой ладони**:

1. Запустить онлайн RGB-режим с `mock` или `tesollo` и открыть веб-интерфейс.
2. Нажать **DISARM**. Показать правую раскрытую ладонь целиком, без предмета,
   достаточно близко к камере; проверить наличие скелета и tracking.
3. Нажать **«Ноль сгиба · раскрытая ладонь»**.
4. Держать пальцы прямыми и неподвижными примерно две секунды. Дождаться
   сообщения **«Нулевой сгиб сохранён»** (`READY`, собрано 20 кадров).
5. Проверить target в раскрытой позе и только затем нажать **ARM**.

Калибровка компенсирует постоянное смещение сгиба указательного, среднего,
безымянного и мизинца. Отведение пальцев и opposition большого пальца не меняются.
Она действует до перезапуска приложения; после перезапуска процедуру повторить.

Если кнопка неактивна — проверить `DISARMED`, свежий tracking и размер ладони.
Если за восемь секунд сбор не завершён — развернуть ладонь к камере, убрать
обрезание/перекрытие пальцев и повторить. Во время сбора target не публикуется.

Ту же процедуру можно запустить через API при выполнении этих условий:

```bash
curl --fail -X POST http://127.0.0.1:8080/api/camera/calibrate
curl --fail http://127.0.0.1:8080/api/state
```

Статус находится в `camera.open_hand_calibration`. Порог «Отсечь дальние руки»
настраивается отдельно по видимому размеру ладони; он не измеряет расстояние
в метрах.

Для Quest отдельной команды калибровки раскрытой ладони в `stack.sh` нет.
Перед ARM проверить tracking и движение MuJoCo; геометрические соглашения
и масштаб ретаргетинга задаются существующим адаптером и конфигурацией.

### Без интернета

После первой сборки модели RGB-демо находятся внутри Docker-образа, и обычный
запуск камеры использует локальные файлы. На той же машине запускать
`docker compose ... up -d --no-build --pull never`. Quest/ROS также использует
локальный образ и собранный `install/`; команды `setup` и `build` повторно
не выполнять без подготовленных зависимостей.

Для переноса RGB-демо на компьютер без сети:

```bash
docker image save -o dg5f-camera-demo-image.tar dg5f-camera-demo:local
```

Перенести архив и загрузить его на целевой машине:

```bash
docker image load -i dg5f-camera-demo-image.tar
docker run --rm --init -p 127.0.0.1:8080:8080 --device /dev/video0:/dev/video0 \
  dg5f-camera-demo:local --backend mock --camera 0
```

Для проверки синтетического режима вообще без сети контейнера:

```bash
docker run --rm --init --network none \
  dg5f-camera-demo:local --synthetic --backend mock --no-background-blur
```

При `--network none` браузер с хоста недоступен; для браузерного просмотра
использовать предыдущий запуск с `-p`. Интернет ему не требуется.
Нативная установка RGB-демо может автоматически скачивать отсутствующую модель:
для неё заранее подготовить модели и зависимости по
[инструкции установки](../dg5f_camera_demo/INSTALL.md).

### Две Azure Kinect и MANO

> Ниже сохранена инструкция отдельного раннего модуля. Текущий путь записи кисти и куба — [форк ViKi](https://github.com/vizitei-dmitri/ViKi_inference); он используется вместе с DexUMR.

Добавлен отдельный модуль [kinect_mano](../kinect_mano/README.md): joint/surface/
temporal MANO-fitting, запись и обработка двух RGB-D потоков, внешняя калибровка
по шахматной доске, калибровка формы, экспорт и ROS replay/live-адаптер.
Инструкция содержит реальные CLI-команды онлайн/офлайн-запуска и калибровки.

Официальные файлы MANO предоставляются отдельно. В доступном окружении нет этих
файлов и подключённых Kinect: численная часть и ROS проверены на синтетических
данных, hardware-захват и точность реальной MANO/Kinect пока не подтверждены.
Результаты и ограничения: [отчёт](../kinect_mano/REPORT.md).

### Если не запускается

| Симптом | Что проверить |
|---|---|
| Нет `/dev/video0` / `Camera read failed` | Найти нужный V4L2 device, проверить права и что камеру не заняло другое приложение |
| В браузере нет руки | Показать правую ладонь целиком; проверить порог отсечения и `camera.error` в `/api/state` |
| Порт 8080 занят | Остановить предыдущий RGB-демо; Compose и `docker run` используют один порт |
| Quest не передаёт точки | Проверить ADB reverse либо IP компьютера, порт 10000 и запущенное Unity-приложение |
| `Another DG5F pipeline is already running` | Остановить предыдущий launch через Ctrl+C |
| Нет окна MuJoCo | Проверить X11, `gui-on`, `/dev/dri`; при необходимости выбрать headless |
| DG5F не подключается | Проверить питание, Ethernet, маршрут к IP и логи; детали в RGB INSTALL |
| Нет пакетов после `up` | Выполнить `bash scripts/stack.sh compile`; `up` только запускает контейнер |

Полная справка ROS-команд: `bash scripts/stack.sh help`. Подробности RGB API,
параметров и нативного запуска: [README RGB-демо](../dg5f_camera_demo/README.md).
Запись демонстраций: [LeRobotDataset](LEROBOT_DATASET_RECORDING.md).


## Isaac Lab

The simulation environments require an Isaac Lab installation:

```bash
cd isaaclab_ext/dg5f_isaaclab
source ~/Documents/work/IsaacLab/env_isaaclab/bin/activate
python scripts/list_envs.py
python scripts/rsl_rl/train.py --task DG5F-Cube-Direct-v0 --num_envs 128 --headless
```

Third-party components and their licenses are listed in
[THIRD_PARTY.md](../THIRD_PARTY.md).
