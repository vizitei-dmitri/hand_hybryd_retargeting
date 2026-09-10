# Запись демонстраций DG5F + Quest в LeRobotDataset

Это отдельный **пассивный ROS-процесс** для training data, не новый драйвер кисти.
Он использует установленный в Docker **LeRobot 0.4.4 / dataset format v3.0**.
Локальная запись не требует интернета, аккаунта Hugging Face или токена.

## Что изменилось и что не изменилось

Добавлены recorder, validator, команды `dataset-*`, offline/mock-тесты.
`ros_bridge_node.py`, `dg5f.py`, `backends.py`, `command_shaper.py`,
`current_guard.py`, DGSDK, launch и параметры движения **не изменены**.
Текущие startup blend, ограничение direct step, current guard, tracking grace
15 секунд, smooth catch-up, recovery и внутренний цикл 200 Hz остаются прежними.

```text
Quest -> retarget -> ros_bridge -> current guard -> shaper/limits/fault map -> DGSDK -> DG5F
             |                                  |                              |
             | raw target                       | accepted effective command   | measured telemetry
             +----------------------------------+------------------------------+
                                                |
                                     passive ROS subscriptions
                                                |
                                     LeRobotDatasetRecorder
                                                |
                              local LeRobot parquet + metadata + optional images
```

Только существующий bridge владеет соединением с физической DG5F.
Recorder не создаёт Robot/backend/DGApi, не открывает TCP 502, не вызывает
arm/disarm/recover и не публикует управляющих сообщений. Его четыре сервиса
относятся исключительно к записи. `debug-record` можно запускать одновременно.

## Источники и единицы

| Feature | Источник | Содержание |
|---|---|---|
| `observation.state` | `/dg5f/lerobot/diagnostics`, `measured_pos` | 20 измеренных углов, **degree** |
| `action` | `/dg5f/lerobot/commanded_joint_states` | 20 разрешённых setpoints, rad → **degree** |
| `expert.raw_action` | `/dg5f/joint_command` | 20 целей ретаргетера до hardware guard/shaper, rad → **degree** |
| `observation.velocity` | diagnostics, `measured_vel` | 20 скоростей, **degree/s** |
| `observation.current` | diagnostics, `measured_current` | 20 токов, **mA** |
| `observation.temperature` | diagnostics, `measured_temp` | 20 температур, **°C** |
| `diagnostic.*` | тот же diagnostics snapshot | состояние bridge/защит |
| `observation.images.<name>` | настроенный `sensor_msgs/Image` | RGB-изображение |

Измерения берутся из единого уже существующего diagnostics snapshot. Это не
состояние MuJoCo и не вычисление положения из команды. В этом snapshot скорость
уже преобразована SDK rpm → degree/s: повторного умножения на 6 нет.

Final-команда действительно доступна без изменений bridge: `_send_latest()`
сначала вызывает current guard, затем `Dg5f.send_action()`, и только после
успешного принятия публикует результат в `commanded_joint_states`.
`send_action()` возвращает последний принятый setpoint после shaper, joint limits
и fault map. Recorder не подменяет его raw target и не сглаживает повторно.
Это **принятая управляющим слоем цель**, а не доказательство, что привод уже
достиг угла или отдельное подтверждение каждого servo-пакета: для этого есть
измеренный `observation.state` и отдельные debug-логи.

Все шесть векторных features — `float32[20]`, порядок строго `JOINT_NAMES`:

```text
rj_dg_1_1 .. rj_dg_1_4   thumb
rj_dg_2_1 .. rj_dg_2_4   index
rj_dg_3_1 .. rj_dg_3_4   middle
rj_dg_4_1 .. rj_dg_4_4   ring
rj_dg_5_1 .. rj_dg_5_4   pinky
```

Для ROS-команд переупорядочивание идёт по именам, а не по позиции в сообщении.
Неисправный `rj_dg_5_1` — индекс **16**. В `action` он всегда `0.0`.
Если bridge вдруг публикует ненулевое значение больше допуска `1e-5 degree`,
эпизод приостанавливается: recorder не маскирует ошибку ложной меткой действия.
Измеренную позицию этого сустава recorder **не обнуляет**. Policy space остаётся 20D.

Отдельные признаки:

```text
diagnostic.armed                   float32[1], 0 или 1
diagnostic.tracking_ok             float32[1], 0 или 1
diagnostic.motion_ready            float32[1], 0 или 1
diagnostic.current_guard_active    float32[1], 0 или 1
diagnostic.current_guard_min_scale float32[1]
diagnostic.disarm_reason           string
recording.monotonic_s              float64[1]
recording.wall_time_s              float64[1]
recording.segment_index            int64[1]
```

Числовой тип флагов совместим со статистикой установленной версии LeRobot.
Диагностика не добавляется внутрь `observation.state`. Официальные `timestamp`,
`frame_index`, `episode_index`, `index`, `task_index` создаёт LeRobot.
Task передаётся в `add_frame` как текст и сохраняется штатно в `meta/tasks.parquet`.

## Подготовка

На хосте:

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
git branch --show-current
# Ожидается feature/lerobot-dataset-recorder

bash scripts/stack.sh up
bash scripts/stack.sh compile
```

Если контейнер уже работает и workspace скомпилирован, повторная сборка образа
не нужна. Исходники смонтированы в `/workspace`. После обновления `setup.py`
нужен `compile`, чтобы появились новые `ros2 run` executables.

## Рабочий запуск: три терминала

Во всех терминалах перейдите в каталог проекта.

Терминал 1 — ваш обычный hardware pipeline:

```bash
bash scripts/stack.sh hardware 10000 hybrid 169.254.186.72
```

Не запускайте второй hardware pipeline или `sdk-check` одновременно с ним:
у кисти должен оставаться один SDK-клиент. Приложение RSL на Quest подключается
как раньше. Проверьте рабочую область, телеметрию и движение в MuJoCo.

Терминал 2 — recorder, пока **без камер**:

```bash
bash scripts/stack.sh dataset-record dg5f_cube_v1 "grasp the cube"
```

Он запускается в `IDLE`. Пока не вызван `dataset-start`, кадры не записываются.
Можно запустить его до ARM: отсутствие effective-команды в этот момент нормально.

Терминал 3 — ручное разрешение движения и начало демонстрации:

```bash
bash scripts/stack.sh arm
bash scripts/stack.sh dataset-status
bash scripts/stack.sh dataset-start
```

Проверьте `success=True` и `state=RECORDING` в ответе. Выполните демонстрацию.
Начинайте после появления свежих tracking и final-команды; recorder сам ARM
не включает. Ответ `dataset-start` подтверждает только начало записи.

Завершить удачную демонстрацию:

```bash
bash scripts/stack.sh dataset-finish
```

Ответ содержит `state=IDLE`, увеличенное число `episodes` и `saved_frames`.
Файлы этого эпизода уже закрыты на диске. Переставьте объект, затем:

```bash
bash scripts/stack.sh dataset-start
# Следующая демонстрация...
bash scripts/stack.sh dataset-finish
```

Неудачную ещё не сохранённую демонстрацию отмените:

```bash
bash scripts/stack.sh dataset-discard
```

Отмена касается **только текущего несохранённого эпизода**, включая его временные
картинки. Ранее сохранённые эпизоды не удаляются. `discard` после `finish`
вернёт ошибку «No active episode», а не удалит предыдущую демонстрацию.

В четвёртом терминале при необходимости:

```bash
bash scripts/stack.sh debug-record
```

Dataset recorder и debug recorder независимы. Они не создают дополнительных
DGSDK-соединений. ROS-подписки dataset recorder используют best-effort/depth=1:
не накапливают очередь старых кадров и не требуют подтверждений от control publisher.
CPU/диск общие, поэтому тяжёлые камеры всё равно могут косвенно создавать нагрузку.

## Завершение и продолжение позже

Нормальный порядок: `dataset-finish`, затем **Ctrl+C в терминале recorder**.
Это не останавливает управление рукой. При необходимости отдельно выполните
`bash scripts/stack.sh disarm`.

Если нажать Ctrl+C/SIGTERM во время непустого `RECORDING` или `PAUSED`, recorder
сохранит уже накопленную часть как эпизод, пометив `interrupted: true` в
`meta/dg5f_recording.json`. Это не автоматически успешная демонстрация.
Если часть не нужна, сначала явно вызовите `dataset-discard`, потом Ctrl+C.
Пустой эпизод не сохраняется.

Для продолжения в другой сессии нужно явно добавить `--resume`:

```bash
bash scripts/stack.sh dataset-record dg5f_cube_v1 "grasp the cube" --resume
```

Без `--resume` существующий каталог не перезаписывается. При продолжении
проверяются FPS, feature schema, порядок суставов и набор/разрешение камер.
Task можно изменить между сессиями — LeRobot добавит новую задачу в метаданные.
Если первый эпизод был только отменён и не было ни одного `finish`, пустой
датасет не открывается штатным reader 0.4.4; используйте новое имя датасета.

Один каталог защищён файловой блокировкой от двух writers. Стандартные сервисы
также резервируются одним recorder на ROS domain. Не запускайте два recorder
с одинаковым service prefix. Для экспертного использования доступен
`--service-prefix /another_dataset`; тогда вызывайте его Trigger-сервисы вручную.
Команды `dataset-start/finish/discard/status` используют стандартный `/dg5f_dataset`.

## Где лежат файлы

```text
В контейнере: /workspace/lerobot_datasets/dg5f_cube_v1/
На хосте:    ~/Documents/work/hand_hybryd_retargeting/lerobot_datasets/dg5f_cube_v1/
```

```text
dg5f_cube_v1/
├── data/chunk-000/file-*.parquet
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    ├── episodes/chunk-000/file-*.parquet
    └── dg5f_recording.json     # дополнительные параметры/происхождение записи
```

Основной формат — официальный LeRobotDataset, не CSV/pickle. RGB-картинки
записываются как штатные image features: сначала временные PNG, затем
изображения встраиваются в parquet. Запись MP4 в этой реализации не включена.
После `finish` временные картинки эпизода убирает официальный API.

`--root` — **базовый каталог**, к нему добавляется `repo-id`:

```bash
bash scripts/stack.sh dataset-record experiments/cube_v2 "grasp the cube" \
  --root /workspace/lerobot_datasets --fps 30 --required-data-max-age-ms 100
```

Файлы окажутся в `lerobot_datasets/experiments/cube_v2`. Локальное имя без
namespace тоже поддерживается версией 0.4.4. Имена не позволяют выход за `root`.
Используйте путь внутри смонтированного `/workspace` либо отдельно настройте
Docker volume: произвольный путь контейнера не обязательно сохранится на хосте.
`lerobot_datasets/` добавлен в `.gitignore`; демонстрации не попадут в git.

Дополнительный manifest хранит `robot_type=dg5f`, `hand=right`, единицы, 20 имён,
fault map, версию LeRobot, git SHA и dirty status, конфигурацию камер. На старте
каждого эпизода сохраняются runtime-параметры из diagnostics, в частности
`startup_blend_s`, `max_direct_step_deg`, `tracking_grace_s`,
`tracking_resume_blend_s` и настройки current guard. Адрес физической кисти не
используется как идентификатор датасета.

## FPS, свежесть, tracking loss и аппаратные сбои

По умолчанию таймер работает на **30 Hz**, допустимо `--fps 1..200`.
На каждом тике выбираются последние полученные значения. Для обязательных
потоков максимальный возраст по умолчанию **100 ms**; он настраивается через
`--required-data-max-age-ms`.

Проверяются возраст ROS-приёма и timestamp у сообщений с header, а также
переданные bridge возраста физической позиции, телеметрии, tracking и команды.
Это не позволяет выдавать регулярно переопубликованную старую SDK-телеметрию
за новое измерение. Для сообщений без stamp используется receive time;
часы источников сообщений с stamp должны быть согласованы с ROS-часами.
Возраст и таймер записи используют monotonic/steady clock.

- При отсутствии/устаревании обязательного потока кадр пропускается.
- `tracking_ok=false` и состояние bridge hold/resume-pending тоже пропускаются.
- После возвращения tracking нужны **новые raw и effective команды**; старые
  цели до потери трекинга не переиспользуются.
- При `motion_ready=false`, disconnect, disarm, небезопасной температуре,
  error diagnostics или recovery-required эпизод переходит в `PAUSED`.
  Хорошие кадры остаются: решите `finish` или `discard`. Новую демонстрацию
  начинайте отдельным `start` после восстановления внешнего pipeline.
- Адаптивное ограничение токов само по себе не fault: его разрешённое действие
  записывается вместе с `current_guard_active` и `current_guard_min_scale`.
- Recorder никогда не восстанавливает соединение и не переармирует кисть.

`dataset-status` показывает `state`, `reason`, `preflight`, `episode_frames`,
`saved_frames`, `episodes`, `skipped_ticks`, FPS и путь. Если кадров нет,
сначала смотрите `preflight`, затем `bash scripts/stack.sh dg-status`.

**Важная особенность времени:** официальный `timestamp` остаётся равномерной
шкалой `frame_index / fps`. Пропущенные тики не заполняются повторами, поэтому
после потери трекинга временная пауза сжимается в этой шкале. Реальный monotonic
и wall time сохранены отдельно; `segment_index` увеличивается после пропусков.
Для ACT/action chunks не используйте без обработки эпизоды с разрывами:
перезапишите их либо при подготовке обучения отберите непрерывные эпизоды.
`dataset-check` показывает количество сегментов. Это latest-value sampling,
не аппаратная синхронизация камер и не жёсткий realtime на загруженном компьютере.
Не ставьте FPS выше частоты исходных потоков без необходимости.

Если диск закончился или запись дала ошибку, состояние становится `ERROR`.
Recorder не удаляет файлы и не пытается автоматически чинить частично записанный
эпизод; остановите его и проверьте датасет. SIGKILL, отключение питания и сбой
диска не равнозначны корректному Ctrl+C: текущий RAM-буфер при них может потеряться.
Ранее завершённые parquet закрываются на каждом `finish`, а не только при выходе.

## Камеры: 0..N sensor_msgs/Image

Без `--cameras` нет обязательных камер и дополнительных image features.
Пример конфигурации:

```yaml
cameras:
  front:
    topic: /camera/front/image_raw
  wrist:
    topic: /camera/wrist/image_raw
```

Готовый файл: `src/lerobot_robot_dg5f/config/dataset_cameras.example.yaml`.
Оставьте только реально существующие потоки. Пример запуска **нового датасета**:

```bash
bash scripts/stack.sh dataset-record dg5f_cube_cameras_v1 "grasp the cube" \
  --cameras /workspace/src/lerobot_robot_dg5f/config/dataset_cameras.example.yaml
```

Поддержаны `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, в том числе строки с padding.
В датасете всё приводится к RGB uint8. Глубина, JPEG/CompressedImage и
CameraInfo здесь не записываются. `cv_bridge` не нужен.

Разрешение определяется по первым свежим кадрам на `start` и фиксируется схемой.
Источник изображений запускается отдельно: `quest-view`/scrcpy сам по себе
не публикует ROS `sensor_msgs/Image` и не становится камерой рекордера автоматически.
Настроенная, но отсутствующая камера **не завершает процесс**: `start` вернёт
понятное `Waiting for image:<name>`. Её stale-кадры пропускаются так же, как
stale-телеметрия. Камера не подменяется чёрной картинкой и не исчезает из схемы
посреди датасета. Для записи без неё уберите её из YAML и используйте другой
repo-id. Для ACT в этой версии выбирайте камеры одинакового разрешения.

Большие RGB-кадры требуют CPU и места на диске; начните с небольшого разрешения
и одной камеры, следите за `skipped_ticks`. Замедление записи не должно приводить
к смене hardware profile или изменению safety thresholds.

## Проверка датасета

После `finish` и Ctrl+C recorder:

```bash
bash scripts/stack.sh dataset-check dg5f_cube_v1
# Или полный путь ВНУТРИ контейнера:
bash scripts/stack.sh dataset-check /workspace/lerobot_datasets/dg5f_cube_v1
# Машиночитаемый отчёт, включая первый кадр:
bash scripts/stack.sh dataset-check dg5f_cube_v1 --json
```

Проверка открывает датасет именно `LeRobotDataset(repo_id=..., root=...)`,
перечитывает все кадры и изображения, проверяет schema/order, конечность чисел,
размер 20D, нулевой action сломанного сустава, количество/длины эпизодов и
номинальные timestamps. Показывает tasks, камеры, per-joint min/max
state/action, сегменты и interrupted episodes. Код выхода — 0 при успехе, 1
при ошибке; исходные эпизоды не удаляются.

Пример результата короткой записи:

```text
episodes: 1
frames: 90
fps: 30
observation.state: [20] float32
action: [20] float32
observation.velocity: [20] float32
observation.current: [20] float32
observation.temperature: [20] float32
expert.raw_action: [20] float32
OK: all numeric values finite; state/action 20D; rj_dg_5_1 action zero
```

Логический пример одного кадра (массивы сокращены только для показа):

```python
{
    "observation.state": np.array([12.0, -85.0, ..., 0.2, 12.0, 40.0, 35.0], dtype=np.float32),
    "expert.raw_action": np.array([15.0, -80.0, ..., 0.0, 15.0, 55.0, 45.0], dtype=np.float32),
    "action": np.array([13.0, -83.0, ..., 0.0, 13.0, 44.0, 38.0], dtype=np.float32),
    "observation.velocity": np.array([...], dtype=np.float32),
    "observation.current": np.array([...], dtype=np.float32),
    "observation.temperature": np.array([...], dtype=np.float32),
    "diagnostic.current_guard_active": np.array([1.0], dtype=np.float32),
    "diagnostic.current_guard_min_scale": np.array([0.4], dtype=np.float32),
    "diagnostic.disarm_reason": "NONE",
    "task": "grasp the cube",
    # Остальные diagnostic/recording features + необязательные images.
}
```

## Самопроверка без физической кисти

```bash
bash scripts/stack.sh test

# Реальные ROS-подписки/Trigger services, но полностью синтетические publishers:
bash scripts/stack.sh dataset-mock

# Камера + Ctrl+C во время второго активного эпизода:
bash scripts/stack.sh dataset-mock --with-camera --interrupt-active
```

По умолчанию mock использует отдельный `ROS_DOMAIN_ID=93`, localhost-only,
не запускает hardware/retarget/Quest и не подключается к DGSDK. Не запускайте
два таких mock одновременно в одном domain. Имена датасетов автоматически
содержат время; своё можно задать `--repo-id my_mock_test` (новое имя).
После теста выполните `dataset-check <напечатанное имя>`.
Синтетические данные проверяют транспорт/формат, **не пригодны для обучения
реальному захвату**. Первую физическую демонстрацию нужно проверить отдельно.

## Использованный официальный API

Проверены исходники установленной версии 0.4.4, а не API другой версии:

1. `LeRobotDataset.create(repo_id, fps, features, root, robot_type="dg5f", use_videos=False)`.
2. `add_frame(frame)` с NumPy-значениями и обязательным `task`.
3. `save_episode(episode_data=...)` — передаётся копия буфера, потому что 0.4.4
   изменяет его при сохранении; затем `clear_episode_buffer()`.
4. `clear_episode_buffer()` — для явной отмены только активного эпизода.
5. `finalize()` — закрывает parquet writers данных и metadata.
6. `LeRobotDataset(repo_id=..., root=..., download_videos=False)` — повторное
   открытие для проверки и продолжения. После каждого `finish` следующий
   эпизод начинается через официальный resume loader; закрытые файлы не
   переоткрываются на перезапись.

`dg5f_recording.json` — только дополнительные provenance/runtime метаданные.
Все training arrays, tasks, статистика и картинки находятся в штатном формате.
Локальный режим принудительно offline, автоматического Push to Hub нет.

## Будущее обучение: только пример, сейчас не запускаем

В контейнере есть стандартный `lerobot-train`. Важная проверенная особенность
**ACT в LeRobot 0.4.4**: одной `observation.state` недостаточно — конфигурация
требует хотя бы одно изображение либо `observation.environment_state`.
Поэтому первая запись без камеры валидна как LeRobotDataset, но это ещё не
готовый визуальный датасет для ACT. Recorder не выдумывает environment state.

После сбора качественных непрерывных демонстраций **с камерой**, из shell контейнера:

```bash
bash scripts/stack.sh shell
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# ПРИМЕР НА БУДУЩЕЕ; не запускается скриптами записи/тестами:
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 lerobot-train \
  --dataset.repo_id=dg5f_cube_cameras_v1 \
  --dataset.root=/workspace/lerobot_datasets/dg5f_cube_cameras_v1 \
  --policy.type=act \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null \
  --output_dir=/workspace/outputs/train/dg5f_cube_act_v1 \
  --job_name=dg5f_cube_act_v1 \
  --wandb.enable=false \
  --eval_freq=0 \
  --num_workers=2 \
  --batch_size=8 \
  --steps=10000
```

Здесь `dataset.root` — уже **полный путь** к датасету (в отличие от `--root`
нашего recorder). `pretrained_backbone_weights=null` исключает скачивание
ImageNet-весов; CPU указан явно, поскольку Compose не настраивался для GPU.
Это пример конфигурации, не рекомендация числа шагов для конкретного навыка.
При необходимости отберите хорошие эпизоды: `--dataset.episodes='[0,2,3]'`.
Не включайте interrupted/paused или многосегментные демонстрации без проверки.
Загрузка/экспорт на Hub, MP4, обучение, inference policy и управление кистью
обученной policy в эту задачу не входят и не реализованы.
