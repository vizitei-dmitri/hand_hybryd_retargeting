# Две Azure Kinect DK → параметрическая кисть MANO

Новый модуль для существующего проекта DG5F. Поддерживает подгонку MANO по
21 суставу, двум наблюдаемым поверхностям и предыдущим кадрам. Есть обработка
сохранённых данных, захват/онлайн-цикл, внешняя калибровка, оценка формы кисти,
локальная визуализация и ROS 2 адаптер.

Все команды ниже выполняются **из корня `hand_hybryd_retargeting`**.
Реализация находится в ветке `feature/kinect-mano-hybrid-tracker`.

**Статус проверки:** численная часть проверяется на отдельно созданной тестовой
геометрии; официальных файлов MANO, двух подключённых Kinect и реальных RGB-D
записей в окружении нет. Численные проверки не являются испытаниями MANO/Kinect.
Результаты и ограничения: [REPORT.md](REPORT.md).

## Какой режим выбрать

| Режим | Команда | Вход |
|---|---|---|
| Самопроверка без устройств и MANO | `demo --run` | Явно обозначенная численная фикстура, не MANO |
| Офлайн-реконструкция | `prepare`, затем `fit` | Записи двух камер, rig, masks/joints или локальный detector |
| Онлайн-реконструкция | `online` | Две Kinect, rig, MANO, локальная модель MediaPipe |
| Внешняя калибровка | `capture --calibration-only`, затем `calibrate` | Запись одной доски одновременно двумя камерами |
| Форма кисти | `calibrate-shape` | Набор наблюдений одной кисти в разных позах |
| Просмотр | `view` или ROS replay | `reconstruction.npz` |

Режимы оптимизации: `joint` (A), `hybrid` (B), `temporal` (C). В A/B временные
штрафы выключены, но одинаково используется предыдущее решение для инициализации.
C добавляет ограничения по скорости изменения состояния и ускорению с учётом
timestamps. Это последовательный causal tracker, а не batch-smoother с будущими
кадрами. Точные параметры сохраняются с результатом.

## 1. Самопроверка в Docker

Первая сборка требует интернета. Она создаёт отдельное CPU-окружение и не меняет
PyTorch/CUDA исходного DG5F-стека:

```bash
docker compose -f compose.kinect.yaml --profile core build kinect_core
docker compose -f compose.kinect.yaml --profile core run --rm kinect_core \
  demo --output /data/demo --frames 4 --points-per-camera 64 --iterations 40 --run
```

На хосте результат находится в `kinect_mano_data/demo/`: входные наблюдения,
`input.png`, сравнение трёх режимов в `benchmark/benchmark.json` и `benchmark.png`,
а также параметры и геометрия каждого режима.

`numerical_fixture.npz` — независимо сгенерированная тестовая геометрия с
16-joint LBS и 778 вершинами. Она **не содержит MANO** и загружается только
при явном `--allow-test-model`. Для повторного fitting:

```bash
docker compose -f compose.kinect.yaml --profile core run --rm kinect_core \
  fit --model /data/demo/numerical_fixture.npz --allow-test-model \
  --input /data/demo/observations.npz --output /data/fit_demo --mode hybrid
```

Выбирать новый каталог для каждого запуска. Для прерванного `fit` повторить
ту же команду с `--resume`: сверяются хеш входа, модель, форма и конфигурация.

Для запуска уже собранного образа с отключённой сетью, из корня проекта:

```bash
mkdir -p kinect_mano_data
docker run --rm --network none \
  -v "$PWD/kinect_mano_data:/data" \
  kinect-mano-core:cpu demo --output /data/no_network --frames 2 --points-per-camera 32
```

## 2. Нативная установка и GPU

Нужно отдельное Python-окружение; для захвата — Linux и Azure Kinect Sensor SDK.
Пример установки CPU-версии, без изменения существующего ROS-окружения:

```bash
python3 -m venv kinect_mano/.venv
kinect_mano/.venv/bin/python -m pip install --upgrade pip 'setuptools==79.0.1' wheel
kinect_mano/.venv/bin/python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
kinect_mano/.venv/bin/python -m pip install -e './kinect_mano[test]'
kinect_mano/.venv/bin/python -m kinect_mano --help
```

Для CUDA создать отдельное окружение с подходящей CUDA-сборкой PyTorch вместо
CPU-wheel. В текущем окружении проверена `torch 2.10.0+cu128`, RTX 3060 Laptop
6 GB. CLI принимает `--device cpu`, `cuda` и `auto`; Docker core выше — CPU.
Для небольшого меша используются один CPU thread и пакетные вычисления по
уровням кинематического дерева. Частота реального времени не гарантирована.

Далее `PY` — путь к интерпретатору этого окружения:

```bash
PY="$PWD/kinect_mano/.venv/bin/python"
"$PY" -m kinect_mano demo --output kinect_mano/outputs/local_demo \
  --frames 4 --points-per-camera 64 --run --device auto
```

## 3. Подготовка настоящей MANO

Получить `MANO_RIGHT.pkl` или `MANO_LEFT.pkl` с
[официального сайта MANO](https://mano.is.tue.mpg.de/) по условиям его лицензии.
Модели в репозиторий не включены и автоматически не скачиваются.

Сконвертировать **доверенный официальный** pickle в числовой NPZ:

```bash
"$PY" -m kinect_mano convert-model \
  --input /absolute/path/to/MANO_RIGHT.pkl \
  --output kinect_mano/models/MANO_RIGHT.npz --hand right
```

Официальные массивы имеют метрический масштаб; `--unit-scale` нужен только
для экспорта с заведомо другими единицами. Старые Chumpy-pickle могут потребовать
исходные зависимости при конвертации: выполнять её в отдельном legacy-окружении,
не переустанавливать старые NumPy/PyTorch в ROS-стек. Pickle может исполнять код —
использовать только доверенные файлы. Runtime читает NPZ без pickle.

Поза — 45 локальных axis-angle параметров относительно flat hand, глобальная
ориентация отдельно. `translation_m` — положение запястья в world. `beta[10]`
общая для последовательности. Landmarks: wrist, thumb, index, middle, ring,
little, по четыре точки на палец. Кончики берутся по явно сохранённым vertex IDs
соглашения manopth; это не 21 независимый сустав MANO.

## 4. Подготовка камер и калибровка

Нужны Azure Kinect Sensor SDK и Python binding
[pyk4a](https://github.com/etiennedub/pyk4a). После установки SDK в отдельное
окружение добавить опциональные зависимости:

```bash
"$PY" -m pip install -e './kinect_mano[kinect,mediapipe]'
```

SDK и USB-доступ в поставляемый CPU Docker-образ не включены: захват выполнять
нативно или в собственном образе с установленным SDK. Hardware-путь здесь ещё
не испытан. Для аппаратной синхронизации соединить Sync Out камеры A с Sync In
камеры B и использовать `--sync wired`. Без кабеля выбрать `standalone` и
измерить сдвиг часов; совпадение индексов кадров синхронизацией не считается.

### 4.1. Записать доску до наличия внешней калибровки

Используется заводская RGB/depth-калибровка Kinect и SDK-регистрация глубины
на RGB. `--calibration-only` явно сохраняет камеры в **раздельных локальных**
системах координат. Эти записи нельзя использовать как готовый калиброванный rig.

```bash
"$PY" -m kinect_mano capture --calibration-only --sync wired \
  --device-a 0 --device-b 1 --frames 90 --output kinect_mano/recordings/board
```

Двигать одну доску так, чтобы обе камеры видели её одновременно, с разными
наклонами и расстояниями. Для обычной шахматной доски пометить угол и сохранять
одинаковый порядок углов: симметрия доски может приводить к ошибочным
соответствиям. Параметры `--columns`/`--rows` — число **внутренних** углов,
`--square-size` — реально измеренная сторона клетки в метрах.

### 4.2. Вычислить внешнюю калибровку

Например, для доски 9×6 внутренних углов и клетки 25 мм:

```bash
"$PY" -m kinect_mano calibrate \
  --camera-a kinect_mano/recordings/board/camera_A.npz \
  --camera-b kinect_mano/recordings/board/camera_B.npz \
  --columns 9 --rows 6 --square-size 0.025 --min-pairs 8 \
  --output kinect_mano/rig.json
```

Результат: intrinsics/distortion, `T_world_from_camera` для каждой камеры,
число пар, RMS reprojection error и происхождение данных. World — color frame
камеры A: x вправо, y вниз, z вперёд. Направление преобразования:
`p_world = T_world_from_camera @ p_camera` для столбцовых векторов.
Отсутствующая калибровка не заменяется единичной матрицей для реконструкции.

При standalone-съёмке добавить в `capture`, `calibrate` и `prepare`
измеренный `--offset-b`: сравниваются `t_A` и `t_B + offset_b`.
`--tolerance` задаётся в секундах, по умолчанию 0.005.

### 4.3. Проверить совмещение на статическом объекте

После калибровки записать статическую сцену:

```bash
"$PY" -m kinect_mano capture --rig kinect_mano/rig.json --sync wired \
  --frames 10 --output kinect_mano/recordings/static
"$PY" -m kinect_mano check-calibration \
  --camera-a kinect_mano/recordings/static/camera_A.npz \
  --camera-b kinect_mano/recordings/static/camera_B.npz \
  --rig kinect_mano/rig.json --output kinect_mano/outputs/alignment
```

Посмотреть `alignment.png` и `alignment.json`. Residuals ближайших соседей
зависят от перекрытия полей зрения и не являются независимой точностью
калибровки. Проверять заводские intrinsics/режим камер и serial mapping;
при изменении камеры, режима или положения rig калибровать повторно.

## 5. Офлайн-реконструкция реальной записи

Запись с известным rig:

```bash
"$PY" -m kinect_mano capture --rig kinect_mano/rig.json --sync wired \
  --frames 120 --output kinect_mano/recordings/hand
```

Каждая камера сохраняется в отдельный NPZ: RGB, зарегистрированная depth в
метрах, исходные timestamps, калибровка и метаданные. Формат буферизует
ограниченный chunk в RAM; длинные сессии записывать отдельными порциями.

Нужен локальный файл MediaPipe Hand Landmarker `.task`. Существующая модель
RGB-демо может использоваться, если она уже подготовлена. Инициализация берёт
только 2D landmarks и настоящую depth; псевдо-3D MediaPipe не используется.

```bash
"$PY" -m kinect_mano prepare \
  --camera-a kinect_mano/recordings/hand/camera_A.npz \
  --camera-b kinect_mano/recordings/hand/camera_B.npz \
  --rig kinect_mano/rig.json \
  --mediapipe-model dg5f_camera_demo/models/hand_landmarker.task \
  --allow-polygon-roi --output kinect_mano/outputs/hand_observations.npz
"$PY" -m kinect_mano fit --model kinect_mano/models/MANO_RIGHT.npz \
  --input kinect_mano/outputs/hand_observations.npz \
  --mode temporal --device auto --iterations 80 \
  --output kinect_mano/outputs/hand_temporal
```

`--allow-polygon-roi` явно разрешает приближённую маску по landmarks для
экспериментов со свободной кистью. Она **не отделяет предмет при хвате**.
Для таких сцен нужны сохранённые semantic hand masks в `CameraFrame.mask`
(при необходимости с исключёнными object pixels). По умолчанию отсутствие
маски не замалчивается. Вместо детектора можно передать `prepare --joints`
с метрическими joints, timestamps и masks; формат задаёт
`initializers.save_joint_estimates()`.

### Форма кисти

По набору разнообразных поз оценить одну общую форму, затем фиксировать её:

```bash
"$PY" -m kinect_mano calibrate-shape \
  --model kinect_mano/models/MANO_RIGHT.npz \
  --input kinect_mano/outputs/hand_observations.npz \
  --max-frames 12 --iterations 100 --output kinect_mano/hand_shape.json
"$PY" -m kinect_mano fit --model kinect_mano/models/MANO_RIGHT.npz \
  --shape kinect_mano/hand_shape.json \
  --input kinect_mano/outputs/hand_observations.npz \
  --mode temporal --output kinect_mano/outputs/hand_calibrated
```

Профиль привязан к хешу модели и стороне кисти. Для повторной калибровки
использовать новый файл профиля; существующие результаты не перезаписываются.

## 6. Онлайн-режим

После калибровки rig и подготовки локальных моделей:

```bash
"$PY" -m kinect_mano online --rig kinect_mano/rig.json --sync wired \
  --model kinect_mano/models/MANO_RIGHT.npz \
  --mediapipe-model dg5f_camera_demo/models/hand_landmarker.task \
  --allow-polygon-roi --mode temporal --iterations 10 --device auto \
  --output kinect_mano/outputs/live
```

По умолчанию работает до Ctrl+C; `--frames N` ограничивает число результатов.
Захват идёт в отдельном потоке с одним последним кадром: медленный optimizer
не создаёт очередь старых поз. `live.jsonl` содержит параметры, joints, меш,
облака каждой камеры и качество. Потерянные/отклонённые оценки отмечены invalid;
старое состояние не выдаётся за свежее наблюдение. После длинного пропуска
временная история сбрасывается.

Измеряются время preprocessing/fitting, пропуски пар и receive-to-fit latency.
Последняя начинается с получения пары на хосте, а не с экспозиции сенсора;
исходные SDK/device timestamps сохраняются отдельно. Полная аппаратная задержка
требует испытания с камерами и согласования часов. Реальная частота пока не
подтверждена; увеличивать budget итераций следует по измерениям, а не по названию
«онлайн».

## 7. Экспорт, просмотр и сравнение

Каждый `fit` сохраняет `reconstruction.npz`, `quality.json`, `config.json`,
`manifest.json`, `checkpoint.jsonl` и `preview.png`. NPZ включает `theta`,
общую `beta`, `global_orient`, `translation_m`, 21 joints, 778 vertices,
faces, timestamps, valid и диагностические облака. Геометрия воспроизводится
из параметров без повторного fitting. Исходные файлы не изменяются.

```bash
"$PY" -m kinect_mano view \
  --input kinect_mano/outputs/hand_temporal/reconstruction.npz \
  --frame 0 --output kinect_mano/outputs/frame_000.png
"$PY" -m kinect_mano benchmark --model kinect_mano/models/MANO_RIGHT.npz \
  --input kinect_mano/outputs/hand_observations.npz \
  --shape kinect_mano/hand_shape.json --output kinect_mano/outputs/comparison
```

Все три режима получают одинаковые данные и форму. Ошибки joints/кончиков
вычисляются только при наличии синтетической truth в файле. На обычной записи
доступны fitting residuals, покрытие, плавность и скорость. Это не независимая
оценка миллиметровой точности человеческой кисти.

## 8. ROS 2, RViz и существующий ретаргетинг

ROS-пакет `src/kinect_mano_ros` отдельно публикует `PoseArray[21]`, параметры MANO
и качество в JSON, triangle mesh/skeleton markers и clouds. Standalone fitting
не импортирует ROS и не подключается к DGSDK.

После сборки ROS-overlay:

```bash
docker compose -f compose.kinect.yaml --profile ros build kinect_ros
KINECT_MANO_DATA_DIR="$PWD/kinect_mano/outputs/hand_temporal" \
  docker compose -f compose.kinect.yaml --profile ros up kinect_ros
```

По умолчанию используется namespace `/kinect_mano`, replay идёт с исходным
темпом. ROS stamps соответствуют времени публикации; исходные device timestamps
остаются в параметрах/качестве. Офлайн-реконструкция явно отмечается в сообщении.
Для RViz задать `KINECT_START_RVIZ=true` и разрешить X11 средствами своего стенда.

Переключение Quest/Kinect для **симуляции**:

```bash
KINECT_MANO_DATA_DIR="$PWD/kinect_mano/outputs/hand_temporal" HAND_SOURCE=kinect \
  docker compose -f compose.kinect.yaml --profile sim up kinect_sim
```

Для Quest установить `HAND_SOURCE=quest` и подключить существующее Unity-приложение.
Selector переиспользует исходный launch, ретаргетинг и MuJoCo, отключая физический
bridge. Он не разрешает `tesollo` или `start_lerobot:=true`. Для live ROS выбрать
`KINECT_ROS_MODE=live`, `KINECT_LIVE_JSONL=/data/live.jsonl` и смонтировать каталог
живого запуска. Детали ROS API: [README пакета](../src/kinect_mano_ros/README.md).

## 9. Тесты и научные ориентиры

```bash
"$PY" -m pytest kinect_mano/tests -q
RUN_SLOW_TESTS=1 "$PY" -m pytest kinect_mano/tests/test_fitting.py -q
docker build --target test -f docker/kinect-mano.Dockerfile -t kinect-mano-test:cpu .
docker run --rm --network none -e RUN_SLOW_TESTS=1 kinect-mano-test:cpu
```

Проверки включают градиенты, LBS, шум и выбросы, видимость, пропуски, единицы,
калибровку, временные условия, checkpoint/resume и CLI. Тест с официальной MANO
включается переменной `MANO_MODEL=/path/to/MANO_RIGHT.npz`; без модели он явно
пропускается. Hardware-тестов и подтверждения точности реальных камер нет.

Архитектурные ориентиры:
[MANO](https://mano.is.tue.mpg.de/),
[manopth](https://github.com/hassony2/manopth),
[manotorch](https://github.com/lixiny/manotorch),
[DexYCB](https://dex-ycb.github.io/),
[HOnnotate/HO3D](https://github.com/shreyashampali/ho3d).
Собственная реализация дифференцируемых blend shapes/LBS не распространяет
лицензируемые массивы MANO. Анатомические штрафы сейчас геометрические и широкие,
не эквивалентны обученному prior или полному anatomy model manotorch.

HOTrack проверен как возможный initializer: его старые CUDA-зависимости
изолированы за файловым/process-интерфейсом, официальный inference в текущую
версию не интегрирован. Аудит и протокол: [HOTRACK.md](docs/HOTRACK.md).
[HandFlowNet](https://www.nature.com/articles/s41598-025-16821-9) не является
зависимостью; переобучение и scene flow в первый этап не входят.
