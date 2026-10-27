# Kinect MANO: реализация и результаты проверки

Дата: 27 октября 2026. Ветка: `feature/kinect-mano-hybrid-tracker`.

Реализован код параметрической реконструкции и инфраструктура захвата, калибровки,
экспорта и ROS-интеграции. **Испытание настоящей MANO на двух физических Kinect
не выполнено:** в доступном окружении нет лицензируемых файлов MANO, подключённых
Kinect или записей этих камер. Это ограничивает подтверждённый результат
численными и интеграционными проверками, а не качеством реального трекинга.

## Аудит исходного проекта

Проверена цепочка `ManoLandmarks` на `/quest/hand_pose` → Unity adapter →
`PoseArray[21]` на `/hands/right/landmarks` → `dg5f_retarget` →
`JointTrajectory` на `/dg5f/joint_command`. Исходный `landmarks_to_mano`
преобразует координаты landmarks, но не восстанавливает параметры MANO.

Повторно используются интерфейс 21 landmarks, исходный launch и ретаргетинг,
MuJoCo-модель и существующий управляющий слой DG5F. Исходники алгоритмов
Quest-адаптера, ретаргетинга, ServoController и vendor SDK не изменялись.
Существующие незакоммиченные изменения Isaac Lab и RGB-демо сохранены отдельно.

Новый вычислительный модуль находится в `kinect_mano/`, ROS-транспорт — в
`src/kinect_mano_ros/`. Импорт standalone-модуля не открывает устройства или SDK.

## Реализованные функции

| Компонент | Реализация |
|---|---|
| MANO | Дифференцируемые shape/pose blend shapes и LBS на PyTorch; 778 vertices, 16 внутренних joints, 5 явно заданных tips, 21 landmark |
| Параметры | 45 локальных axis-angle, отдельная глобальная ориентация, wrist translation в метрах, общая форма `beta[10]` |
| Joint fitting | Confidence-веса, Huber, пропуски, rigid initialization, warm start, pose/геометрический prior |
| Surface fitting | Отдельные камеры, ray-to-frontmost-triangle visibility, point-to-triangle distance, робастная ошибка и depth gate |
| Временной режим | Ограничения изменений joints/вращений и ускорения по реальному Δt; сброс после разрывов и rejected frames |
| Калибровка формы | Совместная оценка одной beta и отдельных поз нескольких кадров; привязка профиля к хешу модели и handedness |
| RGB-D | Числовой NPZ на камеру, RGB-aligned depth в метрах, intrinsics/distortion/extrinsics и исходные timestamps |
| Kinect capture | Опциональный pyk4a, заводская регистрация, standalone/wired, master/subordinate, manual exposure, проверка sync jack и serial |
| Калибровка rig | Реальное OpenCV-обнаружение шахматной доски, stereoCalibrate с фиксированными заводскими intrinsics, RMS и проверка transform |
| Инициализация | Локальный MediaPipe 2D + измеренная depth, консервативное объединение ракурсов, precomputed metric joints |
| Preprocessing | Маски, depth bounds, удаление выбросов, прореживание, сохранение принадлежности камере |
| Онлайн-цикл | Захват независимо от fitting, один последний кадр без очереди, invalid-статусы, JSONL stream и тайминги |
| Офлайн-цикл | Повторяемый fitting по файлам, checkpoint/resume с fingerprint, атомарный итоговый NPZ |
| Просмотр | Headless PNG с мешем, joints и облаками; ROS markers и RViz config |
| ROS | Replay/live JSONL/topic, PoseArray, параметры/качество JSON, combined и A/B PointCloud2, проверка свежести live-ввода |
| DG5F-интеграция | Selector Quest/Kinect переиспользует исходный ретаргетинг и симуляцию; физический bridge отключён |
| Docker | Отдельный CPU core, test target, ROS overlay, Compose profiles и ограниченный build context |

Загрузка официальной модели выполняется через локальную конвертацию доверенного
pickle в NPZ. Лицензируемые файлы не поставляются и не скачиваются автоматически.
Численная фикстура отдельно помечена `numerical_fixture`; её загрузка требует
явного разрешения. Она не является заменителем MANO в реальном запуске.

## Важные решения оптимизации

Surface-loss использует только первые пересечения лучей наблюдения с мешем.
Скрытые участки не имеют model-to-cloud притяжения. Непопавшие/отброшенные лучи
получают ограниченный постоянный штраф, а нормализация сохраняет число исходных
наблюдений: исчезновение correspondences не может само по себе обнулить ошибку.
Лучшее состояние выбирается только по свежему visibility/correspondence score;
промежуточные cached ICP scores между собой не сравниваются.

При полностью непригодном initializer кадр остаётся invalid даже при наличии
отдельных координат. При отсутствии surface hits допустим явно отмеченный
`degraded_joint_only` результат. Недостоверные landmarks не превращаются в
нулевые измерения с ненулевым весом. После rejected frame временная история
сбрасывается. В ROS invalid или stale live-результаты не публикуют landmarks.

Применяются широкие геометрические ограничения изгиба и нормы локальных вращений.
Это не полный анатомический prior manotorch, не гарантия отсутствия
самопересечений и не обученная модель допустимых поз.

## Что проверено

- Forward LBS, rigid equivariance, единицы, индексы landmarks, finite gradients
  в нулевой позе и gradient checks расстояния до треугольника.
- Joint/surface fitting с известной тестовой позой, шумом, выбросами,
  пропущенными суставами, разными наблюдениями и потерей tracking.
- Совместная калибровка общей формы на нескольких тестовых кадрах.
- Временная согласованность, нерегулярные timestamps и разрывы.
- Чтение/запись камер, timestamp matching, отсутствие/неверная калибровка,
  distortion и направления преобразований.
- Actual OpenCV chessboard detection на синтетически отрендеренных изображениях
  через CLI; метрический масштаб и направление восстановленного stereo rig.
- Подготовка masked RGB-D + precomputed joints через CLI, alignment JSON/PNG.
- Checkpoint/resume, незавершённая последняя строка, повреждённый итоговый архив,
  атомарная запись и повторное восстановление геометрии из параметров.
- Реальные ROS pub/sub и launch policy в имеющемся ROS 2 Docker-образе, без сети:
  replay pacing/stamps, invalid frames, live stream, freshness, clouds/mesh и
  запрет физического запуска. ROS-набор: **29 тестов прошли**.
- Изолированный colcon build ROS-пакета в `/tmp` контейнера; host build/install
  не перезаписывались. Simulation smoke получал команды для 20 суставов DG5F.
- CPU Docker image собран; `demo → A/B/C fitting → export` работает с
  `--network none`, без камеры, GPU, MANO и облачных API.
- Настоящая локальная модель MediaPipe успешно инициализирована и вызвана на
  пустом синтетическом RGB-изображении в контейнере без сети: правильно вернула
  отсутствие руки. Положительное обнаружение реальной кисти не проверялось.

Официальный MANO-тест явно пропускается без `MANO_MODEL`; он не засчитывается
как успешная проверка модели. ROS-тесты пропускаются на host без ROS, но отдельно
исполнены внутри ROS-образа. Полный host-прогон с shape-калибровкой:
**143 passed, 8 skipped** (одна настоящая MANO и семь ROS-проверок).
Семь ROS-проверок отдельно прошли в контейнере вместе с остальными transport
тестами: **29/29**. Доступные команды полного прогона:

Docker test target также собран и исполнен с `--network none` и
`RUN_SLOW_TESTS=1`: **121 passed, 1 skipped**. Единственный skip относится к
официальной MANO, которой нет в окружении. С учётом отдельного ROS-набора
получается **150 прошедших разных проверок**, без двойного счёта host transport.

```bash
PYTHONPATH=kinect_mano:src/kinect_mano_ros RUN_SLOW_TESTS=1 \
  kinect_mano/.venv/bin/python -m pytest kinect_mano/tests src/kinect_mano_ros/test -q
docker build --target test -f docker/kinect-mano.Dockerfile -t kinect-mano-test:cpu .
docker run --rm --network none -e RUN_SLOW_TESTS=1 kinect-mano-test:cpu
```

## Измеренный численный benchmark

Среда: Python 3.10.12, PyTorch 2.10.0+cu128, CUDA 12.8, NumPy 2.2.6,
RTX 3060 Laptop 6 GB, один CPU thread. Все результаты ниже относятся к
**независимой численной фикстуре, не к настоящей MANO и не к Kinect**.

Основной прогон: seed 7, четыре кадра, по 64 surface points на камеру,
depth noise 1 мм, joint noise 4 мм, 60 итераций на кадр. Одна фиксированная
нулевая beta одинакова в A/B/C; синтетическая truth-форма ненулевая.
У каждого режима отдельный warm-up до замера, temporal history не переносится
между режимами. Все четыре кадра приняты во всех режимах.

| Метрика | A: joint | B: hybrid | C: temporal |
|---|---:|---:|---:|
| Средняя ошибка 21 joints, мм | 4.608 | 3.835 | 4.007 |
| Средняя ошибка пяти tips, мм | 6.037 | 5.833 | 5.856 |
| Surface residual по hit rays, мм | 1.961 | 1.113 | 1.202 |
| Покрытие surface rays | 87.1% | 91.6% | 92.0% |
| Acceleration RMS, м/с² | 4.387 | 3.443 | 3.115 |
| Fit time mean, мс | 320.3 | 481.5 | 526.3 |
| Fitting throughput, кадров/с | 3.12 | 2.08 | 1.90 |
| Peak PyTorch allocated VRAM, MiB | 21.87 | 22.00 | 22.00 |

В этом коротком прогоне B снижает среднюю ошибку joints примерно на 17%
относительно A. C уменьшает ускорение относительно B, но немного увеличивает
ошибку joints: временная регуляризация не улучшает все метрики автоматически.
Четырёх кадров недостаточно для обобщённых выводов или статистической оценки.

VRAM в таблице — tensor allocations PyTorch, не полный GPU process footprint;
процесс также использует CUDA context. RSS high-water был около 1.5 GiB и
сохраняется в JSON как накопленный максимум процесса. Throughput измеряет fitting
после warm-up, исключая чтение файлов, detector, подготовку облаков, ROS и запись.
Это не sensor-to-command latency.

Влияние размера облака: другая синтетическая запись seed 7 с 4096 точками на
камеру, те же четыре кадра, **10 итераций**, одинаковый вход с разными caps:

| Суммарный cap точек | B: кадров/с | C: кадров/с | B: joints, мм | Peak allocated VRAM B, MiB |
|---|---:|---:|---:|---:|
| 128 | 10.24 | 7.76 | 4.548 | 27.42 |
| 4096 | 6.98 | 6.67 | 4.482 | 28.07 |
| 8192 | 5.16 | 4.91 | 4.150 | 28.75 |

Исходная цель малой задержки ещё требует оптимизации и проверки на hardware.
**30 FPS не достигнуты и не заявляются.** Online-loop обеспечивает свежий кадр
без очереди, но само наличие этого режима не доказывает real-time throughput.

Локальные артефакты: `outputs/validated_benchmark/benchmark.json` и PNG;
`outputs/points_validated_{64,2048,4096}/`; `outputs/fit_smoke/preview.png`;
`outputs/docker_offline/`. Они исключены из Git, поскольку воспроизводятся CLI.
Команда для основной записи и сравнения есть в [README](README.md).

## Требует реального оборудования или внешних ресурсов

1. Официальные файлы MANO: проверить численное соответствие официальному слою,
   суставы/tips и восстановление известных поз уже настоящей модели.
2. Две Azure Kinect и Sensor SDK: проверить USB, режимы RGB/depth, wired sync,
   manual exposure, active-depth interference, serial mapping и timestamps.
3. Реальные парные снимки доски и статического объекта: оценить внешнюю
   калибровку и совмещение облаков.
4. Реальная кисть/предмет: оценить detector, сегментацию и occlusion handling.
5. Независимая разметка: измерить фактическую ошибку joints/tips; surface residual
   не является такой разметкой.
6. Аппаратные задержки: согласовать временные домены сенсоров/хоста/ROS и измерить
   полный путь до публикации. Текущий таймер начинается с получения пары на хосте.

## HOTrack и научные источники

В [официальном HOTrack](https://github.com/PKU-EPIC/HOTrack) проверены README,
код и зависимости: заявлен PyTorch 1.9.1, CUDA-extension требует PyTorch <1.11.
Запросы к официальному зеркалу pretrained archive завершались таймаутом без
заголовков; доступность и размер архива не подтверждены. Старый CUDA/PyTorch
поверх проекта не устанавливался, обучение не выполнялось.

Есть протестированный универсальный NPZ/process adapter и чтение готовых joints;
это **не интеграция официального HOTrack inference**. Основной путь использует
RGB-D initializer. Подробный аудит: [HOTRACK.md](docs/HOTRACK.md).

Использованы как архитектурные ориентиры
[MANO](https://mano.is.tue.mpg.de/),
[manopth](https://github.com/hassony2/manopth),
[manotorch](https://github.com/lixiny/manotorch),
[DexYCB](https://dex-ycb.github.io/),
[HO3D](https://github.com/shreyashampali/ho3d) и
[HandFlowNet](https://www.nature.com/articles/s41598-025-16821-9).
Лицензируемые модели и исследовательские датасеты не копировались в проект.

## Известные ограничения и следующие улучшения

- Polygon ROI не является semantic hand segmentation и не отделяет предмет.
  Для хватов нужны сохранённые качественные masks или отдельный segmenter.
- Depth в проекции 2D landmark измеряет видимую поверхность, не центр сустава.
  Confidence консервативен, но systematic bias не устранён.
- Узкие пальцы, отсутствующая глубина и слабое перекрытие ракурсов могут
  уменьшать число surface hits; coverage нужно анализировать вместе с residuals.
- Простая шахматная доска имеет неоднозначность ориентации углов. Для сложных
  установок целесообразно расширение на ChArUco с уникальными ID.
- Прямое ray casting имеет сложность по числу точек × числу треугольников.
  Возможные ускорения: BVH, projective correspondences, reuse optimizer state,
  смешанная точность после проверки градиентов, адаптивный iteration budget.
- Нет полного collision/anatomical prior, scene flow, batch-smoother с будущими
  кадрами, native MKV reader и обучения сетей.
- NPZ capture буферизует ограниченные chunks в памяти. Для длительных сессий
  полезен потоковый формат/официальная запись MKV с последующим импортом.
- Нет новой аппаратной активации DG5F: существующий ServoController сохранён,
  а интеграционный launch предназначен для симуляции/просмотра.
