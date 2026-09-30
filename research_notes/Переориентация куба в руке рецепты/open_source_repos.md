# Open-source, готовый к запуску код для in-hand reorientation на многопалых кистях (для Isaac Lab + RSL-RL PPO)

> Проверено 2026-09-30. Часть фактов получена **прямым чтением локального клона Isaac Lab
> v2.3.2** (`/home/yoba/Documents/work/IsaacLab`, git `37ddf626` «Bumps version to v2.3.2»),
> часть — через GitHub API и HTTP-проверку CDN с готовыми чекпойнтами. Там, где источник —
> web-поиск/сниппет, а не первичная страница, это помечено.

## Какие репозитории реально существуют и в каком они состоянии

### Takeaway
«Готовые решения» существуют, но делятся на две очень разные группы: (1) **встроенные задачи
Isaac Lab** (Shadow/Allegro repose-cube + новый DexSuite) — живые, поддерживаемые, с
RSL-RL-конфигами и **опубликованными pretrained-чекпойнтами**; (2) **всё исследовательское
наследие 2022–2024** (IsaacGymEnvs, DeXtreme, hora, dexenv/Visual Dexterity, Bi-DexHands) —
сидит на **Isaac Gym Preview, который NVIDIA объявила deprecated, а сам IsaacGymEnvs
заархивирован**, т.е. это источник алгоритмов и гиперпараметров, а не «взял и запустил в Isaac Lab».
AnyRotate и ADEPT кода не опубликовали вообще.

### Cited Findings

**NVIDIA IsaacGymEnvs (ShadowHand, AllegroHand, DeXtreme, AllegroKuka/DexPBT)**
- Репозиторий переехал на `isaac-sim/IsaacGymEnvs`; GitHub API отдаёт `archived: true`,
  последний push `2024-10-26T15:43:53Z`, 2954 звезды, лицензия по API — `NOASSERTION`
  (т.е. кастомный файл лицензии NVIDIA, не распознанный GitHub), 158 открытых issue — https://api.github.com/repos/isaac-sim/IsaacGymEnvs
- README перечисляет dexterous-задачи: `ShadowHand`, `ShadowHandOpenAI_FF`,
  `ShadowHandOpenAI_LSTM`, `AllegroHand`, `AllegroHandDextremeADR`,
  `AllegroHandDextremeManualDR`, `AllegroKukaLSTM`, `AllegroKukaTwoArmsLSTM`, `Trifinger`.
  **`AllegroHandHora` в IsaacGymEnvs нет** (это отдельный репозиторий `HaozhiQi/hora`);
  DexPBT цитируется как метод PBT, DeXtreme — как метод DR, отдельных «DexPBT»-задач в списке
  нет, ближайшее — `AllegroKuka*` — [IsaacGymEnvs README](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/README.md)
- Официальная позиция NVIDIA: IsaacGymEnvs и Isaac Gym Preview Release deprecated, Isaac Lab
  заменяет IsaacGymEnvs / OmniIsaacGymEnvs / Orbit, есть migration guide — [Isaac Lab: From IsaacGymEnvs](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html), [NVIDIA forum: Isaac Gym Deprecation](https://forums.developer.nvidia.com/t/isaac-gym-deprecation-transition-to-isaac-lab/322978)
- По сниппету веб-поиска репозиторий заархивирован владельцем **14 апреля 2026** и стал
  read-only (дата из сниппета, я не открывал баннер архива напрямую; при этом API-поле
  `pushed_at` = 2024-10-26 подтверждает, что кода с 2024 г. не касались) — [web-поиск по IsaacGymEnvs archived](https://github.com/isaac-sim/IsaacGymEnvs)

**NVIDIA DeXtreme**
- Отдельного репозитория `NVlabs/DeXtreme` **не существует** (GitHub API вернул пустой объект) —
  https://api.github.com/repos/NVlabs/DeXtreme
- Код DeXtreme живёт внутри IsaacGymEnvs: задачи `AllegroHandDextremeManualDR` и
  `AllegroHandDextremeADR`, документация — `docs/dextreme.md`. В ней: ADR обучается на
  **8192 env**, ManualDR тестируется на 32–2048 env; «recommended: 8 x NVIDIA A40 на одной
  ноде» — именно на такой конфигурации они экспериментировали; multi-GPU через
  `torchrun --nnodes=1 --nproc_per_node=${GPUS}`. **Pretrained-чекпойнтов документ не
  предоставляет** — описан только флаг `checkpoint=<ckpt_path>` для своих чекпойнтов —
  [IsaacGymEnvs docs/dextreme.md](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/docs/dextreme.md)

**HaozhiQi/hora (In-Hand Object Rotation via Rapid Motor Adaptation, CoRL 2022)**
- Живой: `pushed_at 2026-07-08T00:01:11Z`, 257 звёзд, 0 открытых issue, лицензия по API
  `NOASSERTION` — https://api.github.com/repos/HaozhiQi/hora
- Требует **IsaacGym Preview 4.0**; README прямо предупреждает: «results will be inconsistent
  if you train with IsaacGym Preview 3.0». Рука — **Allegro** (внутренняя версия и публичная
  AllegroHand-v4) — [hora README](https://raw.githubusercontent.com/HaozhiQi/hora/main/README.md)
- **Чекпойнт есть**, качается с Google Drive: `gdown 1oNZAx-B9yGbgeAXBfd_a_sK5RZ--xOTp -O hora_v0.0.2.zip`;
  визуализация `scripts/vis_s2.sh hora_v0.0.2`, оценка `scripts/eval_s2.sh ${GPU_ID} hora_v0.0.2` — [hora README](https://raw.githubusercontent.com/HaozhiQi/hora/main/README.md)
- Два релиза: `v0.0.1` — чекпойнт к статье CoRL 2022 (нужен для воспроизведения чисел из
  статьи), `v0.0.2` — багфиксы и миграция на IsaacGym Preview 4.0 — [hora releases](https://github.com/HaozhiQi/hora/releases) (через веб-поиск)
- Задача — **непрерывное вращение объекта из устойчивого начального захвата** («starting from
  a stable initial grasp»), т.е. palm-supported, и это **не** последовательность дискретных
  целевых ориентаций — [hora README](https://raw.githubusercontent.com/HaozhiQi/hora/main/README.md)

**Visual Dexterity (Chen et al., Science Robotics) — Improbable-AI/dexenv**
- Живой: `pushed_at 2025-11-03T16:08:38Z`, 157 звёзд, **лицензия MIT** — https://api.github.com/repos/Improbable-AI/dexenv
  (зеркало `taochenshh/dexenv` по API не отвечает валидным JSON — считать каноническим Improbable-AI)
- **IsaacGym Preview 3** («results in the paper are trained with Preview 3»); Docker
  поддерживается, но не обязателен. Рука — **D'Claw**, README говорит, что «can be easily
  adapted to other robot hands» — [dexenv README](https://raw.githubusercontent.com/Improbable-AI/dexenv/main/README.md)
- **Pretrained-чекпойнты есть**: `https://huggingface.co/datasets/taochenshh/dexenv/blob/main/pretrained.zip`,
  распаковать в `dexenv/dexenv/`. Пример запуска обучения:
  `python mlp.py alg.num_envs=4000 task.obj.num_objs=10 -cn=dclaw` — [dexenv README](https://raw.githubusercontent.com/Improbable-AI/dexenv/main/README.md)

**Bi-DexHands (PKU-MARL/DexterousHands)**
- `pushed_at 2025-02-18T13:47:57Z`, 1107 звёзд, **Apache-2.0**, 36 открытых issue — https://api.github.com/repos/PKU-MARL/DexterousHands
- Требует **Isaac Gym Preview 3/4**, **Python 3.7/3.8**, драйвер NVIDIA ≥ 470.74. Задача
  `ShadowHandReOrientation` в списке есть, но подробно не расписана. **Pretrained-весов нет**,
  есть только offline-RL датасеты (Hand Over, Door Open Outward). Рука — Shadow Hand, есть
  раздел «How to change the type of dexterous hand» — [Bi-DexHands README](https://raw.githubusercontent.com/PKU-MARL/DexterousHands/main/README.md)
- Заявленная производительность: RTX 3090, 2048 env, 40 000+ FPS (минимум по VRAM не указан) — [Bi-DexHands README](https://raw.githubusercontent.com/PKU-MARL/DexterousHands/main/README.md)

**AnyRotate (Lu, Church, Lin, … Lepora, CoRL 2024)**
- На официальной project page **нет ссылки на код и нет заявления о его публикации** (единственная
  GitHub-ссылка на странице — шаблон сайта nerfies). Рука: «allegro hand (16 DoF) with tactile
  sensors attached on the fingertips»; симулятор на странице не указан — [AnyRotate project page](https://maxyang27896.github.io/anyrotate/)
- Статья: [arXiv:2405.07391](https://arxiv.org/abs/2405.07391)

**ADEPT**
- Найдена только статья: «ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using
  Reinforcement Learning», [arXiv:2608.19182](https://arxiv.org/pdf/2608.19182). Репозитория с
  кодом веб-поиском не обнаружено.

**Isaac Lab (встроенные задачи) — единственная группа, нативная для стека пользователя**
- `isaac-sim/IsaacLab`: не архивный, `pushed_at 2026-09-30`, 8262 звезды, **BSD-3-Clause**,
  default branch `develop` — https://api.github.com/repos/isaac-sim/IsaacLab
- Локальный клон = ровно v2.3.2 (файл `VERSION` = `2.3.2`, HEAD `37ddf626`, 29 Jan 2026), т.е.
  всё ниже — факты именно про ту версию, которую использует пользователь.

**Релизы 2025–2026: Isaac Lab DexSuite**
- В Isaac Lab добавлены dexterous Lift и Reorient среды «following DextrAH and DexPBT», с
  демонстрацией ADR и PBT; это «remake and extension to the original environment
  kuka-allegro-reorientation implemented in the DexPBT paper»; вошло в релиз Isaac Lab 2.3.0 —
  [IsaacLab PR #3378](https://github.com/isaac-sim/IsaacLab/pull/3378) (через веб-поиск)
- Подтверждено локально: пакет `isaaclab_tasks/manager_based/manipulation/dexsuite/` c конфигом
  `config/kuka_allegro/` присутствует в v2.3.2.
- Смежное: `mani-skill/ManiSkill` (Apache-2.0, `pushed_at 2026-08-04`, 3369 звёзд) — https://api.github.com/repos/mani-skill/ManiSkill
  — живая альтернативная GPU-симуляция, но это другой симулятор, не Isaac Lab.

### Inferences
- Практический вывод: **«готовое решение», которое можно запустить на стеке Isaac Lab 2.3.2 /
  Isaac Sim 5.1 без портирования, ровно одно — встроенные задачи Isaac Lab.** Всё остальное
  (hora, DeXtreme, dexenv, Bi-DexHands) — это код под Isaac Gym Preview, который нужно либо
  ставить отдельным устаревшим стеком (Preview 3/4 + Python 3.7/3.8), либо переписывать.
- Isaac Gym Preview deprecated — это блокер не «технический», а «снабженческий»: сам код
  hora/dexenv запустится, если достать дистрибутив Preview 4 и старый Python, но в Isaac Lab
  он не переносится автоматически и совмещать его с Isaac Sim 5.1 в одном окружении нельзя.

### Gaps
- Точный текст лицензий IsaacGymEnvs и hora (API отдаёт `NOASSERTION`) я не читал построчно —
  формулировку «можно ли переиспользовать код» надо проверить в их `LICENSE`.
- Не нашёл подтверждения, доступен ли до сих пор дистрибутив Isaac Gym Preview 4 для скачивания
  (в IsaacGymEnvs есть issue #222 «How to download Isaac Gym Preview 4 release?» — признак, что
  это проблема, но содержимое issue я не открывал).
- Существует ли публичный код AnyRotate вне project page (напр. под другим именем в организации
  Bristol Robotics/Lepora lab) — не проверено.

## Isaac Lab Shadow Hand cube: точные task ID, награда, num_envs, время обучения, бенчмарки

### Takeaway
Task ID и весь reward-конфиг извлечены прямо из локального v2.3.2. Isaac Lab **публикует
бенчмарки пропускной способности и потребления памяти**, но **не публикует ни ожидаемых кривых
награды, ни времени обучения до успеха** — только FPS и GB.

### Cited Findings

**Task ID (файл `source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/__init__.py`, v2.3.2)**
- `Isaac-Repose-Cube-Shadow-Direct-v0` → cfg `ShadowHandEnvCfg`; RSL-RL entry point
  `ShadowHandPPORunnerCfg`; также rl_games и skrl
- `Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0` → `ShadowHandOpenAIEnvCfg`; RSL-RL
  `ShadowHandAsymFFPPORunnerCfg`
- `Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0` → `ShadowHandOpenAIEnvCfg`; **только rl_games**
  (`rsl_rl_cfg_entry_point` отсутствует)
- `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` и `...-Vision-Direct-Play-v0` → класс
  `ShadowHandVisionEnv`; RSL-RL `ShadowHandVisionFFPPORunnerCfg`; в документации помечены
  «Requires running with `--enable_cameras`»
- Все три state-based Shadow-задачи **не имеют Play-варианта** (в таблице документации колонка
  Inference пустая) — [Isaac Lab: Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html) + локальный `docs/source/overview/environments.rst`, строки 975–1012
- Allegro для сравнения: `Isaac-Repose-Cube-Allegro-Direct-v0` (Direct) и manager-based
  `Isaac-Repose-Cube-Allegro-v0` / `-Play-v0`, `Isaac-Repose-Cube-Allegro-NoVelObs-v0` / `-Play-v0`
- DexSuite: `Isaac-Dexsuite-Kuka-Allegro-Reorient-v0` / `-Reorient-Play-v0`,
  `Isaac-Dexsuite-Kuka-Allegro-Lift-v0` / `-Lift-Play-v0`

**`ShadowHandEnvCfg` (v2.3.2, `shadow_hand_env_cfg.py`)**
- `decimation = 2`, `episode_length_s = 10.0`, `action_space = 20`, `observation_space = 157`
  (full), `state_space = 0`, `asymmetric_obs = False`
- Сцена: `num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True`
- Награда: `dist_reward_scale = -10.0`, `rot_reward_scale = 1.0`, `rot_eps = 0.1`,
  `action_penalty_scale = -0.0002`, `reach_goal_bonus = 250`, `fall_penalty = 0`,
  `fall_dist = 0.24`, `success_tolerance = 0.1`, `max_consecutive_success = 0`, `av_factor = 0.1`
- Есть `EventCfg` с событием на `interval_range_s=(36.0, 36.0)` (комментарий в коде:
  «time_s = num_steps * (decimation * dt)»)

**`ShadowHandOpenAIEnvCfg` (наследует предыдущий)**
- `decimation = 3`, `episode_length_s = 8.0`, `action_space = 20`, `observation_space = 42`,
  `state_space = 187`, `asymmetric_obs = True`
- `fall_penalty = -50`, `success_tolerance = 0.4`, **`max_consecutive_success = 50`**,
  остальные веса те же (`dist -10.0`, `rot 1.0`, `rot_eps 0.1`, `action -0.0002`, `bonus 250`,
  `fall_dist 0.24`, `av_factor 0.1`)

**Формула награды (`inhand_manipulation_env.py`, функция `compute_rewards`)**
```
goal_dist = ||object_pos - target_pos||
rot_dist  = rotation_distance(object_rot, target_rot)
dist_rew  = goal_dist * dist_reward_scale
rot_rew   = 1.0 / (|rot_dist| + rot_eps) * rot_reward_scale
reward    = dist_rew + rot_rew + sum(actions^2) * action_penalty_scale
reward   += reach_goal_bonus   where |rot_dist| <= success_tolerance
reward   += fall_penalty       where goal_dist >= fall_dist
resets    = (goal_dist >= fall_dist) | reset_buf
```
- То есть **позиционный член отрицательный (штраф за уход кубика от ладони), ориентационный —
  обратная величина углового расстояния (не потенциал), успех даётся разовым бонусом 250**.

**PPO-гиперпараметры RSL-RL (`shadow_hand/agents/rsl_rl_ppo_cfg.py`)**
- `ShadowHandPPORunnerCfg`: `num_steps_per_env = 16`, `max_iterations = 10000`,
  `experiment_name = "shadow_hand"`, actor/critic `[512, 512, 256, 128]`,
  `num_learning_epochs = 5`, `num_mini_batches = 4`
- `ShadowHandAsymFFPPORunnerCfg`: `num_steps_per_env = 16`, `max_iterations = 10000`,
  `experiment_name = "shadow_hand_openai_ff"`, actor `[400, 400, 200, 100]`,
  critic `[512, 512, 256, 128]`, `num_learning_epochs = 4`, `num_mini_batches = 4`
- `ShadowHandVisionFFPPORunnerCfg`: `num_steps_per_env = 64`, **`max_iterations = 50000`**,
  actor/critic `[1024, 512, 512, 256, 128]`, 5 epochs, 4 minibatches
- rl_games для сравнения: `rl_games_ppo_cfg.yaml` — `horizon_length: 16`,
  `minibatch_size: 32768`, `mini_epochs: 5`, `max_epochs: 5000`;
  `rl_games_ppo_ff_cfg.yaml` — `horizon_length: 16`, `minibatch_size: 16384`,
  `mini_epochs: 4`, `max_epochs: 10000`

**Бенчмарки Isaac Lab для `Isaac-Repose-Cube-Shadow-Direct-v0`**
- Память при **8192 env: 6.7 GB RAM и 6.4 GB VRAM** — [Isaac Lab Performance Benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
- RTX 4090, 8192 env: 200 000 env-step FPS / 190 000 с inference / 170 000 с обучением
- L40, 8192 env: 170 000 / 140 000 / 120 000
- 1 нода × 4 L40: 440 000 / 420 000 / 390 000; 4 ноды × 4 L40: 2 400 000 / 2 300 000 / 1 800 000
- **Страница бенчмарков не содержит ни времени обучения, ни кривых награды** — только
  throughput и память — [Isaac Lab Performance Benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)

### Inferences
- Единственная «официальная» мера длительности обучения, которую можно вывести:
  `max_iterations = 10000 × num_steps_per_env = 16 × 8192 env` ≈ **1.31 млрд шагов среды**. При
  170 000 шагов/с с обучением (RTX 4090) это ≈ 2.1 часа чистого времени. Это оценка «до конца
  расписания», а не «до успеха»; официального подтверждения нет.
- Отсутствие reward-кривых в документации означает, что **сравнивать своё обучение придётся с
  выложенным pretrained-чекпойнтом** (см. следующий раздел), а не с опубликованным графиком.

### Gaps
- Никакого официального «expected reward» / success-rate для Shadow repose-cube в документации
  Isaac Lab я не нашёл. Есть только issue #7398 про регрессию throughput 3.0.0b2 vs v2.3.2, где
  упомянуто «Shadow-Cube unaffected» — но чисел награды там нет (issue не открывал).
- Бенчмарк-таблица даёт память только для 8192 env; значений VRAM для меньших num_envs нет.

## Pretrained-чекпойнты: что реально можно скачать и проиграть

### Takeaway
**Для `Isaac-Repose-Cube-Shadow-Direct-v0` официальный RSL-RL чекпойнт существует и лежит на
публичном S3 NVIDIA** — я проверил HTTP-кодом, не по документации. Для FF- и Vision-вариантов
чекпойнтов нет.

### Cited Findings
- Механизм: `scripts/reinforcement_learning/rsl_rl/play.py` принимает `--use_pretrained_checkpoint`
  и вызывает `get_published_pretrained_checkpoint("rsl_rl", train_task_name)`; путь собирается
  как `{ISAACLAB_NUCLEUS_DIR}/PretrainedCheckpoints/{workflow}/{task_name}/{filename}`, где
  filename = `checkpoint.pt` (rsl_rl), `checkpoint.pth` (rl_games), `checkpoint.zip` (sb3),
  `checkpoint.pt` (skrl); файл кэшируется в `./.pretrained_checkpoints/{workflow}/{task}/`
  (локально: `source/isaaclab_rl/isaaclab_rl/utils/pretrained_checkpoint.py`, v2.3.2)
- Пример из документации: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --use_pretrained_checkpoint` — [Isaac Lab: RL Scripts](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- **HTTP-проверка (2026-09-30)** базы
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/`:

| workflow | task | HTTP | размер |
|---|---|---|---|
| rsl_rl | `Isaac-Repose-Cube-Shadow-Direct-v0` | 200 | 12 249 762 B |
| rl_games | `Isaac-Repose-Cube-Shadow-Direct-v0` | 200 | 6 148 594 B |
| rl_games | `Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0` | 206 (range) | — |
| rsl_rl | `Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0` | **404** | — |
| rsl_rl | `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` | **404** | — |
| rsl_rl | `Isaac-Repose-Cube-Allegro-Direct-v0` | 200 | 19 664 802 B |
| rl_games | `Isaac-Repose-Cube-Allegro-Direct-v0` | 200 | 9 852 848 B |
| rsl_rl | `Isaac-Repose-Cube-Allegro-v0` (manager-based) | 200 | 4 888 738 B |
| rl_games | `Isaac-Repose-Cube-Allegro-v0` | **404** | — |
| rsl_rl | `Isaac-Dexsuite-Kuka-Allegro-Reorient-v0` | 200 | 25 498 997 B |
| rsl_rl | `Isaac-Dexsuite-Kuka-Allegro-Lift-v0` | 200 | 25 499 079 B |
| rl_games | `Isaac-Dexsuite-Kuka-Allegro-Reorient-v0` | **404** | — |
| rl_games | `Isaac-Dexsuite-Kuka-Allegro-Lift-v0` | 200 | 25 487 349 B |
| — | контроль: `Isaac-Bogus-Task-v9` | **404** | — |

  Контрольный запрос по несуществующей задаче даёт 404, поэтому 200/206 — настоящее наличие файла.
  Те же пути отвечают 200 и под `Assets/Isaac/5.0/` и `Assets/Isaac/4.5/`.
- Сторонние чекпойнты: hora — Google Drive `gdown 1oNZAx-B9yGbgeAXBfd_a_sK5RZ--xOTp` — [hora README](https://raw.githubusercontent.com/HaozhiQi/hora/main/README.md);
  dexenv — HuggingFace `taochenshh/dexenv` → `pretrained.zip` — [dexenv README](https://raw.githubusercontent.com/Improbable-AI/dexenv/main/README.md)
- Bi-DexHands готовых весов не даёт — только offline-RL датасеты — [Bi-DexHands README](https://raw.githubusercontent.com/PKU-MARL/DexterousHands/main/README.md)
- DeXtreme (docs/dextreme.md) готовых весов не даёт — [dextreme.md](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/docs/dextreme.md)
- Сообщество также выкладывает обученные на этих задачах политики, напр.
  `cagataydev/strands-isaaclab-shadow-reorient-policy` на HuggingFace (PPO, обучено через
  isaaclab train_policy) — [HuggingFace](https://huggingface.co/cagataydev/strands-isaaclab-shadow-reorient-policy). Качество/происхождение не проверял.

### Inferences
- Для Shadow-задачи чекпойнт существует именно для той конфигурации, где
  `max_consecutive_success = 0` и `success_tolerance = 0.1` — то есть это **не** «OpenAI-подобная»
  политика с ADR, а базовая полностью наблюдаемая (157-мерное наблюдение). Переиспользовать её
  веса на другой руке нельзя (размерности наблюдения/действия жёстко зашиты), но она бесценна
  как **референс для sanity-check самой задачи и как источник целевого поведения для сравнения**.
- Проигрывание чекпойнта (`--num_envs 32`) — самый дешёвый по VRAM способ увидеть эталонное
  поведение на 6 GB.

### Gaps
- Не проверял, какой именно reward/consecutive-success достигает выложенный чекпойнт:
  в Isaac Lab есть файл `pretrained_checkpoint_review.json` рядом с ранами, но публикуется ли он
  на CDN — не проверено.

## Goal stream (поток последовательных целей без reset) vs одна цель на эпизод

### Takeaway
**Встроенные задачи Isaac Lab in-hand — это уже goal stream**, причём реализованный ровно так,
как нужно пользователю: при достижении цели пересэмплируется только целевая ориентация, эпизод
не рестартится. DexSuite — тоже поток, но по таймеру, а не по успеху. hora — вообще непрерывное
вращение (награда за угловую скорость, целей как таковых нет).

### Cited Findings

**Isaac Lab Direct (`inhand_manipulation_env.py`, v2.3.2) — goal stream по успеху**
- `compute_rewards` выставляет
  `goal_resets = where(|rot_dist| <= success_tolerance, 1, reset_goal_buf)`, инкрементирует
  `successes` и добавляет `reach_goal_bonus`.
- В `_get_rewards` затем: `goal_env_ids = self.reset_goal_buf.nonzero(...)` →
  `self._reset_target_pose(goal_env_ids)`. **Цель меняется без вызова `_reset_idx`** — то есть
  без сброса позы руки и кубика. Это и есть поток целей.
- `_get_dones`: эпизод завершается только по `out_of_reach` (`goal_dist >= fall_dist`) или по
  таймауту. При `max_consecutive_success > 0` дополнительно **сбрасывается `episode_length_buf`
  в момент достижения цели** (`self.episode_length_buf = where(|rot_dist| <= tol, 0, ...)`), то
  есть каждой новой цели даётся полный бюджет времени, а эпизод кончается при
  `successes >= max_consecutive_success`.
- Отсюда: `Isaac-Repose-Cube-Shadow-Direct-v0` (`max_consecutive_success = 0`) — поток целей
  внутри фиксированных 10 с; `Isaac-Repose-Cube-Shadow-OpenAI-FF/LSTM-Direct-v0`
  (`max_consecutive_success = 50`) — поток до 50 целей с перезапуском таймера на каждой (это
  прямой аналог OpenAI/DeXtreme-протокола «consecutive successes»).
- Логируется метрика `consecutive_successes` (экспоненциальное среднее с `av_factor = 0.1`):
  `self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()`.

**Isaac Lab manager-based (`manager_based/manipulation/inhand/inhand_env_cfg.py`) — то же, декларативно**
- В конфиге команды: `update_goal_on_success=True`
- Termination: `max_consecutive_success = DoneTerm(func=mdp.max_consecutive_success, params={"num_success": 50, "command_name": "object_pose"})`
- Сцена `num_envs=8192, env_spacing=0.6`, `episode_length_s = 20.0`

**Isaac Lab DexSuite (`dexsuite_env_cfg.py`) — поток по таймеру**
- `mdp.ObjectUniformPoseCommandCfg(..., resampling_time_range=(3.0, 5.0), ranges=Ranges(pos_x=(-0.7,-0.3), pos_y=(-0.25,0.25), pos_z=(0.55,0.95), roll=(-3.14,3.14), pitch=(-3.14,3.14), yaw=(0.0,0.0)))`
  — цель пересэмплируется каждые 3–5 с независимо от успеха; `episode_length_s = 4.0`,
  `num_envs=4096, env_spacing=3, replicate_physics=False`
- Награды: `action_l2` −0.005, `action_rate_l2` −0.005, `fingers_to_object` (std 0.4) +1.0,
  `success_reward` +10, `is_terminated_term(abnormal_robot)` −1; есть ADR-curriculum,
  подкручивающий `pos_tol`/`rot_tol` относительно `pos_std`/`rot_std`
- RSL-RL: `DexsuiteKukaAllegroPPORunnerCfg` — `num_steps_per_env = 32`, `max_iterations = 15000`,
  actor/critic `[512, 256, 128]`, `num_mini_batches = 4`

**hora** — «continuous rotation» из устойчивого захвата, не дискретные цели — [hora README](https://raw.githubusercontent.com/HaozhiQi/hora/main/README.md)

### Inferences
- Для задачи пользователя (поток целей) **буквально нечего изобретать: `_reset_target_pose` +
  `reset_goal_buf` + `max_consecutive_success` из `inhand_manipulation_env.py` — это ~40 строк,
  которые можно перенести один-в-один** (или, в manager-based варианте, взять
  `update_goal_on_success=True` + `mdp.max_consecutive_success`).
- Разница между двумя вариантами важна: при `max_consecutive_success = 0` сложные цели «съедают»
  бюджет эпизода, при `> 0` таймер обнуляется на каждой цели — второй режим заметно мягче для
  curriculum и ближе к «бесконечному» потоку.

### Gaps
- Реализует ли DeXtreme/AllegroHandDextreme поток целей — в `docs/dextreme.md` про
  `max_consecutive_successes` не сказано (fetch не нашёл упоминания), хотя протокол статьи его
  предполагает; надо смотреть YAML-конфиг задачи в IsaacGymEnvs.
- Bi-DexHands `ShadowHandReOrientation`: поток или одна цель — README не раскрывает.

## Fingertip-only / precision grasp vs palm-supported

### Takeaway
Практически весь готовый open source — **palm-supported** (ладонь вверх, кубик лежит в ладони).
Fingertip-only переориентации в готовом виде я не нашёл ни в Isaac Lab, ни в hora, ни в DeXtreme.

### Cited Findings
- Isaac Lab Shadow/Allegro repose-cube: объект удерживается у ладони, термination — по
  `goal_dist >= fall_dist` (`0.24`) от **`in_hand_pos`**, т.е. точки «в ладони»
  (`_get_dones`: `goal_dist = ||object_pos - in_hand_pos||`); наблюдения включают позы кончиков
  пальцев (`fingertip_pos/rot/velocities`) и их силомоменты (`fingertip_force_sensors`,
  `force_torque_obs_scale = 10.0`), но постановка задачи — ладонная (локальный v2.3.2).
- hora: явно «starting from a stable initial grasp» — [hora README](https://raw.githubusercontent.com/HaozhiQi/hora/main/README.md)
- AnyRotate: «tactile sensors attached on the fingertips», rotation about arbitrary axes «in any
  hand direction» (в т.ч. ладонью вниз, гравитационно-инвариантно) — ближайшее к
  fingertip-centric постановке, **но кода нет** — [AnyRotate project page](https://maxyang27896.github.io/anyrotate/)
- DexSuite Reorient: награда `fingers_to_object` (притягивание пальцев к объекту) + рука на
  Kuka-манипуляторе, объект поднимается со стола — это не palm-cradle, но и не precision
  fingertip-reorient (локальный v2.3.2, `dexsuite_env_cfg.py`).

### Inferences
- Готового fingertip-only reorientation-репозитория, который можно взять и запустить, судя по
  всему, нет; ближайшие идейные источники — AnyRotate (без кода) и собственный
  `fingertip_grasp_cache`-подход (уже есть в проекте пользователя, судя по git-статусу).
- Из Isaac Lab можно взять **наблюдения по кончикам пальцев и силомоментные сенсоры** как готовый
  блок — они уже реализованы в `compute_full_state()`.

### Gaps
- Не проверял, есть ли fingertip-precision вариант в DexSuite-конфигах помимо reorient/lift, и
  нет ли в Isaac Lab 3.0.x новых задач такого рода (пользователь на 2.3.2).

## VRAM: что влезает в 6 GB

### Takeaway
Официальная цифра для Shadow repose-cube — **6.4 GB VRAM при 8192 env**, то есть **дефолтный
конфиг на 6 GB не влезает**. Play/inference на 32 env — реалистично; обучение потребует резкого
снижения `num_envs`, и официальных цифр для малых `num_envs` не публикуется.

### Cited Findings
- `Isaac-Repose-Cube-Shadow-Direct-v0`, 8192 env: **6.7 GB RAM, 6.4 GB VRAM** — [Isaac Lab Performance Benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
- Дефолтные `num_envs` в v2.3.2: Shadow Direct `8192`, Allegro Direct `8192`, manager-based
  inhand `8192`, DexSuite Reorient `4096` (локальный v2.3.2)
- rl_games-конфиг Shadow использует `minibatch_size: 32768` — при уменьшении `num_envs` его
  придётся уменьшать вручную, иначе конфиг несогласован (`horizon_length 16 × num_envs` должно
  делиться на minibatch); RSL-RL таких проблем не создаёт, т.к. задан `num_mini_batches = 4`
  (локальный v2.3.2)
- DeXtreme: 8192 env, «recommended 8 × A40» — [dextreme.md](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/docs/dextreme.md)
- Bi-DexHands: заявлено 2048 env на одной RTX 3090 (24 GB) — [Bi-DexHands README](https://raw.githubusercontent.com/PKU-MARL/DexterousHands/main/README.md)
- dexenv: пример `alg.num_envs=4000` — [dexenv README](https://raw.githubusercontent.com/Improbable-AI/dexenv/main/README.md)
- Vision-вариант Shadow: `ShadowHandVisionFFPPORunnerCfg` с сетью `[1024,512,512,256,128]`,
  `num_steps_per_env = 64`, `max_iterations = 50000` и обязательным `--enable_cameras`
  (локальный v2.3.2 + [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html))

### Inferences
- **На 6 GB надо считать так:** VRAM ≈ фиксированная часть (Isaac Sim/PhysX/RTX-контекст) +
  часть, линейная по `num_envs`, + буферы rollout (`num_envs × num_steps_per_env × obs_dim`).
  При 8192 env полный бюджет 6.4 GB — значит запас на фиксированную часть невелик, и снижение
  `num_envs` до 512–1024 должно дать посадку в 6 GB. **Это оценка, официальных цифр для малых
  `num_envs` нет.**
- Vision-вариант (`--enable_cameras`, рендер + сеть 1024-широкая + `num_steps_per_env = 64`) на
  6 GB почти наверняка нереализуем — флагую как «не пытаться».
- DeXtreme/ADR и DexPBT/PBT-режимы на 6 GB нереализуемы по построению: они требуют либо 8 GPU,
  либо популяции агентов.
- Проигрывание опубликованного чекпойнта с `--num_envs 32` — единственный путь, который с
  высокой вероятностью работает на 6 GB без тюнинга.

### Gaps
- Прямых измерений VRAM для Shadow repose-cube при 256/512/1024/2048 env в документации нет; не
  нашёл и issue Isaac Lab с такими измерениями.
- Минимальные системные требования Isaac Sim 5.1 по VRAM я не проверял — а это нижняя граница
  вообще возможности запуска, независимо от задачи.

## Реалистичная трудоёмкость порта на кастомную 20-DOF кисть

### Takeaway
**Isaac Lab-овская in-hand задача hand-agnostic по построению** — она параметризуется тремя
полями конфига, и именно этим один и тот же класс среды обслуживает 16-DOF Allegro и 20-DOF
Shadow. Более того, `action_space` Shadow-задачи **уже равен 20**, так что размерности кастомной
20-DOF кисти совпадают с эталоном. Всё наследие Isaac Gym Preview портируется существенно дороже.

### Cited Findings
- Один и тот же entry point обслуживает обе руки: и `direct/shadow_hand/__init__.py`, и
  `direct/allegro_hand/__init__.py` регистрируют
  `isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv`,
  различаясь только `env_cfg_entry_point` (локальный v2.3.2)
- Точки параметризации в конфиге (оба файла): `robot_cfg: ArticulationCfg = <HAND>_CFG.replace(prim_path=...)`,
  `actuated_joint_names = [...]`, `fingertip_body_names = [...]`, `force_torque_obs_scale = 10.0`
- В `InHandManipulationEnv.__init__` из этого выводится всё остальное:
  `for joint_name in cfg.actuated_joint_names: ...`, `for body_name in self.cfg.fingertip_body_names: ...`,
  `self.num_fingertips = len(self.finger_bodies)`, `self.hand = Articulation(self.cfg.robot_cfg)`;
  наблюдения строятся как `fingertip_pos.view(num_envs, num_fingertips*3)` и т.д. — т.е.
  **число пальцев и число приводов не зашиты в код** (локальный v2.3.2)
- `ShadowHandEnvCfg.action_space = 20`, `observation_space = 157` (full), `state_space = 187`
  в asymmetric-варианте (локальный v2.3.2)
- Другая рука в тех же задачах: Allegro Direct `success_tolerance = 0.2`,
  `max_consecutive_success = 0`, `episode_length_s = 10.0`, `num_envs = 8192` — то есть при
  смене руки в реальности меняли и допуск успеха (локальный v2.3.2)

### Inferences
- **Порт на кастомную 20-DOF кисть в Isaac Lab = написать свой `EnvCfg`**: подставить свой
  `ArticulationCfg`, перечислить 20 `actuated_joint_names`, перечислить `fingertip_body_names`,
  пересчитать `observation_space`/`state_space` под своё число пальцев, подобрать
  `success_tolerance`, `fall_dist` и `in_hand_pos`. Логика среды, награда и поток целей
  переиспользуются без изменений. Это работа порядка одного файла конфигурации, а не среды.
- Формула `observation_space` выводима из `compute_full_observations`/`compute_full_state`:
  члены вида `num_fingertips × (3+4+6)` и `× 6` для силомоментов — значит при числе пальцев
  ≠ 5 обе размерности надо пересчитать вручную (Isaac Lab их не вычисляет сам).
- Порт из hora/dexenv/DeXtreme — это уже не конфиг, а перенос всего env-кода с API Isaac Gym
  Preview на Isaac Lab (`gym.spaces`/tensor API → `DirectRLEnv`/`Articulation`), плюс их
  специфика: у hora — двухстадийное обучение с proprioceptive-history адаптацией, у dexenv —
  teacher-student с point cloud. Их ценность для пользователя — **алгоритм и гиперпараметры**,
  а не код.
- Единственный переносимый «бесплатно» артефакт из не-Isaac-Lab репозиториев — числовые рецепты
  (DR-диапазоны DeXtreme, reward-шейпинг hora, ADR-расписания), потому что они выражены в
  физических единицах, а не в API.

### Gaps
- Не проверял, требует ли `SHADOW_HAND_CFG` в Isaac Lab нативных PhysX-тендонов (в апстриме есть
  PR #7161 «Serve the Shadow Hand from one asset with native PhysX tendons») — если да, кастомная
  кисть без тендонов может потребовать правок актуации; PR я не открывал.
- Нет источника, оценивающего трудоёмкость порта в человеко-часах — любые такие оценки были бы
  выдумкой.
