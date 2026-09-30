# Передача работы Claude — 2026-09-16

Работа остановлена по просьбе пользователя из-за заканчивающегося usage.
**Код нового delta/SysID управления написан, но ещё НЕ проверен в Isaac Sim.**
Успешны только небольшие CPU-тесты, проверка синтаксиса и editable install.
Не выдавать результаты прежней absolute-версии за проверку новой.

## Рабочая папка и границы

Работать ТОЛЬКО в существующем внешнем проекте:

```text
/home/yoba/Documents/work/hand_hybryd_retargeting/isaaclab_ext/dg5f_isaaclab
```

Это самостоятельный git repository. Показанный средой cwd в Downloads устарел.
Не изменять основной ROS/Quest/retargeting проект, IsaacLab или REGRIND.
Не запускать физическую кисть, большой suite или длительное обучение.
Пользователь просит минимум проверок; последний запрос разрешает конкретные
последовательные проверки ниже. Не добавлять камеры/RGB, student, imitation,
domain randomization или менять DirectRLEnv архитектуру.
Куб оставлять **60 мм**, не 600 и не 50 мм.

Есть много незакоммиченных изменений от предыдущей работы. Не делать reset/restore.
В частности, `scripts/rsl_rl/__pycache__/cli_args.cpython-311.pyc` был изменён
до текущего задания. Удалённый generated Cartpole и untracked dg5f_cube/assets —
тоже существующее состояние, а не повод восстанавливать шаблон.
AGENTS.md в соответствующих папках/предках не обнаружен.

Полный текущий запрос пользователя:

```text
/home/yoba/.codex/attachments/f5c52c36-d4e8-4a10-96a6-8a5c870042fb/pasted-text.txt
```

Его надо прочитать. Старые просьбы о contact-aware+dI/dt физической кисти не
являются текущей задачей. Сейчас требуется довести delta-position + SysID
для `DG5F-Cube-Direct-v0` и выполнить указанные проверки.

## Окружение

- Isaac Lab **2.3.2**, Isaac Sim **5.1**, Python 3.11.
- `/home/yoba/Documents/work/IsaacLab/env_isaaclab/bin/python`.
- GPU RTX 3060 Laptop, 6 GB; GUI DISPLAY ранее `:1`.
- CUDA после перезагрузки работала 15 сентября. Старый сбой Xid/reboot из
  README уже был устранён; не считать его актуальным без новой ошибки.
- Никаких наших симуляторов/обучений сейчас не запущено. Последний pip завершился 0.

## Что реализовано

Все пути далее относительно внешнего проекта.

### Управление, размеры и disabled joint

`source/dg5f_isaaclab/dg5f_isaaclab/tasks/direct/dg5f_cube/control.py`
содержит чистые Torch-функции и FIFO.

Default в `dg5f_cube_env_cfg.py`:

```text
control_mode = "delta_position"
delta_action_scale = radians(3)
a[t] = clamp(policy_action, -1, 1)
delivered[t] = a[t-3]
q_target[t] = clamp(q_measured[t] + delivered[t] * radians(3), URDF limits)
```

Именно измерение в момент ПРИМЕНЕНИЯ доставленного действия. Очередь задерживает
нормированные actions, а не старые абсолютные targets. Интегратора команды нет.
Нулевое доставленное действие удерживает измеренное положение; уже находящийся
в очереди ненулевой импульс всё ещё должен дойти.

Сохранён cfg legacy `absolute_position`: старая рабочая кусочная функция с
нулём в initial grasp, -1=lower, +1=upper. Остальные условия SysID/19 actions/delay
те же для A/B. Старые checkpoints 20/72 несовместимы, их не resume.
Zero/random получили CLI `--control_mode`; train использует Hydra override
`env.control_mode=absolute_position` (ещё не проверен в процессе).

Порядок 19 active actions: finger-major rj_dg_1_1..1_4, 2_1..2_4,
3_1..3_4, 4_1..4_4, затем 5_2,5_3,5_4. Сохранены 20 физических DOFs.
Маппинг URDF/native IDs проверяется assert, startup печатает имена и размеры.

`rj_dg_5_1` исключён из policy. Полная команда всегда задаёт ему 0 rad.
Дополнительно через `write_joint_position_limit_to_sim` установлены limits
[0,0], чтобы внешние контакты не двигали его. В reset q=0, dq=0.
**Этот physical lock ещё требует реального sim sanity**: проверить, что
остаётся 20 DOFs и ошибка блокировки мала; не заменять per-step teleport.

Реальный проект подтверждает 0:
`src/lerobot_robot_dg5f/config/bridge.params.yaml` (~117):
disabled_joints=[rj_dg_5_1], disabled_positions_deg=[0.0];
`src/lerobot_robot_dg5f/lerobot_robot_dg5f/dg5f.py` реализует apply_disabled_joints;
`src/dg5f_teleop/dg5f_teleop/retarget_node.py` (~125) также fixed position0.
Основной проект только читали.

### Задержка и observation

`ActionDelayQueue` вызывается ровно один раз в `_pre_physics_step`.
Decimation2 не удваивает задержку. Для всех пальцев D=3 control steps.
History N×3×19, newest-first. До выбора a[t] видны a[t-1],a[t-2],a[t-3].
После шага returned obs содержит a[t],a[t-1],a[t-2].
Полная очередь входит в observation; при reset выбранных сред обнуляется.
Предыдущий action для rate penalty берётся из той же истории.
Для D=0 сохраняется хотя бы один слот, чтобы rate reward имел наблюдаемую память.

Observation: исходные 72 + 57 history = **129**, action_dim=**19**.
Размер вычисляется из joint counts/delay, проверяется shape assertion.
Исходные 72: q20, dq20, cube position3/quaternion4/linear3/angular3,
goal quaternion4, tips15. Камеры отсутствуют.

### SysID

Новый loader `assets/sysid.py`: JSON version1, exact URDF joint_order,
полные четыре словаря, конечные неотрицательные числа, delay целый >=0,
valid disabled joint, duplicate JSON keys, SHA256 для provenance.

Файлы скопированы без изменения из
`/home/yoba/Downloads/Telegram Desktop/hand_sysid_params.{json,pt}`
в `source/dg5f_isaaclab/dg5f_isaaclab/assets/data/`.
JSON — основной источник. PT НЕ загружался/unpickle, только artifact.
`setup.py` включает оба в package_data.

`assets/dg5f.py` и `DG5FCubeEnvCfg.resolve_control_config()` задают через
штатный `ImplicitActuatorCfg` словари stiffness, damping, armature, friction.
В init среды добавлена сверка hand.data соответствующих параметров с JSON.
Config resolve вызывается также до super().__init__, чтобы учесть overrides.

Фактические значения JSON (не копировать в код вручную):

| Параметр | Палец 1, joints1..4 | Пальцы 2–5, joints1..4 |
|---|---|---|
| stiffness | 10.5,7.85,12.5,5.6 | 13.75,12.5,10.5,5.6 |
| damping | .1,.12,.0001,.0001 | .1,.07,.03,.0001 |
| armature | .0001 везде | .0001 везде |
| friction | .001 везде | .001 везде |

Нативный API проверен по установленному исходному коду 2.3.2:
ImplicitActuatorCfg поддерживает эти поля, Articulation пишет их в PhysX.
DelayedPDActuatorCfg — explicit IdealPD и задержка в physics steps, поэтому
для сохранения implicit модели сделан env FIFO в control steps.

**Неоднозначность:** JSON не задаёт единицы/закон трения/backend идентификации.
В Sim5.1 поле friction — static joint friction effort. Значение применено
туда, dynamic/viscous остались defaults; добавлен startup warning.
Нельзя утверждать, что исходный friction law точно воспроизведён.

### Torque

Сохранён прежний cap: `min(0.5 Nm, URDF effort)`; URDF сейчас7.5.
Вынесен cfg `effort_limit_cap_nm=0.5` (None явно выбирает URDF), TODO
"replace with measured DG5F joint torque limits". Не выводить torque из gains.
Velocity limit остаётся π rad/s из URDF.

Логируются signed computed_torque/applied_torque N×20 в extras,
по суставам scalar means в extras['log'], mean/max abs applied и saturation
fraction при finite positive limits. Иначе fraction=-1 +warning.

**Ограничение API:** у ImplicitActuator это приближённые PD estimates и их
clipping, а не фактически измеренные implicit torques PhysX или real hand.
Добавлен явный startup warning. Нельзя выдавать saturation за реальный моторный.

### Reward и reset

Существующая награда сохранена, rate добавлен один раз:

```text
orientation_progress_reward = 10 * (previous_error - error), error rad
palm_distance_penalty       = -1 * distance_from_initial_grasp_region
action_penalty              = -.002 * mean(a[t]^2)
action_rate_penalty          = -.001 * mean((a[t]-a[t-1])^2)
success_bonus               = +2 * first eligible success
drop_penalty                = -20 * fall
```

Все шесть terms и total_reward логируются отдельно; добавлены
orientation_error_deg, success_rate, cube_distance_from_palm,
mean_action_abs, mean_action_delta_abs. Сохранены прежние полезные лог-ключи.
Reset очищает queue/actions/previous/delivered/delta/returns/goal_reached,
восстанавливает исходный grasp/cube и генерирует goal. previous_orientation_error
инициализируется новым goal error. Цель уже в tolerance при reset не даёт bonus.

## Какие файлы изменены в текущем delta/SysID задании

- `source/dg5f_isaaclab/dg5f_isaaclab/assets/sysid.py` — новый loader.
- `.../assets/data/hand_sysid_params.json`, `.pt` — новые artifacts.
- `.../assets/dg5f.py` — native SysID actuator parameters и сохранённый cap.
- `.../tasks/direct/dg5f_cube/control.py` — новый FIFO и mappings.
- `.../tasks/direct/dg5f_cube/dg5f_cube_env_cfg.py` — cfg, dims, SysID resolve.
- `.../tasks/direct/dg5f_cube/dg5f_cube_env.py` — управление, lock, obs/reward/reset/logs.
- `source/dg5f_isaaclab/setup.py` — package_data.
- `scripts/control_sanity.py` — новый helper для zero/random/debug.
- `scripts/debug_delta_control.py` — новый реальный sim impulse/reset debug.
- `scripts/zero_agent.py`, `scripts/random_agent.py` — sanity и mode override.
- `tests/test_delta_control.py` — небольшие CPU contracts.
- `.gitignore` — снято игнорирование tests/.
- Этот `CLAUDE_HANDOFF.md`.

**README.md ещё не обновлён под новые 19/129/SysID/delta.** Его прежние
изменения и старые результаты относятся к прошлой версии; не считать актуальными.
`scripts/list_envs.py`, регистрация задачи, PPO cfg и visual fixes существовали
до этого задания. Train hyperparameters не меняли.

## Что уже проверено — результаты

1. `python -m unittest discover -s tests -v`: **8/8 passed**, 0.008s.
   Лог `logs/delta_sysid/01_static.log`.
   Проверены loader/rejection, импульс t+3, полная history, mixed delays,
   selective/global reset, zero-delay history, delta на текущем измерении/clamp,
   legacy -1/0/+1. Без запуска Kit/физической руки.
2. AST parse **16 Python files**: passed. `git diff --check`: passed.
3. `python -m pip install -e source/dg5f_isaaclab`: **exit0**, установлен0.1.0.
   Лог `logs/delta_sysid/02_install.log`.

**НЕ выполнялись на новой версии:** list_envs, zero, random, sim impulse, PPO.
Новый debug script прошёл только AST, не runtime.

## Продолжить строго последовательно

```bash
cd ~/Documents/work/hand_hybryd_retargeting/isaaclab_ext/dg5f_isaaclab
source ~/Documents/work/IsaacLab/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1

# static и pip уже прошли; повторять только если будут изменения/причина
python scripts/list_envs.py > logs/delta_sysid/03_list_envs.log 2>&1

python scripts/zero_agent.py --task DG5F-Cube-Direct-v0 --num_envs 1 --max_steps 240 --screenshot_dir logs/delta_sysid/zero > logs/delta_sysid/04_zero.log 2>&1

python scripts/random_agent.py --task DG5F-Cube-Direct-v0 --num_envs 4 --max_steps 240 > logs/delta_sysid/05_random.log 2>&1

python scripts/debug_delta_control.py --headless > logs/delta_sysid/06_impulse.log 2>&1

python scripts/rsl_rl/train.py --task DG5F-Cube-Direct-v0 --num_envs 128 --headless --max_iterations 20 > logs/delta_sysid/07_ppo.log 2>&1
```

Не запускать их вместе. Дождаться exit/status, прочитать log, устранить сбой,
только затем следующая проверка. PPO ровно 10–20 iterations, не default150.
Никакого resume старых checkpoints.

Screenshot использует уже существующий GUI viewport, не camera sensor.
Если GUI недоступен, явно сообщить это; численную проверку можно headless без
`--screenshot_dir`. Не притворяться, что изображение проверено.

ControlSanity проверяет dims/history, targets в limits, delta<=3° для
измерений внутри limits, neutral target=current measurement, disabled error
<1e-3 rad, reset history и finite torque. Печатает max target delta,
disabled error и физическое joint-limit нарушение (последнее пока только metric).
Все zero/random finite obs/reward assertions сохранены.

Impulse script даёт +1 одному joint в t0: delivered=[0,0,0,1,0,0,0],
печатает pending/history, q_measured до шага, q_target и delta.
Сверяет формулу на всех actions и reset с ещё pending импульсом,
затем zeros доказывают отсутствие leakage. Печатает estimates torque и limits.

## На что обратить внимание при runtime

- Динамические поля configclass (`active_action_joints`, delays, dims) должны
  нормально проходить Hydra/to_dict/YAML. Env resolve до super уже сделан,
  но это ещё не проверено реальным train.
- Проверить native readback stiffness/damping/armature/friction и cap0.5.
- Проверить lock [0,0], сохранение20 DOFs, disabled actual q, reset в клонах.
- Низкий damping из JSON может влиять на удержание. Не менять SysID молча;
  сначала разобраться и честно сообщить фактическую стабильность.
- Zero current-measured target математически не возвращает к initial pose
  после внешнего смещения; оно может пассивно дрейфовать. Не подменять его
  запрещённым command integrator ради красивого теста.
- При физическом overshoot за предел URDF clamp может дать correction >3°;
  helper проверяет bound при измерениях внутри пределов и отдельно печатает
  actual violation. Не заявлять безусловные3° для out-of-limit q.
- Runtime reset очищает goal/return/history; per-step extras остаются diagnostics
  последнего шага. Проверить адекватность episode logs в PPO.
- Generic simulator warnings не равно ошибка physics, но новые warnings/errors
  расследовать. Полный test suite не нужен.

## Сохраняемая постановка и прежние результаты

Fixed palm-up root (0,0,.6), quat(sqrt(.5),0,-sqrt(.5),0), 120Hz physics,
60Hz control, episode8s, cube60mm/.05kg/friction1,
cube anchor palm-local(.055,0,.1), reset noise1mm/q±.5deg.
Goal вокруг palm X ±20°, tolerance5°. Никаких новых sensors.

URDF `../../models/dg5f/urdf/dg5f_right.urdf`, limits читаются оттуда,
5 tips. Initial joint angles degrees:
`(10,-90,45,35, 0,40,65,30, 0,35,65,30, 0,35,65,30, 0,-10,55,40)`.

Visual проблема уже решена раньше: make_instanceable=False, custom spawner
снимает instancing visuals/collisions, clone_in_fabric=False,
replicate_physics=True, близкий ViewerCfg. Не откатывать эти исправления.
15 сентября старая absolute20/72 версия: zero240steps/0resets, рука и куб
видны, cube held. Артефакты `logs/visual_debug/sixty_mm_grasp/{reset,end}.png`.
Это НЕ результаты новой delta версии.

## Финальный отчёт пользователю после завершения

Обновить README и сообщить: файлы, action formula,19/129, disabled0,
реально применённые SysID, FIFO location/семантика, reward coefficients,
результаты каждого named check и путь нового checkpoint, ограничения
friction/torque API. Не называть smoke готовым обученным Teacher.

Фактические fit metrics из JSON (не «ошибка около1°»):

| Палец | best iteration | best_score | best_rmse_deg |
|---|---:|---:|---:|
| 1 | 125 | 0.003990898374468088 | 3.619578224870321 |
| 2 | 230 | 0.0027273844461888075 | 2.9922357826334274 |
| 3 | 185 | 0.0017874829936772585 | 2.422387360060671 |
| 4 | 70 | 0.0012124880449846387 | 1.9950848236074818 |
| 5 | 30 | 0.001395003986544907 | 2.1399831647284797 |

Это предоставленные fit metrics, не новая sim-to-real оценка.

## Статус 2026-09-16 (продолжение) — все named checks выполнены

03 list_envs, 04 zero (GUI + PNG), 05 random, 06 impulse, 07 PPO 20 iter — пройдены,
результаты и открытые вопросы в README. Изменения в этой сессии:
`scripts/control_sanity.py` (метрики grasp drift; допуск disabled joint 0.25° вместо
1e-3 rad — PhysX-лимит [0,0] даёт до 0.12° при random), README переписан.
Checkpoint: logs/rsl_rl/dg5f_cube_direct/2026-09-16_11-41-23/model_19.pt.
Exit code Kit ненадёжен — смотреть Traceback в логах.

## Статус 2026-09-16, вечер — integrated delta + калибровка физики + Rubik's Cube

Выполнено по новому запросу пользователя (без коммита). Default
`control_mode=integrated_delta_position` (FIFO -> delta -> q_cmd), obs 148 (+q_cmd 19).
DG-5F-M: effort 0.4 Nm (rated), stall/rpm metadata. Коллизии: convex_decomposition +
FilteredPairs rl_dg_1_1/rl_dg_base (источник фантомных скоростей). Хват откалиброван
(`scripts/grasp_hold_sanity.py`), rj_dg_5_1 = limits[0,0] + armature 0.01.
Rubik's Cube: `assets/rubiks_cube.py`. Все результаты и оставшиеся проблемы — README.
Логи: logs/integrated_rubik, logs/physics_sanity, logs/grasp_hold.
PPO smoke: logs/rsl_rl/dg5f_cube_direct/2026-09-16_19-42-29/model_19.pt.

## Статус 2026-09-16, ночь — буквенный куб, цели, первое PPO 500 it

DexCube visual (`assets/object_cube.py`), rejection sampling целей (`goals.py`,
`min_initial_goal_error_deg=10`), успех только после первого доставленного действия,
исправлено логирование extras["log"] (новый dict на шаг, эпизодные ключи всегда).
PPO 500 it: logs/rsl_rl/dg5f_cube_direct/2026-09-16_23-16-44 (+ analysis/). Обучения нет:
std растёт, |a|~±1 растёт, final error хуже zero-baseline. Пользователь просил НЕ запускать
следующий эксперимент до просмотра отчёта; reward/SysID/гиперпараметры не менялись.

## Статус 2026-09-17 — Experiment B (reward v2 + PPO cfg) выполнен

Reward v2 (`success.py`: state reward 0.1·exp(-(err/10°)²), progress 1.0, held success
5° × 18 шагов, bonus один раз), новые метрики, PPO cfg (noise 0.5, entropy 0.001,
[128,128], obs normalization, 32 steps). Физика не менялась. Логи: logs/reward_v2/
(01–08). Run: logs/rsl_rl/dg5f_cube_direct/2026-09-17_01-01-44_reward_v2_expB (+analysis/).
Eval: logs/reward_v2/08_eval.json. Лучший checkpoint model_400: held 0.75, final 6.3°,
в 5° в конце 0.38, drop 0 (zero: 15.0°). Таблица и вывод — README.
Пользователь запретил: новый эксперимент, авто-изменения reward/physics, git commit.

## Статус 2026-09-30 — исследование литературы: причина не в весах награды. Что делать дальше

Полный отчёт и заметки с источниками лежат в корне репозитория-родителя:
`../../reports/Переориентация куба в руке рецепты.md` (69 КБ) и
`../../research_notes/Переориентация куба в руке рецепты/` (5 файлов, 2276 строк).
Здесь — только то, что меняет план работ.

### Контекст: почему исследование понадобилось

Reward v4, стадия A, две детерминированные оценки по 128 эпизодов: `first_goal_held`
0.109 → 0.070, `drop_rate` 0.500 → 0.570, целей/эпизод 0.117 → 0.070, награда −45.7 → −67.1.
Монотонная деградация. **Три гипотезы подряд, все про ВЕСА награды, все опровергнуты
измерением** (детали в JOURNAL.md за 2026-09-30):
1. «Актор держался штрафом reward v3» — рельсы ×10 сдвинули рабочую точку на 3%.
2. «Свежий критик ломает warm-start актора» — заморозка актора на 150 итераций не изменила ничего.
3. «Защита хвата в 11-27 раз слабее задачного терма» — усилил защиту до 125% задачного терма
   (измерено −0.0310/шаг против −0.0247), деградация та же, награда стала хуже (−82 против −47).

Рычаг весов измерен и исчерпан. Дальше его крутить нельзя.

### Три структурные причины, ни одна из них нами не пробовалась

1. **Горизонт слишком короткий.** γ = 0.99 при управлении 60 Hz даёт эффективный горизонт
   100 шагов = **1.7 с** против эпизода 24 с. Механизм зафиксирован в нашем же журнале
   («drop_penalty не дотягивается до действий 900 шагов назад»), но вывод не был сделан.
   DeXtreme на этой же задаче: **γ = 0.998, управление 30 Hz** при физике 60 Hz.
   Hora (`controlFrequencyInv: 6`), AnyRotate, POISE — **20 Hz**.
   Наши 60 Hz — верхняя граница диапазона всей литературы.
   Правка: `gamma` 0.99 → 0.998, `decimation` 2 → 6.
2. **Исследование задавлено.** Ни один работающий конфиг не стартует со std ниже **1.0**,
   и все держат сигму state-independent: `sigma_init: const_initializer val: 0` +
   `fixed_sigma: True` (rl_games), `init_noise_std: 1.0` (Isaac Lab). У нас 0.2 при
   `entropy_coef = 0`, и std сжимается 0.200 → 0.190 → 0.180, то есть политика закрепляется
   в деградирующем поведении. Референс Isaac Lab ровно для repose-cube: `entropy_coef = 0.005`.
   Правка: `entropy_coef` 0.0 → 0.005, `init_noise_std` 0.2 → 1.0 в `PPOStreamRunnerCfg`.
   Возможный сопутствующий механизм (вывод отчёта, не измерение): KL ~ Δμ²/σ², поэтому
   сжимающаяся σ раздувает измеренный KL, KL-adaptive планировщик режет lr, и политика
   замерзает именно тогда, когда перестаёт исследовать. DeXtreme обходит это
   `lr_schedule: linear` вместо адаптивного.
3. **Форма задачного терма, а не его вес.** Нашего baseline-центрирования
   `0.30*(exp(-err/15°) − exp(-20°/15°))` **нет ни у кого**. POISE: `exp(-e_R/30°)` без
   вычитания, положительный везде. Isaac-семейство: неограниченный `1/(|Δθ| + 0.1)` с весом 1.0
   плюс `−10.0·‖p_obj − p_goal‖` и бонус +250. Наша форма отрицательна везде выше величины цели,
   то есть почти всё время. Это третья причина, и она не про вес.

### Запрет угловой градации из спеки надо снимать

Спека запрещала угловую градацию (каждая цель ровно 20°). **Единственная опубликованная
система с такой градацией сообщает ровно тот эффект, который мы измерили сами.**
POISE (arXiv 2609.13761, 2026): цель 5° → 180°, шаг +10° при frontier success 0.4, смесь
выборки near-frontier/exposed/at-limit 0.6/0.3/0.1. Аблация против равномерной выборки полного
диапазона: strict-успех по первой цели **6.2% → 59.5%**, целей/эпизод 0.08 → 1.55.
Наше измерение: 20° → 10° поднял `held` 0.047 → 0.250 и целей/эпизод 0.05 → 0.30 **без
переобучения**, `success_bonus` вырос в 18 раз.

Оговорки, которые надо держать в голове: задача POISE — 6D reaching из хвата, 64000 сред,
аблация одна. AnyRotate использует **фиксированный** шаг 30° (аблация 30/40/50°, 30° лучший),
то есть константу, а не расписание. Все классические системы (Dactyl, Chen 2021, IsaacGymEnvs,
Isaac Lab, DeXtreme, Hora) берут цели равномерно из SO(3) без градации по величине.
В заметках есть внутренний конфликт: `landmark_methods.md` пишет «ни один источник не
использует угловую градацию», `reward_and_curriculum.md` находит POISE. Оба сохранены.

Реализовано на нашей стороне: разовая фаза 10° (`--bootstrap` в `train_stream.py`,
`bootstrap_angle_deg` в `curriculum.py`, `--goal_angle_deg` в `eval_checkpoints.py`).
Чего нет — непрерывной градации и триггера по frontier success.

### Два неприятных факта, которые надо принять

- **Наши 1.6·10⁷ шагов (500 итер. × 1024 сред × 32 шага) — это 0.4% от самого дешёвого
  опубликованного порога.** Все системы, решившие goal-conditioned переориентацию, взяли
  ≥ 4·10⁹ шагов; самая дешёвая (Allshire, TriFinger, 3 пальца, объект на опоре) — 4·10⁹ за ~24 ч
  при 16384 средах. DeXtreme ~8.5·10¹⁰ = **~240 суток** на нашей карте. Политика, которая
  «не учится» на десятках миллионов шагов, находится в том режиме, где все опубликованные
  системы тоже были на нуле.
- **Goal-conditioned fingertip-only переориентация к заданному углу одностадийным PPO не
  опубликована никем и ни при каком бюджете.** Fingertip-only решён только в ослабленных
  формах: неограниченное вращение вокруг оси (Khandate, AnyRotate, Hora) либо цели,
  ограниченные 24 π/2-симметричными ориентациями куба (DLR, плюс фильтр частиц на входе
  политики, 92% в симуляции). Жёсткая цель 20° без симметрии — не бенчмарк ни у кого.
  **Задача, как записана в нашей спеке, жёстче всего опубликованного.**

Противовес, важный для планирования: **«1024 сред мало» не подтверждается ни одним
источником.** Исследование масштабирования Isaac Gym показывает, что число сред покупает
wall-clock (~10× от 256 к 16384) **без** ухудшения финальной награды. Узкое место — батч:
1024 × 32 = 32768 против референсных 131072. Правка: `num_steps_per_env` 32 → 64 даёт 65536,
а при `num_mini_batches = 4` минибатч выходит ровно 16384 — значение DeXtreme. Прецедент
длинного роллаута в апстриме есть: `ShadowHandVisionFFPPORunnerCfg` использует 64.
Сети: наши actor/critic `[128,128]` против `[512,512,256,128]` во всех Isaac-конфигах.

### Находка, бьющая по нашей собственной работе над кэшем хватов

`source/dg5f_isaaclab/dg5f_isaaclab/assets/data/fingertip_grasp_cache_robust.npz` и
`scripts/grasp_cache_robustify.py` — это **по построению Stable Grasp Sampler**, то есть ровно
та арм аблации Khandate (RSS 2023), про которую сказано «fails to learn a viable policy» на
сложных объектах. В кэше устойчивых хватов **нет состояний посреди перехвата**, поэтому шагу
«отпустить палец и переставить» политике не из чего выучиться.

Рабочая альтернатива у Khandate: два RRT (M-RRT с проекцией желаемого движения на многообразие
контактных ограничений через `Δx_proj = (I − NᵀN)Δx_des` плюс QP-проверка способности кисти
создать внутренние силы; G-RRT — модель-агностичные физические роллауты) растятся до 10⁵ узлов,
берутся **десять путей с наибольшим угловым изменением** → ~2·10⁴ состояний → равномерное
распределение сбросов для обычного model-free RL. Аблация безжалостная: Fixed Initialization
«did not learn finger-gaiting even with zero gravity», Explored Restarts не работает даже на
простых объектах, Stable Grasp Sampler не работает на вогнутых. Кода в релизе нет.

Полезное подтверждение нашего выбора: λ_min грасп-грамиана как плотная награда имеет прямое
измерение (arXiv 2607.12105, 2026, fingertip-only Allegro V4, цилиндр R=40 мм): успех
24% → 56% на штатных кончиках и 64% → 83% на кончиках, выровненных под задачу. То есть
`grasp_quality_scale` мы взяли правильно, а неиспользованный рычаг — **геометрия кончиков**.

### Готовые решения: одно живое, четыре тупика

Живое — только встроенные задачи Isaac Lab, BSD-3-Clause, с чекпойнтами. Наличие проверено
HTTP, а не по докам: `Isaac-Repose-Cube-Shadow-Direct-v0` rsl_rl — **12 249 762 байт, 200 OK**
на `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/<task>/checkpoint.pt`
(есть также Allegro-Direct 19.7 МБ, Allegro manager-based 4.9 МБ, Dexsuite-Reorient 25.5 МБ;
**404** для rsl_rl OpenAI-FF и Vision, то есть список не фиктивный). Играется
через `--use_pretrained_checkpoint`.

Что переносится оттуда бесплатно:
- **Поток целей уже реализован, ~40 строк.** В `inhand_manipulation_env.py` `compute_rewards`
  ставит `goal_resets` там, где `|rot_dist| <= success_tolerance`, и `_get_rewards` вызывает
  `_reset_target_pose(goal_env_ids)` — цель перевыдаётся **без** `_reset_idx`, поза кисти и куба
  не трогается, эпизод продолжается. При `max_consecutive_success > 0` дополнительно обнуляется
  `episode_length_buf` на каждой достигнутой цели.
- **Env hand-agnostic по построению:** один `InHandManipulationEnv` обслуживает и 16-DOF Allegro,
  и 20-DOF Shadow, параметризуясь `robot_cfg` / `actuated_joint_names` / `fingertip_body_names`.
  `ShadowHandEnvCfg.action_space = 20` — уже совпадает с нашей кистью. Порт — это написать
  EnvCfg, а не env; ручной шаг один: пересчитать `observation_space` / `state_space`, там
  захардкожены целые числа `num_fingertips × (3+4+6)` и `× 6`.
- Их награда для сверки: `dist -10.0`, `rot 1.0`, `rot_eps 0.1`, `action_penalty -0.0002`,
  `reach_goal_bonus 250`, `fall_penalty 0`, `fall_dist 0.24`, `success_tolerance 0.1`,
  `av_factor 0.1`, `decimation 2`, `episode_length_s 10.0`, 8192 сред.
  Вариант OpenAI: `decimation 3`, 8.0 с, асимметрия, `fall_penalty -50`,
  `success_tolerance 0.4`, `max_consecutive_success 50`.

Тупики: **IsaacGymEnvs заархивирован, Isaac Gym Preview объявлен deprecated** — на нём сидят
DeXtreme (плюс 8×A40 и без весов), hora (Preview 4.0), dexenv / Visual Dexterity (Preview 3),
Bi-DexHands (Python 3.7). Годятся только как источник чисел. **AnyRotate и ADEPT кода не
выложили вообще.** Fingertip-only переориентации в работающем open source нет — задачи Isaac Lab
палм-опорные (терминация по `‖object_pos − in_hand_pos‖ >= fall_dist = 0.24`), hora стартует
«from a stable initial grasp». То есть наша работа по `fingertip_grasp_cache` ничего не дублирует.

Ограничение по VRAM: бенчмарк Isaac Lab даёт Shadow-Direct при 8192 средах **6.4 ГБ VRAM**,
то есть дефолтный конфиг в наши 6 ГБ не влезает. Для малых `num_envs` NVIDIA чисел не
публикует, так что оценки на 512-1024 среды — вывод, не факт. Vision-вариант
(`--enable_cameras`, сеть 1024, 50000 итераций) считать неприменимым. Проигрывание готового
чекпойнта при `--num_envs 32` — единственный путь, который почти наверняка заработает без тюнинга.

### Порядок работ

Правки уровня конфига, каждая измеряется отдельным прогоном, ни одна не про веса награды:

1. `gamma` 0.99 → 0.998 и `decimation` 2 → 6 (60 → 20 Hz).
2. `entropy_coef` 0.0 → 0.005, `init_noise_std` 0.2 → 1.0. Логировать σ и lr на одной оси,
   чтобы увидеть петлю KL → lr.
3. `num_steps_per_env` 32 → 64; критик увеличить и сделать асимметричным.
4. Снять baseline-центрирование с задачного терма либо расширить `orientation_decay_deg`
   15° → 30°. Проверить `fall_penalty = 0` — DexReMoE **намеренно убрали** штраф за падение:
   «such a term suppresses exploratory actions and adversely affects the overall training».
5. Превратить разовую фазу 10° в непрерывный frontier-curriculum по POISE.

Дороже, недели:

6. Заменить распределение сбросов на состояния «после перехвата» по Khandate вместо
   Stable Grasp Sampler.
7. **Начинать стоит с этого, оно дешёвое и решает, нужен ли пункт 6.** Кинематическая
   диагностика: проинтегрировать спроецированное движение от текущего хвата с QP-проверкой
   до потери допустимости и получить **максимальный поворот 60-мм куба без перехвата** для
   нашей кисти. Если 20° недостижимы без gaiting, никакая форма награды не поможет.
   Косвенное основание: KaRMA (arXiv 2605.15548, 16 кистей) — удержание контакта в щипке
   thumb-index покрывает не более ~35.5% из 228 ориентационных ячеек на лучшей кисти,
   то есть gaiting не опционален. Но KaRMA мерил на 10-мм сфере, поэтому утверждение
   «60 мм у границы применимости» остаётся непроверенным: 60 мм попадает внутрь
   продемонстрированного диапазона (Hora 45-75 мм, AnyRotate до 90 мм), но в ту его часть,
   где успехи 24-83%, и геометрия куба названа отдельно сложным случаем.

Прекратить: крутить веса награды; ADR (съедает 40% сред на оценку и покупает надёжность на
железе, а не выборочную эффективность); PBT (нет запаса памяти); Vision-вариант; HER под PPO
(все варианты off-policy, релейблинг ломает importance ratio — это переход на SAC/TD3, то есть
переписывание); портирование кода с Isaac Gym Preview.

Де-скоуп, если пункт 7 скажет, что 20° без перехвата недостижимы: промежуточная веха в классе
Hora — fingertip-only **непрерывное вращение вокруг одной оси**, один класс объектов,
как бинарный тест на способность кисти к gaiting. Контр-довод сохранён: AnyRotate показывает,
что многоосевые цели требуют вспомогательных keypoint-наград, иначе «near-zero successive goals».

### Что НЕ менялось и что запрещено

Физика не менялась ничем из перечисленного. Куб 60 мм. Правки пунктов 1-5 — это
`agents/rsl_rl_ppo_cfg.py`, `dg5f_cube_env_cfg.py` и `curriculum.py`, физику не трогают.
Ничего не закоммичено. Длинный прогон не запущен. Запуск, отключение DR как диагностика,
и любое изменение reward/physics — только с разрешения пользователя.

## Статус 2026-09-30, передача — текущее состояние и решённое направление

**Это последний раздел. Читать его первым.**

### Решение пользователя о направлении работ

После исследования литературы (раздел выше) пользователь выбрал направление:
**воспроизвести УЖЕ РЕШЁННУЮ задачу Isaac Lab на кисти DG5F**, вместо продолжения правок
своей задачи. Причина в том, что три возможные причины провала сейчас неразличимы:
(1) модель кисти/управление, (2) постановка задачи, (3) нехватка вычислений — все три дают
одинаковую картину «не учится». Воспроизведение решённой задачи их разделяет, и результат
информативен в обе стороны:

- **учится** → кисть, физика, управление и пайплайн исправны, проблема в нашей постановке;
- **не учится** → проблема в модели кисти или управлении, и её надо чинить прежде всего.

Задача для воспроизведения: `Isaac-Repose-Cube-Shadow-Direct-v0`. Проверено по локальной копии
`/home/yoba/Documents/work/IsaacLab` (версия ровно 2.3.2):
`source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py`
— `action_space = 20` (совпадает с DG5F), `observation_space = 157`, `state_space = 0`,
`decimation = 2`, `episode_length_s = 10.0`, `num_envs = 8192`, `env_spacing = 0.75`;
награда `dist_reward_scale = -10.0`, `rot_reward_scale = 1.0`, `rot_eps = 0.1`,
`reach_goal_bonus = 250`, `fall_penalty = 0`, `fall_dist = 0.24`, `success_tolerance = 0.1`,
`max_consecutive_success = 0`, `av_factor = 0.1`.
Вариант `Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0`: `decimation = 3`, `episode_length_s = 8.0`,
`observation_space = 42`, `state_space = 187` (асимметрия), `fall_penalty = -50`,
`success_tolerance = 0.4`, `max_consecutive_success = 50`.

`InHandManipulationEnv` hand-agnostic по построению — один и тот же код обслуживает 16-DOF
Allegro и 20-DOF Shadow, различие только в `robot_cfg`, `actuated_joint_names`,
`fingertip_body_names`. **Порт — это написать EnvCfg, а не среду.** Ручной шаг один:
пересчитать `observation_space` / `state_space`, там захардкожены целые числа
`num_fingertips × (3+4+6)` и `× 6`, Isaac Lab их не выводит.

Эталон «как выглядит работающее»: готовый чекпойнт, наличие проверено HTTP (200 OK, 12 249 762 B):
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Repose-Cube-Shadow-Direct-v0/checkpoint.pt`
Играется через `--use_pretrained_checkpoint`. Рекомендуется запустить это ПЕРВЫМ делом:
проверяет стек и даёт визуальный эталон. При 6 ГБ VRAM брать `--num_envs 32`.

Чем задача Isaac Lab легче нашей (это и есть список того, что потом возвращать по одному):

| | Isaac Lab | наша |
|---|---|---|
| хват | куб лежит на ладони, гравитация помогает | висит в щипке на кончиках |
| критерий падения | куб уехал на **0.24 м** | потерян контакт кончиков |
| цели | случайные из SO(3) | поток относительных по 20° |
| допуск | 0.1 рад = 5.7° (Direct) / 0.4 рад = 22.9° (OpenAI) | 5° |
| сред | 8192 | 1024 |

Главная разница — первые две строки, НЕ допуск: 5.7° против наших 5° почти одинаковы, а вот
«уронил = уехал на 24 см» против «уронил = разжал кончики» это разные по трудности задачи.
Отсюда и вся наша борьба с `drop_rate`.

После того как базовая задача поедет, возвращать наши требования **по одному, с замером
каждого**: (1) ладонь → щипок кончиками, (2) случайные цели → поток по 20°, (3) мягкий
критерий падения → наш жёсткий. Смысл — узнать цену каждого требования отдельно; сейчас
мы платим за все сразу и не знаем, за что именно.

Бюджет: эталонный прогон ~1.3·10⁹ шагов, у нас это **2-4 суток** непрерывно при ~8000 шаг/с
(1024 сред × 32 шага / ~4 с на итерацию). 8192 сред в 6 ГБ не влезают (бенчмарк Isaac Lab даёт
6.4 ГБ VRAM), нужно 1024-2048 — меньше батч и, возможно, больше итераций. Реалистичная цель:
**симуляция, один объект, сначала на ладони.** Не железо и не сразу щипок.

### Что закоммичено этим коммитом

Вся работа по reward v4 / goal stream: визуализация цели, поток целей, стадии A/B/C,
робастный кэш хватов, warm start, аудит скоростей, pre-flight награды, драйвер стадий,
фаза 10°, односторонние штрафы защиты хвата, 68 тестов. Плюс отчёт исследования
(`reports/`) и заметки с источниками (`research_notes/`) — они нужны, потому что этот хэндоф
и `JOURNAL.md` на них ссылаются.

### Состояние окружения на момент передачи

- **Обучение не запущено, GPU свободна.** Длинный прогон остановлен вручную на 500 итерациях.
- Состояние провалившегося прогона: `logs/stream/state.json` (оценки @250 и @500 внутри).
  Если запускать `train_stream.py` заново без `--resume_state`, он этот файл перезапишет —
  сначала переименовать.
- Тесты: `python -m pytest tests/test_delta_control.py -q` → 68 passed. Один тест флакует
  (`test_f_degenerate_band_does_not_crash`, ~1 падение на 6 прогонов): `sample_angles_haar`
  отбраковкой не сходится, плотность 1−cos θ вырождается у нуля. На боевом пути НЕ лежит
  (поток целей идёт через `sample_fixed_angle_goals`, Haar нужен только отключённому угловому
  curriculum). Не исправлено сознательно — вне задачи. Если угловой curriculum включат, он
  уронит прогон исключением.
- Чекпойнты для сравнения: `logs/rsl_rl/dg5f_cube_direct/2026-09-29_00-00-00_warm_start_v4/model_0.pt`
  (warm start, актор с политики удержания), `..._v5_smoke/model_99.pt` (100 итераций новой
  награды), `..._stream_A_00250/model_498.pt` (500 итераций, деградировавшая).

### Ловушки окружения, на которые уже наступили

1. **`--resume` через hydra МОЛЧА не работает.** `scripts/rsl_rl/cli_args.py`:
   `if args_cli.resume is not None: agent_cfg.resume = args_cli.resume`, а `--resume` объявлен
   `action="store_true", default=False`, то есть никогда не None и безусловно затирает значение
   в False. Hydra-override `agent.resume=true` принимается без предупреждения и выбрасывается,
   при этом `agent.load_run` и `agent.load_checkpoint` переживают и попадают в
   `params/agent.yaml` — сохранённый конфиг выглядит правильным, а прогон учится с нуля.
   **Только флагами: `--resume --load_run <имя> --checkpoint <файл>.pt`** (именно `--checkpoint`).
   Проверять по строке `Loading model checkpoint from` в логе, а не по yaml. Цена ошибки уже
   заплачена: четыре смока и две неверные диагностики.
2. **При warm start первым делом сверять тензоры чекпойнта с исходником**, а не диагностировать
   динамику обучения. Одно сравнение на 30 секунд отменило два часа рассуждений.
3. **Обучающий `drop_rate` здесь не показатель.** Действия при обучении семплируются и
   интегрируются в `q_cmd`, поэтому команда уходит случайным блужданием: при σ=0.2 это
   0.2 × 1° × √1440 ≈ 7.6° на сустав только от шума. Судить только по детерминированной оценке
   (`scripts/eval_checkpoints.py`).
4. **Упавший скрипт с незавершённым `simulation_app.close()` висит живым процессом и держит
   ~1 ГБ GPU.** Проверять `nvidia-smi --query-compute-apps` и убивать.
5. **GUI и обучение одновременно не запускать** — карта 6 ГБ, обучение берёт ~5.3 ГБ,
   Vulkan OOM убьёт прогон.
6. **Порог критерия задавать по измеренному РАСПРЕДЕЛЕНИЮ величины, а не по её среднему.**
   Три случая одного класса в проекте: `min_mean_tips` (перенёс мгновенный порог на среднее,
   отбраковал 790 из 897 здоровых хватов), reward v3 (вес размерен по предполагаемым 3°
   против измеренных 14.9°), `grasp_quality_floor` (порог по среднему занизил цену штрафа
   в 6 раз — штраф обрезан, то есть выпуклый, и по Йенсену среднее его недооценивает).
   `scripts/reward_preflight.py` теперь печатает перцентили p01…p90 — без них ошибка не видна.
7. **При смене масштаба задачи проверять все термы с порогом в абсолютных единицах этого же
   масштаба.** Их было два: `min_initial_goal_error_deg` (ловится ассертом в env, виден сразу)
   и окно анти-перелёта `goal_velocity_error_deg` (ничем не ловилось, при цели 10° накрыло весь
   подход вместо финала, −0.0025/шаг против −0.0003; исправлено на
   `min(goal_velocity_error_deg, 0.5 * goal_stream_angle_deg)` и только при `goal_stream`).

### Чего НЕ делать без разрешения пользователя

Физика: SysID stiffness/damping, armature/friction, лимиты момента, масса и геометрия куба,
геометрия коллизий, задержка управления, частота управления, отключённый сустав
`rj_dg_5_1`, масштаб delta-действия 1°, определение контакта кончиков. **Куб 60 мм.**
Не менять Dex Hybrid / Vector / DexPilot / MANO / преобразования координат / маппинг суставов.
Не добавлять камеры, RGB, imitation, Quest, teacher/student, domain randomization, новые
объекты, наклон кисти, гравитационный curriculum.
Не запускать длительное обучение и не менять reward/physics по своей инициативе.
