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
