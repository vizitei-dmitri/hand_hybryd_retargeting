# DG5F Cube — Isaac Lab Direct PPO Teacher

Внешний проект для **Isaac Lab v2.3.2 / Isaac Sim 5.1**. Задача Gymnasium:
`DG5F-Cube-Direct-v0`, класс `DG5FCubeEnv(DirectRLEnv)`. Цель — state-based
PPO teacher для DG5F с цветным кубом Рубика 60 mm. Камер, student, imitation и
domain randomization нет; цветной куб подготовлен для будущего vision-этапа.

Рабочая копия: `~/Documents/work/hand_hybryd_retargeting/isaaclab_ext/dg5f_isaaclab`.
Исходники Isaac Lab, REGRIND и ROS/Quest/retargeting не изменяются.

## Модель руки

**Аппарат: TESOLLO DG-5F-M (right).** Выведено из реального стека, не из имени URDF:
`vendor/tesollo_control` использует DGSDK `DG_MODEL_DG_5F_RIGHT = 0x5F22`
(суффикс `..02` в SDK соответствует M-серии: `DG_1F_M=0x1F02`, `DG_3F_M=0x3F02`),
Modbus TCP порт 502; URDF взят из `tesollodelto/delto_m_ros2`.
Табличка на корпусе не проверялась.

| Параметр DG-5F-M | Значение | Источник | Применение |
|---|---|---|---|
| rated joint torque | 0.4 Nm | Knoxlabs DG-5F-M listing | `effort_limit_sim` |
| stall joint torque | 2.0 Nm | Knoxlabs; пресс-материалы Tesollo | только metadata |
| no-load speed | 75 rpm = 7.854 rad/s | Knoxlabs; пресс-материалы Tesollo | только metadata |
| control cycle | 250 Hz | tesollo.com DG-5F-M | только metadata |

Официальная страница tesollo.com DG-5F-M подтверждает 250 Hz, 20 DoF, 1763 g, но
текст torque/rpm в ней не извлекается. Константы: `assets/dg5f.py`.

- `effort_limit_sim = min(0.4, URDF 7.5) = 0.4 Nm` для всех 20 суставов
  (`effort_limit_cap_nm`, `None` = URDF). ImplicitActuator в 2.3.2 имеет один предел
  без continuous/peak и тепловой модели, поэтому взят rated, не stall.
  TODO: current/thermal actuator model по реальной руке.
- Velocity limit оставлен **π rad/s из URDF** (readback PhysX подтверждён), а не
  7.854 rad/s: no-load speed — верхняя граница без нагрузки, URDF консервативнее и
  согласован с шагом 3°/60 Hz = 180°/s. PhysX ограничивает скорость привода, но
  контакт может сдвинуть звено быстрее (random: до 3.2–3.6 rad/s).

Коллизии: `collider_type="convex_decomposition"` (convex hull ладони поглощал
основание большого пальца, постоянный контакт ~1 kN) и
`UsdPhysics.FilteredPairsAPI` для пары `rl_dg_1_1`–`rl_dg_base`
(`DG5F_FILTERED_COLLISION_PAIRS`): PhysX фильтрует только прямые parent/child, а
при `merge_fixed_joints=False` основание большого пальца постоянно проникало в
корпус base (0.2–1.4 MN). `scripts/contact_audit.py` после исправления: самоконтакты
≤ 2 N.

## Управление: integrated delta position (default)

```text
a[t]          = clamp(policy_action, -1, 1)                     # 19 значений
delivered[t]  = a[t-3]                                           # FIFO в control steps
q_cmd[t]      = clamp(q_cmd[t-1] + delivered[t] * delta_action_scale, URDF lower, upper)
PD target     = q_cmd[t];   rj_dg_5_1 target = 0
delta_action_scale = radians(3)
```

Задержка действует на delta **до** интегрирования: `ActionDelayQueue`
(`tasks/direct/dg5f_cube/control.py`) вызывается один раз за control step в
`DG5FCubeEnv._pre_physics_step`, затем `position_targets(...)` добавляет
доставленный delta к `q_cmd`. При нулевом действии `q_cmd` не меняется, и PD
продолжает удерживать последнюю команду.

Режимы для A/B (`control_mode`): `integrated_delta_position` (default),
`measured_delta_position` (прежний: `q_measured + delta`) и `absolute_position`
(grasp-centered −1/0/+1). Checkpoints прежних версий несовместимы.

Порядок 19 действий: `rj_dg_1_1..1_4, 2_1..2_4, 3_1..3_4, 4_1..4_4, 5_2, 5_3, 5_4`.

## Наблюдение (148)

| Индексы | Состояние | Размер |
|---|---|---|
| 0:20 | Положения 20 суставов | 20 |
| 20:40 | Скорости суставов (PhysX) | 20 |
| 40:53 | Куб в системе ладони: position, quaternion wxyz, lin/ang velocity | 13 |
| 53:57 | Целевая ориентация | 4 |
| 57:72 | Пять fingertips относительно ладони | 15 |
| 72:129 | Очередь действий a[t-1], a[t-2], a[t-3] × 19 | 57 |
| 129:148 | Текущая `q_cmd` 19 активных суставов | 19 |

Размер вычисляется в `resolve_control_config()`, форма проверяется assertion.
Reset: `q_cmd` = grasp command, очередь/actions/success/return = 0, `rj_dg_5_1` = 0.

## Стартовый хват

Хранится в `DG5FCubeEnvCfg`: `grasp_joint_pos_deg` (состояние при reset),
`grasp_command_offset_deg` (preload: команда = состояние + offset),
`cube_position_in_palm`. Подобран `scripts/grasp_hold_sanity.py --mode search`:
пальцы закрываются на кубе из раскрытого положения, состояние дважды
пересаживается в найденное равновесие, preload ≤ 0.5° только на сгибающих
суставах (отводящие/оппозиция = равновесный угол), ранжирование по худшему из 4
удержаний с шумом reset. Итог: исходный хват + `rj_dg_5_2` +2°, куб +3 mm по нормали.
Лог: `logs/grasp_hold/06_search_flexion_preload_noisy.log`.

## Неработающий `rj_dg_5_1`

Не входит в action space; target 0 rad (подтверждено `bridge.params.yaml`).
Фиксация: PhysX limits `[0, 0]` + `disabled_joint_lock_armature = 0.01` только на
этой оси (заменяет SysID armature этого сустава). 20 DOF сохраняются. Отклонение при
random: 0.065° без armature → 0.0019° с ней; ужесточение drive (1000 Nm/rad) не
помогает. Допуск `control_sanity` 0.02°. Настоящий fixed joint убрал бы DOF из
артикуляции (19 DOF), поэтому не использован.

## SysID

`assets/sysid.py` загружает `assets/data/hand_sysid_params.json`; через
`ImplicitActuatorCfg` применены stiffness, damping, armature, friction.

Fit metrics из JSON (метрики идентификации, не sim-to-real оценка):
best_rmse_deg 3.620 / 2.992 / 2.422 / 1.995 / 2.140 для пальцев 1–5.

Ограничения: friction без единиц/закона — записан как static joint friction;
`computed/applied_torque` у ImplicitActuator — PD-оценки, не измеренные моменты.

**Недодемпфированные суставы:** damping 0.0001 для 1_3, 1_4 и всех x_4 даёт
ζ ≈ 0.001–0.005 при собственной частоте 40–94 Hz (из матрицы масс PhysX);
у остальных ζ ≈ 0.34–1.3. При скачках команды эти суставы звенят (random: смена
знака скорости в 21–38% шагов). SysID не менялся.

## Скорости: PhysX vs конечная разность

`scripts/velocity_sanity.py` сравнивает `qdot_physx` с `(q_t - q_{t-1})/dt` на
каждом physics substep. Фантомная скорость (например, `rj_dg_1_2` ≈ 2 rad/s при
неизменном угле) возникала только при постоянном проникновении коллизий: без
контактов PhysX совпадает с FD точно. Параметры solver её не устраняли:

| Zero, установившийся режим (t>1 s) | mean\|err\| rad/s | max qdot_physx | толчок после reset |
|---|---:|---:|---:|
| A: старая геометрия и хват | 0.654 | 3.142 | 23.1 |
| A: + external forces every iteration (полный прогон) | 0.780 | 8.90 | 23.1 |
| A + external forces + velocity iterations 1 (рука и куб) | 0.561 | 3.142 | 20.9 |
| A + velocity iterations 4 / 0 | 0.667 / 0.561 | 4.15 / 3.35 | 15.6 / 9.8 |
| A + max_depenetration_velocity 0.05 | 0.585 | 3.143 | 5.0 |
| без куба и без self-collision | 0.000 | 0.000 | 2.4 |
| **итог: исправленная геометрия + хват** | **0.031** | **0.21** | **0.83** |
| итог + external forces every iteration | 0.036 | 0.23 | 0.84 |

Solver оставлен по умолчанию (`enable_external_forces_every_iteration=False`,
8/2 итераций). При random расхождение 0.6 rad/s — реальное быстрое движение
недодемпфированных суставов, FD тоже доходит до π.

## Объект: буквенный куб (NVIDIA DexCube)

`assets/object_cube.py`, `VisualCuboidCfg`:
- visual: штатный `f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"`
  (куб из примеров Isaac Lab Allegro/Shadow in-hand): 6 цветных граней с буквами
  X, R, E, T, M, D. Подключается только подветка `/DexCube/visuals`: корень ассета
  несёт собственные rigid body, массу и коллайдер. Родное ребро 60.0 mm → масштаб 1.0,
  поворот 0, центрирован, дочерний prim тела;
- collision/rigid body: скрытый USD cube 60 mm (`cube_size_m`), масса 50 g,
  friction 1.0/1.0, restitution 0, один rigid body (физика не менялась).

`scripts/cube_visual_check.py`: bbox визуала в системе тела = ±30.0 mm, в визуальной
подветке нет физических API, коллайдер невидим, поза для рендера (Fabric) =
`root_pose_w`; скриншоты `logs/letter_cube/cube/`.

## Цели и метрика успеха

`goals.py`: rejection sampling цели, пока угол до начальной ориентации куба
(identity в системе ладони) не станет ≥ `min_initial_goal_error_deg = 10`.
Режим `axis`: ±20° вокруг X ладони, значит фактически 10–20° обоих знаков.
`scripts/goal_reset_sanity.py`: 10 000 reset через `_reset_idx` без rollout, минимум
10.001°, внутри допуска 5° — 0.

Успех засчитывается только после доставки первого действия политики
(`episode_length > delay`), первый раз за эпизод. Эпизодные логи (среднее по
завершённым эпизодам): `episode_success_rate`, `episode_drop_rate`,
`episode_initial/final_orientation_error_deg`, `episode_time_to_first_success_s`,
`episode_reward`, `episode_length_s`. Итерация без завершённых эпизодов даёт `nan`.
`episode_length_s` включает случайный стартовый сдвиг RSL-RL (`init_at_random_ep_len`).

Исправлено логирование: раньше `extras["log"]` был одним и тем же dict на всех шагах
(RSL-RL усреднял последний шаг), а эпизодные ключи терялись, если первый шаг итерации
был без reset.

## Награда (reward v2, 2026-09-17)

```text
orientation_state_reward    = 0.1   * exp(-(error / 10 deg)^2)        # плотная: «быть у цели»
orientation_progress_reward = 1.0   * (previous_error - error)        # rad; v1: 10.0, без state
palm_distance_penalty       = -1    * distance_from_initial_grasp_region
action_penalty              = -0.002 * mean(a[t]^2)
action_rate_penalty         = -0.001 * mean((a[t]-a[t-1])^2)
success_bonus               = +2    один раз, когда held success впервые выполнен
drop_penalty                = -20   * fall
```

State reward: 0° 0.100, 2.5° 0.094, 5° 0.078, 10° 0.037, 15° 0.011, 20° 0.002, 30° 0.00001.
`success.py`: `HeldSuccessTracker` — успех = ошибка ≤ 5° в течение
`success_hold_steps = round(0.30 s / control_dt) = 18` ПОСЛЕДОВАТЕЛЬНЫХ шагов
(выход из допуска обнуляет счётчик); шаги до доставки первого действия не считаются.
Эпизодные логи: held/entered success, final/mean/min error, time in tolerance,
time to held success, drop, reward, length; `abs_contribution/<term>` — средний |вклад|.
Zero policy (128 эпизодов): held 0, drop 0, final 15.0°, episode reward +6.08.

## Команды

```bash
cd ~/Documents/work/hand_hybryd_retargeting/isaaclab_ext/dg5f_isaaclab
source ~/Documents/work/IsaacLab/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1

python -m unittest discover -s tests -v
python scripts/list_envs.py
python scripts/zero_agent.py --task DG5F-Cube-Direct-v0 --num_envs 1 --max_steps 240 --screenshot_dir logs/letter_cube/zero
python scripts/cube_visual_check.py --screenshot_dir logs/letter_cube/cube
python scripts/goal_reset_sanity.py --headless --num_envs 1000 --rounds 10
python scripts/grasp_hold_sanity.py --headless --mode validate --num_envs 8 --hold_s 5
python scripts/contact_audit.py --headless --num_envs 2
python scripts/velocity_sanity.py --headless --actions zero   # и --actions random
python scripts/debug_delta_control.py --headless
python scripts/random_agent.py --task DG5F-Cube-Direct-v0 --num_envs 4 --max_steps 240 --headless
python scripts/rsl_rl/train.py --task DG5F-Cube-Direct-v0 --num_envs 128 --headless --max_iterations 20
python scripts/analyze_training.py logs/rsl_rl/dg5f_cube_direct/<run>
```

Exit code Kit ненадёжен — смотрите `Traceback` в логах. Диагностические скрипты
печатают traceback до закрытия Kit.

## Результаты 2026-09-16 (`logs/integrated_rubik/`)

| Проверка | Результат |
|---|---|
| A unit | 10/10 |
| B list_envs | задача найдена |
| C zero, 1 env, GUI | 240 шагов, 0 reset, obs (1,148); `q_cmd` не менялась; дрейф хвата 0.49° (было 28.3° при measured delta); куб 1.23 mm; disabled 0.0006°; max qdot 0.54; saturation 0 |
| Rubik check (прежний визуал) | PASS |
| D grasp hold, 8 env, шум, 5 s | 0 падений; дрейф ≤ 0.94°; куб ≤ 1.79 mm / 1.33°; saturation 0; max qdot 0.25 |
| E velocity | zero 0.031 rad/s; random 0.61 rad/s (реальное движение) |
| F delay | delta дошла до `q_cmd` при t=3 (+3.000°), удерживается; reset очистил |
| G random, 4 env | шаг команды ≤ 2.9999°; limits 0.016°; disabled 0.0019°; max qdot 3.16; saturation 19%; 0 reset |
| H PPO, 128 env, 20 it | 40 960 переходов, 18.8 s, 2606 steps/s (было 5402) |

PPO checkpoint: `logs/rsl_rl/dg5f_cube_direct/2026-09-16_19-42-29/model_19.pt`.
Smoke test, **не обученный Teacher**: episode length 171, orientation_error 24.6°,
saturation 25% (отводящие 2_1/3_1/4_1 — 78–96%).

## Буквенный куб, цели и первое PPO-обучение (2026-09-16, `logs/letter_cube/`)

Sanity: unit 12/12; list_envs; zero (дрейф 0.49°, куб 1.26 mm, saturation 0);
random (шаг ≤ 2.9999°, disabled 0.0019°); cube_visual_check PASS; goal reset 10 000
выборок, минимум 10.001°, внутри допуска 0; grasp hold 8 env с шумом — те же цифры, что
до замены визуала (0.94° / 1.79 mm / 1.33°); PPO smoke 20 it.

Обучение 128 env × 500 it (`train.py` без изменений гиперпараметров, 657 s):
`logs/rsl_rl/dg5f_cube_direct/2026-09-16_23-16-44/`, checkpoints 0, 50, …, 450, 499.
Анализ: `analysis/{summary.txt,metrics.csv,curves.png}` (`scripts/analyze_training.py`) и
детерминированная оценка `analysis/eval_deterministic.json` (`scripts/eval_checkpoints.py`,
128 полных эпизодов на политику, одинаковый seed).

| политика | success (хоть раз в 5°) | в 5° в конце | drop | final err | \|a\|≥0.95 | saturation |
|---|---:|---:|---:|---:|---:|---:|
| zero | 0.000 | 0.000 | 0.000 | 15.0° | 0.00 | 0.00 |
| random | 0.305 | 0.031 | 0.273 | 64.0° | 0.05 | 0.19 |
| model_0 | 0.422 | 0.031 | 0.000 | 22.7° | 0.00 | 0.30 |
| model_250 | 0.516 | 0.000 | 0.039 | 26.9° | 0.09 | 0.27 |
| model_499 | 0.008 | 0.008 | 0.000 | 21.1° | 0.56 | 0.33 |

**Вывод: обучения нет.** Ни один checkpoint не лучше «ничего не делать» по финальной
ошибке (15°); success в основном = пролёт через допуск, а не удержание. Action std
монотонно растёт 1.00 → 1.42 (entropy 26.9 → 33.6), доля |a| ≥ 0.95 растёт 0.35 → 0.63;
KL в RSL-RL 3.1.2 не логируется. NaN нет, скорости ≤ 3.28 rad/s, disabled ≤ 0.017°.

## Experiment B: reward v2 + PPO cfg (2026-09-17, `logs/reward_v2/`)

Физика не менялась. PPO: init_noise_std 1.0 → 0.5, entropy_coef 0.005 → 0.001,
actor/critic [32,32] → [128,128], obs normalization выкл → вкл, num_steps_per_env 16 → 32
(остальное без изменений). Sanity: unit 20/20, reward table, zero baseline, goal reset,
smoke 20 it. Обучение 128 env × 500 it (1112 s, NaN/ошибок нет):
`logs/rsl_rl/dg5f_cube_direct/2026-09-17_01-01-44_reward_v2_expB/` (+ `analysis/`).
Детерминированный eval (128 полных эпизодов, одинаковый seed, без шума):
`logs/reward_v2/08_eval.{json,log}`.

| политика | held | entered | в 5° в конце | time in 5° | drop | final err | reward | \|a\|≥0.95 | sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 15.0° | 6.1 | 0.00 | 0.00 |
| old/model_0 | 0.328 | 0.383 | 0.016 | 0.081 | 0.047 | 32.6° | 4.6 | 0.00 | 0.29 |
| old/model_250 | 0.336 | 0.539 | 0.000 | 0.035 | 0.055 | 32.7° | −3.9 | 0.10 | 0.25 |
| old/model_499 | 0.008 | 0.008 | 0.008 | 0.001 | 0.000 | 21.7° | −5.8 | 0.55 | 0.32 |
| new/model_0 | 0.023 | 0.078 | 0.000 | 0.003 | 0.312 | 75.6° | −15.7 | 0.06 | 0.25 |
| new/model_100 | 0.398 | 0.484 | 0.031 | 0.061 | 0.000 | 16.8° | 7.5 | 0.00 | 0.20 |
| new/model_200 | 0.430 | 0.570 | 0.047 | 0.081 | 0.000 | 17.7° | 11.5 | 0.00 | 0.22 |
| new/model_250 | 0.680 | 0.758 | 0.211 | 0.217 | 0.000 | 9.1° | 22.0 | 0.00 | 0.23 |
| new/model_300 | 0.852 | 0.938 | 0.211 | 0.294 | 0.016 | 11.9° | 26.5 | 0.00 | 0.18 |
| new/model_350 | 0.711 | 0.844 | 0.289 | 0.303 | 0.000 | 8.5° | 24.5 | 0.05 | 0.23 |
| **new/model_400** | 0.750 | 0.883 | **0.383** | **0.374** | 0.000 | **6.3°** | **28.0** | 0.06 | 0.20 |
| new/model_450 | 0.719 | 0.844 | 0.352 | 0.373 | 0.000 | 7.9° | 27.6 | 0.03 | 0.22 |
| new/model_499 | 0.719 | 0.781 | 0.297 | 0.263 | 0.000 | 6.9° | 25.4 | 0.04 | 0.23 |

(model_50: held 0.19 / final 36.3°; model_150: 0.16 / 26.4°.)
Лучший checkpoint — **model_400**: лучшая финальная ошибка, доля «в 5° в конце», время в
допуске и reward, без падений; по одному held выше model_300 (0.85), но у него final 11.9°
и 1.6% падений. Обучение (trailing mean 25 it, it0 → it499): action std 0.50 → 0.41,
entropy 13.8 → 10.1, |a|≥0.95 0.07 → 0.11, saturation 0.15 → 0.21 (watch-суставы
0.39 → 0.73; хуже всех 5_2, 1_3, 3_1, 2_1), drop 0.07 (it100) → 0.03, max |qdot| ≤ 3.17.

**Вывод:** PPO научился поворачивать куб и удерживать его у цели (финальная ошибка
6–9° против 15° у zero). Но в конце эпизода внутри 5° только около трети эпизодов, а
held 5°/0.3 s достигается и пролётом (у необученного old/model_0 held = 0.33).
Повторный eval того же old/model_0 дал 32.6° вместо прежних 22.7°: разброс между
запусками оценки заметный, но разрыв new/old больше.

## Оставшиеся проблемы

- Недодемпфированные суставы из SysID (ζ ≈ 0.001–0.005).
- При исследовании PPO отводящие суставы часто на пределе 0.4 Nm: блуждающая
  `q_cmd` упирает их в соседей.
- Torque limit — паспортный rated, не измеренный; нет current/thermal модели.
- Модель руки выведена из SDK/URDF, не с таблички.
- Friction SysID без единиц; torque — PD-оценки.
- Под контактом PhysX qdot всё ещё может расходиться с FD (random 0.6 rad/s).
- GPU PhysX недетерминирован между средами (одинаковые settle дают разные позы).
- Convex decomposition замедлила PPO примерно в 2 раза.
