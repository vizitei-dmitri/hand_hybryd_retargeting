# Reward functions and curricula in published in-hand reorientation systems (reimplementation-level detail)

Scope note: every coefficient below is quoted from either the paper text/appendix or the named source
file. Where a number exists only in code, the file is named. Where a paper does not state something,
it is listed under Gaps rather than inferred.

## Q1. Full reward term lists with numeric coefficients, per system

### Takeaway
Two reward families dominate. Family A (OpenAI Dactyl) is a **progress/difference** term
`d_t - d_{t+1}` plus a sparse bonus. Family B (IsaacGymEnvs ShadowHand → DeXtreme → Isaac Lab →
most 2024-2026 derivatives) is a **reciprocal** dense term `1/(|Δθ| + 0.1)` with weight 1.0, a
position-drift penalty at -10.0, an action penalty around -1e-4…-1e-3, and a **+250** goal bonus.
The rotation-only systems (Hora, AnyRotate) instead reward **angular velocity projected on a target
axis**, clipped, with work/torque/pose regularizers. The newest systems (POISE 2026) switch the dense
term to `exp(-error/scale)` and add an explicit grasp-quality term.

### Cited Findings

**OpenAI Dactyl (Learning Dexterous In-Hand Manipulation, 2018)**
- `r_t = d_t − d_{t+1}`, where `d_t`, `d_{t+1}` are the rotation angles between desired and current
  object orientations before and after the transition. Additional reward of **5** whenever a goal is
  achieved (tolerance 0.4 rad, i.e. `d_{t+1} < 0.4`), and **−20** whenever the object is dropped. No
  other terms — no action penalty, no torque penalty. — [OpenAI et al. 2018, §4.2 and Appendix C.1](https://arxiv.org/abs/1808.00177)
- "we do not use any human demonstrations and do not encode any prior into the reward function" —
  [same](https://arxiv.org/abs/1808.00177)
- Environment step = 80 ms (10 MuJoCo steps of 8 ms). — [Appendix C.1](https://arxiv.org/abs/1808.00177)

**IsaacGymEnvs ShadowHand (shipped code)**
Reward assembled in `compute_hand_reward` as
`reward = goal_dist*dist_reward_scale + (1/(|rot_dist| + rot_eps))*rot_reward_scale + sum(a^2)*action_penalty_scale`
then `+ reach_goal_bonus` if `|rot_dist| <= success_tolerance`, `+ fall_penalty` if
`goal_dist >= fall_dist`, and `+ 0.5*fall_penalty` on timeout (only when `maxConsecutiveSuccesses > 0`).
`rot_dist = 2*asin(clamp(||quat_diff[0:3]||, max=1.0))`.
— [isaacgymenvs/tasks/shadow_hand.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/shadow_hand.py)
Coefficients, from `isaacgymenvs/cfg/task/ShadowHand.yaml`:
| key | value |
|---|---|
| `distRewardScale` | **-10.0** |
| `rotRewardScale` | **1.0** |
| `rotEps` | **0.1** |
| `actionPenaltyScale` | **-0.0002** |
| `reachGoalBonus` | **250** |
| `fallDistance` | **0.24** |
| `fallPenalty` | **0.0** |
| `successTolerance` | **0.1** |
| `maxConsecutiveSuccesses` | **0** |
| `episodeLength` | **600** |
— [cfg/task/ShadowHand.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/ShadowHand.yaml)
- Note `ignore_z_rot` doubles `success_tolerance` when set (pen task). — shadow_hand.py

**Isaac Lab in-hand manipulation (shipped code)**
- Identical formula to IsaacGymEnvs: `dist_rew + rot_rew + action_penalty*scale`, `rot_rew = 1.0/(|rot_dist| + rot_eps)*rot_reward_scale`. — [isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py)
- ShadowHand cfg (`shadow_hand_env_cfg.py`): `dist_reward_scale=-10.0`, `rot_reward_scale=1.0`,
  `rot_eps=0.1`, `action_penalty_scale=-0.0002`, `reach_goal_bonus=250`, `fall_penalty=0`,
  `fall_dist=0.24`, `success_tolerance=0.1`, `max_consecutive_success=0`, `av_factor=0.1`,
  `episode_length_s=10.0`.
- **A second config in the same file** (the OpenAI-observation / "Shadow Hand OpenAI" variant) uses
  `fall_penalty=-50`, `success_tolerance=0.4`, `max_consecutive_success=50`, `episode_length_s=8.0`,
  with the other weights unchanged. This is the Dactyl-like setting.
  — [shadow_hand_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py)
- AllegroHand cfg (`allegro_hand_env_cfg.py`): same weights but `success_tolerance=**0.2**`,
  `max_consecutive_success=0`, `episode_length_s=10.0`.
  — [allegro_hand_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/allegro_hand_env_cfg.py)

**NVIDIA DeXtreme (Handa et al. 2022)** — reward table quoted verbatim from the paper (Table 2):
| Term | Formula | Weight |
|---|---|---|
| Rotation Close to Goal | `1/(d + 0.1)` | **1.0** |
| Position Close to Fixed Target | `‖p_object − p_goal‖` | **-10.0** |
| Action Penalty | `‖a‖²` | **-0.001** |
| Action Delta Penalty | `‖targ_curr − targ_prev‖²` | (see note) |
| Joint Velocity Penalty | `‖v_joints‖²` | **-0.003** |
| Reach Goal Bonus | condition `d < 0.1` | **250.0** |
— [DeXtreme, §2.4 Table 2](https://arxiv.org/abs/2210.13702)
Code-only values (paper Table 2's action-delta weight is not legible in the HTML render; these come
from the configs):
- `AllegroHandDextremeADR.yaml`: `distRewardScale: -10.0`, `rotRewardScale: 1.0`, `rotEps: 0.1`,
  `actionPenaltyScale: **-0.001**`, `actionDeltaPenaltyScale: **-0.2**` (the file contains a
  commented-out `-0.01`), `reachGoalBonus: 250`, `fallDistance: 0.24`, `fallPenalty: 0.0`,
  `successTolerance: **0.1**`, `maxConsecutiveSuccesses: 50`, `num_success_hold_steps: **0**`,
  `resetTime: 8` seconds (overrides `episodeLength: 320`).
- `AllegroHandDextremeManualDR.yaml`: `actionPenaltyScale: **-0.0001**`,
  `actionDeltaPenaltyScale: **-0.01**`, `successTolerance: **0.4**`, `maxConsecutiveSuccesses: 50`,
  `reachGoalBonus: 250`, `fallPenalty: 0.0`, `resetTime: 8`.
- The velocity penalty is **hard-coded in the source, not in the YAML**:
  `max_velocity = 5.0`, `vel_tolerance = 1.0`, `velocity_penalty_coef = -0.05`, and
  `velocity_penalty = -0.05 * Σ (hand_dof_vel/(5.0 - 1.0))²`. Note the paper says -0.003 for the
  joint-velocity weight; the code's effective coefficient is `-0.05/16 = -0.003125`, which reconciles
  the two. — [tasks/dextreme/allegro_hand_dextreme.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/dextreme/allegro_hand_dextreme.py)
- Timeout penalty in code: `timeout_rew = timed_out * 0.5 * fall_penalty` (zero in both shipped
  configs because `fallPenalty: 0.0`). — same file

**Allshire et al. TriFinger (IsaacGymEnvs `trifinger.py` + `cfg/task/Trifinger.yaml`)**
Reward = `finger_movement_penalty + finger_reach_object_reward + pose_reward`, where
- `finger_movement_penalty = w_fm * Σ_{9} (fingertip_vel)²`, `w_fm = **-0.5**`
- `finger_reach_object_reward = w_fr * ft_sched_val * Σ_{i=1..3} (‖p_fi − p_o‖_t − ‖p_fi − p_o‖_{t−1})`,
  `w_fr = **-250**` — this is an explicit **progress/difference** term on fingertip-to-object distance
- keypoint branch (default on): `pose_reward = 2000 * dt * mean_i lgsk_kernel(‖k_i^o − k_i^g‖, scale=**30.**, eps=**2.**)`
  with 8 keypoints at cube size `(0.065, 0.065, 0.065)`
- quaternion branch (default off): `object_dist_reward = 2000 * dt * lgsk_kernel(‖p_o − p_g‖, scale=**50.**, eps=**2.**)`
  plus `object_rot_reward = 2000 * dt / (3.*|angle| + **0.01**)`
- `lgsk_kernel(x, scale, eps) = 1/(exp(scale*x) + eps + exp(-scale*x))`, docstring: "bound input to
  [-0.25, 0)", cited to [arXiv:1901.08652 p.15]
- YAML: `finger_move_penalty.weight: -0.5`, `finger_reach_object_rate.weight: -250` (`norm_p: 2`),
  `object_dist.weight: 2000` (`activate: false`), `object_rot.weight: 2000` (`activate: false`),
  `keypoints_dist.weight: 2000` (`activate: true`), `episodeLength: 750`, `dt: 0.02`,
  `task_difficulty: 4`, `cube_obs_keypoints: true`
— [isaacgymenvs/tasks/trifinger.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/trifinger.py), [cfg/task/Trifinger.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/Trifinger.yaml)
- There is **no drop penalty and no success bonus** in the TriFinger reward; the only reset is timeout
  (`reset = progress_buf >= episode_length - 1`). — trifinger.py

**Chen et al., "A System for General In-Hand Object Re-Orientation" (CoRL 2021)**
Reorientation reward (Eq. 1):
`r(s_t, a_t) = c_θ1 * 1/(|Δθ_t| + ε_θ) + c_θ2 * 1(|Δθ_t| < θ̄) + c_3 * ‖a_t‖²`
with `c_θ1 > 0`, `c_θ2 > 0`, `c_3 < 0`.
Numeric values, Table C.2: **`c_θ1 = 1`, `c_θ2 = 800`, `c_3 = 0.1`** (listed as magnitude; text states
`c_3 < 0`), **`ε_θ = 0.1`**, **`θ̄ = 0.1 rad`**, **Episode length 300**.
Lifting-policy reward (Eq. 2): `c_h1 * 1/(|Δh_t| + ε_h) + c_h2 * 1(|Δh_t| < h̄) + c_3‖a_t‖²` with
**`c_h1 = 0.05`, `c_h2 = 800`, `ε_h = 0.02`, `h̄ = 0.04`**.
— [Chen et al. 2021, §C.2 and Table C.2](https://arxiv.org/abs/2111.03043)
- Action space: relative target joint angle, `q_{t+1}^target = q_t^target + a_t*Δt`, clamped so
  `|Δq_t^target| ≤ 0.33 rad`; control at 60 Hz. — [§2.1](https://arxiv.org/abs/2111.03043)
- No drop penalty term appears in Eq. 1; dropping is handled by episode termination only.

**Chen et al., "Visual Dexterity" (Science Robotics 2024)** — reward Eqs. 1-8:
`r_1t = c_1 * 1(Task successful)` (sparse)
`    + c_2 * 1/(|Δθ_t| + ε_θ)` (dense task reward)
`    + c_3 * Σ_{i=1..G} ‖p_t^{f_i} − p_t^o‖²` (keep fingertip close to object)
`    + c_4 * |q̇_t|ᵀ|τ_t|` (energy)
`    + c_5 * 1(‖p_t^o‖² > p̄)` (penalty for pushing object away)
In-air variant: `r_2t = r_1t + c_6 * 1(object contacts table) + c_7 * Σ_i 1(p_{t,z}^{f_i} > p̄_z)`.
Values, Table S1: **`c_1 = 800`, `c_2 = 1`, `c_3 = -1`, `c_4 = -20`, `c_5 = -100`, `c_6 = -1`,
`c_7 = -2`, `ε_θ = 0.4 radians`, `p̄ = 0.15`, `p̄_z = 0.16`, `q̇̄ = 0.25`, `v̄ = 0.04`, `ω̄ = 0.5`,
`c_d = 15`**. Action smoothing `α = 0.8`; `q^tgt_{t+1} = q_t + ā_t`. Teacher: 32000 envs, batch 64000.
— [Visual Dexterity, Reward section and Table S1](https://arxiv.org/abs/2211.11744)
- Reward ablation (Fig. S6) is reported for `c_1`…`c_5` individually. Qualitatively: increasing `c_1`
  improved learning up to a point then degraded; increasing `c_2` improved learning but "after
  `c_2 = 1.0` the performance began to deteriorate"; learning was "less sensitive" to `c_3` and `c_5`,
  though `c_3 < 0` and `c_5 < 0` were advantageous (higher learning-curve variance otherwise).
  — [Visual Dexterity, Fig. S6 discussion](https://arxiv.org/abs/2211.11744)

**Qi et al. Hora ("In-Hand Object Rotation via Rapid Motor Adaptation", CoRL 2022)** — code-derived:
`reward = rotate_reward_scale * rotate_reward + objLinvelPenaltyScale * ‖v_obj‖_1
          + poseDiffPenaltyScale * Σ(q − q_init)² + torquePenaltyScale * Σ τ²
          + workPenaltyScale * (Σ τ·q̇_finite_diff)²`
where `rotate_reward = clip(ω_obj · k̂, angvelClipMin, angvelClipMax)`, `k̂` is the fixed target axis
(`rot_axis_buf[:, -1] = -1`, i.e. −z), and `ω_obj` is computed by finite difference of the object
quaternion: `axis_angle(q_t ⊗ q_{t−1}^*) / (control_freq_inv * dt)`.
— [hora/tasks/allegro_hand_hora.py, `compute_reward`/`compute_hand_reward`](https://github.com/HaozhiQi/hora/blob/main/hora/tasks/allegro_hand_hora.py)
Values from `configs/task/AllegroHandHora.yaml`:
| key | value |
|---|---|
| `angvelClipMin` | **-0.5** |
| `angvelClipMax` | **0.5** |
| `rotateRewardScale` | **1.0** |
| `objLinvelPenaltyScale` | **-0.3** |
| `poseDiffPenaltyScale` | **-0.3** |
| `torquePenaltyScale` | **-0.1** |
| `workPenaltyScale` | **-2.0** |
| `episodeLength` | **400** |
| `controlFrequencyInv` | **6** (= 20 Hz) |
| `baseObjScale` | **0.8** |
- `configs/task/PublicAllegroHandHora.yaml` overrides `rotateRewardScale: **1.25**` and otherwise
  inherits. — [HaozhiQi/hora configs](https://github.com/HaozhiQi/hora/tree/main/configs/task)
- Torque commands are clipped to `[-0.5, 0.5]` in the torque-control path: `torch.clip(torques, -0.5, 0.5)`.
  — allegro_hand_hora.py
- **No goal orientation, no success bonus, no drop penalty** in Hora's reward; the objective is
  continuous rotation about a fixed axis.

**AnyRotate (Yang et al. 2024)** — full specification from Appendix B.1:
`r = r_rotation + r_contact + r_stable + r_terminate`, with
`r_rotation = λ_kp·r_kp + λ_rot·r_rot + λ_goal·r_goal`
`r_contact  = λ_rew·(λ_gc·r_gc + λ_bc·r_bc)`
`r_stable   = λ_rew·(λ_ω·r_ω + λ_pose·r_pose + λ_work·r_work + λ_torque·r_torque)`
`r_terminate = λ_penalty·r_penalty`
Term formulas:
- `r_kp = d_kp / (e^{a·x} + b + e^{−a·x})` with `kp_dist = (1/N) Σ_{i=1..N} ‖k_i^o − k_i^g‖`,
  **N = 6 keypoints placed 5 cm from the object origin on each principal axis**, **a = 50, b = 2.0**
- `r_rot = clip(ΔΘ · k̂, −c_1, c_1)` with **c_1 = 0.025 rad** (change in object rotation about the
  target axis — a difference/progress term)
- `r_goal = 1 if kp_dist < d_tol else 0`
- `r_gc = 1 if n_tip_contact ≥ 2 else 0`
- `r_bc = 1 if n_non_tip_contact ≥ 0 else 0` (as printed in the appendix)
- `r_ω = −min(‖ω_o‖ − ω_max, 0)` with **ω_max = 0.5**
- `r_pose = −‖q − q_0‖`, `q_0` a canonical grasping pose
- `r_work = −τᵀ q̄`; torque penalty `= −‖τ‖`
- `r_terminate = −1` if `kp_dist > d_max` (**d_max = 0.1**) or the object rotation axis deviates
  beyond **k̂_max = 45°**, else 0
Weights: **λ_kp = 1.0, λ_rot = 5.0, λ_goal = 10.0, λ_gc = 0.1, λ_bc = 0.2, λ_ω = 0.75,
λ_pose = 0.2, λ_work = 2.0, λ_torque = 1.0, λ_penalty = 50.0**.
— [AnyRotate, Appendix B.1](https://arxiv.org/abs/2405.07391)
Alternative (baseline) formulation, Appendix B.2: `r_rotation = λ_av·r_av + λ_rot·r_rot` with
`r_av = clip(ω·k̂, −c_2, c_2)`, **c_2 = 0.5**, plus axis penalty
`r_axis = 1 − (k̂·k̂_o)/(‖k̂‖‖k̂_o‖)` inside `r_stable`, `λ_ω` set to **0**, and
**λ_av = 1.5, λ_axis = 1.0**. — [AnyRotate, Appendix B.2](https://arxiv.org/abs/2405.07391)
Training: 8192 envs, batch 8192, minibatch 32768, 5 mini-epochs, lr 3e-4, γ 0.99, GAE τ 0.95,
clip 0.2, KL threshold 0.02, **Goal Update `d_tol` = 0.25 (teacher) and 0.15 (student)** (Table 5).
— [AnyRotate, Table 5](https://arxiv.org/abs/2405.07391)

**POISE, "Learning In-Hand Object Reaching to General 6D Poses" (2026)** — full reward, Eqs. 3-10:
- `r_t^R = exp(−e_{R,t} / 30°)`
- `r_t^p = α(e_R) · exp(−e_{p,t} / σ(e_R))` with
  `(α, σ) = (0.15, 30 mm)` for `e_R > 45°`; `(0.40, 20 mm)` for `25° < e_R ≤ 45°`;
  `(1.00, 15 mm)` for `e_R ≤ 25°`
- `r_t^pose = (1/6)·r_t^R + (1/4)·r_t^p`
- `r_t^goal = 0.5·s_t + 45·c_t`, where `s_t ∈ {0,1}` means retained and inside both pose tolerances,
  and `c_t` marks the first step at which the condition has held for **N = 20 consecutive steps**
- `Q_t = [ (1/6) Σ_{k=1..6} (m_{t,k} + ε)^{−8} ]^{−1/8} − ε` (generalized mean over six bidirectional
  friction-cone wrench margins); `Q* =` mean quality over the first four control steps, clipped to
  **[0.08, 0.35]**; `r_t^grasp = 1 − clip((Q* − Q_t)_+ / Q*, 0, 1)²`
- total: `r_t = r_t^pose + r_t^goal + **0.05**·r_t^grasp − **80**·d_t − **0.10**·‖τ_t‖²
  − **0.15**·(τ_tᵀ q̇_t)²`, `d_t ∈ {0,1}` a drop indicator
- Physics 120 Hz, policy at 20 Hz, incremental joint-position commands, Isaac Lab, asymmetric PPO,
  actor/critic [512,256,128], 64,000 parallel envs.
— [POISE, §IV-D](https://arxiv.org/abs/2609.13761)

**DexReMoE (2025)** — Isaac-family reward with a hold criterion:
`r_1 = c_success`; `r_2 = c_dist·|δ_p| + c_rot/(|δ_θ| + ε)`;
`r_3 = c_ω Σ_{i=1..n} [|ω_{i,t}| − ω_clip]_+ + c_a·‖a_t‖²`; `R = r_1 + r_2 + r_3`.
Table II values: **`c_success = 800`, `c_dist = -10.0`, `c_rot = -1.0`, `c_a = -0.0002`,
success tolerance 0.4, episode length 600, 32768 envs, lr 5e-3, `τ_θ = 0.1`, `τ_q = 10.0`,
`τ_v = 0.04`, `τ_ω = 0.5`**.
- Explicit statement: "Our reward function does not include the penalty for the object falling, as we
  found during experiments that such a term suppresses exploratory actions and adversely affects the
  overall training performance." — [DexReMoE](https://arxiv.org/abs/2508.01695)

**Rotation-Aware Point-Cloud Embeddings (RAPE, 2026)** — same Family-B structure with a learned
goal representation:
`r(s_t, a_t, g) = w_p‖p_t − p_ref‖₂ + w_R·1/(d_R(q_t, q_g) + ε) + w_a‖a_t‖₂² + b·1(d_R < β)`,
`d_R` = geodesic SO(3) distance via the relative quaternion `q ⊗ q_g*`.
Episodes terminate early on drop/out-of-reach, otherwise time out after **600** steps, with the
counter **reset on each successful reorientation**, and **max consecutive goal successes = 5**.
PPO, 64 steps/env, 3500 iterations, lr 5e-4, actor/critic (1024,512,256,128).
— [RAPE, §2.2 and Appendix A3](https://arxiv.org/abs/2606.21788)

### Inferences
- The `-10.0 / 1.0 / 0.1 / 250 / 0.24` tuple is effectively a de-facto standard: it appears unchanged
  in IsaacGymEnvs ShadowHand, DeXtreme (both configs), Isaac Lab ShadowHand and AllegroHand, and is
  reproduced with the same numbers in DexReMoE (`c_dist = -10`, `c_success = 800` differs only in the
  bonus size and in it being awarded per-step rather than once).
- Chen 2021 and Visual Dexterity are the same reward lineage as the Isaac family (`1/(Δθ + ε)` plus
  sparse bonus) but with a much larger bonus relative to the dense term (`c_2 = 1` vs `c_1 = 800`).
- Hora and AnyRotate are a different objective class (continuous rotation, no fixed goal pose), so
  their coefficients are not directly comparable to the reorientation reward magnitudes.

### Gaps
- The exact `actionDeltaPenaltyScale` weight printed in DeXtreme's paper Table 2 was not legible in
  the HTML render; only the two code values (-0.2 in the ADR config, -0.01 in the manual-DR config)
  are confirmed.
- RAPE does not state numeric values for `w_p`, `w_R`, `w_a`, `b`, `β`, or `ε` anywhere in the HTML
  version; only the functional form is given.
- The OpenAI Rubik's Cube paper (arXiv:1910.07113) was not fetched in this pass; its reward is widely
  described as the Dactyl reward plus a face-rotation goal, but no coefficient from it is quoted here.
- Allshire et al.'s TriFinger *paper* (arXiv:2108.09779) was not fetched; all TriFinger numbers above
  come from the IsaacGymEnvs release.

## Q2. What is the dense shaping term, specifically? Which papers report the choice mattered?

### Takeaway
Four distinct shapes are in use: **reciprocal** `1/(error + ε)` (Isaac family, Chen 2021, Visual
Dexterity, DeXtreme, DexReMoE, RAPE), **progress/difference** `d_t − d_{t+1}` (Dactyl; also the
TriFinger fingertip-reach term and AnyRotate's `r_rot`), **logistic-kernel keypoint distance**
(TriFinger), and **exponential** `exp(−error/scale)` (POISE, and AnyRotate's `r_kp` which is a
sigmoid-like bell `d/(e^{ax} + b + e^{-ax})`). Only Visual Dexterity and AnyRotate publish evidence
that the choice mattered.

### Cited Findings
- Reciprocal, exact: `rot_rew = 1.0/(|rot_dist| + 0.1) * 1.0` with `rot_dist` the quaternion geodesic
  angle. Unbounded as error → 0; maximum value 10 at `rot_dist = 0`. — [shadow_hand.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/shadow_hand.py); DeXtreme paper writes it as `1/(d + 0.1)` with weight 1.0 — [DeXtreme Table 2](https://arxiv.org/abs/2210.13702)
- Progress term, exact: `r_t = d_t − d_{t+1}` (Dactyl). — [OpenAI 2018](https://arxiv.org/abs/1808.00177)
- Bounded logistic kernel, exact: `lgsk_kernel(x, scale, eps) = 1/(e^{scale·x} + eps + e^{−scale·x})`,
  bounded to `[-0.25, 0)` per the docstring; applied to the mean over 8 cube keypoints with
  `scale=30., eps=2.`, weight 2000, multiplied by `dt = 0.02` (so ~≤10 per step). — [trifinger.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/trifinger.py)
- Exponential, exact: `r_t^R = exp(−e_R / 30°)` and `r_t^p = α·exp(−e_p/σ)` with the `(α, σ)` gating
  table above. — [POISE §IV-D](https://arxiv.org/abs/2609.13761)
- Bell-shaped keypoint kernel, exact: `r_kp = d_kp/(e^{50x} + 2.0 + e^{−50x})`, `x = kp_dist`. —
  [AnyRotate App. B.1](https://arxiv.org/abs/2405.07391)
- **Choice mattered (Visual Dexterity)**: the sparse success criterion alone "is insufficient for
  successful learning"; and in the reward ablation, increasing the dense-reward coefficient `c_2`
  improved learning only up to `c_2 = 1.0`, after which "the performance began to deteriorate". —
  [Visual Dexterity, Reward and Fig. S6](https://arxiv.org/abs/2211.11744)
- **Choice mattered (AnyRotate)**: replacing the keypoint-based auxiliary-goal dense term with an
  angular-velocity objective produced "much lower accuracy, obtaining near-zero successive goals
  reached, despite having a rotation axis penalty" in the single-axis setting, and in the multi-axis
  setting "the training was unsuccessful and the agent was unable to maintain stable rotation". —
  [AnyRotate §Results](https://arxiv.org/abs/2405.07391)
- **Choice mattered (DeXtreme, indirectly)**: the paper says the reward is "inspired by the Shadow hand
  environment in Isaac Gym" and is "described and justified in Table 2", i.e. each term is
  individually justified but no ablation numbers for the shaping shape are given. — [DeXtreme §2.4](https://arxiv.org/abs/2210.13702)
- **Choice mattered (POISE, positional gating)**: "When the orientation error is large, the position
  term is broad and downweighted, allowing translation needed for contact rearrangement. As the object
  becomes rotationally aligned, the reward increasingly emphasizes accurate positioning." — [POISE §IV-D](https://arxiv.org/abs/2609.13761)

### Inferences
- The reciprocal term is a strictly-increasing-payoff-near-zero shape that becomes very steep inside
  the tolerance; the logistic/exponential forms are bounded. Papers that stream many goals per episode
  and care about *hold* behaviour (POISE, AnyRotate) use bounded kernels; the single-goal Isaac tasks
  use the unbounded reciprocal.
- Nobody publishes a head-to-head `exp(-k·e)` vs `1/(e+ε)` comparison on the same task; the evidence
  is on coefficient magnitude (Visual Dexterity) and on goal *representation* (AnyRotate, TriFinger).

### Gaps
- No paper found that ablates `exp(−k·e)` against `1/(e + ε)` against a progress term on the same
  reorientation task with matched everything else.

## Q3. Keypoint goal representation vs quaternion/angle-difference: explicit comparisons

### Takeaway
There is one code-level A/B toggle (TriFinger) and one paper-level ablation with learning curves
(AnyRotate), plus a 2026 representation paper (RAPE) whose framing is keypoint/point-cloud vs
privileged pose. Numeric side-by-side success rates for keypoints-vs-quaternion on the *same* task
were found only in AnyRotate's goal-tolerance / increment tables and RAPE's baseline comparison; the
TriFinger toggle ships with keypoints enabled and quaternion disabled but no in-repo numbers.

### Cited Findings
- **TriFinger ships an explicit switch.** `cfg/task/Trifinger.yaml` has
  `keypoints_dist: {activate: true, weight: 2000}` and `object_dist: {activate: false, weight: 2000}`,
  `object_rot: {activate: false, weight: 2000}`; in `trifinger.py` the `use_keypoints` branch computes
  8 cube keypoints and `lgsk_kernel(dist_l2, scale=30., eps=2.).mean(-1)`, and the `else` branch
  computes `lgsk_kernel(object_dist, scale=50., eps=2.)` plus `1/(3.*|angles| + 0.01)` on the
  quaternion difference. The keypoint branch is the shipped default. — [trifinger.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/trifinger.py), [Trifinger.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/Trifinger.yaml)
- **AnyRotate compares the keypoint auxiliary-goal formulation against an angular-rotation objective**
  ("w/o auxiliary goal", the formulation of prior works [5,6,31]) and against "w/o adaptive
  curriculum", reporting learning curves for average rotation and successive goals reached (Fig. 6).
  Result: the angular-rotation-objective baseline reached "near-zero successive goals reached" in the
  single-axis setting and failed entirely in the multi-axis setting. — [AnyRotate](https://arxiv.org/abs/2405.07391)
- **AnyRotate quantifies keypoint tolerance and goal increment** (Table 9, student policy):
  | Goal update tolerance | Rot | TTT (s) | #Success |
  |---|---|---|---|
  | `d_tol = 0.15` | 0.75 | 28.1 | 3.07 |
  | `d_tol = 0.20` | 1.36 | 27.7 | 4.48 |
  | **`d_tol = 0.25`** | **1.77** | **27.2** | **5.26** |

  | Goal increment interval | Rot | TTT (s) | #Success |
  |---|---|---|---|
  | `θ = 50°` | 1.30 | 27.1 | 3.86 |
  | `θ = 40°` | 1.50 | 26.7 | 4.36 |
  | **`θ = 30°`** | **1.77** | **27.2** | **5.26** |
  — [AnyRotate, Appendix K.1 Table 9](https://arxiv.org/abs/2405.07391)
- **RAPE (2026)** argues raw point-cloud goal conditioning is "poorly conditioned for policy learning"
  because "their discrepancy entangles object rotation with permutation, resampling, and unstable
  correspondence structure", and instead learns an embedding whose Euclidean latent distance is
  calibrated to the SO(3) geodesic error; the policy reward still uses the quaternion geodesic
  `1/(d_R + ε)`. It benchmarks against a privileged-pose teacher (object pose + goal pose + relative
  quaternion, 154-dim actor input), a FoundationPose estimator baseline, a flow-PointNet++ baseline,
  a distilled point-cloud student, and a Point-MAE reference, reporting that its interface "matches
  privileged-state and distillation-based baselines" while "a policy using a task-agnostic Point-MAE
  encoder fine-tuned on the same YCB objects fails to learn". — [RAPE](https://arxiv.org/abs/2606.21788)
- **Visual Dexterity explicitly rejects keypoints** as a goal representation: "By directly predicting
  actions from point clouds, our approach bypasses the problem of consistently defining
  pose/keypoints across different objects, allowing for generalization to new shapes." —
  [Visual Dexterity](https://arxiv.org/abs/2211.11744)
- A 2025 follow-up notes the opposite failure mode of point clouds: "the keypoints in [5] refer to the
  vertices ... and point clouds can't handle symmetrical objects well. For instance, even simple
  spheres are difficult to manage because point clouds look identical from any rotation angle." —
  ["From Simple to Complex Skills: The Case of In-Hand Object Reorientation"](https://arxiv.org/abs/2501.05439)

### Inferences
- AnyRotate is the closest thing to a controlled keypoint-vs-angle comparison, but its baseline
  differs in objective (angular velocity about an axis), not merely in goal parameterization, so the
  gap cannot be attributed to the keypoint representation alone.
- TriFinger's code layout (weight 2000 for both branches, identical scaling by `dt`) suggests the two
  branches were intended to be swappable at matched magnitude, which is a useful template for running
  the comparison yourself.

### Gaps
- No paper found that reports paired success rates for "keypoint L2 reward" vs "quaternion
  angle-difference reward" with everything else held fixed. AnyRotate's Fig. 6 gives curves, not a
  table of final numbers, for that ablation.

## Q4. Drop/grasp-loss penalties, their magnitude relative to the success bonus, and explicit grasp-quality rewards

### Takeaway
Penalty-to-bonus ratios span four orders of magnitude and disagree on sign of usefulness: Dactyl uses
−20 against +5 (4:1 penalty-heavy); the shipped Isaac tasks use **0** drop penalty against +250
bonus; the Isaac Lab "OpenAI" ShadowHand variant uses −50 against +250 (1:5); POISE uses −80 against
+45 (≈1.8:1); DexReMoE removed the drop penalty on purpose. Explicit grasp-quality reward terms exist
in exactly two systems found: AnyRotate (contact-count terms) and POISE (a friction-cone wrench-coverage
score).

### Cited Findings
- Dactyl: **−20** drop penalty vs **+5** goal bonus (ratio 4:1 against the bonus). — [OpenAI 2018](https://arxiv.org/abs/1808.00177)
- IsaacGymEnvs ShadowHand: `fallPenalty: 0.0` with `reachGoalBonus: 250`; the fall condition
  (`goal_dist >= fallDistance = 0.24`) only triggers a reset, adding zero reward. — [ShadowHand.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/ShadowHand.yaml)
- DeXtreme (both shipped configs): `fallPenalty: 0.0`, `reachGoalBonus: 250`. The timeout penalty
  `0.5 * fall_penalty` is therefore also zero. — [AllegroHandDextremeADR.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeADR.yaml), [AllegroHandDextremeManualDR.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeManualDR.yaml)
- Isaac Lab ShadowHand "OpenAI" variant: `fall_penalty = -50`, `reach_goal_bonus = 250`,
  `max_consecutive_success = 50`. The base ShadowHand and AllegroHand variants use `fall_penalty = 0`. —
  [shadow_hand_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py), [allegro_hand_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/allegro_hand_env_cfg.py)
- AnyRotate: `r_terminate = −1` scaled by **λ_penalty = 50.0**, i.e. an effective −50 on drop
  (`kp_dist > d_max = 0.1`) or axis deviation > 45°, against a goal bonus of `λ_goal = 10.0` per goal
  — so the drop penalty is **5× the goal bonus**. — [AnyRotate App. B.1](https://arxiv.org/abs/2405.07391)
- AnyRotate **grasp-quality terms**: `r_gc = 1 if n_tip_contact ≥ 2` with **λ_gc = 0.1** ("to encourage
  stable grasping contacts") and a bad-contact penalty `r_bc` over all non-fingertip contacts with
  **λ_bc = 0.2**; plus a pose penalty `−‖q − q_0‖` toward a canonical grasping pose with
  **λ_pose = 0.2**. Both contact terms are multiplied by the curriculum coefficient `λ_rew`. —
  [AnyRotate App. B.1](https://arxiv.org/abs/2405.07391)
- POISE **grasp-quality term is the headline contribution**: a generalized-mean wrench-coverage score
  `Q_t` over six bidirectional friction-cone margins (Eq. 8), converted to
  `r^grasp = 1 − clip((Q* − Q_t)_+/Q*, 0, 1)²` (Eq. 9) with weight **0.05**, against a drop penalty of
  **−80** and a verified-goal bonus of **+45**. Explicitly: "Contact contributions saturate with force,
  preventing the policy from increasing the score merely by squeezing harder", and quality is measured
  *relative to the episode's own stable reset* (`Q*` = mean over the first four control steps, clipped
  to [0.08, 0.35]) rather than an absolute threshold. — [POISE §IV-D](https://arxiv.org/abs/2609.13761)
- POISE reports the hardware effect of this term: "the grasp-maintenance reward improves three-target
  sequence success from 20% to 80%". — [POISE Abstract](https://arxiv.org/abs/2609.13761)
- Visual Dexterity has no drop penalty but two grasp-shaping terms: keep fingertips near the object
  (`c_3 = -1` on `Σ‖p^{f_i} − p^o‖²`) and a penalty for pushing the object away
  (`c_5 = -100` on `1(‖p^o‖² > 0.15)`), plus in-air terms penalizing object-table contact (`c_6 = -1`)
  and use of the penultimate joint instead of the fingertip (`c_7 = -2`). — [Visual Dexterity Table S1](https://arxiv.org/abs/2211.11744)
- DexReMoE: "Our reward function does not include the penalty for the object falling, as we found
  during experiments that such a term suppresses exploratory actions and adversely affects the overall
  training performance." — [DexReMoE](https://arxiv.org/abs/2508.01695)
- TriFinger has no drop penalty at all (cube sits on a table; reset only on timeout), but has the
  `-250`-weighted fingertip-approach progress term as the contact-shaping mechanism. — [trifinger.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/trifinger.py)
- Hora has no drop penalty in the reward; instead `objLinvelPenaltyScale = -0.3` on `‖v_obj‖_1` and
  `poseDiffPenaltyScale = -0.3` on `Σ(q − q_init)²` keep the object seated in the hand. —
  [AllegroHandHora.yaml](https://github.com/HaozhiQi/hora/blob/main/configs/task/AllegroHandHora.yaml)

### Inferences
- The two systems that stream goals within an episode and terminate on drop (Dactyl, AnyRotate, POISE)
  all use a drop penalty **larger than** the per-goal bonus. The systems with `fallPenalty: 0`
  (IsaacGymEnvs/Isaac Lab defaults) rely on the loss of future reward from the episode reset, plus the
  `-10.0 * goal_dist` position term, to discourage dropping.
- Grasp quality as an explicit reward term is a very recent development (AnyRotate 2024 contact counts;
  POISE 2026 wrench coverage). No pre-2024 system in this survey has one.

### Gaps
- DeXtreme's paper does not discuss why `fallPenalty` is 0; no ablation on drop-penalty magnitude was
  found in any of the surveyed papers except POISE's abstract-level grasp-reward ablation.

## Q5. Dwell/hold reward for staying inside tolerance, vs instantaneous success

### Takeaway
Success is **instantaneous** in Dactyl, IsaacGymEnvs ShadowHand, all Isaac Lab in-hand tasks, Chen
2021, and Hora (no goal at all). A hold mechanism exists as **dead code with the constant set to 0**
in DeXtreme's released source; is a **velocity-gated** success criterion in Visual Dexterity and
DexReMoE; and is an **explicit N-step dwell plus per-step in-region reward** in POISE.

### Cited Findings
- Isaac Lab / IsaacGymEnvs ShadowHand: goal reset fires the same step the condition holds —
  `goal_resets = where(|rot_dist| <= success_tolerance, 1, reset_goal_buf)`. No hold counter. —
  [inhand_manipulation_env.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py), [shadow_hand.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/shadow_hand.py)
- **DeXtreme's code does implement a hold counter**:
  `hold_count_buf = where(goal_reached, hold_count_buf + 1, 0)` and
  `goal_resets = where(hold_count_buf > num_success_hold_steps, 1, reset_goal_buf)`,
  driven by `self.num_success_hold_steps = self.cfg["env"]["num_success_hold_steps"]`. In the shipped
  `AllegroHandDextremeADR.yaml` this is **`num_success_hold_steps: 0`**, so the bonus fires on the
  first step past the tolerance. — [allegro_hand_dextreme.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/dextreme/allegro_hand_dextreme.py), [AllegroHandDextremeADR.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeADR.yaml)
- **Visual Dexterity replaces dwell with a velocity gate**: "A straightforward success criterion is
  judging whether an object's orientation is close to the target orientation (orientation criterion).
  However, a controller trained using this criterion tends to cause the object to oscillate around the
  target orientation. To address this issue, the success criterion is expanded to explicitly penalize
  finger and object movements." The criterion requires object-motion, orientation-distance and
  fingertip-motion conditions; the associated thresholds in Table S1 are **`q̇̄ = 0.25`, `v̄ = 0.04`,
  `ω̄ = 0.5`**. — [Visual Dexterity](https://arxiv.org/abs/2211.11744)
- **DexReMoE uses a sustained-hold criterion**: "A straightforward criterion that declares success
  whenever the orientation error falls below a specified tolerance can be misled by incidental
  collisions ... we require both precise orientation and a sustained halt at the desired
  configuration." Four conditions: rotational distance < `τ_θ = 0.1`, each finger joint velocity below
  `τ_q = 10.0`, object linear velocity under `τ_v = 0.04`, angular velocity ≤ `τ_ω = 0.5`, and "we
  enforce that all four conditions hold continuously throughout the final control cycle of the
  episode". — [DexReMoE](https://arxiv.org/abs/2508.01695)
- **POISE has both a dwell requirement and a per-step in-region reward**: "A goal is verified only after
  this condition holds for **N = 20** consecutive steps", and `r_t^goal = 0.5·s_t + 45·c_t` where
  "the per-step term encourages the policy to remain inside the target region, while the one-time bonus
  rewards verified completion". At 20 Hz control this is a **1-second dwell**. — [POISE §IV-D](https://arxiv.org/abs/2609.13761)
- **AnyRotate** does not use a dwell: `r_goal = 1 if kp_dist < d_tol`, evaluated per step, and the goal
  is then incremented by another 30° rotation. — [AnyRotate App. B.1 and Table 9](https://arxiv.org/abs/2405.07391)
- Dactyl: the goal is achieved the moment `d_{t+1} < 0.4`, with an 8-second (simulated) budget per
  goal. — [OpenAI 2018 App. C.1](https://arxiv.org/abs/1808.00177)

### Inferences
- The oscillation failure mode that Visual Dexterity, DexReMoE and POISE all describe is a direct
  consequence of instantaneous success plus an unbounded reciprocal dense term: the policy is paid
  most for passing through the tolerance repeatedly rather than settling. The three fixes are
  (a) velocity gating on success, (b) sustained-hold gating, (c) a per-step in-region reward + N-step
  verification. POISE is the only one that pays per step for dwelling.
- DeXtreme's `num_success_hold_steps` plumbing existing but set to 0 suggests the mechanism was tried
  and not needed for their reported result — but the paper does not discuss it.

### Gaps
- DeXtreme's paper does not mention `num_success_hold_steps` at all; whether a non-zero value was used
  for the reported policies cannot be established from the release.
- No ablation found that quantifies the benefit of a dwell reward against instantaneous success.

## Q6. Documented curricula: gravity, ADR, reward-weight scheduling, tolerance tightening, object-set expansion

### Takeaway
Five distinct documented curricula: Chen 2021's **gravity curriculum** (full algorithm and constants
published), DeXtreme's **vectorised ADR** (full algorithm, thresholds and per-parameter deltas
published in the config), AnyRotate's **reward-weight curriculum** `λ_rew` keyed to goals-reached,
DexReMoE's baseline **tolerance-tightening + object-set expansion** (stated but not quantified), and
POISE's **goal-magnitude curriculum** (see Q7). Isaac Lab and IsaacGymEnvs ShadowHand/AllegroHand ship
**no curriculum** of any kind; TriFinger ships only a reward-term on/off schedule.

### Cited Findings

**Chen 2021 gravity curriculum (Algorithm 1, §C.4)** — verbatim constants:
`w̄ = 0.8`, `g_0 = 1 m/s²` (i.e. gravity initially points *up*), `Δg = −0.5 m/s²`, `K = 3`, `L = 20`,
`ΔT_min = 40`, decreasing to `g = −9.81 m/s²`. Mechanism: a FIFO queue `Q` of size `K` holds surrogate
evaluation success rates; "we only test on the training objects once (one random initial and goal
orientation) to get the surrogate average success rate `w` on all the objects during training"; `g` is
stepped by `Δg` "if the evaluation success rate (w) is above a threshold value (w̄)". —
[Chen et al. 2021 §4.2, §C.4](https://arxiv.org/abs/2111.03043)
- Reported effect: "adding gravity curriculum (g-curr) significantly boost the success rates on the
  YCB dataset" (Table D.6, Exp Q vs T). Without the curriculum and without stable initialization, the
  downward-facing task had **0% success**. — [Chen et al. 2021 §4.2](https://arxiv.org/abs/2111.03043)
- Only applied to YCB (the harder set), not EGAD: "Since `π^E` already performs very well on EGAD
  objects, we apply gravity curriculum to train `π^E` on YCB objects." — [same](https://arxiv.org/abs/2111.03043)
- Visual Dexterity later frames the gravity curriculum as something it *removed*: "Prior work used a
  specialized training procedure of configuring the object in a good pose at the start of each training
  episode and a manually designed gravity curriculum [7] to learn in-air ... reorientation controllers.
  Consequently, it was necessary to train separate controllers for reorientation with a supporting
  surface and in the air." Visual Dexterity instead used a four-fingered hand plus the object-table
  contact penalty `c_6 = -1`. — [Visual Dexterity](https://arxiv.org/abs/2211.11744)

**DeXtreme vectorised ADR (VADR)** — full trigger conditions:
- Each ADR dimension is `d^n ~ U(p^{2n}, p^{2n+1})`; `Δ^n` (step size) is chosen **separately for each
  parameter**, unlike OpenAI's ADR: "we choose the size of step `Δ^n` separately for each parameter.
  This trades off more tuning work for more stable training".
- **40%** of the vectorised environments are boundary-evaluation workers; in each, one dimension is
  pinned to the current lower or upper boundary.
- Consecutive-success counts are queued per boundary with max length **N = 256**; if the mean is above
  **`t_H = 20`** the range is widened by `Δ^n`, if below **`t_L = 5`** it is tightened by `Δ^n`; on any
  bound change the queue is cleared.
— [DeXtreme §2.6, Algorithms 1-2](https://arxiv.org/abs/2210.13702)
- Config mirror (`AllegroHandDextremeADR.yaml`): `use_adr: True`, `update_adr_ranges: True`,
  `adr_extended_boundary_sample: False`, `worker_adr_boundary_fraction: 0.4`,
  `adr_queue_threshold_length: 256`, `adr_objective_threshold_low: 5`,
  `adr_objective_threshold_high: 20`, `adr_rollout_perf_alpha: 0.99`. Per-parameter entries carry
  `init_range`, `limits`, `delta`, `delta_style`; e.g. `hand_damping: init_range [0.5, 2.0],
  limits [0.01, 20.0], delta 0.01, additive`; `hand_stiffness: [0.8, 1.2], limits [0.01, 20.0],
  delta 0.01`; `hand_effort: [0.9, 1.1], limits [0.4, 10.0], delta 0.01`;
  `hand_lower/hand_upper: init [0.0, 0.0], limits [-5.0, 5.0], delta 0.02`;
  `hand_mass: [0.8, 1.2], limits [0.01, 10.0]`. — [AllegroHandDextremeADR.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeADR.yaml)
- Published initial vs ADR-discovered ranges (paper Table 3), selected rows:
  Hand Mass `[0.4, 1.5] → [0.4, 1.5]`; Scale `[0.95, 1.05] → [0.95, 1.05]`;
  Friction `[0.8, 1.2] → [0.54, 1.58]`; Armature `[0.8, 1.02] → [0.31, 1.24]`;
  Effort `[0.9, 1.1] → [0.9, 2.49]`; Joint Stiffness (loguniform) `[0.3, 3.0] → [0.3, 3.52]`;
  Joint Damping (loguniform) `[0.75, 1.5] → [0.43, 1.6]`; Restitution `[0.0, 0.4] → [0.0, 0.4]`.
  — [DeXtreme Table 3](https://arxiv.org/abs/2210.13702)
- DeXtreme also schedules **non-reward** quantities on a step counter: action moving-average
  (`actionsMovingAverage.schedule_steps` / `schedule_freq`, with
  `sched_scaling = (1/schedule_steps)*min(last_step, schedule_steps)`) and action latency
  (`cur_action_latency = (1/action_latency_scheduled_steps)*min(last_step, ...)`). No reward weight is
  scheduled. — [allegro_hand_dextreme.py lines ~900, ~1569](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/dextreme/allegro_hand_dextreme.py)

**AnyRotate reward-weight curriculum**
- "these terms can hinder the learning process, resulting in local optima where the object will be
  stably grasped without being rotated. To alleviate this issue, we apply a reward curriculum
  coefficient `λ_rew (r_contact + r_stable)`". The coefficient is a **linear schedule in the number of
  successive goals reached per episode**, `λ_rew = (g_eval − g_min)/(g_max − g_min)` with
  **`[g_min, g_max] = [1.0, 2.0]`**. — [AnyRotate §Method / App. B](https://arxiv.org/abs/2405.07391)
- Ablated: "we also compare learning without adaptive curriculum (w/o curriculum)"; "The agent also
  failed to learn wi[thout the curriculum]" (Fig. 6). — [AnyRotate](https://arxiv.org/abs/2405.07391)
- AnyRotate additionally **loosens** the student's goal tolerance relative to the teacher's: "we
  increase the goal update tolerance `d_tol` during student training" — Table 5 gives `d_tol = 0.25`
  (teacher) and `0.15` (student), and Table 9 shows the student does best at the *looser* 0.25. —
  [AnyRotate](https://arxiv.org/abs/2405.07391)

**TriFinger reward-term schedule**
- `ft_sched_start = 0`, `ft_sched_end = 5e7` hard-coded in `compute_trifinger_reward`;
  `ft_sched_val = 1.0 if ft_sched_start <= env_steps_count <= ft_sched_end else 0.0` gates the
  `-250`-weighted fingertip-approach term off after 5×10⁷ env steps. This is the only curriculum in
  the TriFinger task and it is **in code, not in the YAML**. — [trifinger.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/trifinger.py)
- TriFinger's `Trifinger.yaml` has `randomization_params` with `schedule: "linear"` / `schedule_steps`
  entries for observation noise, action noise, gravity and DOF properties, but **all of them are
  commented out** in the shipped file. — [Trifinger.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/Trifinger.yaml)

**Tolerance tightening + object-set expansion (DexReMoE's reproduced baseline)**
- "When initially reproducing the baseline DR and ADR [1, 2] under our strict success criterion, which
  requires maintaining a stable hold at the goal orientation, training failed to converge ... To
  restore performance, we implemented curriculum learning. This process began with a relaxed success
  test using a single cube, then progressively tightened the criterion while incrementally introducing
  all 100 objects. Only after this staged progression did the baseline achieve comparable results."
  — [DexReMoE §IV](https://arxiv.org/abs/2508.01695)

**Hora's object-scale randomization (not a curriculum)**
- `randomizeScale: True`, `randomizeScaleList: [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86]`
  with per-env jitter `uniform(scale − 0.025, scale + 0.025)`; `randomizeScaleLower: 0.75`,
  `randomizeScaleUpper: 0.8` (used only when not `scale_list_init`); `jointNoiseScale: 0.02`;
  `forceScale: 0.0` (random-force disturbance disabled by default; applied as
  `randn * obj_mass * forceScale` when enabled). These are fixed-range randomizations, not scheduled.
  — [AllegroHandHora.yaml](https://github.com/HaozhiQi/hora/blob/main/configs/task/AllegroHandHora.yaml), [allegro_hand_hora.py](https://github.com/HaozhiQi/hora/blob/main/hora/tasks/allegro_hand_hora.py)

**Chen 2021 dynamics randomization (fixed, not scheduled)**
- Table C.4: state observation `+U(−0.001, 0.001)`, action `+N(0, 0.01)`, joint stiffness
  `×E(0.75, 1.5)`, object mass `×U(0.5, 1.5)`. Table C.3: EGAD bounding-box longest side
  `[0.05, 0.08] m`, YCB `[0.05, 0.12] m`, object mass `[0.05, 0.15] kg`, 2282 EGAD meshes, 78 YCB
  meshes, 5 variants per mesh, voxelization 0.003 m. — [Chen et al. 2021](https://arxiv.org/abs/2111.03043)
- Visual Dexterity Table S3: state observation `+N(−0.002, 0.002)`, action `+N(0, 0.05)`, joint
  stiffness `×U(0.8, 1.2)`, joint damping `×U(0.8, 1.2)`, link mass `×U(0.8, 1.2)`. —
  [Visual Dexterity Table S3](https://arxiv.org/abs/2211.11744)
- RAPE uses an **occlusion curriculum on the encoder, not the policy**: "the first 10 epochs are
  unoccluded, followed by 10-epoch stages with occlusion probabilities 0.3, 0.6, and 0.8", 16
  relative-angle bins, 600 pairs per object per epoch, Adam lr 1e-3, cosine annealing with linear
  warmup. — [RAPE Appendix A2](https://arxiv.org/abs/2606.21788)

### Inferences
- No system in this survey anneals a *reward weight* on a step counter. The two weight-schedule
  mechanisms found are keyed to **performance** (AnyRotate's `λ_rew` on goals-reached) or to a raw
  **env-step threshold that switches a term off** (TriFinger's `ft_sched`).
- The "as-shipped" Isaac Lab and IsaacGymEnvs reorientation tasks contain no curriculum at all, which
  means any curriculum in your own implementation is above baseline and needs its own justification.

### Gaps
- DexReMoE does not give the schedule for its baseline's tolerance tightening or object-set expansion
  (no thresholds, no step counts, no tolerance values per stage).
- OpenAI's original ADR (arXiv:1910.07113) constants — boundary-sampling probability, performance
  thresholds `t_L`/`t_H`, per-parameter step sizes — were not fetched in this pass; only DeXtreme's
  reimplementation constants are quoted above.

## Q7. Is there any published system with a goal-ANGLE-MAGNITUDE curriculum (small rotations first, then larger)?

### Takeaway
**Yes — this is a confirmed positive finding, not a negative one.** POISE (arXiv:2609.13761, 2026)
schedules the maximum goal rotation magnitude from 5° to 180°, with a published trigger condition and
step size, and publishes the ablation against uniform full-range sampling. AnyRotate has a related but
different mechanism (fixed 30° goal increments, ablated at 30/40/50°, not scheduled). No pre-2026
system in this survey has a goal-angle-magnitude curriculum.

### Cited Findings
- **POISE, "Adaptive 6D Goal Curriculum" (§IV-C), verbatim**: "Sampling the full range of 6D goals
  from the outset makes early exploration difficult. We therefore use separate curricula that expand
  the maximum rotation and translation magnitudes **from 5° and 10 mm to 180° and 30 mm**,
  respectively. Only the magnitudes are scheduled; axes and directions remain randomized, with the
  same limits applied to all targets in a rollout. Targets outside the palm-frame workspace or
  intersecting the fixed hand base are resampled. At each stage, magnitudes are sampled **near the
  frontier, from the exposed range, or at the current limit in a 0.6 / 0.3 / 0.1 ratio**. When
  **frontier success reaches 0.4**, the rotation or translation limit **increases by 10° or 10 mm**.
  **Each object maintains independent curriculum progress.**" — [POISE §IV-C](https://arxiv.org/abs/2609.13761)
- POISE's ablation (Table II, hexagonal prism, 64,000 envs per run), against "a variant that samples
  target magnitudes uniformly over the full range throughout training":
  | Training | Full range: Strict | Near | Goals/ep. | Max change: Strict | Near | Goals/ep. |
  |---|---|---|---|---|---|---|
  | Without curriculum | **6.2%** | 18.8% | 0.08 | **3.4%** | 13.9% | 0.04 |
  | Curriculum | **59.5%** | 77.0% | 1.55 | **55.3%** | 72.8% | 1.08 |

  Evaluation suites: full-range targets with rotations in `[5°, 180°]` and translations in `[10, 30]`
  mm; maximum-change targets of `180°` and `30` mm. "Strict" = 10°/10 mm, "Near" = 20°/20 mm
  first-target success. Conclusion as written: "directly training on the full goal range rarely
  produces successful 6D reaching ... progressively expanding the goal range is important for learning
  large coupled translations and rotations." — [POISE §V-B2, Table II](https://arxiv.org/abs/2609.13761)
- **AnyRotate uses a fixed goal increment, not a curriculum on it**: goals advance by successive
  rotations of a fixed angle, and the paper ablates the increment interval at 50°, 40° and 30°,
  choosing **30°** (best: 1.77 rotations, 5.26 successive goals). "Increasing the goal increment
  intervals also resulted in fewer rotations achieved." This is a *constant*, not a schedule. —
  [AnyRotate Table 9](https://arxiv.org/abs/2405.07391)
- **Negative for all the classic systems**: Dactyl samples goal orientations by "a uniformly-sampled
  axis and ... 0.4 rad"-tolerance targets with no magnitude schedule; Chen 2021 states "The initial and
  goal orientation are randomly sampled from SO(3) space in **all** the experiments"; the IsaacGymEnvs
  and Isaac Lab tasks call `randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor)` with no
  magnitude parameter and no schedule; Hora has no goal orientation at all; DeXtreme samples a fresh
  target whenever within 0.4 rad, with ADR acting on *dynamics* parameters only, never on goal
  difficulty. — [OpenAI 2018](https://arxiv.org/abs/1808.00177); [Chen et al. 2021 §Evaluation criterion](https://arxiv.org/abs/2111.03043); [shadow_hand.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/shadow_hand.py); [DeXtreme §2](https://arxiv.org/abs/2210.13702)

### Inferences
- POISE's design is directly transferable: the trigger is a *frontier* success rate (success measured
  specifically at the current limit) rather than overall success, which avoids the classic failure
  where an overall-success trigger stalls because easy goals dominate the average. The 0.6/0.3/0.1
  frontier/exposed/limit sampling mix is the anti-forgetting mechanism.
- POISE's task is 6D *reaching from a grasp* (with drop recovery), not free in-hand reorientation from
  arbitrary poses, so the 6.2% → 59.5% figure should not be read as a claim about cube reorientation
  specifically.

### Gaps
- POISE reports the curriculum ablation on a single object (hexagonal prism) in Table II; per-object
  curriculum-versus-no-curriculum numbers across its full object set were not located in the HTML.

## Q8. Success tolerance, hold time, and goal streaming without episode reset

### Takeaway
Tolerances cluster at **0.1 rad** (tight/Isaac default, Chen 2021, DexReMoE's hold criterion) and
**0.4 rad** (Dactyl-compatible: Dactyl, DeXtreme paper + manual-DR config, DexReMoE, Isaac Lab
"OpenAI" ShadowHand, TriFinger's orientation tolerance). Goal streaming without episode reset is
standard in Dactyl, DeXtreme, AnyRotate, POISE, RAPE and DexReMoE, and is available but **disabled by
default** in IsaacGymEnvs ShadowHand and Isaac Lab (`maxConsecutiveSuccesses: 0`).

### Cited Findings

| System | Orientation tolerance | Position tolerance | Hold | Goals per episode | Per-goal / episode budget |
|---|---|---|---|---|---|
| Dactyl | **0.4 rad** | none (drop check only) | none | up to **50** consecutive, streamed | 8 s simulated per goal; 80 ms/step |
| DeXtreme (paper) | 0.4 rad to resample target; **`d < 0.1`** for the +250 bonus (Table 2) | fall at 0.24 m | code supports `num_success_hold_steps`, shipped as **0** | **50** (`maxConsecutiveSuccesses: 50`) | `resetTime: 8` s |
| DeXtreme ADR yaml | `successTolerance: **0.1**` | `fallDistance: 0.24` | 0 | 50 | resetTime 8 s |
| DeXtreme ManualDR yaml | `successTolerance: **0.4**` | `fallDistance: 0.24` | not present in file | 50 | resetTime 8 s |
| IsaacGymEnvs ShadowHand | **0.1** (doubled to 0.2 if `ignore_z_rot`) | `fallDistance: 0.24` | none | **0 → streaming disabled**; one goal is re-sampled but `progress_buf` is not reset | `episodeLength: 600` |
| Isaac Lab ShadowHand | **0.1** | 0.24 | none | `max_consecutive_success = 0` | `episode_length_s = 10.0` |
| Isaac Lab ShadowHand (OpenAI variant) | **0.4** | 0.24 | none | **50** | `episode_length_s = 8.0` |
| Isaac Lab AllegroHand | **0.2** | 0.24 | none | 0 | 10.0 s |
| TriFinger | `orientation_tolerance: **0.4**` | `position_tolerance: **0.02**` m | none | 1 (reset only on timeout) | `episodeLength: 750`, `dt: 0.02` |
| Chen 2021 | **`θ̄ = 0.1 rad`** (non-vision); **0.2 rad** + `d̄_C = 0.01` (vision) | — | none | 1 (episode ends on success, drop, or timeout) | Episode length **300** |
| Visual Dexterity | success is orientation + object-motion + finger-motion criteria; `ε_θ = 0.4 rad` in the dense term; `q̇̄ = 0.25`, `v̄ = 0.04`, `ω̄ = 0.5` | `p̄ = 0.15`, `p̄_z = 0.16` | velocity-gated, not time-gated | 1 ("The robot stops when it is deemed successful") | — |
| Hora | no goal | — | — | continuous rotation | `episodeLength: 400` at 20 Hz |
| AnyRotate | keypoint `d_tol = **0.25**` (teacher), **0.15** (student); drop at `d_max = 0.1`; axis deviation `k̂_max = 45°` | — | none | **streamed**, goal advanced by **30°** each time | episode 600 steps ≈ 30 s |
| POISE | **10°/10 mm** strict (20°/20 mm "near") | 10 mm | **N = 20 consecutive steps** (1 s at 20 Hz) | **streamed** ("reaches successive 6D targets without manual reset") | — |
| DexReMoE | `success tolerance 0.4`; hold criterion `τ_θ = 0.1`, `τ_q = 10.0`, `τ_v = 0.04`, `τ_ω = 0.5` | — | **sustained hold through the final control cycle** | **streamed**, "consecutive successful reorientations ... within each fixed time window" (>15 typical, some higher) | episode length **600** |
| RAPE | `β` (unspecified value) | drop/out-of-reach termination | none | **max 5 consecutive**, "This count is reset upon a successful reorientation" | 600 env steps, counter reset per success |

Sources for the table rows above, in order: [OpenAI 2018](https://arxiv.org/abs/1808.00177);
[DeXtreme](https://arxiv.org/abs/2210.13702) and its [ADR](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeADR.yaml)/[ManualDR](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeManualDR.yaml) configs;
[ShadowHand.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/ShadowHand.yaml);
[shadow_hand_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py);
[allegro_hand_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/allegro_hand_env_cfg.py);
[Trifinger.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/Trifinger.yaml);
[Chen et al. 2021](https://arxiv.org/abs/2111.03043); [Visual Dexterity](https://arxiv.org/abs/2211.11744);
[AllegroHandHora.yaml](https://github.com/HaozhiQi/hora/blob/main/configs/task/AllegroHandHora.yaml);
[AnyRotate](https://arxiv.org/abs/2405.07391); [POISE](https://arxiv.org/abs/2609.13761);
[DexReMoE](https://arxiv.org/abs/2508.01695); [RAPE](https://arxiv.org/abs/2606.21788).

Mechanics of goal streaming, exactly as implemented:
- IsaacGymEnvs/Isaac Lab: the goal is re-sampled on `goal_resets`, and **only when
  `max_consecutive_successes > 0`** is the episode clock reset too:
  `progress_buf = where(|rot_dist| <= success_tolerance, 0, progress_buf)`, with
  `resets = where(successes >= max_consecutive_successes, 1, resets)`. With the default
  `maxConsecutiveSuccesses: 0`, the episode clock keeps running, so the effective budget for all goals
  in an episode is one `episodeLength`. — [shadow_hand.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/shadow_hand.py), [inhand_manipulation_env.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py)
- DeXtreme: same plumbing plus `hold_count_buf`; `resetTime: 8` seconds is described in the config as
  "Max time till reset, in seconds, **if a goal wasn't achieved**", i.e. a per-goal timeout, and it
  overrides `episodeLength`. — [AllegroHandDextremeADR.yaml](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeADR.yaml)
- DeXtreme's success metric: "the number of consecutive target orientations achieved without dropping
  the object or having the object stuck in the same configuration for more than 80 seconds.
  Importantly, each consecutive success becomes increasingly harder to achieve". The fingers "continue
  from the current configuration" — no state reset between goals. — [DeXtreme §2](https://arxiv.org/abs/2210.13702)
- Dactyl: "The episode ends when either the policy achieves 50 consecutive goals, the policy fails to
  achieve the current goal within 8 seconds of simulated time, or the object is dropped." Real-robot
  evaluation cut off at 80 s per goal. Reported result: **median 13 consecutive goals** with all
  randomizations, vs medians of **0, 2, 2** for no randomizations / no physics randomizations / no
  unmodeled effects. — [OpenAI 2018 §C.1, §6.2, Table 4](https://arxiv.org/abs/1808.00177)
- The `consecutive_successes` metric that ADR reads is an EMA:
  `cons_successes = av_factor * finished_cons_successes/num_resets + (1 − av_factor) * consecutive_successes`
  with `av_factor = 0.1` in the Isaac configs (`averFactor`, default 0.1 in DeXtreme). — shadow_hand.py, allegro_hand_dextreme.py

### Inferences
- Goal streaming is the substrate that makes ADR possible: the ADR objective *is* the consecutive-goal
  count, so the streaming episode structure and the curriculum are coupled in DeXtreme. If you stream
  goals, you get a natural difficulty signal for free.
- The `maxConsecutiveSuccesses: 0` default in the shipped Isaac tasks means the widely-copied baseline
  is **not** a goal-streaming task, even though the code supports it — a common source of confusion
  when comparing against Dactyl/DeXtreme numbers.
- A tolerance of 0.1 rad with `maxConsecutiveSuccesses: 0` (the IsaacGymEnvs ShadowHand default) and a
  tolerance of 0.4 rad with 50 streamed goals (Dactyl/DeXtreme-manual-DR) are different tasks; success
  rates between the two are not comparable.

### Gaps
- RAPE does not publish the numeric value of its success threshold `β`.
- DeXtreme's paper (0.4 rad to resample, `d < 0.1` for the bonus) and its ADR config
  (`successTolerance: 0.1`) do not obviously agree; the release does not explain the discrepancy, and
  the manual-DR config's `0.4` is the value consistent with the paper's stated resampling threshold.
  This is a genuine source conflict, flagged rather than resolved.
- Visual Dexterity's exact success thresholds (the orientation threshold `θ̄` used for the training
  success indicator, and how the three criteria are combined) are deferred to "Supplementary Methods"
  and were not located in the ar5iv render; `c_d = 15` appears in Table S1 without an in-text
  definition in the portion retrieved.
