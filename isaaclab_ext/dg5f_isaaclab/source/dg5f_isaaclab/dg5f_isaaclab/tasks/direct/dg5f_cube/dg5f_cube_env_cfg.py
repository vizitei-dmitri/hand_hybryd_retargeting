"""State-only DG5F cube teacher for the installed Isaac Lab 2.3.2 API."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from dg5f_isaaclab.assets.dg5f import (
    DG5F_CFG, DG5F_DEFAULT_EFFORT_CAP_NM, DG5F_FINGERTIP_NAMES, DG5F_JOINT_EFFORT_LIMITS,
    DG5F_JOINT_LIMITS, DG5F_JOINT_NAMES, DG5F_PALM_NAME,
)
from dg5f_isaaclab.assets.object_cube import (
    DEX_CUBE_NATIVE_EDGE_M, DEX_CUBE_USD_PATH, VisualCuboidCfg,
)
from dg5f_isaaclab.assets.grasp_cache import DEFAULT_GRASP_CACHE_PATH, ROBUST_GRASP_CACHE_PATH
from dg5f_isaaclab.assets.sysid import DEFAULT_SYSID_PATH, load_sysid
from .control import CONTROL_MODES
from .goals import STREAM_STAGES


@configclass
class DG5FCubeEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 2
    # Long enough for a chain of goals: the single-goal task used 8 s, a solved goal takes ~0.5 s
    # plus the reorientation itself, so 24 s leaves room for several in a row.
    episode_length_s = 24.0
    # Derived again from the JSON by resolve_control_config, including CLI overrides.
    action_space = len(DG5F_JOINT_NAMES) - 1
    privileged_observation_space = 2 * len(DG5F_JOINT_NAMES) + 3 + 4 + 3 + 3 + 4 + 3 * len(DG5F_FINGERTIP_NAMES)
    observation_space = privileged_observation_space
    state_space = 0
    act_moving_average = 1.0
    # Also "measured_delta_position" (target = q_measured + delta) and "absolute_position" for A/B.
    control_mode = "integrated_delta_position"
    delta_action_scale = math.radians(3.0)
    sysid_path = str(DEFAULT_SYSID_PATH)
    # rj_dg_5_1 is locked by PhysX limits [0, 0]. The limit alone is solved iteratively
    # (random rollouts: up to 0.065 deg); extra armature on THIS axis only brings it to
    # ~0.002 deg without touching the 20-DOF layout (logs/physics_sanity/vel_O_*).
    # Overrides the (meaningless for a locked joint) SysID armature of the disabled joint.
    disabled_joint_lock_armature = 0.01
    # DG-5F-M rated joint torque, independent of stiffness/damping. None selects the URDF 7.5 Nm.
    # TODO: replace with a current/thermal actuator model identified on the real hand.
    effort_limit_cap_nm: float | None = DG5F_DEFAULT_EFFORT_CAP_NM
    # Frame the hand in the existing GUI viewport (no camera sensor).
    viewer = ViewerCfg(
        eye=(-0.42, -0.32, 0.86), lookat=(-0.12, 0.0, 0.65), origin_type="env",
    )
    # GUI-only goal visualization, off for training: it adds stage prims and a per-step USD write
    # that thousands of headless envs should not pay for. play.py goes through hydra, so watching a
    # policy only needs `env.goal_marker=true` on the command line.
    goal_marker = False
    # Palm frame: +X is the palm normal (see cube_position_in_palm), so this draws the ghost above
    # the hand, visible from the configured camera and clear of the neighbouring env (spacing 0.75).
    # 0.22 m clears the cube itself: the cache seats it 0.085-0.129 m out along the same axis.
    goal_marker_offset_in_palm = (0.22, 0.0, 0.0)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            # The same stock asset as the cube's own visual, so the ghost's coloured faces and
            # letters can be read against the real cube's directly. VisualizationMarkers strips the
            # rigid-body/collider APIs from the prototype, so this stays purely visual.
            # scale is fitted to cube_size_m in __post_init__.
            "goal": sim_utils.UsdFileCfg(usd_path=DEX_CUBE_USD_PATH),
        },
    )
    # Lamp above the ghost: the orientation error as a colour, so "is it getting closer" is legible
    # without reading numbers. Green is the success tolerance, yellow is this multiple of it.
    goal_marker_near_scale = 3.0
    goal_marker_status_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_status",
        markers={
            "far": sim_utils.SphereCfg(
                radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.1, 0.1))),
            "near": sim_utils.SphereCfg(
                radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.7, 0.1))),
            "inside": sim_utils.SphereCfg(
                radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2))),
        },
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        # The default rigid patch budget (163840) overflows past ~1024 envs: PhysX then drops
        # contact patches, which silently changes the finger/cube contacts this task depends on
        # (logs/scaling/probe_1024.log asked for 165827). 4x the default keeps headroom.
        physx=PhysxCfg(bounce_threshold_velocity=0.2, gpu_max_rigid_patch_count=655360),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False,
    )
    robot_cfg: ArticulationCfg = DG5F_CFG.copy()
    actuated_joint_names = list(DG5F_JOINT_NAMES)
    fingertip_body_names = list(DG5F_FINGERTIP_NAMES)
    palm_body_name = DG5F_PALM_NAME

    # Reset state of all 20 joints, degrees, URDF order. The PD command starts at
    # grasp_joint_pos_deg + grasp_command_offset_deg (the offset is the grasp preload).
    # Calibrated by scripts/grasp_hold_sanity.py --mode search on the corrected collision geometry
    # (logs/grasp_hold/06_search_flexion_preload_noisy.log): the hand-designed grasp
    # (10,-90,45,35, 0,40,65,30, 0,35,65,30, 0,35,65,30, 0,-10,55,40) with rj_dg_5_2 +2 deg,
    # settled on the 60 mm cube, re-seated twice, preload <= 0.5 deg on flexion joints only
    # (abduction/opposition commands = settled angles), ranked by the worst of 4 noisy holds.
    grasp_joint_pos_deg = (10.0, -89.86, 44.19, 34.56, -0.26, 39.68, 64.81, 29.94, -1.17, 35.01, 64.85, 29.75,
                           -2.14, 34.72, 39.28, 30.0, 0.0, -4.52, 55.22, 40.06)
    grasp_command_offset_deg = (0.0, 0.0, 0.5, 0.44, 0.0, 0.32, 0.19, 0.06, 0.0, -0.01, 0.15, 0.25,
                                0.0, 0.28, 0.5, 0.0, 0.0, 0.0, -0.22, -0.06)

    # One rigid body: hidden box collider + stock DexCube visual (colored faces, letters X/R/E/T/M/D).
    cube_size_m = 0.06
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=VisualCuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, enable_gyroscopic_forces=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=2,
                max_depenetration_velocity=0.2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            ),
        ),
        # Initial spawn before reset; root->palm fixed translation is 0.0738 m.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.1738, 0.0, 0.655)),
    )
    # URDF palm coordinates: +X is the palm normal, +Z points toward the fingers.
    # Settled cube center for the calibrated grasp (previously (0.055, 0.0, 0.100)).
    cube_position_in_palm = (0.0548, -0.0015, 0.076)
    cube_reset_position_noise_m = 0.001
    # 0 keeps the cube aligned with the palm at reset. Used by the grasp-cache robustness search,
    # which validates a reset under perturbation instead of trusting a static snapshot.
    cube_reset_orientation_noise_deg = 0.0
    reset_joint_position_noise_rad = math.radians(0.5)

    target_orientation_mode = "uniform"  # "axis" for the single-axis task, "uniform" for full SO(3)
    target_axis_in_palm = (1.0, 0.0, 0.0)
    target_angle_range_deg = (-20.0, 20.0)
    success_tolerance_rad = math.radians(5.0)
    # Goals are rejection-sampled so the episode never starts inside (or near) the tolerance.
    min_initial_goal_error_deg = 10.0
    # Success = inside the tolerance for this long without leaving (v1: first entry counted).
    success_hold_time_s = 0.30
    # Consecutive goals: a completed goal is immediately replaced by a new one (sampled against
    # the CURRENT cube pose, same minimum error), so the episode keeps asking for more rotation
    # instead of ending in a hold. This is what produces finger gaiting rather than a squeeze.
    resample_goal_on_success = True

    # Reward v2: dense state term + small progress shaping (v1 used progress scale 10, no state term).
    orientation_state_scale = 0.1
    orientation_sigma_deg = 10.0
    orientation_progress_scale = 1.0
    palm_region_distance_scale = 1.0
    action_penalty_scale = 0.002
    action_rate_penalty_scale = 0.001
    success_bonus = 2.0
    fall_penalty = 20.0
    max_cube_distance_from_palm_m = 0.25
    # --- Goal-difficulty curriculum (POISE 2026: full-range success 6.2% -> 59.5%).
    # The limit is the maximum rotation a goal may demand; it starts at the difficulty that is
    # already solved (night_5000 reached 1.74 deg final error on axis +-20 deg) and only widens
    # once the frontier is being solved, so a hard task simply keeps training at 20 deg.
    goal_curriculum = False
    goal_curriculum_start_deg = 20.0
    goal_curriculum_max_deg = 180.0
    goal_curriculum_step_deg = 10.0
    goal_curriculum_band_deg = 10.0
    goal_curriculum_promote_rate = 0.4
    goal_curriculum_min_samples = 1024
    # Sampling split: near the frontier / anywhere inside the exposed range / exactly at the limit.
    goal_frontier_fraction = 0.6
    goal_inside_fraction = 0.3
    # Promotion needs a host sync, so it is checked on an interval rather than every step.
    goal_curriculum_check_steps = 120

    # --- Precision (fingertip) grasp task, reward v3. Everything here is OFF by default, so the
    # power-grasp task above is bit-for-bit unchanged unless it is switched on. Motivation: the
    # configured cfg grasp is an enclosing grasp -- logs/*/04b_contact_audit.log has the palm at
    # 0.38-1.11 N against a 0.49 N cube and the tips at 0.02-0.53 N -- so the contact and
    # grasp-quality terms below are undefined from it and have to start from a grasp cache.
    # None keeps the single configured grasp; a path makes every episode start from a cached grasp.
    grasp_cache_path: str | None = None
    # Required by every contact term. 6 cube-filtered sensors (5 tips + palm), not all 28 links.
    enable_contact_sensors = False
    tip_contact_force_n = 0.05
    # AnyRotate (2024): reward >= 2 tips (we have 5 fingers, so 3) and penalise non-tip support.
    min_tip_contacts = 3
    tip_contact_reward = 0.0
    palm_contact_penalty = 0.0
    # Robust In-Hand Manipulation via Priors (2026): dense lambda_min of the grasp Gramian
    # M = G G^T built from the fingertip contact positions, bounded and EMA-normalised so it
    # cannot swamp the task term. POISE weights grasp maintenance at 0.05x the pose terms.
    grasp_quality_scale = 0.0
    grasp_quality_clip = 0.5
    grasp_quality_margin = 1e-4
    grasp_quality_ema = 0.97
    # --- grasp DEFENCE: one-sided deficit penalties with a dead zone.
    #
    # Measured over stage A of the reward-v4 run, per step: orientation_state reached -0.0273 while
    # tip_contact paid +0.0025 and grasp_quality +0.0010. The dense landscape asked for less
    # orientation error with 11-27x the weight of anything protecting the pinch, so the only defence
    # was drop_penalty -50 -- a single terminal event that at gamma=0.99 (horizon 100 steps) cannot
    # reach the actions 900 steps earlier that actually lost the grasp. Result: goals/episode 0.117
    # -> 0.070 while drop_rate went 0.078 -> 0.570.
    #
    # Why a one-sided penalty and not a larger scale on the existing terms. Both existing terms are
    # purely POSITIVE functions of state that the zero policy already maximises (+13 per episode on
    # the robust cache, which is why the zero policy holds the highest reward of any policy at
    # +5.5). Scaling them up scales up the payoff for holding still.
    #
    # Why a dead zone and not centering on the intact value. Measured (scripts/reward_preflight.py,
    # logs/reward_preflight/preflight_grasp_ref.json): the zero policy holds tips>=3 on 0.906 of
    # steps at quality 0.361; the warm-start policy flickers down to 0.718 / 0.255 WITHOUT dropping
    # (drop 0.039); the scripted motion probe collapses to 0.243 / -0.139 and drops 0.984. Centering
    # on the still-hand value would tax the flicker that legitimate motion produces at roughly -19
    # per episode, more than a completed goal pays -- reward v3's exact failure, a penalty active at
    # the grasp itself. The floors are therefore set at what the non-dropping policy already holds,
    # so maintaining it is free and only degradation toward the failure region costs.
    grasp_deficit_scale = 0.0        # on the fingertip COUNT below min_tip_contacts, graded
    grasp_quality_deficit_scale = 0.0
    grasp_quality_floor = 0.25       # warm-start level; below it the Gramian is collapsing
    # POISE (2026): -0.10||tau||^2 and -0.15(tau^T qdot)^2. Reward v2 penalises actions, not effort,
    # which is what matters at a rated 0.4 Nm.
    torque_penalty_scale = 0.0
    work_penalty_scale = 0.0
    # Keeps q_cmd near the grasp the episode started from. Both 2607.12105 (lambda_pose on
    # ||q - q_cache||) and AnyRotate (lambda_pose = 0.2 on ||q - q0||) carry this term, and its
    # stated job is exactly this: stop exploration from wandering out of a validated grasp.
    # Without it a precision grasp cannot survive PPO's exploration, because the delta actions
    # integrate into q_cmd as a random walk (std 0.5 x 3 deg over 90 steps is ~14 deg per joint).
    pose_penalty_scale = 0.0
    # A fingertip grasp can actually be dropped, unlike the cradle: 0.25 m never fired, so every
    # policy logged drop_rate 0.000. Tightened only when the precision task is enabled.
    max_cube_distance_from_grasp_m: float | None = None
    # Terminate after this long with fewer than one fingertip on the cube (None disables).
    max_time_without_tip_contact_s: float | None = None

    # --- Reward v4 / fixed-angle goal stream. OFF by default: the reward-v3 fingertip task above
    # stays bit-for-bit reproducible, because its night run is the reference this is measured
    # against. Motivation from that run: the policy learned to hold and to FLY THROUGH the 5 deg
    # tolerance (entered 0.508, held 0.000), and two stability penalties were 69 of the 90 units
    # of per-episode reward magnitude, so not moving was the optimum.
    # Every goal is a relative rotation of exactly this angle -- there is no angle curriculum.
    # Repeated local 20 deg goals about changing axes cover SO(3) without one large jump.
    goal_stream = False
    goal_stream_angle_deg = 20.0
    # Directional curriculum: A = the already-solved palm axis, B = the three palm principal axes,
    # C = a uniformly random axis. One policy runs through all three.
    goal_stream_stage = "A"
    # Baseline-centred dense orientation term: scale * (exp(-err/sigma) - exp(-stream_angle/sigma)).
    orientation_baseline = False
    orientation_decay_deg = 15.0
    # Clipped progress: a contact impulse or an overshoot must not pay a large one-step reward.
    # 0 disables the clip. 0.025 rad is 1.43 deg per control step, i.e. 86 deg/s of object rotation.
    orientation_progress_clip_rad = 0.0
    # Per-step reward for being inside the tolerance before the hold completes: this is what
    # teaches braking rather than passing through. Capped below success_bonus by construction
    # (success_hold_steps * goal_dwell_reward must stay smaller).
    goal_dwell_reward = 0.0
    # Anti-overshoot, deliberately NOT a global motion penalty: only near the goal, and only the
    # object angular velocity ABOVE a threshold.
    goal_velocity_scale = 0.0
    goal_velocity_error_deg = 10.0
    goal_velocity_threshold_rad_s = 0.6
    # Dead zones turn the two dominant penalties into safety rails. Sized from the night run, not
    # assumed: the measured trained deviation was mean|q_cmd - q_grasp| = 14.9 deg (pose_penalty
    # 0.026 at scale 0.1) and cube_drift_from_grasp 21.5 mm mean, 32.8 mm max, against 6.7 mm of
    # passive settling. So the rails engage just past where manipulation actually operates, and
    # the first 12 deg / 20 mm of rearrangement is free.
    pose_penalty_deadband_deg = 0.0
    palm_region_deadband_m = 0.0
    # Reset curriculum: the primary cache plus an optional fraction drawn from a second one, so a
    # policy can be introduced to the fragile resets only after the robust ones are stable.
    mix_grasp_cache_path: str | None = None
    mix_cache_fraction = 0.0

    # Diagnostics only (no reward): SysID damping 1e-4 joints and joints that saturated in the smoke test.
    velocity_watch_joints = ["rj_dg_1_3", "rj_dg_1_4", "rj_dg_2_4", "rj_dg_3_4", "rj_dg_4_4", "rj_dg_5_4"]
    saturation_watch_joints = ["rj_dg_2_1", "rj_dg_3_1", "rj_dg_4_1", "rj_dg_5_2"]
    action_near_limit = 0.95
    min_cube_world_height_m = 0.45

    def __post_init__(self):
        self.resolve_control_config()

    def resolve_control_config(self):
        """Apply the JSON once per config resolution and derive the policy layout."""
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"Unknown control_mode: {self.control_mode}")
        if not math.isfinite(self.delta_action_scale) or self.delta_action_scale <= 0:
            raise ValueError("delta_action_scale must be positive radians")
        data = load_sysid(self.sysid_path, DG5F_JOINT_NAMES)
        # Confirmed by src/lerobot_robot_dg5f/config/bridge.params.yaml:
        # disabled_joints: [rj_dg_5_1], disabled_positions_deg: [0.0].
        real_disabled_positions = {"rj_dg_5_1": 0.0}
        if data.disabled_joint not in real_disabled_positions:
            raise ValueError("No verified real-robot fixed position for this SysID disabled_joint")
        self.sysid_sha256 = data.sha256
        self.disabled_joint = data.disabled_joint
        self.disabled_joint_position = real_disabled_positions[data.disabled_joint]
        self.active_action_joints = [name for name in DG5F_JOINT_NAMES if name != self.disabled_joint]
        assert self.disabled_joint not in self.active_action_joints
        self.action_space = len(self.active_action_joints)
        self.delay_control_steps_by_finger = data.delay_control_steps_by_finger.copy()
        self.action_delay_steps = data.delays_for(self.active_action_joints)
        self.action_history_steps = max(1, max(self.action_delay_steps))
        # + delay queue (history x actions) + current integrated command q_cmd (actions).
        self.observation_space = (
            self.privileged_observation_space + self.action_space * self.action_history_steps + self.action_space
        )
        actuator = self.robot_cfg.actuators["fingers"]
        for name, values in data.parameters.items():
            setattr(actuator, name, values.copy())
        if not math.isfinite(self.disabled_joint_lock_armature) or self.disabled_joint_lock_armature < 0:
            raise ValueError("disabled_joint_lock_armature must be nonnegative")
        actuator.armature[data.disabled_joint] = self.disabled_joint_lock_armature
        cap = self.effort_limit_cap_nm
        if cap is not None and (not math.isfinite(cap) or cap <= 0):
            raise ValueError("effort_limit_cap_nm must be positive or None")
        actuator.effort_limit_sim = {
            name: limit if cap is None else min(cap, limit) for name, limit in DG5F_JOINT_EFFORT_LIMITS.items()
        }
        if len(self.grasp_joint_pos_deg) != len(DG5F_JOINT_NAMES) or len(self.grasp_command_offset_deg) != len(
            DG5F_JOINT_NAMES
        ):
            raise ValueError("Grasp state/offset must list all 20 URDF joints")
        state = dict(zip(DG5F_JOINT_NAMES, map(math.radians, self.grasp_joint_pos_deg)))
        offset = dict(zip(DG5F_JOINT_NAMES, map(math.radians, self.grasp_command_offset_deg)))
        state[self.disabled_joint] = self.disabled_joint_position
        offset[self.disabled_joint] = 0.0
        self.grasp_command = {}
        for name, (lower, upper) in zip(DG5F_JOINT_NAMES, DG5F_JOINT_LIMITS):
            if not lower <= state[name] <= upper:
                raise ValueError(f"Grasp state for {name} is outside the URDF limits")
            self.grasp_command[name] = min(max(state[name] + offset[name], lower), upper)
        self.robot_cfg.init_state.joint_pos = state
        if math.radians(self.min_initial_goal_error_deg) <= self.success_tolerance_rad:
            raise ValueError("min_initial_goal_error_deg must exceed the success tolerance")
        control_dt = self.sim.dt * self.decimation
        self.success_hold_steps = max(1, round(self.success_hold_time_s / control_dt))
        if self.orientation_sigma_deg <= 0 or self.orientation_state_scale < 0:
            raise ValueError("orientation_sigma_deg must be positive and orientation_state_scale nonnegative")
        if not math.isfinite(self.cube_size_m) or self.cube_size_m <= 0:
            raise ValueError("cube_size_m must be positive")
        self.object_cfg.spawn.size = (self.cube_size_m,) * 3
        # The ghost must be the same size as the cube or the comparison is misleading.
        self.goal_marker_cfg.markers["goal"].scale = (self.cube_size_m / DEX_CUBE_NATIVE_EDGE_M,) * 3
        # Half the goal magnitude, so "near the goal" keeps its meaning when the goal shrinks: at the
        # 10 deg bootstrap angle the absolute 10 deg window covered the whole approach and braked it
        # (-0.0025 per step against -0.0001..-0.0006 at 20 deg). Only with the goal stream, because
        # without it the goal magnitude comes from target_angle_range_deg, not from this field.
        self.goal_velocity_window_deg = self.goal_velocity_error_deg
        if self.goal_stream:
            self.goal_velocity_window_deg = min(self.goal_velocity_error_deg,
                                                0.5 * self.goal_stream_angle_deg)
        if self.grasp_quality_deficit_scale and not self.grasp_quality_scale:
            raise ValueError(
                "grasp_quality_deficit_scale needs grasp_quality_scale non-zero: the quality is only "
                "computed inside that term's branch, and the deficit penalty reads it.")
        if self.goal_stream:
            if self.goal_stream_stage not in STREAM_STAGES:
                raise ValueError(f"goal_stream_stage must be one of {STREAM_STAGES}")
            if self.goal_stream_angle_deg <= math.degrees(self.success_tolerance_rad):
                raise ValueError("goal_stream_angle_deg must exceed the success tolerance")
            if self.goal_curriculum:
                raise ValueError("goal_stream replaces the angle curriculum; goal_curriculum must be False")
            # The dwell shaping must not outgrow the event it leads up to, or reaching the goal and
            # loitering just inside the tolerance beats completing and taking the next goal.
            dwell_total = self.success_hold_steps * self.goal_dwell_reward
            if dwell_total >= self.success_bonus:
                raise ValueError(
                    f"dwell shaping {dwell_total:.2f} over {self.success_hold_steps} steps must stay "
                    f"below success_bonus {self.success_bonus}")
        if self.orientation_baseline and self.orientation_decay_deg <= 0:
            raise ValueError("orientation_decay_deg must be positive")
        if self.mix_grasp_cache_path is not None and not 0.0 <= self.mix_cache_fraction <= 1.0:
            raise ValueError("mix_cache_fraction must be in [0, 1]")
        return data


@configclass
class DG5FCubeFingertipEnvCfg(DG5FCubeEnvCfg):
    """Precision-grasp variant: episodes start from a cached fingertip grasp, reward v3 is on.

    A separate registered task rather than a dozen CLI overrides, so the configuration is
    reproducible and the power-grasp task above keeps its defaults untouched.

    Weights are relative to orientation_state_scale (0.1), the dense task term. AnyRotate's
    contact reward of 0.1 stands against a rotation reward of 5.0, i.e. ~2% of the task term;
    taken literally here it would make "hold still and do not rotate" worth up to +144 over a
    24 s episode, which is the "stably grasped without being rotated" failure that paper reports.
    """

    grasp_cache_path = str(DEFAULT_GRASP_CACHE_PATH)
    enable_contact_sensors = True
    # 1 deg, not the power-grasp task's 3 deg. At 60 Hz control 3 deg/step is 180 deg/s of command
    # slew, and because the deltas integrate into q_cmd, exploration is a random walk: std 0.2 x 3 deg
    # over 90 steps reaches ~14 deg per joint, which a precision grasp cannot survive (smoke runs 09,
    # 10 and 12 all lost the grasp). POISE controls at 20 Hz for comparison.
    delta_action_scale = math.radians(1.0)
    # 0.01, not AnyRotate's 0.1: measured on the cache hold (logs/fingertip_grasp/07_cache_sanity.log)
    # 0.02 already made this the LARGEST positive term (+0.0164/step vs +0.0099 for the task term),
    # i.e. +23.6 per 24 s episode for doing nothing. It is also partly redundant -- losing the grasp
    # is already punished by fall_penalty and by termination on losing every fingertip.
    tip_contact_reward = 0.01
    palm_contact_penalty = 0.2      # a penalty cannot be farmed, so it stays strong
    grasp_quality_scale = 0.01      # POISE weights grasp maintenance at 0.05x the pose terms
    torque_penalty_scale = 0.02
    # work_penalty_scale stays 0: POISE's -0.15 (tau^T qdot)^2 reaches ~94 per step at our
    # tau <= 0.4 Nm and qdot <= pi rad/s, which would swamp every other term.
    success_bonus = 10.0            # chained goals: the bonus must outweigh the progress term
    fall_penalty = 50.0             # AnyRotate uses 50; the old 20 went with an unreachable threshold
    # A fingertip grasp can actually be dropped, unlike the cradle -- but a DROP is losing the
    # contacts, not translating. At 0.04 m this terminated 1.07% of steps while lost_tip_contact
    # fired on 0.09% (logs/fingertip_grasp/11_diag.log): it was killing valid grasps that had
    # merely slid, and in-hand manipulation moves the object by definition (POISE even makes
    # translation a goal). 0.10 m is left only as a backstop; drift is discouraged smoothly by
    # palm_distance_penalty, and the real drop detector is the loss of every fingertip.
    max_cube_distance_from_grasp_m = 0.10
    max_time_without_tip_contact_s = 0.25
    # 0.1 x mean|q_cmd - q_grasp| in rad: ~0.005/step at 3 deg of deviation and 0.017 at 10 deg,
    # so small finger motion (which rotating the cube requires) stays cheap while wandering does not.
    pose_penalty_scale = 0.1
    # Starts at the difficulty night_5000 already solved (axis +-20 deg, final error 1.74 deg) and
    # widens only when the frontier is solved, so a hard task simply keeps training at 20 deg.
    goal_curriculum = True


@configclass
class DG5FCubeStreamEnvCfg(DG5FCubeFingertipEnvCfg):
    """Reward v4: a stream of 20 deg goals that must be REACHED, BRAKED INTO and HELD.

    Registered separately from the reward-v3 fingertip task so that task, and the night run that
    measured it, stay reproducible. Physics, control and contact definitions are inherited
    unchanged; only the reward, the goal generator and the reset distribution differ.

    What the reward-v3 night run established, and what each change here answers:
      * the grasp is learnable and actively stabilised (drop 0.078 vs 0.172 for zero actions),
        so grasp stability becomes a CONSTRAINT here, not the objective -- no hold-only stage;
      * the policy reached the tolerance ever more often (entered 0.008 -> 0.508) but never held
        it (held 0.000, time in tolerance <= 1.8%), i.e. it flew through -> dwell reward and a
        near-goal angular-velocity penalty;
      * pose_penalty and palm_distance_penalty carried 69 of ~90 units of per-episode reward
        magnitude against +21 for the task, making "do not move" optimal -> both become dead-zone
        rails, and the task terms are raised;
      * exploration inflated (action_std 0.20 -> 0.795) -> entropy_coef 0 and a std watchdog in
        the agent config.
    """

    goal_stream = True
    goal_curriculum = False          # the angle is fixed; the curriculum is directional
    goal_stream_angle_deg = 20.0
    goal_stream_stage = "A"
    # A stream goal is always 20 deg away, so the rejection minimum is what the stream itself gives.
    min_initial_goal_error_deg = 20.0

    # --- task terms. The baseline-centred form is worth exactly 0 at 20 deg, +0.221 at the goal
    # and negative beyond 20 deg, so loitering at the start distance earns nothing.
    orientation_baseline = True
    orientation_state_scale = 0.30
    orientation_decay_deg = 15.0
    orientation_progress_scale = 2.0
    orientation_progress_clip_rad = 0.025
    goal_dwell_reward = 0.20         # 18 hold steps x 0.20 = 3.6, well under success_bonus 10
    success_bonus = 10.0

    # --- anti-overshoot, active only inside 10 deg and only above 0.6 rad/s.
    goal_velocity_scale = 0.03
    goal_velocity_error_deg = 10.0
    goal_velocity_threshold_rad_s = 0.6

    # --- grasp constraint. Unchanged quality term; contact shaping halved so that holding
    # without rotating cannot pay for itself (AnyRotate's "stably grasped without being rotated").
    grasp_quality_scale = 0.01
    tip_contact_reward = 0.005
    palm_contact_penalty = 0.2       # a penalty cannot be farmed, so the cradle stays forbidden
    fall_penalty = 50.0

    # --- grasp defence, sized from MEASURED distributions, not means
    # (logs/reward_preflight/preflight_v5_pct.json).
    #
    # First sizing attempt used the mean quality (0.255) against a floor of 0.25 and predicted -7 per
    # episode for the warm-start policy. Measured: -43. The penalty is clipped, hence convex, so
    # E[max(0, floor - q)] > max(0, floor - E[q]) for anything that fluctuates -- the same error as
    # sizing min_mean_tips from an instantaneous threshold, and as reward v3's assumed 3 deg.
    #
    # The percentiles show why: grasp_quality is BIMODAL, not spread around its mean. Warm-start
    # policy p01..p25 = -0.245..-0.175, p50..p90 = +0.425..+0.500, with almost nothing between
    # -0.17 and +0.42. The two modes are "pinch intact" and "pinch degenerate", and the tip-count
    # percentiles identify them: p50 = 3 tips against p10..p25 = 2. Time spent in the degenerate
    # mode: zero policy ~7%, warm-start ~30-40%, scripted probe ~80% (it drops 0.984).
    #
    # grasp_quality_floor therefore sits in the measured GAP, where its exact value does not matter
    # -- the term is effectively "is the pinch degenerate", and no small change to the floor moves
    # which samples it selects.
    #
    # The scales are what had to come down. At the measured occupancies they give, per step:
    #                          defence     orientation_state    defence as % of task
    #   zero    (drop 0.031)    0.003            0.017                  18%
    #   warm    (drop 0.039)    0.013            0.035                  38%
    #   probe   (drop 0.984)    0.046            0.051                  89%
    # So the defence grows from a fifth of the task term at an intact grasp to nearly all of it as
    # the pinch collapses, which is the gradient stage A did not have (defence was 0.0035 against
    # 0.027). The cost to the warm-start policy's current behaviour is about -19 per episode, under
    # two completed goals, so rotating still pays.
    grasp_deficit_scale = 0.015
    grasp_quality_deficit_scale = 0.06
    grasp_quality_floor = 0.25

    # --- rails. Dead zones sized from the night run's MEASURED values (pose deviation 14.9 deg,
    # cube drift 21.5 mm mean / 32.8 mm max against 6.7 mm of passive settling), so the first
    # 12 deg / 20 mm of finger rearrangement is free.
    #
    # The gradient outside the dead zone is a tenth of reward v3's, which is the point: v3 DID hold
    # its grasp (model_4000: deterministic drop 0.078) but could not rotate, because its penalty was
    # active everywhere including at the grasp itself.
    #
    # These were briefly raised to v3's full gradient (0.1 / 1.0) on the theory that the warm-start
    # actor needed that restoring force. Two 1024-env smokes, one per setting, were
    # indistinguishable (pose deviation 20.9 vs 20.3 deg, drift 30.4 vs 30.7 mm, tips 2.03 vs 1.97),
    # and the theory was void anyway: neither run had actually loaded the warm start, because
    # cli_args.py silently forces agent_cfg.resume to False. Reverted to the prescribed values;
    # nothing has been measured that argues against them.
    pose_penalty_scale = 0.01
    pose_penalty_deadband_deg = 12.0
    palm_region_distance_scale = 0.3
    palm_region_deadband_m = 0.020

    # --- reset distribution: the perturbation-validated subset (scripts/grasp_cache_robustify.py).
    grasp_cache_path = str(ROBUST_GRASP_CACHE_PATH)
    mix_grasp_cache_path: str | None = None
    mix_cache_fraction = 0.0
