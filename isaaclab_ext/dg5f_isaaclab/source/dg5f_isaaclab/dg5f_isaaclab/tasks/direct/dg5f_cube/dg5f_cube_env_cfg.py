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
from dg5f_isaaclab.assets.grasp_cache import DEFAULT_GRASP_CACHE_PATH
from dg5f_isaaclab.assets.sysid import DEFAULT_SYSID_PATH, load_sysid
from .control import CONTROL_MODES


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
