"""State-only DG5F cube teacher for the installed Isaac Lab 2.3.2 API."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from dg5f_isaaclab.assets.dg5f import (
    DG5F_CFG, DG5F_DEFAULT_EFFORT_CAP_NM, DG5F_FINGERTIP_NAMES, DG5F_JOINT_EFFORT_LIMITS,
    DG5F_JOINT_LIMITS, DG5F_JOINT_NAMES, DG5F_PALM_NAME,
)
from dg5f_isaaclab.assets.object_cube import VisualCuboidCfg
from dg5f_isaaclab.assets.sysid import DEFAULT_SYSID_PATH, load_sysid
from .control import CONTROL_MODES


@configclass
class DG5FCubeEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 2
    episode_length_s = 8.0
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

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
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

    target_orientation_mode = "axis"  # "axis" initially, "uniform" for full SO(3) later
    target_axis_in_palm = (1.0, 0.0, 0.0)
    target_angle_range_deg = (-20.0, 20.0)
    success_tolerance_rad = math.radians(5.0)
    # Goals are rejection-sampled so the episode never starts inside (or near) the tolerance.
    min_initial_goal_error_deg = 10.0
    # Success = inside the tolerance for this long without leaving (v1: first entry counted).
    success_hold_time_s = 0.30

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
        return data
