"""Port only the asset/layout; upstream reward, resets and actions stay upstream.

DG5F has 20 revolute joints in its URDF, but rj_dg_5_1 is unavailable. The stream
uses a zero-width PhysX limit. Upstream unscale divides by (upper - lower), so
that representation would produce NaN observations. In this separate asset the
already-disabled axis is fixed at the same zero position, retaining both bodies,
inertias and collisions (merge_fixed_joints=False). All 19 movable axes retain
the original SysID gains, friction, armature and torque/velocity limits.

This benchmark uses upstream absolute position targets. It does not test the
stream's delta integrator / command delay; that controller is not modified.
"""
from .asset_adapter import fixed_axis_urdf

from isaaclab.utils import configclass
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg
from isaaclab_tasks.direct.shadow_hand.agents.rsl_rl_ppo_cfg import ShadowHandPPORunnerCfg

from dg5f_isaaclab.assets.dg5f import DG5F_JOINT_NAMES, DG5F_FINGERTIP_NAMES, URDF_PATH, EXTERNAL_PROJECT_ROOT
from ..dg5f_cube.dg5f_cube_env_cfg import DG5FCubeEnvCfg




@configclass
class DG5FReposeEnvCfg(ShadowHandEnvCfg):
    seed = 42
    # full = 2 * movable DOFs + object(13) + goal(11) + tips(65) + actions(19)
    action_space = 19
    observation_space = 2 * 19 + 13 + 11 + 5 * (3 + 4 + 6) + 19  # 146
    state_space = 0

    def __post_init__(self):
        source = DG5FCubeEnvCfg()
        disabled = source.disabled_joint
        self.actuated_joint_names = [name for name in DG5F_JOINT_NAMES if name != disabled]
        self.fingertip_body_names = list(DG5F_FINGERTIP_NAMES)
        self.action_space = len(self.actuated_joint_names)
        self.observation_space = 2 * self.action_space + 24 + 13 * len(self.fingertip_body_names) + self.action_space
        self.robot_cfg = source.robot_cfg.copy()
        cache = EXTERNAL_PROJECT_ROOT / 'assets/repose_cache'
        self.robot_cfg.spawn.asset_path = fixed_axis_urdf(URDF_PATH, cache / 'dg5f_right_fixed_axis.urdf', disabled)
        self.robot_cfg.spawn.usd_dir = str(cache)
        self.robot_cfg.spawn.usd_file_name = 'dg5f_right_fixed_axis.usd'
        self.robot_cfg.init_state.joint_pos.pop(disabled)
        actuator = self.robot_cfg.actuators['fingers']
        actuator.joint_names_expr = self.actuated_joint_names.copy()
        for attribute in ('stiffness', 'damping', 'armature', 'friction', 'effort_limit_sim', 'velocity_limit_sim'):
            values = getattr(actuator, attribute)
            if isinstance(values, dict):
                values.pop(disabled, None)
        # Keep the 60-mm, 50-g cube and its collision/material/solver properties.
        self.object_cfg = source.object_cfg.copy()
        self.object_cfg.prim_path = '/World/envs/env_.*/object'
        self.goal_object_cfg = source.goal_marker_cfg.copy()
        self.scene.num_envs = 1024
        self.scene.clone_in_fabric = False  # editable URDF collisions, same as stream
        # Capacity, not physics: DG5F's mesh contacts exceed the stock patch budget.
        self.sim.physx.gpu_max_rigid_patch_count = source.sim.physx.gpu_max_rigid_patch_count
        self.viewer = source.viewer.copy()
        # Deliberately inherit ALL upstream reward, tolerance, reset and action settings.


@configclass
class DG5FReposePPORunnerCfg(ShadowHandPPORunnerCfg):
    experiment_name = 'dg5f_repose'
