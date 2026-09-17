"""Direct DG5F in-hand orientation task, adapted from the generated Direct template.

Control/reset ordering follows Isaac Lab 2.3.2's InHandManipulationEnv.
No camera observations, teleoperation reference, or action-history filter.
"""

from collections.abc import Sequence
import logging
import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply, quat_apply_inverse, quat_conjugate, quat_error_magnitude, quat_mul, sample_uniform,
)

from dg5f_isaaclab.assets.dg5f import (
    DG5F_HARDWARE_MODEL, DG5F_JOINT_EFFORT_LIMITS, DG5F_JOINT_LIMITS, DG5F_NO_LOAD_SPEED_RPM,
    DG5F_RATED_JOINT_TORQUE_NM, DG5F_STALL_JOINT_TORQUE_NM,
)
from .dg5f_cube_env_cfg import DG5FCubeEnvCfg
from .control import ActionDelayQueue, position_targets
from .goals import sample_goal_quats
from .success import HeldSuccessTracker, hold_steps, orientation_state_reward


logger = logging.getLogger(__name__)

# Per-episode logs (1-D, possibly empty): present in EVERY step's log dict.
EPISODE_LOG_KEYS = (
    "episode_reward", "episode_held_success_rate", "episode_entered_tolerance_rate", "episode_drop_rate",
    "episode_initial_orientation_error_deg", "episode_final_orientation_error_deg",
    "episode_mean_orientation_error_deg", "episode_min_orientation_error_deg",
    "episode_time_in_tolerance_fraction", "episode_time_to_held_success_s", "episode_length_s",
)


class DG5FCubeEnv(DirectRLEnv):
    cfg: DG5FCubeEnvCfg

    def __init__(self, cfg: DG5FCubeEnvCfg, render_mode: str | None = None, **kwargs):
        sysid = cfg.resolve_control_config()
        if cfg.act_moving_average != 1.0:
            raise ValueError("The teacher uses no action smoothing: act_moving_average must be 1.0")
        if cfg.action_space != len(cfg.actuated_joint_names) - 1:
            raise ValueError("Each active joint must have exactly one policy action")
        if cfg.target_orientation_mode not in ("axis", "uniform"):
            raise ValueError("target_orientation_mode must be 'axis' or 'uniform'")
        lo, hi = cfg.target_angle_range_deg
        if not -180 <= lo < hi <= 180:
            raise ValueError("Invalid target angle range")
        if sum(value * value for value in cfg.target_axis_in_palm) <= 0:
            raise ValueError("Target rotation axis must be nonzero")
        super().__init__(cfg, render_mode, **kwargs)
        # The Direct integer space shorthand is unbounded in Isaac Lab 2.3.2.
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(cfg.action_space,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self.joint_ids, names = self.hand.find_joints(cfg.actuated_joint_names, preserve_order=True)
        self.tip_ids, tips = self.hand.find_bodies(cfg.fingertip_body_names, preserve_order=True)
        (self.palm_id,), _ = self.hand.find_bodies(cfg.palm_body_name)
        assert names == cfg.actuated_joint_names and tips == cfg.fingertip_body_names
        assert self.hand.num_joints == len(self.joint_ids) == 20
        limits = self.hand.data.joint_pos_limits[:, self.joint_ids]
        expected = torch.tensor(DG5F_JOINT_LIMITS, device=self.device)
        torch.testing.assert_close(limits[0], expected, atol=1e-5, rtol=1e-5)
        self.active_indices = [names.index(name) for name in cfg.active_action_joints]
        self.active_joint_ids, active_names = self.hand.find_joints(cfg.active_action_joints, preserve_order=True)
        assert active_names == cfg.active_action_joints
        assert self.active_joint_ids == [self.joint_ids[i] for i in self.active_indices]
        assert cfg.disabled_joint not in active_names
        self.disabled_index = names.index(cfg.disabled_joint)
        self.disabled_joint_id = self.joint_ids[self.disabled_index]
        self.lower = limits[:, self.active_indices, 0].clone()
        self.upper = limits[:, self.active_indices, 1].clone()
        # Keep 20 physical DOFs/state entries, but lock the unavailable joint at zero.
        # A fixed target alone would still permit movement under external contact.
        fixed_limits = torch.full((self.num_envs, 1, 2), cfg.disabled_joint_position, device=self.device)
        self.hand.write_joint_position_limit_to_sim(fixed_limits, joint_ids=[self.disabled_joint_id])
        for parameter, data_name in (
            ("stiffness", "joint_stiffness"), ("damping", "joint_damping"),
            ("armature", "joint_armature"), ("friction", "joint_friction_coeff"),
        ):
            actual = getattr(self.hand.data, data_name)[0, self.joint_ids]
            values = dict(sysid.parameters[parameter])
            if parameter == "armature":
                values[cfg.disabled_joint] = cfg.disabled_joint_lock_armature
            expected = torch.tensor([values[name] for name in names], device=self.device)
            torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-5)
        self.torque_limits = self.hand.data.joint_effort_limits[:, self.joint_ids].clone()
        effort_cap = math.inf if cfg.effort_limit_cap_nm is None else cfg.effort_limit_cap_nm
        expected_limits = torch.tensor(
            [min(effort_cap, DG5F_JOINT_EFFORT_LIMITS[name]) for name in names], device=self.device)
        torch.testing.assert_close(self.torque_limits[0], expected_limits)
        self._valid_torque_limits = bool(torch.all(torch.isfinite(self.torque_limits) & (self.torque_limits > 0)))

        self.palm_pos_w = self.hand.data.body_pos_w[:, self.palm_id].clone()
        self.palm_quat_w = self.hand.data.body_quat_w[:, self.palm_id].clone()
        self.cube_anchor = torch.tensor(cfg.cube_position_in_palm, device=self.device).expand(self.num_envs, -1)
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_quat[:, 0] = 1
        self.actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.applied_actions = torch.zeros_like(self.actions)
        self.action_delta = torch.zeros_like(self.actions)
        self.action_queue = ActionDelayQueue(self.num_envs, cfg.action_delay_steps, self.device)
        grasp_command = torch.tensor([cfg.grasp_command[name] for name in names], device=self.device)
        self.grasp_command = grasp_command.expand(self.num_envs, -1).clone()
        self.joint_targets = self.grasp_command.clone()
        # Integrated command state q_cmd for the 19 active joints (observed by the policy).
        self.joint_command = self.grasp_command[:, self.active_indices].clone()
        self.previous_joint_command = self.joint_command.clone()
        self.measured_at_command = self.hand.data.joint_pos[:, self.active_joint_ids].clone()
        self.previous_orientation_error = torch.zeros(self.num_envs, device=self.device)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        required = hold_steps(cfg.success_hold_time_s, self.step_dt)
        assert required == cfg.success_hold_steps, (required, cfg.success_hold_steps)
        self.success = HeldSuccessTracker(self.num_envs, required, self.device)
        self.success_eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.error_sum = torch.zeros(self.num_envs, device=self.device)
        self.error_min = torch.full((self.num_envs,), math.inf, device=self.device)
        self.initial_orientation_error = torch.zeros(self.num_envs, device=self.device)
        # A success only counts once a policy action has actually been delivered (FIFO delay).
        self.success_start_step = max(cfg.action_delay_steps) + 1
        self.velocity_watch = [names.index(n) for n in cfg.velocity_watch_joints]
        self.saturation_watch = [names.index(n) for n in cfg.saturation_watch_joints]
        self.raw_action_clip_fraction = torch.zeros((), device=self.device)
        self._compute_state()
        print(f"[DG5F] joints={names}")
        print(f"[DG5F] active_action_joints={active_names}")
        print(f"[DG5F] fingertips={tips}; disabled_joint={cfg.disabled_joint} fixed={cfg.disabled_joint_position} rad"
              f" lock=limits[0,0]+armature {cfg.disabled_joint_lock_armature}")
        print(f"[DG5F] action_dim={cfg.action_space} observation_dim={cfg.observation_space}"
              f" control_mode={cfg.control_mode} delta_action_scale_deg={math.degrees(cfg.delta_action_scale):g}"
              f" delay_steps={cfg.delay_control_steps_by_finger}")
        print(f"[DG5F] grasp_state_deg={cfg.grasp_joint_pos_deg}")
        print(f"[DG5F] grasp_command_offset_deg={cfg.grasp_command_offset_deg}")
        print(f"[DG5F] reward_v2 state_scale={cfg.orientation_state_scale} sigma_deg={cfg.orientation_sigma_deg}"
              f" progress_scale={cfg.orientation_progress_scale} success_tol_deg="
              f"{math.degrees(cfg.success_tolerance_rad):g} hold={cfg.success_hold_time_s}s={required} steps"
              f" (control_dt={self.step_dt:.5f}s) bonus={cfg.success_bonus}")
        print(f"[DG5F] cube_size_m={cfg.cube_size_m} cube_mass_kg={cfg.object_cfg.spawn.mass_props.mass}")
        print(f"[DG5F] SysID={sysid.source} sha256={sysid.sha256}; applied stiffness/damping/armature/friction")
        print(f"[DG5F] SysID fit_info_by_finger={sysid.fit_info_by_finger}")
        velocity_limits = self.hand.data.joint_vel_limits[0, self.joint_ids]
        print(f"[DG5F] hardware={DG5F_HARDWARE_MODEL} rated_torque_nm={DG5F_RATED_JOINT_TORQUE_NM}"
              f" stall_torque_nm={DG5F_STALL_JOINT_TORQUE_NM} (metadata only)"
              f" no_load_speed={DG5F_NO_LOAD_SPEED_RPM} rpm={DG5F_NO_LOAD_SPEED_RPM * math.tau / 60:.4f} rad/s")
        print(f"[DG5F] effort_limit_sim_nm={self.torque_limits[0].tolist()}"
              f" (effort_limit_cap_nm={cfg.effort_limit_cap_nm})")
        print(f"[DG5F] velocity_limit_sim_rad_s(readback)={velocity_limits.tolist()}")
        logger.warning("ImplicitActuator computed_torque/applied_torque and saturation are PD estimates,"
                       " not measured PhysX joint torques or real DG5F saturation.")
        logger.warning("SysID JSON does not specify friction units/model; friction is applied through"
                       " ImplicitActuatorCfg.friction (static joint friction effort in Isaac Sim 5.1)."
                       " Dynamic/viscous friction retain the existing asset defaults.")
        if not self._valid_torque_limits:
            logger.warning("Invalid torque limits: torque_saturation_fraction will be -1 (unavailable).")

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["cube"] = self.cube
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor):
        assert actions.shape == (self.num_envs, self.cfg.action_space)
        # Fresh dict every step: RSL-RL stores a reference per step and averages them.
        self.extras["log"] = {key: torch.empty(0, device=self.device) for key in EPISODE_LOG_KEYS}
        self.raw_action_clip_fraction = (actions.abs() > 1.0).float().mean()
        self.actions = actions.clamp(-1.0, 1.0)
        self.previous_actions.copy_(self.action_queue.history[:, 0])
        self.action_delta = self.actions - self.previous_actions
        # One FIFO push per CONTROL step, not per decimated physics substep.
        self.applied_actions = self.action_queue.push(self.actions)
        self.measured_at_command = self.hand.data.joint_pos[:, self.active_joint_ids].clone()
        self.previous_joint_command.copy_(self.joint_command)
        # Delay acts on the delta BEFORE integration: FIFO -> delivered delta -> q_cmd.
        self.joint_command = position_targets(
            self.applied_actions, self.measured_at_command, self.joint_command, self.lower, self.upper,
            self.grasp_command[:, self.active_indices], self.cfg.control_mode, self.cfg.delta_action_scale,
        )
        self.joint_targets[:, self.active_indices] = self.joint_command
        self.joint_targets[:, self.disabled_index] = self.cfg.disabled_joint_position

    def _apply_action(self):
        self.hand.set_joint_position_target(self.joint_targets, joint_ids=self.joint_ids)

    def _compute_state(self):
        self.cube_pos = quat_apply_inverse(self.palm_quat_w, self.cube.data.root_pos_w - self.palm_pos_w)
        self.cube_quat = quat_mul(quat_conjugate(self.palm_quat_w), self.cube.data.root_quat_w)
        self.orientation_error = quat_error_magnitude(self.cube_quat, self.goal_quat)
        self.cube_distance_from_palm = torch.linalg.vector_norm(self.cube_pos, dim=-1)
        self.palm_region_distance = torch.linalg.vector_norm(self.cube_pos - self.cube_anchor, dim=-1)

    def _get_observations(self):
        self._compute_state()
        tip_positions = self.hand.data.body_pos_w[:, self.tip_ids] - self.palm_pos_w[:, None, :]
        tip_positions = quat_apply_inverse(
            self.palm_quat_w[:, None, :].expand(-1, 5, -1).reshape(-1, 4),
            tip_positions.reshape(-1, 3),
        ).reshape(self.num_envs, 15)
        obs = torch.cat((
            self.hand.data.joint_pos[:, self.joint_ids],
            self.hand.data.joint_vel[:, self.joint_ids],
            self.cube_pos, self.cube_quat,
            quat_apply_inverse(self.palm_quat_w, self.cube.data.root_lin_vel_w),
            quat_apply_inverse(self.palm_quat_w, self.cube.data.root_ang_vel_w),
            self.goal_quat, tip_positions, self.action_queue.history.flatten(start_dim=1),
            self.joint_command,
        ), dim=-1)
        assert obs.shape == (self.num_envs, self.cfg.observation_space), obs.shape
        return {"policy": obs}

    def _get_dones(self):
        self._compute_state()
        fallen = (self.cube_distance_from_palm > self.cfg.max_cube_distance_from_palm_m) | (
            self.cube.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2] < self.cfg.min_cube_world_height_m
        )
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, timeout

    def _get_rewards(self):
        started = self.episode_length_buf >= self.success_start_step
        in_tolerance = (self.orientation_error <= self.cfg.success_tolerance_rad) & ~self.reset_terminated
        in_tolerance &= started & self.success_eligible
        first_success = self.success.update(in_tolerance, self.episode_length_buf)
        self.error_sum += self.orientation_error
        self.error_min = torch.minimum(self.error_min, self.orientation_error)
        terms = {
            "orientation_state_reward": orientation_state_reward(
                self.orientation_error, self.cfg.orientation_state_scale, math.radians(self.cfg.orientation_sigma_deg)),
            "orientation_progress_reward": self.cfg.orientation_progress_scale * (
                self.previous_orientation_error - self.orientation_error),
            "palm_distance_penalty": -self.cfg.palm_region_distance_scale * self.palm_region_distance,
            "action_penalty": -self.cfg.action_penalty_scale * self.actions.square().mean(dim=-1),
            "action_rate_penalty": -self.cfg.action_rate_penalty_scale * self.action_delta.square().mean(dim=-1),
            "success_bonus": self.cfg.success_bonus * first_success.float(),
            "drop_penalty": -self.cfg.fall_penalty * self.reset_terminated.float(),
        }
        reward = sum(terms.values())
        self.previous_orientation_error.copy_(self.orientation_error)
        self.episode_returns += reward
        computed = self.hand.data.computed_torque[:, self.joint_ids].detach()
        applied = self.hand.data.applied_torque[:, self.joint_ids].detach()
        # Full signed tensors are available to debug tools; scalar means go to PPO.
        self.extras["computed_torque"] = computed
        self.extras["applied_torque"] = applied
        log = self.extras.setdefault("log", {})
        log.update({name: value.mean() for name, value in terms.items()})
        log.update({f"abs_contribution/{name}": value.abs().mean() for name, value in terms.items()})
        joint_vel = self.hand.data.joint_vel[:, self.joint_ids]
        velocity_rms = joint_vel.square().mean(dim=0).sqrt()
        command_step = (self.joint_command - self.previous_joint_command).abs()
        log.update({
            "total_reward": reward.mean(),
            "orientation_error": self.orientation_error.mean(),
            "orientation_error_deg": torch.rad2deg(self.orientation_error).mean(),
            "cube_distance_from_palm": self.cube_distance_from_palm.mean(),
            "held_success_rate": self.success.succeeded.float().mean(),
            "in_tolerance_fraction": in_tolerance.float().mean(),
            "consecutive_tolerance_steps": self.success.consecutive.float().mean(),
            "current_orientation_error_deg": torch.rad2deg(self.orientation_error).mean(),
            "mean_action_magnitude": self.actions.abs().mean(),
            "mean_action_abs": self.actions.abs().mean(),
            "mean_action_delta_abs": self.action_delta.abs().mean(),
            "mean_abs_applied_torque": applied.abs().mean(),
            "max_abs_applied_torque": applied.abs().max(),
            "max_abs_joint_velocity": joint_vel.abs().max(),
            "mean_abs_joint_velocity": joint_vel.abs().mean(),
            "mean_abs_command_delta_deg": torch.rad2deg(command_step).mean(),
            "action_near_limit_fraction": (self.actions.abs() >= self.cfg.action_near_limit).float().mean(),
            "raw_action_clip_fraction": self.raw_action_clip_fraction,
            "disabled_joint_abs_deviation_deg": torch.rad2deg(
                (self.hand.data.joint_pos[:, self.disabled_joint_id] - self.cfg.disabled_joint_position).abs()).max(),
        })
        for index, name in enumerate(self.cfg.actuated_joint_names):
            log[f"velocity_rms/{name}"] = velocity_rms[index]
        watched = velocity_rms[self.velocity_watch]
        log["velocity_rms_low_damping_joints"] = watched.mean()
        log["velocity_rms_other_joints"] = velocity_rms[
            [i for i in range(len(velocity_rms)) if i not in self.velocity_watch]].mean()
        # A PD estimate at/above the limit means the implicit drive is clipped.
        saturated = (computed.abs() >= self.torque_limits * (1 - 1e-6)).float()
        if not self._valid_torque_limits:
            saturated = torch.full_like(saturated, -1.0)
        log["torque_saturation_fraction"] = saturated.mean()
        log["torque_saturation_fraction_watch_joints"] = saturated[:, self.saturation_watch].mean()
        for index, name in enumerate(self.cfg.actuated_joint_names):
            log[f"computed_torque/{name}"] = computed[:, index].mean()
            log[f"applied_torque/{name}"] = applied[:, index].mean()
            log[f"abs_applied_torque/{name}"] = applied[:, index].abs().mean()
            log[f"max_abs_applied_torque/{name}"] = applied[:, index].abs().max()
            log[f"torque_saturation_fraction/{name}"] = saturated[:, index].mean()
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        completed = env_ids[self.episode_length_buf[env_ids] > 0]
        # Per-episode values (possibly empty) under every key, every step: RSL-RL takes its keys
        # from the first step of an iteration and concatenates, i.e. an episode-weighted mean.
        success = self.success.succeeded[completed]
        steps = self.episode_length_buf[completed].float()
        episode_log = {
            "episode_reward": self.episode_returns[completed],
            "episode_held_success_rate": success.float(),
            "episode_entered_tolerance_rate": self.success.entered[completed].float(),
            "episode_drop_rate": self.reset_terminated[completed].float(),
            "episode_initial_orientation_error_deg": torch.rad2deg(self.initial_orientation_error[completed]),
            "episode_final_orientation_error_deg": torch.rad2deg(self.orientation_error[completed]),
            "episode_mean_orientation_error_deg": torch.rad2deg(self.error_sum[completed] / steps),
            "episode_min_orientation_error_deg": torch.rad2deg(self.error_min[completed]),
            "episode_time_in_tolerance_fraction": self.success.steps_in_tolerance[completed].float() / steps,
            "episode_time_to_held_success_s": self.success.success_step[completed][success].float() * self.step_dt,
            "episode_length_s": steps * self.step_dt,
        }
        assert set(episode_log) == set(EPISODE_LOG_KEYS)
        self.extras.setdefault("log", {}).update(episode_log)
        super()._reset_idx(env_ids)
        count = len(env_ids)
        joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        noise = self.cfg.reset_joint_position_noise_rad
        joint_pos += sample_uniform(-noise, noise, joint_pos.shape, device=self.device)
        native_limits = self.hand.data.joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, native_limits[..., 0], native_limits[..., 1])
        joint_pos[:, self.disabled_joint_id] = self.cfg.disabled_joint_position
        joint_vel = torch.zeros_like(joint_pos)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # The command restarts at the (noise-free) grasp target, including its preload offset.
        self.joint_targets[env_ids] = self.grasp_command[env_ids]
        self.hand.set_joint_position_target(self.joint_targets[env_ids], joint_ids=self.joint_ids, env_ids=env_ids)
        self.joint_command[env_ids] = self.grasp_command[env_ids][:, self.active_indices]
        self.previous_joint_command[env_ids] = self.joint_command[env_ids]
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.applied_actions[env_ids] = 0
        self.action_delta[env_ids] = 0
        self.action_queue.reset(env_ids)
        self.measured_at_command[env_ids] = joint_pos[:, self.active_joint_ids]

        cube_state = self.cube.data.default_root_state[env_ids].clone()
        noise = self.cfg.cube_reset_position_noise_m
        local_pos = self.cube_anchor[env_ids] + sample_uniform(-noise, noise, (count, 3), device=self.device)
        cube_state[:, :3] = self.palm_pos_w[env_ids] + quat_apply(self.palm_quat_w[env_ids], local_pos)
        cube_state[:, 3:7] = self.palm_quat_w[env_ids]
        cube_state[:, 7:] = 0
        self.cube.write_root_pose_to_sim(cube_state[:, :7], env_ids=env_ids)
        self.cube.write_root_velocity_to_sim(cube_state[:, 7:], env_ids=env_ids)

        # The cube restarts aligned with the palm: identity in the palm frame.
        initial_quat = torch.zeros((count, 4), device=self.device)
        initial_quat[:, 0] = 1
        self.goal_quat[env_ids] = sample_goal_quats(
            initial_quat, self.cfg.target_orientation_mode, self.cfg.target_axis_in_palm,
            self.cfg.target_angle_range_deg, math.radians(self.cfg.min_initial_goal_error_deg),
        )
        initial_error = quat_error_magnitude(initial_quat, self.goal_quat[env_ids])
        assert torch.all(initial_error >= math.radians(self.cfg.min_initial_goal_error_deg) - 1e-4)
        self.previous_orientation_error[env_ids] = initial_error
        self.initial_orientation_error[env_ids] = initial_error
        # Defensive: never a free bonus for a goal already within tolerance.
        self.success_eligible[env_ids] = initial_error >= self.cfg.success_tolerance_rad
        self.success.reset(env_ids)
        self.error_sum[env_ids] = 0
        self.error_min[env_ids] = math.inf
        self.episode_returns[env_ids] = 0
