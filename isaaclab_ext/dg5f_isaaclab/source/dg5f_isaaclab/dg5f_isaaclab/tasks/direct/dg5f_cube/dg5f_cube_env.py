"""Direct DG5F in-hand orientation task, adapted from the generated Direct template.

Control/reset ordering follows Isaac Lab 2.3.2's InHandManipulationEnv.
No camera observations, teleoperation reference, or action-history filter.
"""

from collections.abc import Sequence
import logging
import math

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply, quat_apply_inverse, quat_conjugate, quat_error_magnitude, quat_mul, sample_uniform,
)

from dg5f_isaaclab.assets.grasp_cache import load_grasp_cache
from dg5f_isaaclab.assets.dg5f import (
    DG5F_HARDWARE_MODEL, DG5F_JOINT_EFFORT_LIMITS, DG5F_JOINT_LIMITS, DG5F_NO_LOAD_SPEED_RPM,
    DG5F_RATED_JOINT_TORQUE_NM, DG5F_STALL_JOINT_TORQUE_NM,
)
from .dg5f_cube_env_cfg import DG5FCubeEnvCfg
from .control import ActionDelayQueue, position_targets
from .goals import (
    axis_angle_quat_axes, axis_velocity_reward, joint_limit_pressure, random_axes, relocation_events,
    rotation_vector,
    sample_curriculum_goals, sample_fixed_angle_goals, sample_goal_quats,
)
from .grasp import GraspQualityReward
from .success import (
    HeldSuccessTracker, hold_steps, orientation_baseline_reward, orientation_reward_table,
    orientation_state_reward,
)


logger = logging.getLogger(__name__)


# Per-episode logs (1-D, possibly empty): present in EVERY step's log dict.
EPISODE_LOG_KEYS = (
    "episode_reward", "episode_held_success_rate", "episode_entered_tolerance_rate", "episode_drop_rate",
    "episode_initial_orientation_error_deg", "episode_final_orientation_error_deg",
    "episode_mean_orientation_error_deg", "episode_min_orientation_error_deg",
    "episode_time_in_tolerance_fraction", "episode_time_to_held_success_s", "episode_length_s",
    "episode_goals_completed", "episode_mean_tip_contacts",
    # Stream metrics: with a chain of goals, "success" alone no longer describes the behaviour.
    "episode_goals_issued", "episode_target_completion_rate", "episode_time_per_completed_goal_s",
    "episode_first_goal_held_success_rate", "episode_commanded_rotation_deg",
    # Net SIGNED rotation about target_axis_in_palm, integrated from the cube's angular velocity.
    # commanded_rotation counts goals, which a reversible (+/-) stream completes by rocking.
    "episode_axis_rotation_palm_deg", "episode_axis_rotation_body_deg",
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
        needs_contacts = (cfg.gait_task_gate or cfg.gait_observation_mode != "none"
                          or cfg.axis_velocity_reward_scale or cfg.relocation_bonus
                          or cfg.tip_contact_reward or cfg.palm_contact_penalty or cfg.grasp_quality_scale
                          or cfg.grasp_deficit_scale or cfg.grasp_quality_deficit_scale
                          or cfg.max_time_without_tip_contact_s is not None)
        if needs_contacts and not cfg.enable_contact_sensors:
            raise ValueError("Contact-based reward/termination terms require enable_contact_sensors")
        if cfg.tip_collider_approximation not in ("convexHull", "convexDecomposition"):
            raise ValueError("tip_collider_approximation must be 'convexHull' or 'convexDecomposition'")
        # Read by assets/dg5f.py:spawn_dg5f when the articulation spawns in _setup_scene.
        cfg.robot_cfg.spawn.tip_collider_approximation = cfg.tip_collider_approximation
        if cfg.enable_contact_sensors:
            # Must be set before the articulation spawns in _setup_scene.
            cfg.robot_cfg.spawn.activate_contact_sensors = True
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
        # clone(): an expand() view shares one row of memory, so per-env anchors (grasp cache,
        # grasp searches) cannot be written into it.
        self.cube_anchor = torch.tensor(
            cfg.cube_position_in_palm, device=self.device).expand(self.num_envs, -1).clone()
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_quat[:, 0] = 1
        # Optional per-env reset orientation of the cube (palm frame), set by grasp searches the
        # same way they set cube_anchor. None keeps the configured reset.
        self.cube_reset_quat: torch.Tensor | None = None
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
        # Goals completed within the current episode (the consecutive-successes metric), and goals
        # HANDED OUT, whose ratio is the completion rate of the stream.
        self.goals_completed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goals_issued = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Always defined so the log schema does not depend on the contact sensors being on.
        self.tip_contact_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tip_in_contact = torch.zeros((self.num_envs, len(self.tip_ids)), dtype=torch.bool, device=self.device)
        self.gait_timers = torch.zeros((self.num_envs, 5, 3), device=self.device)
        self.gait_previous_contacts = self.tip_in_contact.clone()
        self.palm_contact_force = torch.zeros(self.num_envs, device=self.device)
        self.tip_contact_sum = torch.zeros(self.num_envs, device=self.device)
        self.steps_without_tip_contact = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.grasp_quality = GraspQualityReward(
            cfg.grasp_quality_margin, cfg.grasp_quality_clip, cfg.grasp_quality_ema)
        # A plain float: goals are resampled on every completion, and reading a device scalar
        # there would force a host sync on most steps.
        self.goal_angle_limit_rad = math.radians(cfg.goal_curriculum_start_deg)
        self.goal_at_frontier = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.frontier_attempts = torch.zeros((), device=self.device)
        self.frontier_successes = torch.zeros((), device=self.device)
        if cfg.goal_curriculum:
            if not cfg.goal_curriculum_start_deg >= cfg.min_initial_goal_error_deg:
                raise ValueError("goal_curriculum_start_deg must be at least min_initial_goal_error_deg")
            if not 0 < cfg.goal_frontier_fraction + cfg.goal_inside_fraction <= 1:
                raise ValueError("goal frontier/inside fractions must sum into (0, 1]")
        self.grasp_cache = None
        # Entries [0, primary_grasp_count) come from grasp_cache_path, the rest from the optional
        # mix cache: one tensor set, so _pick_grasps only has to choose which range to draw from.
        self.primary_grasp_count = 0
        if cfg.grasp_cache_path is not None:
            caches = [load_grasp_cache(cfg.grasp_cache_path, cfg.actuated_joint_names, DG5F_JOINT_LIMITS)]
            self.primary_grasp_count = len(caches[0])
            if cfg.mix_grasp_cache_path is not None and cfg.mix_cache_fraction > 0:
                caches.append(load_grasp_cache(
                    cfg.mix_grasp_cache_path, cfg.actuated_joint_names, DG5F_JOINT_LIMITS))
            self.grasp_cache = {
                key: torch.as_tensor(np.concatenate([getattr(c, attribute) for c in caches]),
                                     device=self.device)
                for key, attribute in (("joint_pos", "joint_pos"), ("joint_command", "joint_command"),
                                       ("cube_pos", "cube_pos"), ("cube_quat", "cube_quat"))
            }
            for cache in caches:
                print(f"[DG5F] grasp_cache n={len(cache)} sha256={cache.sha256[:16]} source={cache.source}")
            if len(caches) > 1:
                print(f"[DG5F] reset draw: {1 - cfg.mix_cache_fraction:.0%} primary"
                      f" ({self.primary_grasp_count}) / {cfg.mix_cache_fraction:.0%} mix"
                      f" ({len(caches[1])})")
        self.error_sum = torch.zeros(self.num_envs, device=self.device)
        self.gait_cache = None
        self.gait_cache_sha256 = None
        self.gait_last_picks = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        if not 0 <= cfg.gait_transition_fraction <= 1:
            raise ValueError("gait_transition_fraction must be in [0,1]")
        if cfg.gait_transition_fraction > 0:
            if not cfg.gait_transition_cache_path:
                raise ValueError("Transition resets require an explicit validated cache")
            from dg5f_isaaclab.assets.gait_cache import load_gait_cache, DYNAMIC_FIELDS
            arrays, self.gait_cache_sha256 = load_gait_cache(
                cfg.gait_transition_cache_path, cfg.actuated_joint_names, self.action_queue.history_steps)
            self.gait_cache = {k: torch.as_tensor(arrays[k], device=self.device) for k in DYNAMIC_FIELDS}
            weights = torch.as_tensor(cfg.gait_phase_weights, device=self.device)
            if weights.shape != (7,) or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("Invalid gait phase distribution")
            phases = torch.as_tensor(arrays["phase"], device=self.device)
            counts = torch.bincount(phases, minlength=7)
            if ((weights > 0) & (counts == 0)).any():
                raise ValueError("Configured transition phase absent from validated cache; explicitly revise phase weights")
            self.gait_sample_weights = weights[phases] / counts[phases]
            print(f"[GAIT_CACHE] sha256={self.gait_cache_sha256} n={len(phases)} fraction={cfg.gait_transition_fraction}"
                  f" phase_weights={weights.tolist()} phase_counts={counts.tolist()}")
        self.error_min = torch.full((self.num_envs,), math.inf, device=self.device)
        axis = torch.tensor(cfg.target_axis_in_palm, dtype=torch.float32, device=self.device)
        self.rotation_axis = axis / axis.norm()
        self.axis_angular_velocity_palm = torch.zeros(self.num_envs, device=self.device)
        self.axis_rotation_palm = torch.zeros(self.num_envs, device=self.device)
        self.axis_rotation_body = torch.zeros(self.num_envs, device=self.device)
        # Orientation at the previous control step. A freshly reset env has none yet (a transition
        # reset may rewrite the pose after _reset_idx), so its first step only re-seeds it.
        self.previous_cube_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.previous_cube_quat[:, 0] = 1
        self.rotation_tracking_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.relocation_previous_contact = torch.zeros((self.num_envs, 5), dtype=torch.bool, device=self.device)
        self.relocation_release_position = torch.zeros((self.num_envs, 5, 3), device=self.device)
        self.relocation_release_steps = torch.zeros((self.num_envs, 5), dtype=torch.long, device=self.device)
        self.relocation_count = torch.zeros(self.num_envs, device=self.device)
        # Own flag: rotation_tracking_fresh is already cleared by the time the relocation term runs.
        self.relocation_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.initial_orientation_error = torch.zeros(self.num_envs, device=self.device)
        # A success only counts once a policy action has actually been delivered (FIFO delay).
        self.success_start_step = max(cfg.action_delay_steps) + 1
        self.velocity_watch = [names.index(n) for n in cfg.velocity_watch_joints]
        self.saturation_watch = [names.index(n) for n in cfg.saturation_watch_joints]
        self.raw_action_clip_fraction = torch.zeros((), device=self.device)
        self.reward_terms: dict[str, torch.Tensor] = {}
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
        if cfg.goal_stream:
            print(f"[DG5F] goal_stream angle={cfg.goal_stream_angle_deg} deg stage={cfg.goal_stream_stage}"
                  f" dwell={cfg.goal_dwell_reward}/step x {cfg.success_hold_steps} steps"
                  f" = {cfg.goal_dwell_reward * cfg.success_hold_steps:.2f} vs bonus {cfg.success_bonus}"
                  f"; progress scale={cfg.orientation_progress_scale} clip={cfg.orientation_progress_clip_rad} rad"
                  f" ({math.degrees(cfg.orientation_progress_clip_rad):.2f} deg/step)")
            print(f"[DG5F] rails: pose {cfg.pose_penalty_scale} beyond {cfg.pose_penalty_deadband_deg} deg,"
                  f" anchor {cfg.palm_region_distance_scale} beyond {1000 * cfg.palm_region_deadband_m:.0f} mm,"
                  f" near-goal omega {cfg.goal_velocity_scale} above {cfg.goal_velocity_threshold_rad_s} rad/s"
                  f" inside {cfg.goal_velocity_error_deg} deg")
        if cfg.orientation_baseline:
            table = orientation_reward_table(
                cfg.orientation_state_scale, math.radians(cfg.orientation_decay_deg),
                math.radians(cfg.goal_stream_angle_deg))
            # Printed, not assumed: the sign change at the stream angle is the whole point of the term.
            print("[DG5F] orientation_state_reward(err): "
                  + "  ".join(f"{deg:g}deg={value:+.4f}" for deg, value in table))
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
        self.contact_sensors = {}
        if self.cfg.enable_contact_sensors:
            # Six cube-filtered sensors (5 tips + palm), not one per link as in contact_audit.py:
            # the reward only needs "is a tip on the cube" and "is the palm carrying it".
            for link in list(self.cfg.fingertip_body_names) + [self.cfg.palm_body_name]:
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{link}",
                    track_contact_points=self.cfg.track_gait_contact_points,
                    filter_prim_paths_expr=["/World/envs/env_.*/Cube"]))
                self.scene.sensors[f"contact_{link}"] = sensor
                self.contact_sensors[link] = sensor
        # Created here, not after super().__init__(): __init__ ends with _compute_state(), which
        # already draws them. /Visuals is outside /World/envs, so clone_environments ignores it.
        self.goal_markers = None
        self.status_markers = None
        if self.cfg.goal_marker:
            self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self.status_markers = VisualizationMarkers(self.cfg.goal_marker_status_cfg)

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
        tips = self.hand.data.body_pos_w[:, self.tip_ids] - self.palm_pos_w[:, None, :]
        self.tip_pos_palm = quat_apply_inverse(
            self.palm_quat_w[:, None, :].expand(-1, len(self.tip_ids), -1).reshape(-1, 4),
            tips.reshape(-1, 3),
        ).view(self.num_envs, len(self.tip_ids), 3)
        if self.contact_sensors:
            forces = torch.stack(
                [s.data.force_matrix_w[:, 0, 0].norm(dim=-1) for s in self.contact_sensors.values()], dim=-1)
            self.tip_in_contact = forces[:, :len(self.tip_ids)] > self.cfg.tip_contact_force_n
            self.tip_contact_count = self.tip_in_contact.sum(dim=-1)
            self.palm_contact_force = forces[:, len(self.tip_ids)]

    def _get_observations(self):
        self._compute_state()
        # Palm-frame fingertip positions, computed once in _compute_state and shared with the
        # grasp-quality reward (which runs before observations in the DirectRLEnv step order).
        tip_positions = self.tip_pos_palm.reshape(self.num_envs, -1)
        obs = torch.cat((
            self.hand.data.joint_pos[:, self.joint_ids],
            self.hand.data.joint_vel[:, self.joint_ids],
            self.cube_pos, self.cube_quat,
            quat_apply_inverse(self.palm_quat_w, self.cube.data.root_lin_vel_w),
            quat_apply_inverse(self.palm_quat_w, self.cube.data.root_ang_vel_w),
            self.goal_quat, tip_positions, self.action_queue.history.flatten(start_dim=1),
            self.joint_command,
        ), dim=-1)
        if self.cfg.gait_observation_mode != "none":
            from .gait_features import observation_features
            features = observation_features(self.cfg.gait_observation_mode, self.tip_in_contact,
                self.tip_pos_palm - self.cube_pos[:, None, :], self.gait_timers)
            obs = torch.cat((obs, features), dim=-1)
        assert obs.shape == (self.num_envs, self.cfg.observation_space), obs.shape
        if self.goal_markers is not None:
            self._visualize_goals()
        return {"policy": obs}

    def _visualize_goals(self):
        """GUI only: the goal orientation as a ghost cube above the hand, plus an error lamp.

        Called from _get_observations so it runs once per control step on the freshly computed
        state, after resets have already resampled the goals.
        """
        offset = torch.tensor(self.cfg.goal_marker_offset_in_palm, device=self.device).expand(self.num_envs, 3)
        ghost_pos = self.palm_pos_w + quat_apply(self.palm_quat_w, offset)
        # goal_quat is a palm-frame orientation; the marker is placed in world coordinates.
        self.goal_markers.visualize(ghost_pos, quat_mul(self.palm_quat_w, self.goal_quat))
        # Lamp clear of the ghost along the same palm normal.
        lamp_offset = offset.clone()
        lamp_offset[:, 0] += self.cfg.cube_size_m
        index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        index[self.orientation_error <= self.cfg.goal_marker_near_scale * self.cfg.success_tolerance_rad] = 1
        index[self.orientation_error <= self.cfg.success_tolerance_rad] = 2
        self.status_markers.visualize(
            self.palm_pos_w + quat_apply(self.palm_quat_w, lamp_offset), marker_indices=index)

    def _get_dones(self):
        self._compute_state()
        far = self.cube_distance_from_palm > self.cfg.max_cube_distance_from_palm_m
        low = self.cube.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2] < self.cfg.min_cube_world_height_m
        fallen = far | low
        # Which criterion fires is the first thing to know when everything terminates at once.
        self.termination_reasons = {"term/far_from_palm": far, "term/below_height": low,
                                   "term/left_grasp_region": torch.zeros_like(far),
                                   "term/lost_tip_contact": torch.zeros_like(far)}
        # A fingertip grasp can genuinely be dropped; the 0.25 m cradle threshold never fired, which
        # is why every policy so far logged drop_rate 0.000.
        if self.cfg.max_cube_distance_from_grasp_m is not None:
            left = self.palm_region_distance > self.cfg.max_cube_distance_from_grasp_m
            self.termination_reasons["term/left_grasp_region"] = left
            fallen |= left
        if self.cfg.max_time_without_tip_contact_s is not None:
            self.steps_without_tip_contact = torch.where(
                self.tip_contact_count > 0,
                torch.zeros_like(self.steps_without_tip_contact),
                self.steps_without_tip_contact + 1)
            limit = round(self.cfg.max_time_without_tip_contact_s / self.step_dt)
            lost = self.steps_without_tip_contact > limit
            self.termination_reasons["term/lost_tip_contact"] = lost
            fallen |= lost
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, timeout

    def _get_rewards(self):
        started = self.episode_length_buf >= self.success_start_step
        in_tolerance = (self.orientation_error <= self.cfg.success_tolerance_rad) & ~self.reset_terminated
        in_tolerance &= started & self.success_eligible
        if self.cfg.gait_task_gate:
            in_tolerance &= self.tip_contact_count >= 3
        if self.cfg.gait_observation_mode == "timers":
            from .gait_features import update_timers
            self.gait_timers = update_timers(self.gait_timers, self.gait_previous_contacts, self.tip_in_contact, self.step_dt)
            self.gait_previous_contacts.copy_(self.tip_in_contact)
        first_success = self.success.update(in_tolerance, self.episode_length_buf)
        self.error_sum += self.orientation_error
        self.error_min = torch.minimum(self.error_min, self.orientation_error)
        self.tip_contact_sum += self.tip_contact_count.float()
        # Rotation about the target axis from the FINITE DIFFERENCE of orientation between control
        # steps (as HORA computes its rotation reward), not from the sampled angular velocity: the
        # bang-bang policies vibrate the cube, and one velocity sample per control step aliases that
        # vibration into a drift (measured: 4.4 completed +20 deg goals read as -70 deg). The step
        # increments are far below pi, so their sum is the exact net rotation, full turns included.
        # Hand-centric (palm axis, as Khandate rewards it) and object-centric (body axis, as the
        # goal stream applies its deltas: current * delta).
        step_palm = rotation_vector(quat_mul(self.cube_quat, quat_conjugate(self.previous_cube_quat)))
        step_body = rotation_vector(quat_mul(quat_conjugate(self.previous_cube_quat), self.cube_quat))
        fresh = self.rotation_tracking_fresh
        step_palm[fresh] = 0
        step_body[fresh] = 0
        self.axis_angular_velocity_palm = (step_palm * self.rotation_axis).sum(dim=-1) / self.step_dt
        self.axis_rotation_palm += (step_palm * self.rotation_axis).sum(dim=-1)
        self.axis_rotation_body += (step_body * self.rotation_axis).sum(dim=-1)
        self.previous_cube_quat.copy_(self.cube_quat)
        self.rotation_tracking_fresh[:] = False
        # Read before the terms: the effort penalties below use them (reward v2 penalised actions,
        # not effort, which is what the 0.4 Nm rating actually constrains).
        computed = self.hand.data.computed_torque[:, self.joint_ids].detach()
        applied = self.hand.data.applied_torque[:, self.joint_ids].detach()
        joint_velocity = self.hand.data.joint_vel[:, self.joint_ids].detach()
        if self.cfg.orientation_baseline:
            state_term = orientation_baseline_reward(
                self.orientation_error, self.cfg.orientation_state_scale,
                math.radians(self.cfg.orientation_decay_deg), math.radians(self.cfg.goal_stream_angle_deg))
        else:
            state_term = orientation_state_reward(
                self.orientation_error, self.cfg.orientation_state_scale,
                math.radians(self.cfg.orientation_sigma_deg))
        raw_progress = self.previous_orientation_error - self.orientation_error
        if self.cfg.orientation_progress_clip_rad:
            # A contact impulse or an overshoot must not pay a large one-step reward; the direction
            # of motion is the information wanted here, not its magnitude.
            progress = raw_progress.clamp(-self.cfg.orientation_progress_clip_rad,
                                          self.cfg.orientation_progress_clip_rad)
        else:
            progress = raw_progress
        # Dead zone: the first pose_penalty_deadband_deg of finger rearrangement is free, so the
        # term is a rail against integrated-delta drift rather than a pull back to the exact
        # cached grasp. Rotating the cube REQUIRES leaving that pose.
        pose_deviation = (self.joint_command - self.grasp_command[:, self.active_indices]).abs().mean(dim=-1)
        pose_excess = (pose_deviation - math.radians(self.cfg.pose_penalty_deadband_deg)).clamp_min(0.0)
        anchor_excess = (self.palm_region_distance - self.cfg.palm_region_deadband_m).clamp_min(0.0)
        terms = {
            "orientation_state_reward": state_term,
            "orientation_progress_reward": self.cfg.orientation_progress_scale * progress,
            "palm_distance_penalty": -self.cfg.palm_region_distance_scale * anchor_excess,
            "action_penalty": -self.cfg.action_penalty_scale * self.actions.square().mean(dim=-1),
            "action_rate_penalty": -self.cfg.action_rate_penalty_scale * self.action_delta.square().mean(dim=-1),
            "success_bonus": self.cfg.success_bonus * first_success.float(),
            "drop_penalty": -self.cfg.fall_penalty * self.reset_terminated.float(),
        }
        if self.cfg.gait_task_gate:
            supported = (self.tip_contact_count >= 3).float()
            terms["orientation_state_reward"] *= supported
            terms["orientation_progress_reward"] *= supported
        if self.cfg.axis_velocity_reward_scale:
            terms["axis_velocity_reward"] = self.cfg.axis_velocity_reward_scale * axis_velocity_reward(
                self.axis_angular_velocity_palm, self.tip_contact_count,
                self.cfg.axis_velocity_clip_rad_s, self.cfg.axis_velocity_min_tips)
        if self.cfg.relocation_bonus:
            tip_local = quat_apply_inverse(
                self.cube_quat[:, None, :].expand(-1, 5, -1).reshape(-1, 4),
                (self.tip_pos_palm - self.cube_pos[:, None, :]).reshape(-1, 3)).view(self.num_envs, 5, 3)
            fresh = self.relocation_fresh[:, None]  # first step after a reset: no contact history yet
            previous = torch.where(fresh, self.tip_in_contact, self.relocation_previous_contact)
            events, self.relocation_release_position, self.relocation_release_steps = relocation_events(
                previous, self.tip_in_contact, self.relocation_release_position, self.relocation_release_steps,
                tip_local, self.cfg.relocation_min_release_steps, self.cfg.relocation_min_displacement_m,
                self.cfg.relocation_min_support_tips)
            self.relocation_previous_contact.copy_(self.tip_in_contact)
            self.relocation_fresh[:] = False
            count = events.sum(dim=-1).float()
            self.relocation_count += count
            terms["relocation_bonus"] = self.cfg.relocation_bonus * count
        if self.cfg.joint_limit_penalty_scale:
            q = self.hand.data.joint_pos[:, self.active_joint_ids]
            terms["joint_limit_penalty"] = -self.cfg.joint_limit_penalty_scale * joint_limit_pressure(
                q, self.lower, self.upper, math.radians(self.cfg.joint_limit_margin_deg))
        if self.cfg.tip_contact_reward:
            # AnyRotate (2024): reward enough fingertips on the object, not a distance proxy.
            terms["tip_contact_reward"] = self.cfg.tip_contact_reward * (
                self.tip_contact_count >= self.cfg.min_tip_contacts).float()
        if self.cfg.palm_contact_penalty:
            # The cradle is the cheap solution; this is what forbids it.
            terms["palm_contact_penalty"] = -self.cfg.palm_contact_penalty * (
                self.palm_contact_force > self.cfg.tip_contact_force_n).float()
        if self.cfg.grasp_quality_scale:
            quality, lambda_min = self.grasp_quality(
                self.tip_pos_palm - self.cube_pos[:, None, :], self.tip_in_contact)
            self.grasp_lambda_min = lambda_min
            # Kept on the env so reward_preflight.py can report the value an intact grasp holds:
            # that measurement is what grasp_quality_floor has to be set from, and the deficit
            # penalty below reads it rather than recomputing the Gramian.
            self.grasp_quality_value = quality
            terms["grasp_quality_reward"] = self.cfg.grasp_quality_scale * quality
        if self.cfg.grasp_deficit_scale:
            # Graded on the COUNT, not the >=3 indicator: 3+ tips is free, 2 tips costs one unit,
            # 1 tip two. The indicator would make the whole flicker band cost the same as a
            # collapsing grasp, and it is the slide from 3 to 1 that predicts the drop.
            deficit = (self.cfg.min_tip_contacts - self.tip_contact_count).clamp_min(0).float()
            terms["grasp_deficit_penalty"] = -self.cfg.grasp_deficit_scale * deficit
        if self.cfg.grasp_quality_deficit_scale:
            # Only below the floor, so the quality an intact grasp holds is never taxed.
            shortfall = (self.cfg.grasp_quality_floor - self.grasp_quality_value).clamp_min(0.0)
            terms["grasp_quality_deficit_penalty"] = (
                -self.cfg.grasp_quality_deficit_scale * shortfall)
        if self.cfg.goal_dwell_reward:
            # The reward-v3 policy learned to pass THROUGH the tolerance (entered 0.508, held
            # 0.000). Paying per step while inside it is what makes braking and staying worth more
            # than flying past, and the total is capped below success_bonus by resolve_control_config.
            terms["goal_dwell_reward"] = self.cfg.goal_dwell_reward * in_tolerance.float()
        if self.cfg.goal_velocity_scale:
            # Anti-overshoot only: near the goal, and only the part of the object's angular speed
            # above a threshold. A global motion penalty is what broke the previous reward.
            # The window is capped at half the goal magnitude, because an ABSOLUTE window stops
            # meaning "near the goal" once the goal is small: measured at a 10 deg goal with the
            # 10 deg window, this term went to -0.0025 per step against -0.0001..-0.0006 at 20 deg,
            # i.e. it was braking the entire approach instead of the arrival.
            near = self.orientation_error < math.radians(self.cfg.goal_velocity_window_deg)
            speed = torch.linalg.vector_norm(self.cube.data.root_ang_vel_w, dim=-1)
            excess = (speed - self.cfg.goal_velocity_threshold_rad_s).clamp_min(0.0)
            terms["goal_velocity_penalty"] = -self.cfg.goal_velocity_scale * excess * near.float()
        if self.cfg.pose_penalty_scale:
            terms["pose_penalty"] = -self.cfg.pose_penalty_scale * pose_excess
        if self.cfg.torque_penalty_scale:
            # Mean, not sum, so the weight does not depend on the DOF count.
            terms["torque_penalty"] = -self.cfg.torque_penalty_scale * applied.square().mean(dim=-1)
        if self.cfg.work_penalty_scale:
            terms["work_penalty"] = -self.cfg.work_penalty_scale * (
                applied * joint_velocity).sum(dim=-1).square()
        reward = sum(terms.values())
        # Per-env terms for diagnostics (the log below only keeps their means, which cannot be
        # restricted to the envs that are still holding the cube). No host sync, no extra compute.
        self.reward_terms = terms
        self.previous_orientation_error.copy_(self.orientation_error)
        self.episode_returns += reward
        # Counted before _resample_goals overwrites goal_at_frontier for these envs.
        self.frontier_successes += (first_success & self.goal_at_frontier).sum()
        # A promotion here applies to the goals handed out just below, which is intended.
        self._update_goal_curriculum()
        # Only after the reward for THIS step is settled: a completed goal is replaced in place.
        # previous_orientation_error is re-seeded with the new error, otherwise the next step's
        # progress term would read the goal switch as a huge regression and punish the success.
        if self.cfg.resample_goal_on_success and bool(first_success.any()):
            self._resample_goals(first_success.nonzero(as_tuple=False).squeeze(-1))
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
            "held_success_rate": (self.goals_completed > 0).float().mean(),
            "goals_completed": self.goals_completed.float().mean(),
            **{name: mask.float().mean() for name, mask in self.termination_reasons.items()},
            "cube_drift_from_grasp_mm": 1000 * self.palm_region_distance.mean(),
            # Both forms of the progress term: if the clip is binding often, the object is being
            # moved faster than the term can see and the clip is the wrong size.
            "orientation_progress_raw_rad": raw_progress.mean(),
            "orientation_progress_abs_raw_rad": raw_progress.abs().mean(),
            "orientation_progress_clipped_fraction": (
                (raw_progress.abs() > self.cfg.orientation_progress_clip_rad).float().mean()
                if self.cfg.orientation_progress_clip_rad else torch.zeros((), device=self.device)),
            "pose_deviation_deg": torch.rad2deg(pose_deviation).mean(),
            "pose_excess_deg": torch.rad2deg(pose_excess).mean(),
            "anchor_excess_mm": 1000 * anchor_excess.mean(),
            "cube_ang_speed_rad_s": torch.linalg.vector_norm(self.cube.data.root_ang_vel_w, dim=-1).mean(),
            "goals_issued": self.goals_issued.float().mean(),
            "goal_angle_limit_deg": torch.tensor(
                math.degrees(self.goal_angle_limit_rad), device=self.device),
            "tip_contacts": self.tip_contact_count.float().mean(),
            "tip_contacts_ok_fraction": (self.tip_contact_count >= self.cfg.min_tip_contacts).float().mean(),
            "palm_contact_fraction": (self.palm_contact_force > self.cfg.tip_contact_force_n).float().mean(),
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

    def _sample_goals_for(self, env_ids: torch.Tensor, reference_quat: torch.Tensor) -> torch.Tensor:
        """Goals for env_ids, measured from reference_quat, honouring the curriculum if enabled."""
        if self.cfg.goal_stream:
            # Fixed magnitude, stage-dependent direction. reference_quat is the cube's ACTUAL
            # orientation (at reset, or at the moment the previous goal was completed), so the new
            # goal is exactly goal_stream_angle_deg away from where the cube really is.
            self.goals_issued[env_ids] += 1
            return sample_fixed_angle_goals(
                reference_quat, math.radians(self.cfg.goal_stream_angle_deg),
                self.cfg.goal_stream_stage, self.cfg.target_axis_in_palm)
        if not self.cfg.goal_curriculum:
            return sample_goal_quats(
                reference_quat, self.cfg.target_orientation_mode, self.cfg.target_axis_in_palm,
                self.cfg.target_angle_range_deg, math.radians(self.cfg.min_initial_goal_error_deg),
            )
        quats, at_frontier = sample_curriculum_goals(
            reference_quat, math.radians(self.cfg.min_initial_goal_error_deg),
            self.goal_angle_limit_rad, math.radians(self.cfg.goal_curriculum_band_deg),
            self.cfg.goal_frontier_fraction, self.cfg.goal_inside_fraction,
        )
        self.goal_at_frontier[env_ids] = at_frontier
        # Every issued frontier goal is an attempt; completions are counted in _get_rewards.
        self.frontier_attempts += at_frontier.sum()
        return quats

    def _update_goal_curriculum(self):
        """Widen the goal limit once the frontier band is being solved (POISE: promote at 40%)."""
        if not self.cfg.goal_curriculum:
            return
        if int(self.common_step_counter) % self.cfg.goal_curriculum_check_steps:
            return
        attempts = self.frontier_attempts.item()
        if attempts < self.cfg.goal_curriculum_min_samples:
            return
        rate = self.frontier_successes.item() / max(attempts, 1.0)
        if rate >= self.cfg.goal_curriculum_promote_rate:
            limit = min(self.goal_angle_limit_rad + math.radians(self.cfg.goal_curriculum_step_deg),
                        math.radians(self.cfg.goal_curriculum_max_deg))
            self.goal_angle_limit_rad = limit
            logger.info("Goal curriculum promoted to %.1f deg (frontier success %.2f over %d goals)",
                        math.degrees(limit), rate, int(attempts))
        self.frontier_attempts.zero_()
        self.frontier_successes.zero_()

    def _resample_goals(self, env_ids: torch.Tensor):
        """Hand a new goal to envs that just completed one, without resetting the episode.

        The new goal is sampled against the cube's CURRENT orientation with the same minimum
        error, so every goal demands a real rotation from wherever the cube now sits.
        """
        current_quat = self.cube_quat[env_ids]
        self.goal_quat[env_ids] = self._sample_goals_for(env_ids, current_quat)
        new_error = quat_error_magnitude(current_quat, self.goal_quat[env_ids])
        self.orientation_error[env_ids] = new_error
        # Seeded with the new error so the progress term sees no jump across the goal switch.
        self.previous_orientation_error[env_ids] = new_error
        self.goals_completed[env_ids] += 1
        self.success_eligible[env_ids] = new_error >= self.cfg.success_tolerance_rad
        self.success.new_goal(env_ids)

    def _pick_grasps(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Which cache entries these envs reset to, as indices into self.grasp_cache.

        A seam, not a policy: the robustness search prescribes an exact entry per env and the
        reset curriculum restricts the draw to a subset, without either of them re-implementing
        reset. env_ids is passed because a mid-episode reset covers only some envs.
        """
        total = self.grasp_cache["joint_pos"].shape[0]
        count = len(env_ids)
        if total == self.primary_grasp_count or self.cfg.mix_cache_fraction <= 0:
            return torch.randint(self.primary_grasp_count, (count,), device=self.device)
        # Two ranges rather than one weighted draw over all entries: the fraction then means what
        # it says regardless of how differently sized the two caches are.
        from_mix = torch.rand(count, device=self.device) < self.cfg.mix_cache_fraction
        picks = torch.randint(self.primary_grasp_count, (count,), device=self.device)
        mix = self.primary_grasp_count + torch.randint(
            total - self.primary_grasp_count, (count,), device=self.device)
        return torch.where(from_mix, mix, picks)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        completed = env_ids[self.episode_length_buf[env_ids] > 0]
        # Per-episode values (possibly empty) under every key, every step: RSL-RL takes its keys
        # from the first step of an iteration and concatenates, i.e. an episode-weighted mean.
        # `succeeded` is cleared on every new goal, so the episode's first success is the one
        # recorded in success_step (-1 when the episode never reached a held success).
        success = self.success.success_step[completed] >= 0
        steps = self.episode_length_buf[completed].float()
        episode_log = {
            "episode_reward": self.episode_returns[completed],
            "episode_held_success_rate": (self.goals_completed[completed] > 0).float(),
            "episode_goals_completed": self.goals_completed[completed].float(),
            "episode_mean_tip_contacts": self.tip_contact_sum[completed] / steps,
            "episode_entered_tolerance_rate": self.success.entered[completed].float(),
            "episode_drop_rate": self.reset_terminated[completed].float(),
            "episode_initial_orientation_error_deg": torch.rad2deg(self.initial_orientation_error[completed]),
            "episode_final_orientation_error_deg": torch.rad2deg(self.orientation_error[completed]),
            "episode_mean_orientation_error_deg": torch.rad2deg(self.error_sum[completed] / steps),
            "episode_min_orientation_error_deg": torch.rad2deg(self.error_min[completed]),
            "episode_time_in_tolerance_fraction": self.success.steps_in_tolerance[completed].float() / steps,
            "episode_time_to_held_success_s": self.success.success_step[completed][success].float() * self.step_dt,
            "episode_length_s": steps * self.step_dt,
            # Stream metrics. completion rate = completed / issued, so loitering on one goal reads
            # differently from completing several; time per goal is only defined once one completed.
            "episode_goals_issued": self.goals_issued[completed].float(),
            "episode_target_completion_rate":
                self.goals_completed[completed].float() / self.goals_issued[completed].clamp_min(1).float(),
            "episode_time_per_completed_goal_s":
                (steps[self.goals_completed[completed] > 0] * self.step_dt
                 / self.goals_completed[completed][self.goals_completed[completed] > 0].float()),
            "episode_first_goal_held_success_rate": (self.goals_completed[completed] >= 1).float(),
            # Cumulative COMMANDED rotation, not net object orientation: the axes change, so this
            # is path length along the stream, and it is the honest way to say "how much it turned".
            "episode_commanded_rotation_deg":
                self.goals_completed[completed].float() * self.cfg.goal_stream_angle_deg,
            "episode_axis_rotation_palm_deg": torch.rad2deg(self.axis_rotation_palm[completed]),
            "episode_axis_rotation_body_deg": torch.rad2deg(self.axis_rotation_body[completed]),
        }
        assert set(episode_log) == set(EPISODE_LOG_KEYS)
        self.extras.setdefault("log", {}).update(episode_log)
        super()._reset_idx(env_ids)
        count = len(env_ids)
        if self.grasp_cache is not None:
            # Draw a validated fingertip grasp per episode. Diverse initialisation is the single
            # ingredient POISE (2026) credits with post-drop recovery 33.8% -> 72.9%.
            picks = self._pick_grasps(env_ids)
            self.hand.data.default_joint_pos[env_ids[:, None], self.joint_ids] = \
                self.grasp_cache["joint_pos"][picks]
            self.grasp_command[env_ids] = self.grasp_cache["joint_command"][picks]
            self.cube_anchor[env_ids] = self.grasp_cache["cube_pos"][picks]
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
        # The cube restarts aligned with the palm (identity in the palm frame) unless a perturbation
        # is configured. The grasp-cache robustness search needs one; 0 deg leaves the historical
        # aligned reset unchanged, and the goal is sampled against the ACTUAL start pose either way,
        # so the minimum-initial-error guarantee holds regardless.
        initial_quat = torch.zeros((count, 4), device=self.device)
        initial_quat[:, 0] = 1
        if self.cfg.cube_reset_orientation_noise_deg:
            angles = sample_uniform(0.0, math.radians(self.cfg.cube_reset_orientation_noise_deg),
                                    (count,), device=self.device)
            initial_quat = axis_angle_quat_axes(angles, random_axes(count, self.device))
        if self.grasp_cache is not None and self.cfg.grasp_cache_orientation:
            initial_quat = self.grasp_cache["cube_quat"][picks]
        elif self.cube_reset_quat is not None:
            initial_quat = self.cube_reset_quat[env_ids]
        cube_state[:, :3] = self.palm_pos_w[env_ids] + quat_apply(self.palm_quat_w[env_ids], local_pos)
        cube_state[:, 3:7] = quat_mul(self.palm_quat_w[env_ids], initial_quat)
        cube_state[:, 7:] = 0
        self.cube.write_root_pose_to_sim(cube_state[:, :7], env_ids=env_ids)
        self.cube.write_root_velocity_to_sim(cube_state[:, 7:], env_ids=env_ids)

        self.goal_quat[env_ids] = self._sample_goals_for(env_ids, initial_quat)
        initial_error = quat_error_magnitude(initial_quat, self.goal_quat[env_ids])
        assert torch.all(initial_error >= math.radians(self.cfg.min_initial_goal_error_deg) - 1e-4)
        self.previous_orientation_error[env_ids] = initial_error
        self.initial_orientation_error[env_ids] = initial_error
        # Defensive: never a free bonus for a goal already within tolerance.
        self.success_eligible[env_ids] = initial_error >= self.cfg.success_tolerance_rad
        self.success.reset(env_ids)
        self.goals_completed[env_ids] = 0
        # _sample_goals_for above already counted this episode's first goal, so clear before it.
        self.goals_issued[env_ids] = 1 if self.cfg.goal_stream else 0
        self.error_sum[env_ids] = 0
        self.gait_timers[env_ids] = 0
        self.gait_previous_contacts[env_ids] = False
        self.tip_contact_sum[env_ids] = 0
        self.steps_without_tip_contact[env_ids] = 0
        self.error_min[env_ids] = math.inf
        self.axis_rotation_palm[env_ids] = 0
        self.axis_rotation_body[env_ids] = 0
        self.rotation_tracking_fresh[env_ids] = True
        self.relocation_release_steps[env_ids] = 0
        self.relocation_fresh[env_ids] = True
        self.relocation_count[env_ids] = 0
        self.episode_returns[env_ids] = 0
        self.gait_last_picks[env_ids] = -1
        if self.gait_cache is not None:
            selected = env_ids[torch.rand(count, device=self.device) < self.cfg.gait_transition_fraction]
            if len(selected):
                from dg5f_isaaclab.assets.gait_cache import restore
                picks = torch.multinomial(self.gait_sample_weights, len(selected), replacement=True)
                restore(self, self.gait_cache, picks, selected)
                self.gait_last_picks[selected] = picks
