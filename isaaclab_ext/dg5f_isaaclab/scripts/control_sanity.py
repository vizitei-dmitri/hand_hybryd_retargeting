"""Lightweight checks for the DG5F zero/random rollouts; not used by PPO."""

import math

import torch

# Lock = PhysX limits [0, 0] + extra armature on that axis. Random rollouts measured
# <= 0.002 deg with the lock (0.065 deg without armature, 0.12 deg with the old colliders).
DISABLED_JOINT_TOLERANCE_RAD = math.radians(0.02)


class ControlSanity:
    def __init__(self, env):
        self.env = env
        self.steps = 0
        self.max_target_step = 0.0
        self.max_disabled_angle = 0.0
        self.max_limit_violation = 0.0
        self.max_grasp_drift = 0.0
        self.final_grasp_drift = 0.0
        self.max_command_drift = 0.0
        self.max_cube_offset = 0.0
        self.max_joint_speed = 0.0
        self.max_abs_torque = 0.0
        self.saturation_sum = 0.0

    def check(self, obs):
        env, cfg = self.env, self.env.cfg
        policy = obs["policy"]
        assert policy.shape == (env.num_envs, cfg.observation_space)
        assert torch.isfinite(policy).all()
        queue_end = cfg.privileged_observation_space + env.action_queue.history[0].numel()
        torch.testing.assert_close(policy[:, cfg.privileged_observation_space:queue_end],
                                   env.action_queue.history.flatten(start_dim=1), atol=0, rtol=0)
        torch.testing.assert_close(policy[:, queue_end:], env.joint_command, atol=0, rtol=0)

        target = env.joint_targets[:, env.active_indices]
        torch.testing.assert_close(target, env.joint_command, atol=0, rtol=0)
        assert torch.all((target >= env.lower - 1e-6) & (target <= env.upper + 1e-6))
        assert torch.all(env.joint_targets[:, env.disabled_index] == cfg.disabled_joint_position)
        disabled_error = (env.hand.data.joint_pos[:, env.disabled_joint_id] - cfg.disabled_joint_position).abs().max()
        assert disabled_error < DISABLED_JOINT_TOLERANCE_RAD, f"Disabled joint moved by {disabled_error.item()} rad"
        self.max_disabled_angle = max(self.max_disabled_angle, disabled_error.item())

        valid = env.episode_length_buf > 0
        base = None
        if cfg.control_mode == "integrated_delta_position":
            base = env.previous_joint_command
        elif cfg.control_mode == "measured_delta_position":
            base = env.measured_at_command
        if base is not None:
            step = (target - base).abs()
            # A measured base outside the limits can need a larger corrective clamp.
            in_limits = (base >= env.lower) & (base <= env.upper) & valid[:, None]
            if in_limits.any():
                self.max_target_step = max(self.max_target_step, step[in_limits].max().item())
            assert torch.all(step[in_limits] <= cfg.delta_action_scale + 1e-6)
            neutral = (env.applied_actions == 0) & in_limits
            torch.testing.assert_close(target[neutral], base[neutral], atol=1e-6, rtol=0)

        positions = env.hand.data.joint_pos[:, env.active_joint_ids]
        violation = torch.maximum(env.lower - positions, positions - env.upper).clamp_min(0).max().item()
        self.max_limit_violation = max(self.max_limit_violation, violation)
        drift = (positions - env.hand.data.default_joint_pos[:, env.active_joint_ids]).abs().max(dim=-1).values
        self.max_grasp_drift = max(self.max_grasp_drift, drift.max().item())
        self.final_grasp_drift = drift.max().item()
        command_drift = (env.joint_command - env.grasp_command[:, env.active_indices]).abs().max().item()
        self.max_command_drift = max(self.max_command_drift, command_drift)
        self.max_cube_offset = max(self.max_cube_offset, env.palm_region_distance.max().item())
        self.max_joint_speed = max(self.max_joint_speed, env.hand.data.joint_vel[:, env.joint_ids].abs().max().item())

        reset = ~valid
        if reset.any():
            assert torch.count_nonzero(env.action_queue.history[reset]) == 0
            assert torch.count_nonzero(env.previous_actions[reset]) == 0
            assert torch.count_nonzero(env.applied_actions[reset]) == 0
            torch.testing.assert_close(env.joint_command[reset], env.grasp_command[reset][:, env.active_indices])
        for key in ("computed_torque", "applied_torque"):
            if key in env.extras:
                assert torch.isfinite(env.extras[key]).all()
        if "applied_torque" in env.extras:
            self.max_abs_torque = max(self.max_abs_torque, env.extras["applied_torque"].abs().max().item())
            self.saturation_sum += env.extras["log"]["torque_saturation_fraction"].item()
        self.steps += 1

    def report(self):
        print(f"[CONTROL] checks={self.steps} mode={self.env.cfg.control_mode}"
              f" max_target_step_deg={math.degrees(self.max_target_step):.6f}"
              f" max_disabled_error_deg={math.degrees(self.max_disabled_angle):.6f}"
              f" max_measured_limit_violation_deg={math.degrees(self.max_limit_violation):.6f}")
        print(f"[CONTROL] grasp_drift_deg max={math.degrees(self.max_grasp_drift):.4f}"
              f" final={math.degrees(self.final_grasp_drift):.4f}"
              f" max_command_change_from_grasp_deg={math.degrees(self.max_command_drift):.6f}"
              f" max_cube_offset_mm={1000 * self.max_cube_offset:.2f}")
        print(f"[CONTROL] max_abs_joint_velocity={self.max_joint_speed:.4f} rad/s"
              f" max_abs_applied_torque={self.max_abs_torque:.4f} Nm"
              f" mean_torque_saturation_fraction={self.saturation_sum / max(1, self.steps - 1):.4f}")
