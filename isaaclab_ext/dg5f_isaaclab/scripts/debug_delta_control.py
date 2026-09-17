"""One-joint delta impulse through the actual DG5F loop: FIFO -> delta -> q_cmd (no training)."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="DG5F-Cube-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401
from control_sanity import ControlSanity


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    assert cfg.control_mode == "integrated_delta_position"
    obs, _ = env.reset()
    checks = ControlSanity(raw)
    checks.check(obs)
    joint = 0
    delay = cfg.action_delay_steps[joint]
    print(f"[IMPULSE] joint={cfg.active_action_joints[joint]} delay={delay} control steps")
    print("[IMPULSE] t policy delivered pending(newest-first) q_measured_rad q_cmd_prev_rad q_cmd_rad dq_cmd_deg")
    with torch.inference_mode():
        for step in range(delay + 4):
            action = torch.zeros((1, cfg.action_space), device=raw.device)
            action[0, joint] = float(step == 0)
            measured = raw.hand.data.joint_pos[:, raw.active_joint_ids].clone()
            command = raw.joint_command.clone()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert not (terminated | truncated).any(), "Unexpected reset during impulse"
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            checks.check(obs)
            expected_action = torch.zeros_like(action)
            expected_action[0, joint] = float(step == delay)
            torch.testing.assert_close(raw.applied_actions, expected_action, atol=0, rtol=0)
            expected_target = (command + cfg.delta_action_scale * expected_action).clamp(raw.lower, raw.upper)
            target = raw.joint_targets[:, raw.active_indices]
            torch.testing.assert_close(target, expected_target, atol=1e-6, rtol=0)
            torch.testing.assert_close(obs["policy"][:, -cfg.action_space:], target, atol=0, rtol=0)
            print(f"[IMPULSE] {step} {action[0, joint].item():.0f} {raw.applied_actions[0, joint].item():.0f}"
                  f" {raw.action_queue.history[0, :, joint].tolist()}"
                  f" {measured[0, joint].item():.7f} {command[0, joint].item():.7f} {target[0, joint].item():.7f}"
                  f" {math.degrees((target - command)[0, joint].item()):.6f}")
        grasp = raw.grasp_command[:, raw.active_indices]
        expected_cmd = grasp.clone()
        expected_cmd[0, joint] = (grasp[0, joint] + cfg.delta_action_scale).clamp(raw.lower[0, joint], raw.upper[0, joint])
        torch.testing.assert_close(raw.joint_command, expected_cmd, atol=1e-6, rtol=0)
        print(f"[IMPULSE] q_cmd kept the single +{math.degrees(cfg.delta_action_scale):g} deg step after the impulse")

        # Reset with an impulse still pending, then prove it cannot leak into the next episode.
        pending = torch.zeros((1, cfg.action_space), device=raw.device)
        pending[0, joint] = 1
        env.step(pending)
        obs, _ = env.reset()
        checks.check(obs)
        for value in (raw.action_queue.history, raw.actions, raw.previous_actions,
                      raw.applied_actions, raw.action_delta, raw.episode_returns,
                      raw.success.succeeded, raw.success.consecutive, raw.success.steps_in_tolerance):
            assert torch.count_nonzero(value) == 0, "Reset left episode history behind"
        for _ in range(delay + 1):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros_like(pending))
            assert not (terminated | truncated).any()
            assert torch.count_nonzero(raw.applied_actions) == 0
            checks.check(obs)
        torch.testing.assert_close(raw.joint_command, grasp, atol=0, rtol=0)
        print("[IMPULSE] PASS: delta reached q_cmd at t+3; reset restored q_cmd and removed the pending impulse")
        print(f"[TORQUE] limits_nm={raw.torque_limits[0].tolist()}")
        for key in ("computed_torque", "applied_torque"):
            print(f"[TORQUE] {key}={raw.extras[key][0].tolist()} (implicit PD estimate)")
        for key in ("mean_abs_applied_torque", "max_abs_applied_torque", "torque_saturation_fraction"):
            print(f"[TORQUE] {key}={raw.extras['log'][key].item()}")
    checks.report()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
