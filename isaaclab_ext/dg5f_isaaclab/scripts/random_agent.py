# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--control_mode", choices=("integrated_delta_position", "measured_delta_position", "absolute_position"), help="DG5F control mapping override.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps; 0 runs until closed.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401
from control_sanity import ControlSanity


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.control_mode is not None:
        env_cfg.control_mode = args_cli.control_mode
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, _ = env.reset()
    assert torch.isfinite(obs["policy"]).all(), "Non-finite reset observation"
    checks = ControlSanity(env.unwrapped) if hasattr(env.unwrapped, "action_queue") else None
    if checks:
        checks.check(obs)
    steps, resets = 0, 0
    # simulate environment
    while simulation_app.is_running() and (args_cli.max_steps <= 0 or steps < args_cli.max_steps):
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all(), "Non-finite observation"
            assert torch.isfinite(reward).all(), "Non-finite reward"
            steps += 1
            resets += int((terminated | truncated).sum().item())
            if checks:
                checks.check(obs)

    print(f"[SANITY] random: steps={steps}, resets={resets}, observation_shape={tuple(obs['policy'].shape)}")
    if checks:
        checks.report()
    if args_cli.max_steps > 0:
        assert steps == args_cli.max_steps, "Simulator closed before the requested check finished"

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    simulation_app.close()
