# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--control_mode", choices=("integrated_delta_position", "measured_delta_position", "absolute_position"), help="DG5F control mapping override.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps; 0 runs until closed.")
parser.add_argument("--screenshot_dir", type=Path, help="Save reset/end GUI viewport images for visual debugging.")
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


def capture_viewport(env, name):
    """Capture the existing GUI viewport without adding a camera sensor."""
    if args_cli.screenshot_dir is None:
        return
    if not env.unwrapped.sim.has_gui():
        raise ValueError("--screenshot_dir requires the GUI (omit --headless)")
    import asyncio
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from PIL import Image

    args_cli.screenshot_dir.mkdir(parents=True, exist_ok=True)
    path = args_cli.screenshot_dir / f"{name}.png"
    path.unlink(missing_ok=True)
    for _ in range(5):
        env.unwrapped.sim.render()
    capture = capture_viewport_to_file(get_active_viewport(), str(path.resolve()))
    result = asyncio.ensure_future(capture.wait_for_result())
    for _ in range(120):
        env.unwrapped.sim.render()
        if result.done():
            result.result()
            # Kit's capture future completes before its asynchronous PNG writer.
            try:
                with Image.open(path) as saved:
                    saved.verify()
            except (OSError, SyntaxError):
                continue
            print(f"[VIEWPORT] {path}")
            return
    raise RuntimeError(f"Viewport capture timed out: {path}")


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
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
    capture_viewport(env, "reset")
    steps, resets = 0, 0
    # simulate environment
    while simulation_app.is_running() and (args_cli.max_steps <= 0 or steps < args_cli.max_steps):
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all(), "Non-finite observation"
            assert torch.isfinite(reward).all(), "Non-finite reward"
            steps += 1
            resets += int((terminated | truncated).sum().item())
            if checks:
                checks.check(obs)

    print(f"[SANITY] zero: steps={steps}, resets={resets}, observation_shape={tuple(obs['policy'].shape)}")
    if checks:
        checks.report()
    if args_cli.max_steps > 0:
        assert steps == args_cli.max_steps, "Simulator closed before the requested check finished"
    capture_viewport(env, "end")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    simulation_app.close()
