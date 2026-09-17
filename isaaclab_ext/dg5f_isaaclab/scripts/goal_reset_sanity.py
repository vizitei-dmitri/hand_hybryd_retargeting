"""Goal sampling through the real DG5FCubeEnv._reset_idx, without any physics rollout."""

import argparse
import math
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--rounds", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_error_magnitude

from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401


def main():
    cfg = parse_env_cfg("DG5F-Cube-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make("DG5F-Cube-Direct-v0", cfg=cfg)
    raw = env.unwrapped
    errors = []
    with torch.inference_mode():
        for _ in range(args_cli.rounds):
            raw._reset_idx(None)  # samples goals for every env; no sim.step()
            identity = torch.zeros_like(raw.goal_quat)
            identity[:, 0] = 1
            error = quat_error_magnitude(identity, raw.goal_quat)
            torch.testing.assert_close(error, raw.initial_orientation_error)
            errors.append(error)
            assert not raw.success.succeeded.any() and not raw.success.consecutive.any() and bool(raw.success_eligible.all())
    error = torch.rad2deg(torch.cat(errors))
    inside = (error < math.degrees(cfg.success_tolerance_rad)).sum().item()
    print(f"[GOAL] samples={error.numel()} mode={cfg.target_orientation_mode}"
          f" range_deg={cfg.target_angle_range_deg} min_required_deg={cfg.min_initial_goal_error_deg}")
    print(f"[GOAL] initial error deg: min={error.min().item():.4f} max={error.max().item():.4f}"
          f" mean={error.mean().item():.3f}; inside {math.degrees(cfg.success_tolerance_rad):g} deg tolerance={inside}")
    counts = torch.histc(error, bins=10, min=0, max=20).int().tolist()
    print(f"[GOAL] histogram 0..20 deg (2 deg bins)={counts}")
    assert error.min().item() >= cfg.min_initial_goal_error_deg - 1e-3 and inside == 0
    print("[GOAL] PASS")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
