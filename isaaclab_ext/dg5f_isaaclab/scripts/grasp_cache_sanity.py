"""Zero-action check of the fingertip grasp cache over a FULL episode, plus reward-term scales.

The cache is validated by fingertip_grasp_search.py over a 2 s release hold, but an episode is
24 s: a grasp that quietly fails at 5 s would make every episode a stub and waste a training
run. This holds zero actions for the whole episode and reports when envs start dropping.

It also prints the mean magnitude of every reward term, which is the cheap way to see whether
the configured weights are in a sane range relative to each other before committing hours of PPO.
"""

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="DG5F-Cube-Fingertip-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grasp_cache_path", default=None, help="Override the registered cache path")
parser.add_argument("--report_every_s", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env import DG5FCubeEnv


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # The registered DG5FCubeFingertipEnvCfg already carries the reward-v3 weights, the cache path
    # and the curriculum, so this checks the exact configuration the training run uses.
    if args_cli.grasp_cache_path is not None:
        cfg.grasp_cache_path = args_cli.grasp_cache_path
    assert cfg.grasp_cache_path is not None and cfg.enable_contact_sensors, "not the fingertip task"
    env = DG5FCubeEnv(cfg)
    steps = int(env.max_episode_length)
    report = max(1, int(args_cli.report_every_s / env.step_dt))
    zero = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
    print(f"[CACHE] envs={env.num_envs} episode_steps={steps} dt={env.step_dt:.5f}s"
          f" goal_limit={cfg.goal_curriculum_start_deg} deg", flush=True)

    # Terms are logged per step by the env; accumulate their means over the hold.
    term_sums, term_steps = {}, 0
    ever_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
        env.reset()
        start_cube = env.cube_pos.clone()
        for step in range(steps):
            _, reward, terminated, truncated, _ = env.step(zero)
            ever_dropped |= terminated
            log = env.extras.get("log", {})
            for key, value in log.items():
                if value.numel() == 1 and not key.startswith("episode_"):
                    term_sums[key] = term_sums.get(key, 0.0) + float(value)
            term_steps += 1
            if step % report == 0 or step == steps - 1:
                drift = (env.cube_pos - start_cube).norm(dim=-1)
                alive = ~ever_dropped
                print(f"[CACHE] t={step * env.step_dt:5.2f}s dropped={int(ever_dropped.sum())}/{env.num_envs}"
                      f" tips={env.tip_contact_count.float().mean():.2f}"
                      f" tips>=3={(env.tip_contact_count >= cfg.min_tip_contacts).float().mean():.2f}"
                      f" palm_contact={(env.palm_contact_force > cfg.tip_contact_force_n).float().mean():.3f}"
                      f" cube_drift_mm(alive)={1000 * drift[alive].mean() if bool(alive.any()) else float('nan'):.2f}"
                      f" reward={reward.mean():+.3f}", flush=True)
            if bool(ever_dropped.all()):
                print("[CACHE] every env dropped; stopping early", flush=True)
                break

    survived = int((~ever_dropped).sum())
    print(f"[CACHE] SURVIVED_FULL_EPISODE {survived}/{env.num_envs} "
          f"({survived / env.num_envs:.1%}) with zero actions", flush=True)
    print("[CACHE] mean per-step reward terms and diagnostics over the hold:", flush=True)
    for key in sorted(term_sums):
        print(f"[CACHE]   {key:44s} {term_sums[key] / max(term_steps, 1):+.5f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
