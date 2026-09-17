"""Deterministic evaluation of RSL-RL checkpoints vs zero/random baselines (no training).

Every env runs exactly one full episode from step 0 (no random episode offsets), with the
same seed (same goals / reset noise up to GPU PhysX nondeterminism) for every policy.
Each checkpoint is rebuilt from ITS run's params/agent.yaml (network size, normalization).
Per-episode results are captured right before the env resets itself.
Policies are always deterministic (mean action, no exploration noise).
"""

import argparse
import json
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", action="append", default=[], metavar="LABEL=RUN_DIR[:ITERS]",
                    help="e.g. new=logs/.../2026-09-17_x or old=logs/...:0,250,499 (default: all checkpoints)")
parser.add_argument("--task", default="DG5F-Cube-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--baselines", default="zero,random")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401


def parse_runs():
    policies = [(name, None, None) for name in args_cli.baselines.split(",") if name]
    for spec in args_cli.run:
        label, rest = spec.split("=", 1)
        run_dir, _, iters = rest.partition(":")
        run_dir = Path(run_dir)
        available = sorted(int(p.stem[6:]) for p in run_dir.glob("model_*.pt"))
        chosen = [int(i) for i in iters.split(",")] if iters else available
        for it in chosen:
            if it not in available:
                raise FileNotFoundError(f"{run_dir}/model_{it}.pt")
            policies.append((f"{label}/model_{it}", run_dir, run_dir / f"model_{it}.pt"))
    return policies


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    raw = env.unwrapped
    tol_deg = math.degrees(env_cfg.success_tolerance_rad)

    records = {}
    reset_idx = raw._reset_idx

    def recorded_reset(env_ids):
        ids = raw.hand._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=raw.device)
        tracker = raw.success
        for i in ids[raw.episode_length_buf[ids] > 0].tolist():
            if i not in records:
                steps = int(raw.episode_length_buf[i])
                records[i] = {
                    "held": bool(tracker.succeeded[i]), "entered": bool(tracker.entered[i]),
                    "drop": bool(raw.reset_terminated[i]),
                    "initial": math.degrees(raw.initial_orientation_error[i]),
                    "final": math.degrees(raw.orientation_error[i]),
                    "mean": math.degrees(raw.error_sum[i] / steps), "min": math.degrees(raw.error_min[i]),
                    "in_tol": int(tracker.steps_in_tolerance[i]) / steps,
                    "held_step": int(tracker.success_step[i]), "steps": steps,
                    "reward": float(raw.episode_returns[i]),
                }
        reset_idx(env_ids)

    raw._reset_idx = recorded_reset
    runners = {}
    results = {}
    for name, run_dir, path in parse_runs():
        if path is not None:
            if run_dir not in runners:
                agent = load_yaml(str(run_dir / "params" / "agent.yaml"))
                runners[run_dir] = OnPolicyRunner(env, agent, log_dir=None, device=raw.device)
            runner = runners[run_dir]
            runner.load(str(path))
            policy = runner.get_inference_policy(device=raw.device)
        torch.manual_seed(args_cli.seed)
        with torch.inference_mode():
            env.reset()
            records.clear()  # the reset above closed the previous policy's episodes
            obs = env.get_observations()
            near, sat, vel, steps = 0.0, 0.0, 0.0, 0
            while len(records) < raw.num_envs and steps < raw.max_episode_length + 5:
                if name == "zero":
                    actions = torch.zeros((raw.num_envs, env_cfg.action_space), device=raw.device)
                elif name == "random":
                    actions = 2 * torch.rand((raw.num_envs, env_cfg.action_space), device=raw.device) - 1
                else:
                    actions = policy(obs)
                obs, _, _, extras = env.step(actions)
                log = extras["log"]
                near += log["action_near_limit_fraction"].item()
                sat += log["torque_saturation_fraction"].item()
                vel = max(vel, log["max_abs_joint_velocity"].item())
                steps += 1
        values = list(records.values())
        n = len(values)
        mean = lambda key: sum(v[key] for v in values) / n
        held_times = [v["held_step"] * raw.step_dt for v in values if v["held"]]
        kept = [v for v in values if not v["drop"]]
        results[name] = {
            "episodes": n,
            "held_success_rate": mean("held"),
            "entered_tolerance_rate": mean("entered"),
            "drop_rate": mean("drop"),
            "initial_error_deg": mean("initial"),
            "final_error_deg": mean("final"),
            "final_error_deg_not_dropped": sum(v["final"] for v in kept) / max(1, len(kept)),
            "final_within_tolerance": sum(v["final"] <= tol_deg for v in kept) / n,
            "mean_error_deg": mean("mean"),
            "min_error_deg": mean("min"),
            "time_in_tolerance_fraction": mean("in_tol"),
            "time_to_held_success_s": sum(held_times) / len(held_times) if held_times else float("nan"),
            "episode_reward": mean("reward"),
            "episode_length_s": mean("steps") * raw.step_dt,
            "action_near_limit_fraction": near / steps,
            "torque_saturation_fraction": sat / steps,
            "max_abs_joint_velocity": vel,
        }
        r = results[name]
        print(f"[EVAL] {name:>18} n={n} held={r['held_success_rate']:.3f} entered={r['entered_tolerance_rate']:.3f}"
              f" in_tol={r['time_in_tolerance_fraction']:.3f} drop={r['drop_rate']:.3f}"
              f" final={r['final_error_deg']:.1f} final_kept={r['final_error_deg_not_dropped']:.1f}"
              f" mean={r['mean_error_deg']:.1f} min={r['min_error_deg']:.1f} reward={r['episode_reward']:.2f}"
              f" near_pm1={r['action_near_limit_fraction']:.3f} sat={r['torque_saturation_fraction']:.3f}"
              f" maxvel={r['max_abs_joint_velocity']:.2f}", flush=True)
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(results, indent=2))
    print(f"[EVAL] wrote {args_cli.output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
