"""Measure what the reward ACTUALLY pays before spending hours of PPO on it.

The reward-v3 night run failed for a reason that was visible in one number and nobody had looked:
pose_penalty and palm_distance_penalty together carried 69 of ~90 units of per-episode reward
magnitude against +21 for the task, so "do not move" was the optimum. Worse, the weights had been
sized from an ASSUMED deviation (3 deg) rather than the measured one (14.9 deg).

This reports mean |term| per step under several policies, restricted to the envs that are still
holding the cube -- an env that dropped contributes only drop_penalty and would flatter the ratio.

    task_abs        = |orientation_state| + |orientation_progress| + |goal_dwell|
    stability_abs   = |pose| + |palm_distance| + |action| + |action_rate| + |torque| + |near-goal omega|

The gate is task_abs / stability_abs roughly 2-4. Below 1 the penalties are in charge again; far
above 10 the policy has nothing holding it back and is worth a look before committing.
"""

import argparse
import json
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="DG5F-Cube-Stream-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=600, help="Control steps per condition (10 s at 60 Hz)")
parser.add_argument("--checkpoint", type=Path, default=None, help="Warm-start checkpoint for conditions B and C")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--scripted_amplitude_deg", type=float, default=0.6,
                    help="Peak per-step delta of the scripted gaiting probe, in action units x scale")
parser.add_argument("--scripted_period_s", type=float, default=1.5)
parser.add_argument("--output", type=Path, default=Path("logs/reward_preflight/preflight.json"))
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
import isaaclab_tasks  # noqa: F401

import dg5f_isaaclab.tasks  # noqa: F401

TASK_TERMS = ("orientation_state_reward", "orientation_progress_reward", "goal_dwell_reward")
STABILITY_TERMS = ("pose_penalty", "palm_distance_penalty", "action_penalty", "action_rate_penalty",
                   "torque_penalty", "goal_velocity_penalty")


QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)


def percentiles(samples):
    """Quantiles of a pooled sample, as {"p10": value, ...}; empty when nothing was collected."""
    if not samples:
        return {}
    pooled = torch.cat(samples).float()
    values = torch.quantile(pooled, torch.tensor(QUANTILES, device=pooled.device))
    return {f"p{round(q * 100):02d}": float(v) for q, v in zip(QUANTILES, values)}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    raw = env.unwrapped
    period_steps = max(2, round(args_cli.scripted_period_s / raw.step_dt))
    # The scripted probe commands a slow out-of-phase sinusoid per joint: it is a MOTION probe, not
    # a controller, and its job is to show what the reward pays when the fingers actually move.
    phase = torch.linspace(0, 2 * math.pi, env_cfg.action_space + 1, device=raw.device)[:-1]
    amplitude = math.radians(args_cli.scripted_amplitude_deg) / env_cfg.delta_action_scale

    conditions = ["zero", "scripted"]
    runner = None
    if args_cli.checkpoint is not None:
        from isaaclab_tasks.utils import load_cfg_from_registry
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw.device)
        runner.load(str(args_cli.checkpoint))
        conditions += ["warm_deterministic", "warm_stochastic"]
        print(f"[PRE] loaded {args_cli.checkpoint}", flush=True)

    results = {"meta": {"task": args_cli.task, "num_envs": raw.num_envs, "steps": args_cli.steps,
                        "checkpoint": str(args_cli.checkpoint) if args_cli.checkpoint else None}}
    for condition in conditions:
        policy = None
        if condition == "warm_deterministic":
            policy = runner.get_inference_policy(device=raw.device)
        torch.manual_seed(args_cli.seed)
        sums, grasp_sums, alive_steps = {}, {}, 0.0
        samples_quality, samples_tips = [], []
        dropped_total, rotation = 0, 0.0
        with torch.inference_mode():
            env.reset()
            obs = env.get_observations()
            dropped = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
            start_error = raw.orientation_error.clone()
            for step in range(args_cli.steps):
                if condition == "zero":
                    actions = torch.zeros((raw.num_envs, env_cfg.action_space), device=raw.device)
                elif condition == "scripted":
                    actions = amplitude * torch.sin(2 * math.pi * step / period_steps + phase)
                    actions = actions.expand(raw.num_envs, -1).clamp(-1.0, 1.0)
                elif condition == "warm_deterministic":
                    actions = policy(obs)
                else:
                    actions = runner.alg.policy.act(obs)  # sampled, i.e. exploration at std 0.2
                obs, _, _, _ = env.step(actions)
                alive = (~dropped).float()
                for name, value in raw.reward_terms.items():
                    sums[name] = sums.get(name, 0.0) + float((value.abs() * alive).sum())
                # The two grasp quantities SIGNED and uncentered: these are what
                # tip_contact_reference and grasp_quality_reference have to be set from. Taking them
                # from the term would divide by the scale and silently break once the term is
                # centered, and taking them from an assumption is the error that sized reward v3.
                tips_ok = (raw.tip_contact_count >= raw.cfg.min_tip_contacts).float()
                grasp_sums["tip_contact_ok_fraction"] = (
                    grasp_sums.get("tip_contact_ok_fraction", 0.0) + float((tips_ok * alive).sum()))
                quality = getattr(raw, "grasp_quality_value", None)
                if quality is not None:
                    # The DISTRIBUTION, not just the mean. A floor is a threshold on a clipped,
                    # convex penalty, so E[max(0, floor - q)] > max(0, floor - E[q]) whenever the
                    # value fluctuates: sizing grasp_quality_floor from the mean understated the
                    # warm-start policy's cost by 4.5x (-7 predicted, -43 measured per episode).
                    # Percentiles are what the floor has to be read off.
                    samples_quality.append(quality[alive > 0].detach().clone())
                    samples_tips.append(raw.tip_contact_count[alive > 0].detach().clone())
                    grasp_sums["grasp_quality"] = (
                        grasp_sums.get("grasp_quality", 0.0) + float((quality * alive).sum()))
                    grasp_sums["grasp_lambda_min"] = (
                        grasp_sums.get("grasp_lambda_min", 0.0)
                        + float((raw.grasp_lambda_min * alive).sum()))
                alive_steps += float(alive.sum())
                # reset_terminated is the DROP, not the time limit; the wrapper's `dones` mixes both.
                dropped |= raw.reset_terminated
            dropped_total = int(dropped.sum())
            rotation = float(torch.rad2deg((raw.orientation_error - start_error).abs()).mean())

        means = {name: total / max(alive_steps, 1.0) for name, total in sums.items()}
        task_abs = sum(means.get(name, 0.0) for name in TASK_TERMS)
        stability_abs = sum(means.get(name, 0.0) for name in STABILITY_TERMS)
        ratio = task_abs / stability_abs if stability_abs > 0 else float("inf")
        results[condition] = {
            "mean_abs_terms": means, "task_abs": task_abs,
            "stability_regularization_abs": stability_abs, "ratio": ratio,
            "drop_rate": dropped_total / raw.num_envs,
            "alive_step_fraction": alive_steps / (raw.num_envs * args_cli.steps),
            "mean_abs_orientation_change_deg": rotation,
            "grasp_reference_candidates": {
                name: total / max(alive_steps, 1.0) for name, total in grasp_sums.items()},
            "grasp_quality_percentiles": percentiles(samples_quality),
            "tip_contact_count_percentiles": percentiles(samples_tips),
        }
        print(f"[PRE] {condition:>18}  task={task_abs:.5f}  stability={stability_abs:.5f}"
              f"  ratio={ratio:6.2f}  drop={dropped_total / raw.num_envs:.3f}"
              f"  alive={alive_steps / (raw.num_envs * args_cli.steps):.3f}", flush=True)
        for name in sorted(means, key=lambda k: -means[k]):
            group = "task" if name in TASK_TERMS else ("stab" if name in STABILITY_TERMS else "----")
            print(f"[PRE]     {group} {name:32s} {means[name]:+.6f}", flush=True)
        for name, value in results[condition]["grasp_reference_candidates"].items():
            print(f"[PRE]     ref  {name:32s} {value:+.6f}", flush=True)
        for name in ("grasp_quality_percentiles", "tip_contact_count_percentiles"):
            spread = results[condition][name]
            if spread:
                print(f"[PRE]     pct  {name:32s} "
                      + "  ".join(f"{k}={v:+.3f}" for k, v in spread.items()), flush=True)

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(results, indent=2))
    print(f"[PRE] wrote {args_cli.output}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
