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
parser.add_argument("--stage", default=None,
                    help="Goal-stream directional stage to evaluate at (A/B/C). This script does not "
                         "go through hydra, so the stage cannot be passed as env.goal_stream_stage.")
parser.add_argument("--goal_angle_deg", type=float, default=None,
                    help="Goal magnitude to evaluate at, for the bootstrap phase. Sets BOTH "
                         "goal_stream_angle_deg and min_initial_goal_error_deg: the env asserts the "
                         "sampled initial error is at least the latter, so changing one alone trips "
                         "that assert. A checkpoint must be judged at the angle it is being trained "
                         "at, or the gate measures a different task.")
parser.add_argument("--orientation_baseline", choices=("true", "false"), default=None,
                    help="Match E4 training reward when reporting episode return")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Evaluation-only episode length. A longer episode separates a rotation CEILING "
                         "(net turn plateaus, then the cube drops) from the time limit of the 24 s episode.")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--smoothing_telemetry", action="store_true", help="Collect first-episode action/reward/drop diagnostics")
parser.add_argument("--gait_telemetry", action="store_true")
parser.add_argument("--gait_thresholds", type=Path, default=None)
parser.add_argument("--gait_hypothesis_config", type=Path, default=None)
parser.add_argument("--gait_reset_config", type=Path, default=None)
parser.add_argument("--action_penalty_scale", type=float, default=None)
parser.add_argument("--action_rate_penalty_scale", type=float, default=None)
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
    env_cfg.track_gait_contact_points = args_cli.gait_telemetry
    if args_cli.gait_hypothesis_config:
        for key, value in json.loads(args_cli.gait_hypothesis_config.read_text()).items():
            if key not in ("gait_task_gate", "gait_observation_mode"):
                raise ValueError(f"Unexpected hypothesis override: {key}")
            setattr(env_cfg, key, value)
        env_cfg.resolve_control_config()
    if args_cli.gait_reset_config:
        for key,value in json.loads(args_cli.gait_reset_config.read_text()).items():
            if key not in ('gait_transition_cache_path','gait_transition_fraction','gait_phase_weights'):
                raise ValueError(f'Unexpected reset override: {key}')
            setattr(env_cfg,key,value)
    for key in ("action_penalty_scale", "action_rate_penalty_scale"):
        value = getattr(args_cli, key)
        if value is not None:
            if value < 0:
                raise ValueError(f"{key} must be nonnegative")
            setattr(env_cfg, key, value)
    if args_cli.orientation_baseline is not None:
        env_cfg.orientation_baseline = args_cli.orientation_baseline == "true"
    if args_cli.episode_length_s is not None:
        if args_cli.episode_length_s <= 0:
            raise ValueError("episode_length_s must be positive")
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.stage is not None:
        env_cfg.goal_stream_stage = args_cli.stage
    if args_cli.goal_angle_deg is not None:
        env_cfg.goal_stream_angle_deg = args_cli.goal_angle_deg
        env_cfg.min_initial_goal_error_deg = args_cli.goal_angle_deg
    if args_cli.stage is not None or args_cli.goal_angle_deg is not None:
        env_cfg.resolve_control_config()  # re-validates the stage name and the angle vs tolerance
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    raw = env.unwrapped
    telemetry = None
    gait = None
    if args_cli.gait_telemetry:
        from gait_metrics import GaitTelemetry
        gait_rewards = raw._get_rewards
        def sample_gait_rewards():
            value = gait_rewards()
            if raw.cfg.gait_task_gate:
                unsupported = raw.tip_contact_count < 3
                for term in ("orientation_state_reward", "orientation_progress_reward", "goal_dwell_reward", "success_bonus"):
                    if term in raw.reward_terms and bool((raw.reward_terms[term][unsupported] != 0).any()):
                        raise RuntimeError(f"H2 task gate violation: {term}")
            if gait is not None:
                gait.sample()
            return value
        raw._get_rewards = sample_gait_rewards
    if args_cli.smoothing_telemetry:
        from smoothing_telemetry import Telemetry
        original_rewards, original_apply = raw._get_rewards, raw._apply_action
        def measured_rewards():
            value = original_rewards()
            if telemetry is not None:
                telemetry.reward_sample()
            return value
        def measured_apply():
            original_apply()
            if telemetry is not None:
                telemetry.physics_sample()
        raw._get_rewards, raw._apply_action = measured_rewards, measured_apply
    tol_deg = math.degrees(env_cfg.success_tolerance_rad)

    records = {}
    reset_idx = raw._reset_idx

    def recorded_reset(env_ids):
        ids = raw.hand._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=raw.device)
        tracker = raw.success
        for i in ids[raw.episode_length_buf[ids] > 0].tolist():
            if i not in records:
                steps = int(raw.episode_length_buf[i])
                completed = int(raw.goals_completed[i])
                records[i] = {
                    # NOT tracker.succeeded: that flag is cleared on every new goal, so with a
                    # goal stream it describes only the goal in progress. goals_completed is the
                    # episode-level fact, and success_step keeps the FIRST success of the episode.
                    "held": completed > 0, "entered": bool(tracker.entered[i]),
                    "drop": bool(raw.reset_terminated[i]),
                    "initial": math.degrees(raw.initial_orientation_error[i]),
                    "final": math.degrees(raw.orientation_error[i]),
                    "mean": math.degrees(raw.error_sum[i] / steps), "min": math.degrees(raw.error_min[i]),
                    "in_tol": int(tracker.steps_in_tolerance[i]) / steps,
                    "held_step": int(tracker.success_step[i]), "steps": steps,
                    "reward": float(raw.episode_returns[i]),
                    # Stream metrics: with a chain of goals, "did it succeed" no longer describes
                    # the behaviour. goals_completed is consecutive within an episode by
                    # construction, because a completed goal never resets the episode.
                    "goals": completed,
                    "issued": int(raw.goals_issued[i]),
                    "tips": float(raw.tip_contact_sum[i]) / steps,
                    "axis_palm": math.degrees(raw.axis_rotation_palm[i]),
                    "axis_body": math.degrees(raw.axis_rotation_body[i]),
                }
        reset_idx(env_ids)

    raw._reset_idx = recorded_reset
    runners = {}
    results = {}
    for name, run_dir, path in parse_runs():
        if path is not None:
            if run_dir not in runners:
                # Each checkpoint is rebuilt from ITS run's agent config. A synthetic run directory
                # (the actor-only warm start) has no params/, so fall back to the task's registered
                # config -- correct there precisely because that checkpoint was BUILT for this task.
                params = run_dir / "params" / "agent.yaml"
                if params.exists():
                    agent = load_yaml(str(params))
                else:
                    from isaaclab_tasks.utils import load_cfg_from_registry
                    agent = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point").to_dict()
                    print(f"[EVAL] {run_dir.name} has no params/agent.yaml; using the registered "
                          f"config for {args_cli.task}", flush=True)
                runners[run_dir] = OnPolicyRunner(env, agent, log_dir=None, device=raw.device)
            runner = runners[run_dir]
            print(f"[INFO]: Loading model checkpoint from: {path.resolve()}", flush=True)
            runner.load(str(path))
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
            from resume_verification import actor_fingerprint
            fingerprint = actor_fingerprint(runner.alg.policy)
            print(f"[ACTOR_FINGERPRINT] {fingerprint}", flush=True)
            policy = runner.get_inference_policy(device=raw.device)
            # The learned exploration std. Not used by the deterministic rollout below, but it is
            # what the training watchdog reads: std growing while these metrics do not improve is
            # the signature of the failed reward-v3 run (0.20 -> 0.795).
            std = getattr(runner.alg.policy, "std", None)
            action_std = float(std.mean()) if std is not None else float(
                runner.alg.policy.log_std.exp().mean())
        torch.manual_seed(args_cli.seed)
        with torch.inference_mode():
            env.reset()
            records.clear()  # the reset above closed the previous policy's episodes
            if args_cli.gait_telemetry:
                gait = GaitTelemetry(raw)
            if args_cli.smoothing_telemetry:
                telemetry = Telemetry(raw)
            obs = env.get_observations()
            near, sat, vel, steps = 0.0, 0.0, 0.0, 0
            while len(records) < raw.num_envs and steps < raw.max_episode_length + 5:
                if name == "zero":
                    actions = torch.zeros((raw.num_envs, env_cfg.action_space), device=raw.device)
                elif name == "random":
                    actions = 2 * torch.rand((raw.num_envs, env_cfg.action_space), device=raw.device) - 1
                else:
                    actions = policy(obs)
                if telemetry is not None:
                    telemetry.mu = actions
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
            # --- stream metrics. first_goal_held_success_rate is the stage gate; it is the same
            # quantity as held_success_rate but named for what it means in a chain of goals.
            "first_goal_held_success_rate": mean("held"),
            "goals_completed_per_episode": mean("goals"),
            "max_consecutive_goals": max(v["goals"] for v in values),
            "target_completion_rate": sum(v["goals"] for v in values) / max(1, sum(v["issued"] for v in values)),
            "mean_time_per_completed_goal_s": (
                sum(v["steps"] * raw.step_dt / v["goals"] for v in values if v["goals"])
                / max(1, sum(1 for v in values if v["goals"]))),
            "commanded_rotation_deg_per_episode": mean("goals") * getattr(env_cfg, "goal_stream_angle_deg", 0.0),
            "mean_tip_contacts": mean("tips"),
            # Net signed rotation about the target axis (see env.axis_rotation_*). With stage M the
            # goals all point one way, so goals x angle and this integral should agree; with a
            # reversible stream they need not, and the gap is the rocking.
            "axis_rotation_palm_deg": mean("axis_palm"),
            "axis_rotation_body_deg": mean("axis_body"),
            "axis_rotation_palm_abs_deg": sum(abs(v["axis_palm"]) for v in values) / n,
            "axis_rotation_body_abs_deg": sum(abs(v["axis_body"]) for v in values) / n,
            "axis_rotation_palm_deg_not_dropped": sum(v["axis_palm"] for v in kept) / max(1, len(kept)),
            "goals_histogram": {str(k): sum(1 for v in values if v["goals"] == k)
                                for k in sorted({v["goals"] for v in values})},
            "per_episode": {key: [round(v[key], 2) if isinstance(v[key], float) else v[key] for v in values]
                            for key in ("goals", "drop", "steps", "axis_palm", "axis_body")},
            "action_std": action_std if path is not None else None,
            "stage": getattr(env_cfg, "goal_stream_stage", None) if getattr(env_cfg, "goal_stream", False) else None,
        }
        r = results[name]
        r["actor_output_fingerprint"] = fingerprint if path is not None else None
        if gait is not None:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            thresholds = json.loads(args_cli.gait_thresholds.read_text()) if args_cli.gait_thresholds else None
            r["gait"] = gait.save(args_cli.output.with_name(args_cli.output.stem + '_' + name.replace('/', '_') + '_trajectory.npz'), thresholds)
            gait = None
        if telemetry is not None:
            r["smoothing"] = telemetry.summary()
            telemetry = None
        print(f"[EVAL] {name:>18} n={n} held={r['held_success_rate']:.3f} entered={r['entered_tolerance_rate']:.3f}"
              f" in_tol={r['time_in_tolerance_fraction']:.3f} drop={r['drop_rate']:.3f}"
              f" final={r['final_error_deg']:.1f} final_kept={r['final_error_deg_not_dropped']:.1f}"
              f" mean={r['mean_error_deg']:.1f} min={r['min_error_deg']:.1f} reward={r['episode_reward']:.2f}"
              f" near_pm1={r['action_near_limit_fraction']:.3f} sat={r['torque_saturation_fraction']:.3f}"
              f" maxvel={r['max_abs_joint_velocity']:.2f}", flush=True)
        print(f"[EVAL] {'':>18}   stream: goals/ep={r['goals_completed_per_episode']:.2f}"
              f" max_consecutive={r['max_consecutive_goals']}"
              f" completion={r['target_completion_rate']:.3f}"
              f" s/goal={r['mean_time_per_completed_goal_s']:.2f}"
              f" tips={r['mean_tip_contacts']:.2f}"
              f" axis_palm={r['axis_rotation_palm_deg']:+.1f}deg axis_body={r['axis_rotation_body_deg']:+.1f}deg"
              f" goals_hist={r['goals_histogram']}", flush=True)
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
