"""Summarize an RSL-RL run from its TensorBoard events (no simulator needed).

Writes <run>/analysis/{metrics.csv, summary.txt} and one PNG per learning curve. Only scalars that RSL-RL /
the env actually logged are used; missing ones are reported as missing.
"""

import argparse
import csv
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "held_success_rate": "Episode/episode_held_success_rate",
    "entered_tolerance_rate": "Episode/episode_entered_tolerance_rate",
    "final_orientation_error_deg": "Episode/episode_final_orientation_error_deg",
    "mean_orientation_error_deg": "Episode/episode_mean_orientation_error_deg",
    "min_orientation_error_deg": "Episode/episode_min_orientation_error_deg",
    "time_in_tolerance_fraction": "Episode/episode_time_in_tolerance_fraction",
    "initial_orientation_error_deg": "Episode/episode_initial_orientation_error_deg",
    "drop_rate": "Episode/episode_drop_rate",
    "episode_reward": "Episode/episode_reward",
    "time_to_held_success_s": "Episode/episode_time_to_held_success_s",
    "train_mean_reward": "Train/mean_reward",
    "train_mean_episode_length": "Train/mean_episode_length",
    "cube_distance_from_palm": "Episode/cube_distance_from_palm",
    "orientation_state_reward": "Episode/orientation_state_reward",
    "orientation_progress_reward": "Episode/orientation_progress_reward",
    "palm_distance_penalty": "Episode/palm_distance_penalty",
    "action_penalty": "Episode/action_penalty",
    "action_rate_penalty": "Episode/action_rate_penalty",
    "success_bonus": "Episode/success_bonus",
    "drop_penalty": "Episode/drop_penalty",
    "entropy": "Loss/entropy",
    "action_std": "Policy/mean_noise_std",
    "mean_abs_action": "Episode/mean_action_abs",
    "mean_abs_command_delta_deg": "Episode/mean_abs_command_delta_deg",
    "action_near_limit_fraction": "Episode/action_near_limit_fraction",
    "raw_action_clip_fraction": "Episode/raw_action_clip_fraction",
    "value_loss": "Loss/value_function",
    "surrogate_loss": "Loss/surrogate",
    "learning_rate": "Loss/learning_rate",
    "kl": "Loss/kl",
    "torque_saturation_fraction": "Episode/torque_saturation_fraction",
    "torque_saturation_watch_joints": "Episode/torque_saturation_fraction_watch_joints",
    "mean_abs_applied_torque": "Episode/mean_abs_applied_torque",
    "max_abs_joint_velocity": "Episode/max_abs_joint_velocity",
    "velocity_rms_low_damping_joints": "Episode/velocity_rms_low_damping_joints",
    "disabled_joint_abs_deviation_deg": "Episode/disabled_joint_abs_deviation_deg",
}
ABS_PREFIX = "abs_contribution/"
# One figure per question; several series only where they share a meaning and unit.
FIGURES = [
    ("held_success", "held success rate (5 deg for 0.3 s)", ["held_success_rate", "entered_tolerance_rate"]),
    ("final_error", "orientation error [deg]", ["final_orientation_error_deg", "mean_orientation_error_deg",
                                                "initial_orientation_error_deg"]),
    ("time_in_tolerance", "fraction of episode inside 5 deg", ["time_in_tolerance_fraction"]),
    ("episode_reward", "episode reward", ["episode_reward"]),
    ("entropy", "policy entropy", ["entropy"]),
    ("action_std", "action std", ["action_std"]),
    ("action_near_limit", "fraction of actions with |a| >= 0.95", ["action_near_limit_fraction"]),
    ("torque_saturation", "torque saturation fraction", ["torque_saturation_fraction",
                                                         "torque_saturation_watch_joints"]),
    ("drop_rate", "drop rate", ["drop_rate"]),
]


def smooth(values, window):
    out = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1):i + 1] if math.isfinite(v)]
        out.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=20, help="Trailing mean window (iterations).")
    parser.add_argument("--marks", default="0,100,200,300,400,499")
    args = parser.parse_args()
    events = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
    events.Reload()
    available = set(events.Tags()["scalars"])
    data, missing = {}, []
    for name, tag in TAGS.items():
        if tag in available:
            data[name] = {e.step: e.value for e in events.Scalars(tag)}
        else:
            missing.append(f"{name} ({tag})")
    joint_tags = sorted(t for t in available if t.startswith(("torque_saturation_fraction/", "velocity_rms/", ABS_PREFIX)))
    for tag in joint_tags:
        data[tag] = {e.step: e.value for e in events.Scalars(tag)}
    steps = sorted(set().union(*(d.keys() for d in data.values())))
    out = args.run_dir / "analysis"
    out.mkdir(exist_ok=True)
    columns = list(data)
    with open(out / "metrics.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", *columns])
        for step in steps:
            writer.writerow([step, *(data[c].get(step, "") for c in columns)])

    def series(name):
        return [data.get(name, {}).get(s, math.nan) for s in steps]

    lines = [f"run: {args.run_dir}", f"iterations logged: {steps[0]}..{steps[-1]}",
             f"trailing mean window: {args.window} iterations",
             "missing (not logged by RSL-RL/env): " + (", ".join(missing) or "none")]
    marks = [int(m) for m in args.marks.split(",")]
    header = ["metric", *[f"it{m}" for m in marks]]
    rows = []
    for name in TAGS:
        if name not in data:
            continue
        smoothed = smooth(series(name), args.window)
        values = []
        for mark in marks:
            index = min(range(len(steps)), key=lambda i: abs(steps[i] - mark))
            # Iteration 0 is a single untrained sample; later marks use the trailing mean.
            values.append(series(name)[index] if mark == 0 else smoothed[index])
        rows.append([name, *[f"{v:.4g}" for v in values]])
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    last = [s for s in steps if s > steps[-1] - args.window]
    for prefix in ("torque_saturation_fraction/", "velocity_rms/", ABS_PREFIX):
        per_joint = {t[len(prefix):]: sum(data[t].get(s, 0) for s in last) / len(last)
                     for t in joint_tags if t.startswith(prefix)}
        worst = sorted(per_joint.items(), key=lambda kv: -kv[1])[:6]
        lines.append(f"worst {prefix[:-1]} (last {len(last)} it): " +
                     ", ".join(f"{k}={v:.3f}" for k, v in worst))
    checkpoints = sorted(int(p.stem.split("_")[1]) for p in args.run_dir.glob("model_*.pt"))
    lines.append(f"checkpoints: {checkpoints}")
    ranking = []
    for ckpt in checkpoints:
        window = [s for s in steps if ckpt - args.window < s <= ckpt] or [ckpt]
        sr = [data["held_success_rate"].get(s, math.nan) for s in window]
        fe = [data["final_orientation_error_deg"].get(s, math.nan) for s in window]
        dr = [data["drop_rate"].get(s, math.nan) for s in window]
        mean = lambda v: sum(x for x in v if math.isfinite(x)) / max(1, sum(math.isfinite(x) for x in v))
        ranking.append((ckpt, mean(sr), mean(fe), mean(dr)))
    lines.append("checkpoint task performance (training-rollout trailing mean, not a separate evaluation):")
    for ckpt, sr, fe, dr in ranking:
        lines.append(f"  model_{ckpt}: held_success={sr:.3f} final_err_deg={fe:.2f} drop={dr:.3f}")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for filename, title, names in FIGURES:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.set_title(title)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=0.3)
        for color, name in enumerate(names):
            if name not in data:
                continue
            raw = series(name)
            ax.plot(steps, raw, color=f"C{color}", alpha=0.25, linewidth=0.8)
            ax.plot(steps, smooth(raw, args.window), color=f"C{color}", linewidth=2,
                    label=f"{name} (mean of {args.window})")
        ax.legend(fontsize=8)
        fig.savefig(out / f"{filename}.png", dpi=110)
        plt.close(fig)
    print(f"[ANALYSIS] wrote {out}/metrics.csv, summary.txt and {len(FIGURES)} PNG curves")


if __name__ == "__main__":
    main()
