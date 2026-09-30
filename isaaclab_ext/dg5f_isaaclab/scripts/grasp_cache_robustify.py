"""Score every cached fingertip grasp under perturbation and write a ROBUST subset.

The existing cache was accepted on a static 2 s release hold, and 19.5% of its entries still fail
a full 24 s zero-action episode: a fifth of resets are fragile before RL does anything. This is the
difference from the pipeline of 2607.12105, which keeps a grasp only if it survives perturbation.

Each grasp is replayed `--trials` times with independent reset noise (joint, cube position, cube
orientation) and held with zero actions for `--hold_s`. Perturbation is deliberately LARGER than
the training reset noise (0.5 deg / 1 mm / 0 deg): a grasp that only survives the exact snapshot it
was found in is not a usable reset state.

The original cache is never overwritten; the subset is a separate artefact, so both remain
reproducible and a run can state which one it used.
"""

import argparse
import json
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="DG5F-Cube-Fingertip-Direct-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--trials", type=int, default=8, help="Independent perturbations per grasp")
parser.add_argument("--hold_s", type=float, default=4.0)
parser.add_argument("--cache", default=None, help="Defaults to the task's configured cache")
parser.add_argument("--out", type=Path,
                    default=Path("source/dg5f_isaaclab/dg5f_isaaclab/assets/data/fingertip_grasp_cache_robust.npz"))
parser.add_argument("--report", type=Path, default=Path("logs/grasp_robustness/report.json"))
# Perturbation, ~2-5x the training reset noise so the score means something.
parser.add_argument("--joint_noise_deg", type=float, default=1.0)
parser.add_argument("--cube_pos_noise_mm", type=float, default=3.0)
parser.add_argument("--cube_quat_noise_deg", type=float, default=5.0)
# Acceptance criteria.
parser.add_argument("--min_survival", type=float, default=0.75)
parser.add_argument("--max_palm_fraction", type=float, default=0.05,
                    help="Persistent palm support disqualifies a grasp; the full cache averages 0.008")
parser.add_argument("--min_mean_tips", type=float, default=2.8,
                    help="Sustained average, NOT the task's instantaneous min_tip_contacts=3: the "
                         "measured distribution over the full cache has median 2.72 and p90 3.00, "
                         "so a threshold of 3.0 sits above the 88th percentile and rejected 790 of "
                         "897 otherwise healthy grasps")
parser.add_argument("--from_report", type=Path, default=None,
                    help="Re-select from an existing report instead of simulating again: the "
                         "criteria are a judgement call and re-running 7176 trials to move a "
                         "threshold would be waste. Skips Isaac entirely.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True


def select(survival, mean_tips, palm_fraction):
    """The acceptance mask, shared by the simulating and the re-selecting paths."""
    import numpy as _np
    return (_np.asarray(survival) >= args_cli.min_survival) \
        & (_np.asarray(palm_fraction) <= args_cli.max_palm_fraction) \
        & (_np.asarray(mean_tips) >= args_cli.min_mean_tips)


if args_cli.from_report is not None:
    # No simulator on this path: it only re-applies the thresholds to measurements already taken.
    import json as _json

    import numpy as _np

    report = _json.loads(args_cli.from_report.read_text())
    stats = report["per_grasp"]
    keep = select(stats["survival"], stats["mean_tips"], stats["palm_fraction"])
    with _np.load(report["source_cache"], allow_pickle=False) as data:
        arrays = {name: data[name] for name in ("q", "q_cmd", "cube_pos", "cube_quat", "joint_names")}
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    _np.savez(args_cli.out, q=arrays["q"][keep], q_cmd=arrays["q_cmd"][keep],
              cube_pos=arrays["cube_pos"][keep], cube_quat=arrays["cube_quat"][keep],
              joint_names=arrays["joint_names"])
    survival = _np.asarray(stats["survival"])
    print(f"[ROBUST] re-selected from {args_cli.from_report}: kept {int(keep.sum())}/{len(keep)}"
          f" ({keep.mean():.1%}) survival(kept)={survival[keep].mean():.3f}"
          f" criteria: survival>={args_cli.min_survival} palm<={args_cli.max_palm_fraction}"
          f" tips>={args_cli.min_mean_tips}")
    print(f"[ROBUST] wrote {args_cli.out}")
    report["criteria"] = {"min_survival": args_cli.min_survival,
                          "max_palm_fraction": args_cli.max_palm_fraction,
                          "min_mean_tips": args_cli.min_mean_tips}
    report["grasps_kept"] = int(keep.sum())
    report["per_grasp"]["kept"] = keep.tolist()
    args_cli.from_report.write_text(_json.dumps(report, indent=1))
    raise SystemExit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401
from dg5f_isaaclab.assets.dg5f import DG5F_JOINT_LIMITS
from dg5f_isaaclab.assets.grasp_cache import load_grasp_cache
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env import DG5FCubeEnv


class PrescribedResetEnv(DG5FCubeEnv):
    """Resets each env to a prescribed cache entry instead of a random one."""

    def _setup_scene(self):
        super()._setup_scene()
        # Must exist before __init__ runs a reset through the base class.
        self.forced_picks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _pick_grasps(self, env_ids):
        return self.forced_picks[env_ids]


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.cache is not None:
        cfg.grasp_cache_path = args_cli.cache
    assert cfg.grasp_cache_path is not None and cfg.enable_contact_sensors, "not the fingertip task"
    source_cache = Path(cfg.grasp_cache_path).resolve()
    # The perturbation is the whole point of this script, so it overrides the task's reset noise.
    cfg.reset_joint_position_noise_rad = math.radians(args_cli.joint_noise_deg)
    cfg.cube_reset_position_noise_m = args_cli.cube_pos_noise_mm / 1000.0
    cfg.cube_reset_orientation_noise_deg = args_cli.cube_quat_noise_deg

    env = PrescribedResetEnv(cfg)
    raw_count = env.grasp_cache["joint_pos"].shape[0]
    hold_steps = max(1, round(args_cli.hold_s / env.step_dt))
    envs = env.num_envs
    zero = torch.zeros((envs, cfg.action_space), device=env.device)
    min_tips = args_cli.min_mean_tips

    print(f"[ROBUST] cache={source_cache} n={raw_count} trials={args_cli.trials}"
          f" hold={args_cli.hold_s}s={hold_steps} steps envs={envs}"
          f" noise: joint={args_cli.joint_noise_deg} deg cube={args_cli.cube_pos_noise_mm} mm"
          f" quat={args_cli.cube_quat_noise_deg} deg", flush=True)

    # Grasp g appears at positions g, n+g, 2n+g ..., i.e. once per trial with independent noise.
    jobs = torch.arange(raw_count, device=env.device).repeat(args_cli.trials)
    totals = {name: np.zeros(raw_count) for name in
              ("trials", "survived", "tips", "palm", "drift_mm", "saturation")}

    with torch.inference_mode():
        for start in range(0, jobs.numel(), envs):
            chunk = jobs[start:start + envs]
            real = chunk.numel()
            if real < envs:
                # Padding keeps the batch shape; padded envs are simply not recorded. Tiled, because
                # the final chunk can be far smaller than the padding it needs.
                repeats = -(-(envs - real) // real)
                chunk = torch.cat((chunk, chunk.repeat(repeats)[: envs - real]))
            env.forced_picks.copy_(chunk)
            env.reset()

            dropped = torch.zeros(envs, dtype=torch.bool, device=env.device)
            alive_steps = torch.zeros(envs, device=env.device)
            tips = torch.zeros(envs, device=env.device)
            palm = torch.zeros(envs, device=env.device)
            saturation = torch.zeros(envs, device=env.device)
            drift = torch.zeros(envs, device=env.device)
            for _ in range(hold_steps):
                _, _, terminated, _, extras = env.step(zero)
                # Read BEFORE folding in this step's termination: an env that dies on this step was
                # alive during it, and its auto-reset only happens on the next step.
                alive = (~dropped).float()
                tips += env.tip_contact_count.float() * alive
                palm += (env.palm_contact_force > cfg.tip_contact_force_n).float() * alive
                computed = extras["computed_torque"]
                saturated = (computed.abs() >= env.torque_limits * (1 - 1e-6)).float().mean(dim=-1)
                saturation += saturated * alive
                drift = torch.where(alive.bool(), env.palm_region_distance, drift)
                alive_steps += alive
                dropped |= terminated

            steps = alive_steps.clamp_min(1.0)
            index = chunk[:real].cpu().numpy()
            np.add.at(totals["trials"], index, 1.0)
            np.add.at(totals["survived"], index, (~dropped)[:real].float().cpu().numpy())
            np.add.at(totals["tips"], index, (tips / steps)[:real].cpu().numpy())
            np.add.at(totals["palm"], index, (palm / steps)[:real].cpu().numpy())
            np.add.at(totals["drift_mm"], index, (1000 * drift)[:real].cpu().numpy())
            np.add.at(totals["saturation"], index, (saturation / steps)[:real].cpu().numpy())
            done = min(start + envs, jobs.numel())
            print(f"[ROBUST] {done}/{jobs.numel()} trials  chunk survived="
                  f"{float((~dropped)[:real].float().mean()):.3f}", flush=True)

    trials = totals["trials"]
    assert np.all(trials > 0), "every grasp must be tried"
    survival = totals["survived"] / trials
    mean_tips = totals["tips"] / trials
    palm_fraction = totals["palm"] / trials
    drift_mm = totals["drift_mm"] / trials
    saturation = totals["saturation"] / trials

    keep = select(survival, mean_tips, palm_fraction)
    kept = int(keep.sum())
    print(f"[ROBUST] survival mean={survival.mean():.3f}"
          f" >=0.5: {(survival >= 0.5).mean():.1%} >=0.75: {(survival >= 0.75).mean():.1%}"
          f" ==1.0: {(survival >= 0.999).mean():.1%}", flush=True)
    print(f"[ROBUST] rejected by survival={int((survival < args_cli.min_survival).sum())}"
          f" by palm={int((palm_fraction > args_cli.max_palm_fraction).sum())}"
          f" by tips={int((mean_tips < min_tips).sum())}", flush=True)
    print(f"[ROBUST] KEPT {kept}/{raw_count} ({kept / raw_count:.1%})", flush=True)

    if kept == 0:
        raise RuntimeError("No grasp passed; relax the criteria rather than shipping an empty cache")

    # Same schema and joint order as the source, so load_grasp_cache validates it unchanged.
    # Loaded WITH the limits, as the env does, so the stored entries are the ones actually tested.
    cache = load_grasp_cache(source_cache, cfg.actuated_joint_names, DG5F_JOINT_LIMITS)
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args_cli.out,
        q=cache.joint_pos[keep], q_cmd=cache.joint_command[keep],
        cube_pos=cache.cube_pos[keep], cube_quat=cache.cube_quat[keep],
        joint_names=np.array(cache.joint_names),
    )
    print(f"[ROBUST] wrote {args_cli.out} with {kept} grasps", flush=True)

    args_cli.report.parent.mkdir(parents=True, exist_ok=True)
    args_cli.report.write_text(json.dumps({
        "source_cache": str(source_cache),
        "source_sha256": cache.sha256,
        "output_cache": str(args_cli.out),
        "grasps_in": raw_count,
        "grasps_kept": kept,
        "trials_per_grasp": args_cli.trials,
        "hold_s": args_cli.hold_s,
        "perturbation": {"joint_deg": args_cli.joint_noise_deg,
                         "cube_position_mm": args_cli.cube_pos_noise_mm,
                         "cube_orientation_deg": args_cli.cube_quat_noise_deg},
        "criteria": {"min_survival": args_cli.min_survival,
                     "max_palm_fraction": args_cli.max_palm_fraction,
                     "min_mean_tips": min_tips},
        "summary": {"survival_mean": float(survival.mean()),
                    "survival_kept_mean": float(survival[keep].mean()),
                    "mean_tips_kept": float(mean_tips[keep].mean()),
                    "palm_fraction_kept": float(palm_fraction[keep].mean()),
                    "drift_mm_kept": float(drift_mm[keep].mean()),
                    "saturation_kept": float(saturation[keep].mean())},
        "per_grasp": {"survival": survival.tolist(), "mean_tips": mean_tips.tolist(),
                      "palm_fraction": palm_fraction.tolist(), "drift_mm": drift_mm.tolist(),
                      "saturation": saturation.tolist(), "kept": keep.tolist()},
    }, indent=1))
    print(f"[ROBUST] wrote {args_cli.report}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
