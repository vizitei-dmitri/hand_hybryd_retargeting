"""Zero-action grasp hold check and a small grasp/preload search (no training).

validate: reset from the configured grasp (with the configured reset noise) and hold.
search:   one env per candidate. Candidate = flexion-joint command offset (closing +),
          cube offset along the palm normal. Phase 1 starts the fingers opened by
          --open_deg and lets them close on the cube under the candidate command.
          Phase 2 resets to that settled joint state / cube position (cube orientation
          reset to the palm frame, as the cfg does) and holds with zero actions. The final
          command adds a bounded preload on flexion joints only.
The chosen values are printed as a DG5FCubeEnvCfg snippet; they are stored in the cfg,
not here.
"""

import argparse
import itertools
import math
import traceback
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="DG5F-Cube-Direct-v0")
parser.add_argument("--mode", choices=("validate", "search"), default="validate")
parser.add_argument("--num_envs", type=int, default=4, help="validate only")
parser.add_argument("--hold_s", type=float, default=4.0)
parser.add_argument("--settle_s", type=float, default=3.0)
parser.add_argument("--open_deg", type=float, default=15.0)
parser.add_argument("--curl_offsets_deg", default="-6,-3,0,3")
parser.add_argument("--cube_offsets_mm", default="-3,0,3")
parser.add_argument("--preload_caps_deg", default="0.5,1,2",
                    help="Final command = settled state + clamp(candidate command - settled, +-cap).")
parser.add_argument("--base_grasp_deg", type=float, nargs=20,
                    default=(10, -90, 45, 35, 0, 40, 65, 30, 0, 35, 65, 30, 0, 35, 65, 30, 0, -10, 55, 40),
                    help="Hand-designed starting grasp (URDF order) that the search perturbs.")
parser.add_argument("--base_cube_position_in_palm", type=float, nargs=3, default=(0.055, 0.0, 0.100))
parser.add_argument("--extra_joint", default="rj_dg_5_2", help="Joint with an extra offset grid (state and command).")
parser.add_argument("--extra_offsets_deg", default="0")
parser.add_argument("--hold_noise", action="store_true",
                    help="Use the cfg reset noise in the final scored hold (robustness), settle stays noise-free.")
parser.add_argument("--repeats", type=int, default=1, help="Envs per candidate (use with --hold_noise).")
parser.add_argument("--refine_iters", type=int, default=1,
                    help="Re-seat state/cube at the end of a hold and re-bound the preload.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401

# Positive = closing, as documented in src/lerobot_robot_dg5f/lerobot_robot_dg5f/constants.py.
FLEXION_JOINTS = (
    "rj_dg_1_1", "rj_dg_1_3", "rj_dg_1_4", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
    "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
    "rj_dg_5_3", "rj_dg_5_4",
)
NEAR_LIMIT = 0.95


def tip_surface_distance(raw):
    """Signed distance (m) from each fingertip body origin to the cube box surface."""
    tips = raw.hand.data.body_pos_w[:, raw.tip_ids] - raw.cube.data.root_pos_w[:, None]
    quat = raw.cube.data.root_quat_w[:, None].expand(-1, tips.shape[1], -1)
    local = quat_apply_inverse(quat.reshape(-1, 4), tips.reshape(-1, 3)).reshape(tips.shape)
    q = local.abs() - raw.cfg.cube_size_m / 2
    outside = q.clamp_min(0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp_max(0)
    return outside + inside


def hold(env, steps):
    """Zero actions; returns per-env metrics relative to the post-reset state."""
    raw = env.unwrapped
    ids = raw.joint_ids
    active = raw.active_joint_ids
    q0 = raw.hand.data.joint_pos[:, active].clone()
    cube0 = raw.cube_pos.clone()
    quat0 = raw.cube_quat.clone()
    command0 = raw.joint_command.clone()
    limits = raw.torque_limits
    abs_torque = torch.zeros_like(raw.torque_limits)
    near_limit = torch.zeros_like(raw.torque_limits)
    max_drift = torch.zeros_like(q0)
    max_cube = torch.zeros(raw.num_envs, device=raw.device)
    max_rot = torch.zeros(raw.num_envs, device=raw.device)
    fell = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    max_speed = torch.zeros(raw.num_envs, device=raw.device)
    zero = torch.zeros((raw.num_envs, raw.cfg.action_space), device=raw.device)
    start = time.monotonic()
    with torch.inference_mode():
        for step in range(steps):
            _, _, terminated, truncated, _ = env.step(zero)
            fell |= terminated | truncated
            drift = (raw.hand.data.joint_pos[:, active] - q0).abs()
            max_drift = torch.maximum(max_drift, drift)
            max_cube = torch.maximum(max_cube, (raw.cube_pos - cube0).norm(dim=-1))
            max_rot = torch.maximum(max_rot, quat_error_magnitude(raw.cube_quat, quat0))
            applied = raw.hand.data.applied_torque[:, ids]
            abs_torque += applied.abs()
            near_limit += (applied.abs() >= NEAR_LIMIT * limits).float()
            if step % 60 == 0:
                print(f"[HOLD] progress step={step}/{steps} t={time.monotonic() - start:.1f}s", flush=True)
            if step >= int(0.5 / raw.step_dt):  # after the first 0.5 s transient
                max_speed = torch.maximum(max_speed, raw.hand.data.joint_vel[:, ids].abs().amax(dim=-1))
        assert torch.equal(raw.joint_command, command0), "q_cmd changed under zero actions"
    return {
        "max_joint_drift_deg": torch.rad2deg(max_drift).amax(dim=-1),
        "final_joint_drift_deg": torch.rad2deg((raw.hand.data.joint_pos[:, active] - q0).abs()).amax(dim=-1),
        "per_joint_max_drift_deg": torch.rad2deg(max_drift),
        "max_cube_disp_mm": 1000 * max_cube,
        "max_cube_rot_deg": torch.rad2deg(max_rot),
        "mean_abs_torque": abs_torque / steps,
        "near_limit_fraction": near_limit / steps,
        "computed_torque": raw.hand.data.computed_torque[:, ids].clone(),
        "applied_torque": raw.hand.data.applied_torque[:, ids].clone(),
        "tip_distance_mm": 1000 * tip_surface_distance(raw),
        "max_speed_after_0.5s": max_speed,
        "fell": fell,
    }


def report(tag, raw, metrics, env_id):
    names = raw.cfg.actuated_joint_names
    active_names = raw.cfg.active_action_joints
    m = {k: v[env_id] for k, v in metrics.items()}
    worst = torch.argsort(m["per_joint_max_drift_deg"], descending=True)[:4].tolist()
    print(f"[HOLD] {tag} fell={bool(m['fell'])} max_joint_drift_deg={m['max_joint_drift_deg']:.3f}"
          f" final={m['final_joint_drift_deg']:.3f} worst=" +
          ",".join(f"{active_names[i]}:{m['per_joint_max_drift_deg'][i]:.2f}" for i in worst))
    print(f"[HOLD] {tag} cube_disp_mm={m['max_cube_disp_mm']:.2f} cube_rot_deg={m['max_cube_rot_deg']:.2f}"
          f" max|qdot|(t>0.5s)={m['max_speed_after_0.5s']:.3f} rad/s"
          f" tip_to_cube_surface_mm={[round(v, 1) for v in m['tip_distance_mm'].tolist()]}")
    print(f"[HOLD] {tag} mean_abs_applied_torque={m['mean_abs_torque'].mean():.3f} Nm"
          f" near_limit_fraction(>={NEAR_LIMIT:.0%})={m['near_limit_fraction'].mean():.3f}")
    print(f"[HOLD] {tag} per_joint name:mean|tau|/near_limit/computed/applied " + " ".join(
        f"{n}:{m['mean_abs_torque'][i]:.3f}/{m['near_limit_fraction'][i]:.2f}/"
        f"{m['computed_torque'][i]:+.3f}/{m['applied_torque'][i]:+.3f}" for i, n in enumerate(names)))


def validate(env):
    raw = env.unwrapped
    env.reset()
    metrics = hold(env, int(args_cli.hold_s / raw.step_dt))
    for env_id in range(raw.num_envs):
        report(f"validate env={env_id}", raw, metrics, env_id)
    agg = {k: metrics[k].amax().item() for k in ("max_joint_drift_deg", "max_cube_disp_mm", "max_cube_rot_deg")}
    print(f"[HOLD] validate SUMMARY hold_s={args_cli.hold_s} fell={int(metrics['fell'].sum())}/{raw.num_envs} "
          f"worst_max_joint_drift_deg={agg['max_joint_drift_deg']:.3f} worst_cube_disp_mm={agg['max_cube_disp_mm']:.2f}"
          f" worst_cube_rot_deg={agg['max_cube_rot_deg']:.2f}"
          f" near_limit_fraction={metrics['near_limit_fraction'].mean().item():.3f}"
          f" max|qdot|(t>0.5s)={metrics['max_speed_after_0.5s'].amax().item():.3f}")


def search(env, candidates, noise):
    raw = env.unwrapped
    cfg = raw.cfg
    names = cfg.actuated_joint_names
    flex = torch.tensor([name in FLEXION_JOINTS for name in names], device=raw.device)
    lower = raw.hand.data.joint_pos_limits[0, raw.joint_ids, 0]
    upper = raw.hand.data.joint_pos_limits[0, raw.joint_ids, 1]
    base_command = torch.tensor([math.radians(v) for v in args_cli.base_grasp_deg], device=raw.device)
    base_command[raw.disabled_index] = cfg.disabled_joint_position
    base_anchor = torch.tensor(args_cli.base_cube_position_in_palm, device=raw.device)
    raw.cube_anchor = base_anchor.expand(raw.num_envs, -1).clone()
    extra = torch.tensor([name == args_cli.extra_joint for name in names], device=raw.device)
    for env_id, (curl, dx, _, extra_deg) in enumerate(candidates):
        command = torch.clamp(base_command + flex * math.radians(curl) + extra * math.radians(extra_deg), lower, upper)
        command[raw.disabled_index] = cfg.disabled_joint_position
        raw.grasp_command[env_id] = command
        state = torch.clamp(command - flex * math.radians(args_cli.open_deg), lower, upper)
        raw.hand.data.default_joint_pos[env_id, raw.joint_ids] = state
        raw.cube_anchor[env_id, 0] = base_anchor[0] + dx / 1000
    env.reset()
    settle_fell = hold(env, int(args_cli.settle_s / raw.step_dt))["fell"]
    settled_q = raw.hand.data.joint_pos[:, raw.joint_ids].clone()
    settled_q[:, raw.disabled_index] = cfg.disabled_joint_position
    settled_cube = raw.cube_pos.clone()
    settled_rot = torch.rad2deg(quat_error_magnitude(raw.cube_quat, torch.tensor(
        [[1.0, 0, 0, 0]], device=raw.device).expand(raw.num_envs, -1)))
    target_command = raw.grasp_command.clone()
    for iteration in range(1 + args_cli.refine_iters):
        if iteration:
            settled_q = raw.hand.data.joint_pos[:, raw.joint_ids].clone()
            settled_q[:, raw.disabled_index] = cfg.disabled_joint_position
            settled_cube = raw.cube_pos.clone()
        if iteration == args_cli.refine_iters and args_cli.hold_noise:
            cfg.reset_joint_position_noise_rad, cfg.cube_reset_position_noise_m = noise
        for env_id, (_, _, cap, _) in enumerate(candidates):
            raw.hand.data.default_joint_pos[env_id, raw.joint_ids] = settled_q[env_id]
            raw.cube_anchor[env_id] = settled_cube[env_id]
            # Bounded preload on FLEXION joints only (they press the cube). Abduction/opposition
            # joints command their settled angle, so neighbouring fingers do not squeeze each other.
            preload = (target_command[env_id] - settled_q[env_id]).clamp(-math.radians(cap), math.radians(cap))
            preload = preload * flex
            raw.grasp_command[env_id] = torch.clamp(settled_q[env_id] + preload, lower, upper)
        env.reset()
        metrics = hold(env, int(args_cli.hold_s / raw.step_dt))
        fell = metrics["fell"] | settle_fell
        print(f"[SEARCH] refine iteration {iteration}: median max_joint_drift_deg="
              f"{metrics['max_joint_drift_deg'][~fell].median().item():.3f}")
        settle_fell = fell
    scores = []
    for env_id, (curl, dx, cap, extra_deg) in enumerate(candidates):
        tag = f"curl={curl:+g}deg cube_dx={dx:+g}mm preload_cap={cap:g}deg {args_cli.extra_joint}={extra_deg:+g}deg"
        print(f"[SEARCH] {tag} settle: cube_palm_mm={[round(1000 * v, 1) for v in settled_cube[env_id].tolist()]}"
              f" cube_rot_deg={settled_rot[env_id]:.2f} settle_fell={bool(settle_fell[env_id])}"
              f" preload_deg(cmd-state) max={math.degrees((raw.grasp_command[env_id] - settled_q[env_id]).abs().max()):.2f}")
        report(tag, raw, metrics, env_id)
        m = {k: v[env_id].item() if v[env_id].numel() == 1 else v[env_id] for k, v in metrics.items()}
        ok = not (m["fell"] or settle_fell[env_id])
        # Hold quality first, then low saturation, then a nontrivial grasp (fingers near the surface).
        score = (m["max_joint_drift_deg"] + m["max_cube_disp_mm"] + m["max_cube_rot_deg"]
                 + 20 * m["near_limit_fraction"].mean().item()
                 + 0.2 * m["tip_distance_mm"].clamp_min(0).mean().item())
        scores.append((score if ok else math.inf, env_id))
    # Repeated candidates (noisy holds) are ranked by their worst replica.
    worst = {}
    for score, env_id in scores:
        key = candidates[env_id]
        if key not in worst or score > worst[key][0]:
            worst[key] = (score, candidates.index(key))
    scores = sorted(worst.values())
    print("[SEARCH] ranking (lower is better): " + ", ".join(
        "curl={:+g}/dx={:+g}/cap={:g}/extra={:+g}:{:.2f}".format(*candidates[i], s) for s, i in scores))
    best = scores[0][1]
    curl, dx, cap, extra_deg = candidates[best]
    state_deg = [round(math.degrees(v), 2) for v in settled_q[best].tolist()]
    offset_deg = [round(math.degrees(v), 2) for v in (raw.grasp_command[best] - settled_q[best]).tolist()]
    cube = [round(v, 4) for v in settled_cube[best].tolist()]
    print(f"[SEARCH] BEST curl={curl:+g}deg cube_dx={dx:+g}mm preload_cap={cap:g}deg"
          f" {args_cli.extra_joint}={extra_deg:+g}deg score={scores[0][0]:.3f}")
    print(f"[SEARCH] cfg grasp_joint_pos_deg = {tuple(state_deg)}")
    print(f"[SEARCH] cfg grasp_command_offset_deg = {tuple(offset_deg)}")
    print(f"[SEARCH] cfg cube_position_in_palm = {tuple(cube)}")


def main():
    curls = [float(v) for v in args_cli.curl_offsets_deg.split(",")]
    offsets = [float(v) for v in args_cli.cube_offsets_mm.split(",")]
    caps = [float(v) for v in args_cli.preload_caps_deg.split(",")]
    extras = [float(v) for v in args_cli.extra_offsets_deg.split(",")]
    candidates = list(itertools.product(curls, offsets, caps, extras)) * args_cli.repeats
    num_envs = len(candidates) if args_cli.mode == "search" else args_cli.num_envs
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    cfg.episode_length_s = max(cfg.episode_length_s, args_cli.hold_s + args_cli.settle_s + 1)
    noise = (cfg.reset_joint_position_noise_rad, cfg.cube_reset_position_noise_m)
    if args_cli.mode == "search":
        cfg.reset_joint_position_noise_rad = 0.0
        cfg.cube_reset_position_noise_m = 0.0
    env = gym.make(args_cli.task, cfg=cfg)
    print(f"[HOLD] mode={args_cli.mode} control_mode={cfg.control_mode} num_envs={num_envs}"
          f" effort_limit={env.unwrapped.torque_limits[0, 0].item():.3f} Nm")
    # Isaac Lab buffers become inference tensors inside hold(); keep every reset inside too.
    with torch.inference_mode():
        if args_cli.mode == "search":
            search(env, candidates, noise)
        else:
            validate(env)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Print before closing: Kit shutdown can hang after a failure and swallow the traceback.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
