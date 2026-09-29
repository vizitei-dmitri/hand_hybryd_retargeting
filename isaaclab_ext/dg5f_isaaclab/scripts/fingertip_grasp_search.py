"""Search for PRECISION (fingertip) grasps of the 60 mm cube; no training, no reward changes.

The configured cfg grasp is a power grasp: logs/*/04b_contact_audit.log shows the palm
carrying 0.38-1.11 N (the cube weighs 0.49 N) with the load on the middle phalanges, while
rl_dg_*_tip register 0.02-0.53 N and are absent from half the samples. Every modern
in-hand reward (fingertip contact terms, grasp-quality lambda_min(G G^T)) is undefined or
gradient-free from such a state, so a cache of real fingertip grasps has to exist first.

One candidate per env, three phases:
  A approach/close  the cube is held kinematically at the candidate anchor while the flexion
                    joints are driven toward the candidate closing command through the normal
                    control path (integrated_delta_position: action = delta on q_cmd, so the
                    FIFO delay and the joint clamps stay in force). A finger's actions stop
                    once one of its own tips registers contact.
  B release/hold    the cube is let go and the command is held with zero actions.
  C score           accept only grasps that survive B on the fingertips.

Accepted (settled q, q_cmd, cube pose in the palm frame) are written to an .npz cache for use
as a reset distribution, which is what Chen et al. (CoRL 2021) obtain from a lifting policy and
what the 2026 works call a grasp cache. Nothing here changes physics, SysID or the reward.
"""

import argparse
import math
import traceback
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="DG5F-Cube-Direct-v0")
parser.add_argument("--num_envs", type=int, default=256, help="Candidates evaluated in parallel")
parser.add_argument("--rounds", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--close_s", type=float, default=1.5, help="Phase A")
parser.add_argument("--hold_s", type=float, default=2.0, help="Phase B")
parser.add_argument("--score_s", type=float, default=0.5, help="Scoring window at the end of B")
parser.add_argument("--contact_force_n", type=float, default=0.05,
                    help="Fingertip/palm contact threshold. The cube weighs 0.49 N.")
parser.add_argument("--min_tips", type=int, default=3)
parser.add_argument("--max_palm_force_n", type=float, default=0.05)
# A precision grasp loads the tips, not the phalanges. Without this the search happily returns
# enclosing grasps that merely miss the palm: the first smoke run accepted one with 3.2 N on the
# tips and 19.7 N on the other links, i.e. the cube jammed between phalanges (it weighs 0.49 N).
parser.add_argument("--max_nontip_links", type=int, default=1,
                    help="How many non-tip, non-palm links may touch the cube. A precision grasp "
                         "loads the tips only. Counting links is used instead of a force threshold "
                         "because the absolute forces are not trustworthy: contact_audit reports "
                         "~9 N of total contact on a 0.49 N cube resting in the cradle.")
# The action FIFO delays a delta by 3 control steps, so closing at the full 3 deg/step commits
# up to 9 deg of extra travel after the tip touches; with SysID stiffness 10-13 Nm/rad the stiff PD
# then squeezes the cube (first search: 6-18 N on the phalanges while the joint torques stayed at
# 5-10% of the limit). Closing slowly bounds that overshoot.
parser.add_argument("--close_rate", type=float, default=0.15,
                    help="Fraction of delta_action_scale per step, i.e. 0.15 -> 0.45 deg/step")
parser.add_argument("--preload_deg", type=float, default=0.5,
                    help="After closing, re-seat to the settled state and command only this much "
                         "preload on the flexion joints (the pattern from grasp_hold_sanity.py).")
parser.add_argument("--block_force_n", type=float, default=0.5,
                    help="A finger stops closing once its own non-tip links take this much: a blocked "
                         "finger that keeps driving is what builds up the jammed internal forces.")
parser.add_argument("--max_cube_disp_mm", type=float, default=15.0)
parser.add_argument("--max_cube_rot_deg", type=float, default=20.0)
parser.add_argument("--max_near_limit_fraction", type=float, default=0.5)
# The key search axis: the cfg anchor sits at x=0.0548 m along the palm normal, which is close
# enough for the phalanges to wrap. Pushing the cube out leaves only the distal links in reach.
parser.add_argument("--cube_x_mm", default="55,125", help="min,max along the palm normal")
parser.add_argument("--cube_z_mm", default="62,96", help="min,max toward the fingers")
parser.add_argument("--cube_y_mm", default="-12,12")
parser.add_argument("--open_deg", default="15,55", help="Flexion opened by this much before closing")
parser.add_argument("--curl_deg", default="-25,12", help="Closing command offset on flexion joints")
parser.add_argument("--abduction_deg", default="-12,12", help="Spread offset on *_1 joints")
parser.add_argument("--thumb_deg", default="-18,18", help="Offset on rj_dg_1_1 / rj_dg_1_2")
parser.add_argument("--output", default="logs/fingertip_grasp/cache.npz")
parser.add_argument("--report_rejects", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_error_magnitude

from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401
from dg5f_isaaclab.assets.dg5f import URDF_PATH
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env import DG5FCubeEnv

import xml.etree.ElementTree as ET

# Positive = closing, as in scripts/grasp_hold_sanity.py (1_2 and 5_2 are opposition, *_1 abduction).
FLEXION_JOINTS = (
    "rj_dg_1_1", "rj_dg_1_3", "rj_dg_1_4", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
    "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
    "rj_dg_5_3", "rj_dg_5_4",
)
LINKS = [link.get("name") for link in ET.parse(URDF_PATH).getroot().findall("link")
         if link.find("collision") is not None]
ENV = "/World/envs/env_.*"
NEAR_LIMIT = 0.95


class GraspSearchEnv(DG5FCubeEnv):
    """Adds one cube-filtered ContactSensor per hand link (the proven pattern from contact_audit)."""

    def _setup_scene(self):
        super()._setup_scene()
        self.link_sensors = {}
        for link in LINKS:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"{ENV}/Robot/{link}", filter_prim_paths_expr=[f"{ENV}/Cube"]))
            self.scene.sensors[f"contact_{link}"] = sensor
            self.link_sensors[link] = sensor

    def cube_contact_forces(self) -> torch.Tensor:
        """(num_envs, len(LINKS)) contact force magnitude against the cube, N."""
        return torch.stack(
            [s.data.force_matrix_w[:, 0, 0].norm(dim=-1) for s in self.link_sensors.values()], dim=-1)


def pair(text):
    lo, hi = (float(v) for v in text.split(","))
    assert lo <= hi, text
    return lo, hi


def uniform(generator, lo, hi, shape, device):
    return torch.rand(shape, generator=generator, device=device) * (hi - lo) + lo


def sample_candidates(raw, generator):
    """Per-env pre-grasp state, closing command and cube anchor. Returns (state, command, anchor)."""
    cfg = raw.cfg
    names = cfg.actuated_joint_names
    n = raw.num_envs
    device = raw.device
    lower = raw.hand.data.joint_pos_limits[0, raw.joint_ids, 0]
    upper = raw.hand.data.joint_pos_limits[0, raw.joint_ids, 1]
    base = torch.tensor([cfg.grasp_command[name] for name in names], device=device).expand(n, -1).clone()

    flex = torch.tensor([name in FLEXION_JOINTS for name in names], device=device).float()
    abduction = torch.tensor([name.endswith("_1") and name != "rj_dg_1_1" for name in names],
                             device=device).float()
    thumb = torch.tensor([name in ("rj_dg_1_1", "rj_dg_1_2") for name in names], device=device).float()

    curl = uniform(generator, *pair(args_cli.curl_deg), (n, 1), device)
    spread = uniform(generator, *pair(args_cli.abduction_deg), (n, 1), device)
    thumb_offset = uniform(generator, *pair(args_cli.thumb_deg), (n, 1), device)
    open_by = uniform(generator, *pair(args_cli.open_deg), (n, 1), device)

    command = base + torch.deg2rad(curl * flex + spread * abduction + thumb_offset * thumb)
    # The pre-grasp only extends the flexion joints; spread/opposition already hold their value,
    # so the fingers approach the cube from an open pose instead of colliding on the way in.
    state = command - torch.deg2rad(open_by) * flex
    command = torch.clamp(command, lower, upper)
    state = torch.clamp(state, lower, upper)
    command[:, raw.disabled_index] = cfg.disabled_joint_position
    state[:, raw.disabled_index] = cfg.disabled_joint_position

    anchor = torch.stack([
        uniform(generator, *pair(args_cli.cube_x_mm), (n,), device) / 1000,
        uniform(generator, *pair(args_cli.cube_y_mm), (n,), device) / 1000,
        uniform(generator, *pair(args_cli.cube_z_mm), (n,), device) / 1000,
    ], dim=-1)
    return state, command, anchor


def close_on_cube(env, target_command, steps):
    """Phase A: hold the cube in place and drive the flexion joints in until the tips touch."""
    raw = env.unwrapped
    device = raw.device
    names = raw.cfg.actuated_joint_names
    active_names = raw.cfg.active_action_joints
    # Only flexion joints close; a frozen finger stops driving all of its own joints.
    drives = torch.tensor([name in FLEXION_JOINTS for name in active_names], device=device).float()
    finger_of_action = torch.tensor([int(name.split("_")[2]) for name in active_names], device=device)
    tip_columns = torch.tensor([LINKS.index(name) for name in raw.cfg.fingertip_body_names], device=device)
    tip_finger = torch.tensor([int(name.split("_")[2]) for name in raw.cfg.fingertip_body_names], device=device)
    # rl_dg_{finger}_{segment}: the non-tip segments of each finger, used to detect a blocked finger.
    segment_columns, segment_finger = [], []
    for column, link in enumerate(LINKS):
        parts = link.split("_")
        if len(parts) == 4 and parts[3].isdigit():
            segment_columns.append(column)
            segment_finger.append(int(parts[2]))
    segment_columns = torch.tensor(segment_columns, device=device)
    segment_finger = torch.tensor(segment_finger, device=device)

    target_active = target_command[:, raw.active_indices]
    frozen = torch.zeros((raw.num_envs, 5), dtype=torch.bool, device=device)
    # Pose the cube where the candidate wants it and keep it there while the fingers arrive.
    pose = raw.cube.data.root_pose_w.clone()
    zero_velocity = torch.zeros((raw.num_envs, 6), device=device)
    for _ in range(steps):
        remaining = target_active - raw.joint_command
        action = (remaining / raw.cfg.delta_action_scale).clamp(-1.0, 1.0) * drives * args_cli.close_rate
        finger_frozen = frozen.gather(1, (finger_of_action - 1).expand(raw.num_envs, -1))
        env.step(torch.where(finger_frozen, torch.zeros_like(action), action))
        raw.cube.write_root_pose_to_sim(pose)
        raw.cube.write_root_velocity_to_sim(zero_velocity)
        forces = raw.cube_contact_forces()
        touched = forces[:, tip_columns] > args_cli.contact_force_n
        blocked = forces[:, segment_columns] > args_cli.block_force_n
        for finger in range(5):
            frozen[:, finger] |= touched[:, tip_finger == finger + 1].any(dim=-1)
            frozen[:, finger] |= blocked[:, segment_finger == finger + 1].any(dim=-1)
    return frozen


def reseat(env, target_command):
    """Settle the squeeze out of a closed grasp and make the result a valid reset state.

    Holding the closing command keeps pressing; instead the hand re-resets to the state it
    actually settled into and commands that state plus a bounded preload on the flexion joints
    only, so neighbouring fingers stop driving into each other. Returns what the cache stores.
    """
    raw = env.unwrapped
    device = raw.device
    names = raw.cfg.actuated_joint_names
    flex = torch.tensor([name in FLEXION_JOINTS for name in names], device=device).float()
    lower = raw.hand.data.joint_pos_limits[0, raw.joint_ids, 0]
    upper = raw.hand.data.joint_pos_limits[0, raw.joint_ids, 1]

    settled_q = raw.hand.data.joint_pos[:, raw.joint_ids].clone()
    settled_q[:, raw.disabled_index] = raw.cfg.disabled_joint_position
    settled_cube = raw.cube_pos.clone()
    cap = math.radians(args_cli.preload_deg)
    preload = (target_command - settled_q).clamp(-cap, cap) * flex
    command = torch.clamp(settled_q + preload, lower, upper)
    command[:, raw.disabled_index] = raw.cfg.disabled_joint_position

    raw.hand.data.default_joint_pos[:, raw.joint_ids] = settled_q
    raw.grasp_command[:] = command
    raw.cube_anchor = settled_cube.clone()
    env.reset()
    return settled_q, command, settled_cube


def hold_and_score(env, steps, score_steps):
    """Phase B/C: release the cube, hold with zero actions and score the last score_steps."""
    raw = env.unwrapped
    device = raw.device
    n = raw.num_envs
    tip_columns = torch.tensor([LINKS.index(name) for name in raw.cfg.fingertip_body_names], device=device)
    palm_column = LINKS.index(raw.cfg.palm_body_name)
    other_columns = torch.tensor(
        [i for i in range(len(LINKS)) if i != palm_column and i not in tip_columns.tolist()], device=device)

    zero = torch.zeros((n, raw.cfg.action_space), device=device)
    released_pos = raw.cube_pos.clone()
    released_quat = raw.cube_quat.clone()
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    max_disp = torch.zeros(n, device=device)
    max_rot = torch.zeros(n, device=device)
    near_limit = torch.zeros((n, len(raw.joint_ids)), device=device)
    # Scored over a window so a grasp that is only momentarily in contact cannot pass.
    min_tips = torch.full((n,), 5.0, device=device)
    max_palm = torch.zeros(n, device=device)
    tip_force_sum = torch.zeros(n, device=device)
    other_force_sum = torch.zeros(n, device=device)
    max_nontip_links = torch.zeros(n, device=device)
    scored = 0
    for step in range(steps):
        _, _, terminated, truncated, _ = env.step(zero)
        fell |= terminated | truncated
        max_disp = torch.maximum(max_disp, (raw.cube_pos - released_pos).norm(dim=-1))
        max_rot = torch.maximum(max_rot, quat_error_magnitude(raw.cube_quat, released_quat))
        applied = raw.hand.data.applied_torque[:, raw.joint_ids].abs()
        near_limit += (applied >= NEAR_LIMIT * raw.torque_limits).float()
        if step >= steps - score_steps:
            forces = raw.cube_contact_forces()
            tips = (forces[:, tip_columns] > args_cli.contact_force_n).sum(dim=-1).float()
            min_tips = torch.minimum(min_tips, tips)
            max_palm = torch.maximum(max_palm, forces[:, palm_column])
            tip_force_sum += forces[:, tip_columns].sum(dim=-1)
            other_force_sum += forces[:, other_columns].sum(dim=-1)
            touching = (forces[:, other_columns] > args_cli.contact_force_n).sum(dim=-1).float()
            max_nontip_links = torch.maximum(max_nontip_links, touching)
            scored += 1
    return {
        "fell": fell,
        "min_tips_in_contact": min_tips,
        "max_palm_force_n": max_palm,
        "mean_tip_force_n": tip_force_sum / max(scored, 1),
        "mean_other_force_n": other_force_sum / max(scored, 1),
        "nontip_links_in_contact": max_nontip_links,
        "cube_disp_mm": 1000 * max_disp,
        "cube_rot_deg": torch.rad2deg(max_rot),
        "near_limit_fraction": near_limit.mean(dim=-1) / steps,
    }


def accept(metrics):
    return (
        ~metrics["fell"]
        & (metrics["min_tips_in_contact"] >= args_cli.min_tips)
        & (metrics["max_palm_force_n"] < args_cli.max_palm_force_n)
        & (metrics["nontip_links_in_contact"] <= args_cli.max_nontip_links)
        & (metrics["cube_disp_mm"] < args_cli.max_cube_disp_mm)
        & (metrics["cube_rot_deg"] < args_cli.max_cube_rot_deg)
        & (metrics["near_limit_fraction"] < args_cli.max_near_limit_fraction)
    )


def reject_reasons(metrics):
    """Per-criterion failure counts, so a round that finds nothing still says why."""
    return {
        "terminated_or_fell": metrics["fell"],
        f"tips<{args_cli.min_tips}": metrics["min_tips_in_contact"] < args_cli.min_tips,
        "palm_in_contact": metrics["max_palm_force_n"] >= args_cli.max_palm_force_n,
        "phalanges_touching": metrics["nontip_links_in_contact"] > args_cli.max_nontip_links,
        "cube_slipped": metrics["cube_disp_mm"] >= args_cli.max_cube_disp_mm,
        "cube_rotated": metrics["cube_rot_deg"] >= args_cli.max_cube_rot_deg,
        "torque_pinned": metrics["near_limit_fraction"] >= args_cli.max_near_limit_fraction,
    }


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.robot_cfg.spawn.activate_contact_sensors = True
    # The candidate IS the reset state: noise would blur which configuration was scored.
    cfg.reset_joint_position_noise_rad = 0.0
    cfg.cube_reset_position_noise_m = 0.0
    # Long enough that the close+hold sequence is never cut short by the time limit.
    cfg.episode_length_s = 4 * (args_cli.close_s + args_cli.hold_s)
    env = GraspSearchEnv(cfg)
    raw = env.unwrapped
    generator = torch.Generator(device=raw.device).manual_seed(args_cli.seed)
    close_steps = int(args_cli.close_s / raw.step_dt)
    hold_steps = int(args_cli.hold_s / raw.step_dt)
    score_steps = max(1, int(args_cli.score_s / raw.step_dt))
    print(f"[SEARCH] envs={raw.num_envs} rounds={args_cli.rounds} close={close_steps} hold={hold_steps}"
          f" score_window={score_steps} steps dt={raw.step_dt:.5f}s links={len(LINKS)}", flush=True)
    print(f"[SEARCH] cube weight={0.05 * 9.81:.3f} N contact_threshold={args_cli.contact_force_n} N"
          f" torque_cap={raw.torque_limits[0].amax().item():.3f} Nm", flush=True)

    kept = {key: [] for key in ("q", "q_cmd", "cube_pos", "cube_quat", "anchor")}
    totals = {}
    start = time.monotonic()
    for round_id in range(args_cli.rounds):
        state, command, anchor = sample_candidates(raw, generator)
        raw.hand.data.default_joint_pos[:, raw.joint_ids] = state
        raw.grasp_command[:] = command
        # cfg.cube_position_in_palm is stored as an expand() view, so every row shares memory:
        # writing per-env anchors in place raises. Replace the tensor instead.
        raw.cube_anchor = anchor.clone()
        env.reset()
        frozen = close_on_cube(env, command, close_steps)
        settled_q, settled_command, settled_cube = reseat(env, command)
        metrics = hold_and_score(env, hold_steps, score_steps)
        good = accept(metrics)
        for name, mask in reject_reasons(metrics).items():
            totals[name] = totals.get(name, 0) + int((mask & ~good).sum())
        ids = good.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            # Exactly the state that reseat() reset to and hold_and_score() then validated.
            identity = torch.zeros((ids.numel(), 4), device=raw.device)
            identity[:, 0] = 1
            kept["q"].append(settled_q[ids].cpu().numpy())
            kept["q_cmd"].append(settled_command[ids].cpu().numpy())
            kept["cube_pos"].append(settled_cube[ids].cpu().numpy())
            kept["cube_quat"].append(identity.cpu().numpy())
            kept["anchor"].append(anchor[ids].cpu().numpy())
        print(f"[SEARCH] round={round_id} closed_fingers={frozen.sum(dim=-1).float().mean():.2f}"
              f" accepted={ids.numel()}/{raw.num_envs}"
              f" best_min_tips={metrics['min_tips_in_contact'].amax().item():.0f}"
              f" palm_force_min={metrics['max_palm_force_n'].amin().item():.3f} N"
              f" elapsed={time.monotonic() - start:.0f}s", flush=True)
        for env_id in metrics["min_tips_in_contact"].argsort(descending=True)[:args_cli.report_rejects].tolist():
            print(f"[SEARCH]   env={env_id} accepted={bool(good[env_id])}"
                  f" tips={metrics['min_tips_in_contact'][env_id]:.0f}"
                  f" palm={metrics['max_palm_force_n'][env_id]:.3f} N"
                  f" tip_force={metrics['mean_tip_force_n'][env_id]:.3f} N"
                  f" other_force={metrics['mean_other_force_n'][env_id]:.3f} N"
                  f" nontip_links={metrics['nontip_links_in_contact'][env_id]:.0f}"
                  f" disp={metrics['cube_disp_mm'][env_id]:.1f} mm"
                  f" rot={metrics['cube_rot_deg'][env_id]:.1f} deg"
                  f" near_limit={metrics['near_limit_fraction'][env_id]:.3f}"
                  f" cube_x_mm={1000 * anchor[env_id, 0]:.1f}", flush=True)

    total = sum(a.shape[0] for a in kept["q"])
    attempted = args_cli.rounds * raw.num_envs
    print(f"[SEARCH] ANSWER fingertip_grasps_found={total}/{attempted}", flush=True)
    print(f"[SEARCH] reject_counts={totals}", flush=True)
    if total:
        arrays = {k: np.concatenate(v, axis=0) for k, v in kept.items()}
        arrays["joint_names"] = np.array(raw.cfg.actuated_joint_names)
        os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
        np.savez(args_cli.output, **arrays)
        x = 1000 * arrays["anchor"][:, 0]
        print(f"[SEARCH] wrote {args_cli.output} n={total}"
              f" cube_x_mm min/median/max={x.min():.1f}/{np.median(x):.1f}/{x.max():.1f}", flush=True)
    else:
        print("[SEARCH] no fingertip grasp survived the release; widen the ranges or fall back to"
              " an RL pinch policy from the cradle", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
