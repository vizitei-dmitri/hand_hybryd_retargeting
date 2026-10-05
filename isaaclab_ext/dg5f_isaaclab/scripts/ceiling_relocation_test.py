"""Can a finger be relocated at the in-grasp rotation ceiling? Offline test, no training.

Stage M (monotonic +20 deg goals) showed every policy turning the cube in-grasp to ~120 deg, where
rj_dg_4_1 / 1_1 / 5_2 / 5_4 sit at their limits and the grasp collapses (JOURNAL 2026-10-05). The
cube repeats every 90 deg about a face axis, so a gait cycle could be: turn ~90 deg in-grasp, then
put each finger back where it was RELATIVE TO THE CUBE CENTRE at the start. This script checks
whether that relocation is physically possible before any training is spent on it.

  1 harvest   a policy runs stage M deterministically; when an env's net rotation first passes each
              of --bins with >= 3 tips on the cube, its full dynamic state is snapshotted, together
              with where each fingertip sat relative to the cube centre (palm frame) at 0.5 s.
  2 relocate  every (snapshot, finger in contact) pair is restored and that finger alone is driven
              by Isaac Lab DLS IK (the generate_gait_transition_cache.py primitive): detach along the
              face normal, transfer to the start-relative position projected onto the nearest face,
              approach, hold. The other fingers keep their commands. All actions go through the
              unchanged FIFO / integrated-delta / PD path.
  3 save      accepted relocations are written as phase-labelled reset snapshots (gait cache schema),
              plus the harvested ceiling states themselves as phase 0.
"""
import argparse, json, math, traceback
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', required=True, help='RUN_DIR:ITER of the harvesting policy')
p.add_argument('--num_envs', type=int, default=256)
p.add_argument('--seed', type=int, default=1234)
p.add_argument('--bins', default='80,95,110', help='net body-axis rotation (deg) at which to snapshot')
p.add_argument('--speed', type=float, default=0.4, help='max |action| of the moving finger')
p.add_argument('--detach_m', type=float, default=0.010)
p.add_argument('--out', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args); simulation_app = app.app

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.io import load_yaml
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym
import dg5f_isaaclab.tasks  # noqa: F401  (also registers BoundedActorCritic)
from dg5f_isaaclab.assets.gait_cache import snapshot, restore, DYNAMIC_FIELDS, PHASES

TASK = 'DG5F-Cube-Stream-Direct-v0'


def finger_joint_limit_margin(env, finger):
    """Smallest distance (deg) of any active joint of `finger` (0-based) to its nearest limit."""
    q = env.hand.data.joint_pos[:, env.active_joint_ids]
    lo, hi = env.lower, env.upper
    margin = torch.minimum(q - lo, hi - q)
    names = env.cfg.active_action_joints
    out = torch.empty(env.num_envs, device=env.device)
    for f in range(5):
        cols = [i for i, n in enumerate(names) if n.startswith(f'rj_dg_{f + 1}_')]
        sel = finger == f
        if sel.any():
            out[sel] = torch.rad2deg(margin[sel][:, cols].min(dim=-1).values)
    return out


def harvest(env, policy, bins):
    n, dev = env.num_envs, env.device
    captured = torch.zeros((n, len(bins)), dtype=torch.bool, device=dev)
    canonical = torch.zeros((n, 5, 3), device=dev)
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    snaps = []
    obs = wrapper.get_observations()
    for step in range(env.max_episode_length):
        obs, _, dones, _ = wrapper.step(policy(obs))
        finished |= dones.bool()
        if step == 29:  # 0.5 s: the start grasp has settled
            canonical[:] = env.tip_pos_palm - env.cube_pos[:, None, :]
        if step < 30:
            continue
        net = torch.rad2deg(env.axis_rotation_body)
        ok = (~finished) & (env.tip_contact_count >= 3)
        for b, threshold in enumerate(bins):
            new = ok & ~captured[:, b] & (net >= threshold)
            if new.any():
                s = snapshot(env)
                ids = new.nonzero().squeeze(-1).cpu().numpy()
                for e in ids:
                    snaps.append({**{k: v[e] for k, v in s.items()}, 'canonical': canonical[e].cpu().numpy(),
                                  'net_deg': float(net[e]), 'bin': b})
                captured[:, b] |= new
        if finished.all():
            break
    return snaps


def relocate(env, snaps, pairs, ik):
    """Run one batch of (snapshot, finger) relocations. Returns per-pair metrics and phase snapshots."""
    n, dev = env.num_envs, env.device
    m = len(pairs)
    idx = torch.arange(n, device=dev)
    snap_ids = np.array([pairs[i % m][0] for i in range(n)])
    finger = torch.tensor([pairs[i % m][1] for i in range(n)], device=dev)
    active = idx < m
    arrays = {k: np.stack([snaps[j][k] for j in snap_ids]) for k in DYNAMIC_FIELDS}
    env.reset()
    restore(env, arrays, torch.arange(n, device=dev), idx)
    zero = torch.zeros((n, env.cfg.action_space), device=dev)
    for _ in range(2):  # let the contact sensors and the restored state register
        wrapper.step(zero)
    canonical = torch.as_tensor(np.stack([snaps[j]['canonical'] for j in snap_ids]), device=dev)
    tip_ids = torch.tensor(env.tip_ids, device=dev)[finger]
    names = env.cfg.active_action_joints
    mask = torch.stack([torch.tensor([nm.startswith(f'rj_dg_{int(f) + 1}_') for nm in names], device=dev)
                        for f in finger.tolist()]).float()
    def tip_in_cube():
        tips = env.hand.data.body_pos_w[idx, tip_ids]
        return quat_apply_inverse(env.cube.data.root_quat_w, tips - env.cube.data.root_pos_w)
    start_local = tip_in_cube()
    start_contacts = env.tip_in_contact.clone()
    valid = active & start_contacts[idx, finger] & (env.tip_contact_count >= 3)
    axis = start_local.abs().argmax(-1)
    normal = torch.zeros_like(start_local); normal[idx, axis] = torch.sign(start_local[idx, axis])
    height = start_local[idx, axis].abs()  # tip-origin height above the face it touches
    # Target: the start-relative tip position (palm frame) expressed in the CURRENT cube frame,
    # projected onto the face it points at, at the same height the tip currently has.
    target_palm = canonical[idx, finger]
    target = quat_apply_inverse(env.cube_quat, target_palm)  # cube_quat is the palm-frame orientation
    t_axis = target.abs().argmax(-1)
    t_normal = torch.zeros_like(target); t_normal[idx, t_axis] = torch.sign(target[idx, t_axis])
    target = target.clamp(-0.025, 0.025); target[idx, t_axis] = t_normal[idx, t_axis] * height
    requested = (target - start_local).norm(dim=-1)
    margin_before = finger_joint_limit_margin(env, finger)
    dt = env.step_dt
    detach_steps, transfer_steps, approach_steps, hold_steps = round(.6 / dt), round(1.2 / dt), round(.7 / dt), round(.5 / dt)
    total = detach_steps + transfer_steps + approach_steps + hold_steps
    dead = torch.zeros(n, dtype=torch.bool, device=dev)
    released = torch.zeros_like(dead); survived_release = torch.ones_like(dead)
    phase_snaps = {}
    recontact_hold = torch.zeros(n, device=dev)
    for step in range(total):
        if step < detach_steps:
            a = (step + 1) / detach_steps; desired = start_local + normal * args.detach_m * a
        elif step < detach_steps + transfer_steps:
            a = (step - detach_steps + 1) / transfer_steps
            lifted_target = target + t_normal * args.detach_m
            desired = start_local + normal * args.detach_m + (lifted_target - start_local - normal * args.detach_m) * a
        elif step < detach_steps + transfer_steps + approach_steps:
            a = (step - detach_steps - transfer_steps + 1) / approach_steps
            desired = target + t_normal * (args.detach_m * (1 - a) - 0.002 * a)
        else:
            desired = target - t_normal * 0.002
        current = env.hand.data.body_pos_w[idx, tip_ids]
        world_target = env.cube.data.root_pos_w + quat_apply(env.cube.data.root_quat_w, desired)
        jac = env.hand.root_physx_view.get_jacobians()
        body = tip_ids - 1 if env.hand.is_fixed_base else tip_ids
        J = jac[idx, body, :3][:, :, env.active_joint_ids].clone() * mask[:, None, :]
        quat = env.hand.data.body_quat_w[idx, tip_ids]
        ik.set_command(world_target, ee_quat=quat)
        desired_q = ik.compute(current, quat, J, env.hand.data.joint_pos[:, env.active_joint_ids])
        predicted = env.joint_command + env.action_queue.history.sum(1) * env.cfg.delta_action_scale
        action = ((desired_q - predicted) / env.cfg.delta_action_scale).clamp(-args.speed, args.speed) * mask
        action[~valid | dead] = 0
        if step >= detach_steps + transfer_steps + approach_steps:
            action[env.tip_in_contact[idx, finger]] = 0
        _, _, dones, _ = wrapper.step(action)
        dead |= dones.bool()
        contact = env.tip_in_contact[idx, finger]
        released |= ~contact
        if step == detach_steps - 1:
            survived_release = ~dead
            phase_snaps[1] = snapshot(env)
        if step == detach_steps + transfer_steps // 2:
            phase_snaps[3] = snapshot(env)
        if step == detach_steps + transfer_steps + approach_steps // 2:
            phase_snaps[4] = snapshot(env)
        if step == detach_steps + transfer_steps + approach_steps:
            phase_snaps[5] = snapshot(env)
        if step >= total - round(.3 / dt):
            recontact_hold += contact.float()
    phase_snaps[6] = snapshot(env)
    margin_after = finger_joint_limit_margin(env, finger)
    end_local = tip_in_cube()
    moved = (end_local - start_local).norm(dim=-1)
    recontacted = recontact_hold >= 0.5 * round(.3 / dt)
    success = valid & ~dead & released & recontacted & (env.tip_contact_count >= 3)
    rows = []
    for e in range(m):
        rows.append({'snapshot': int(snap_ids[e]), 'finger': int(finger[e]), 'valid': bool(valid[e]),
                     'net_deg': snaps[snap_ids[e]]['net_deg'], 'bin': snaps[snap_ids[e]]['bin'],
                     'survived_release': bool(survived_release[e]), 'dropped': bool(dead[e]),
                     'released': bool(released[e]), 'recontacted': bool(recontacted[e]),
                     'tips_end': int(env.tip_contact_count[e]), 'requested_m': float(requested[e]),
                     'moved_m': float(moved[e]), 'margin_before_deg': float(margin_before[e]),
                     'margin_after_deg': float(margin_after[e]), 'success': bool(success[e])})
    saved = []
    for phase, s in phase_snaps.items():
        for e in range(m):
            if success[e]:
                saved.append({**{k: s[k][e] for k in DYNAMIC_FIELDS}, 'phase': phase,
                              'moving_finger': int(finger[e]), 'source_snapshot': int(snap_ids[e])})
    return rows, saved


def main():
    global wrapper
    run_dir, _, it = args.run.rpartition(':')
    run_dir = Path(run_dir)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.goal_stream_stage = 'M'
    cfg.track_gait_contact_points = True
    cfg.seed = args.seed
    cfg.resolve_control_config()
    env_gym = gym.make(TASK, cfg=cfg)
    wrapper = RslRlVecEnvWrapper(env_gym)
    env = env_gym.unwrapped
    agent = load_yaml(str(run_dir / 'params' / 'agent.yaml'))
    runner = OnPolicyRunner(wrapper, agent, log_dir=None, device=env.device)
    runner.load(str(run_dir / f'model_{it}.pt'))
    policy = runner.get_inference_policy(device=env.device)
    torch.manual_seed(args.seed)
    bins = [float(b) for b in args.bins.split(',')]
    wrapper.reset()
    snaps = harvest(env, policy, bins)
    print(f'[CEILING] harvested {len(snaps)} snapshots; per bin '
          + str({b: sum(s["bin"] == i for s in snaps) for i, b in enumerate(bins)}), flush=True)
    if not snaps:
        print('[CEILING] no ceiling state reached with >= 3 tips', flush=True); return
    pairs = [(i, f) for i, s in enumerate(snaps) for f in range(5) if s['contact_mask'][f]]
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type='position', ik_method='dls',
                                                              ik_params={'lambda_val': .003}), env.num_envs, env.device)
    # Nothing in the relocation run may time out; max_episode_length is a property read from the cfg.
    env.cfg.episode_length_s = 1e4
    rows, saved = [], []
    for start in range(0, len(pairs), env.num_envs):
        r, s = relocate(env, snaps, pairs[start:start + env.num_envs], ik)
        rows += r; saved += s
        print(f'[CEILING] batch {start // env.num_envs}: pairs {len(r)} success {sum(x["success"] for x in r)}', flush=True)
    summary = {}
    for f in range(5):
        rr = [x for x in rows if x['finger'] == f and x['valid']]
        if not rr:
            summary[f] = {'attempts': 0}; continue
        ok = [x for x in rr if x['success']]
        summary[f] = {'attempts': len(rr),
                      'survived_release': sum(x['survived_release'] for x in rr) / len(rr),
                      'success': len(ok) / len(rr),
                      'margin_gain_deg_success': float(np.mean([x['margin_after_deg'] - x['margin_before_deg'] for x in ok])) if ok else None,
                      'moved_mm_success': float(1000 * np.mean([x['moved_m'] for x in ok])) if ok else None,
                      'requested_mm': float(1000 * np.mean([x['requested_m'] for x in rr]))}
    for b in range(len(bins)):
        rr = [x for x in rows if x['bin'] == b and x['valid']]
        summary[f'bin_{bins[b]:g}'] = {'attempts': len(rr), 'success': sum(x['success'] for x in rr) / max(1, len(rr))}
    print('[CEILING] summary ' + json.dumps(summary), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps({'summary': summary, 'rows': rows, 'bins': bins,
                                                         'run': args.run, 'seed': args.seed}, indent=1))
    # Reset cache: harvested ceiling states as phase 0 + accepted relocation phases.
    entries = [{**{k: s[k] for k in DYNAMIC_FIELDS}, 'phase': 0, 'moving_finger': -1, 'source_snapshot': i}
               for i, s in enumerate(snaps)] + saved
    out = {k: np.stack([np.asarray(e[k]) for e in entries]) for k in entries[0]}
    out.update(schema_version=np.array(1), joint_names=np.array(env.cfg.actuated_joint_names), phase_names=np.array(PHASES))
    np.savez_compressed(args.out, **out)
    counts = np.bincount(out['phase'], minlength=7).tolist()
    print(f'[CEILING] wrote {args.out} n={len(entries)} phase_counts={counts}', flush=True)
    env_gym.close()


if __name__ == '__main__':
    try:
        # Everything under inference mode, as generate_gait_transition_cache.py does: tensors made
        # in the harvest are inference tensors, and restore() later updates them in place.
        with torch.inference_mode():
            main()
    except Exception:
        traceback.print_exc(); raise
    finally:
        simulation_app.close()
