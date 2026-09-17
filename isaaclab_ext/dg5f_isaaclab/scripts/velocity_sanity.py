"""Compare PhysX joint velocity with a finite difference of joint position (diagnostic only).

Samples every PHYSICS substep (before each _apply_action the articulation buffers hold
the state after the previous substep), so 120 Hz chatter is not hidden by 60 Hz sampling.
Solver options are overridden one at a time from the CLI for before/after comparison.
"""

import argparse
import json
import math
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="DG5F-Cube-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=240, help="Control steps.")
parser.add_argument("--actions", choices=("zero", "random"), default="zero")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--external_forces_every_iteration", type=int, choices=(0, 1))
parser.add_argument("--articulation_velocity_iterations", type=int)
parser.add_argument("--articulation_position_iterations", type=int)
parser.add_argument("--cube_velocity_iterations", type=int)
parser.add_argument("--no_cube_contact", action="store_true", help="Diagnostic: drop the cube beside the hand.")
parser.add_argument("--no_self_collision", action="store_true", help="Diagnostic: disable hand self-collision.")
parser.add_argument("--collider_type", choices=("convex_hull", "convex_decomposition"))
parser.add_argument("--max_depenetration_velocity", type=float, help="Hand and cube, m/s.")
parser.add_argument("--disabled_stiffness", type=float, help="Diagnostic: drive stiffness of the disabled joint only.")
parser.add_argument("--disabled_armature", type=float, help="Diagnostic: armature of the disabled joint only.")
parser.add_argument("--settle_s", type=float, default=1.0, help="Steady-state statistics start after this time.")
parser.add_argument("--label", default="run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    articulation = cfg.robot_cfg.spawn.articulation_props
    if args_cli.external_forces_every_iteration is not None:
        cfg.sim.physx.enable_external_forces_every_iteration = bool(args_cli.external_forces_every_iteration)
    if args_cli.articulation_velocity_iterations is not None:
        articulation.solver_velocity_iteration_count = args_cli.articulation_velocity_iterations
    if args_cli.articulation_position_iterations is not None:
        articulation.solver_position_iteration_count = args_cli.articulation_position_iterations
    cube = cfg.object_cfg.spawn
    if args_cli.cube_velocity_iterations is not None:
        cube.rigid_props.solver_velocity_iteration_count = args_cli.cube_velocity_iterations
    if args_cli.no_cube_contact:
        cfg.cube_position_in_palm = (0.25, 0.0, 0.10)
        cfg.max_cube_distance_from_palm_m = 1e3
        cfg.min_cube_world_height_m = -1e3
    if args_cli.max_depenetration_velocity is not None:
        cfg.robot_cfg.spawn.rigid_props.max_depenetration_velocity = args_cli.max_depenetration_velocity
        cube.rigid_props.max_depenetration_velocity = args_cli.max_depenetration_velocity
    if args_cli.no_self_collision:
        articulation.enabled_self_collisions = False
    if args_cli.collider_type is not None and args_cli.collider_type != cfg.robot_cfg.spawn.collider_type:
        # Separate USD cache so the default converted hand is not overwritten.
        cfg.robot_cfg.spawn.collider_type = args_cli.collider_type
        cfg.robot_cfg.spawn.usd_dir = f"{cfg.robot_cfg.spawn.usd_dir}_{args_cli.collider_type}"
    settings = {
        "solver_type": cfg.sim.physx.solver_type,
        "enable_external_forces_every_iteration": cfg.sim.physx.enable_external_forces_every_iteration,
        "enable_stabilization": cfg.sim.physx.enable_stabilization,
        "articulation_pos_iters": articulation.solver_position_iteration_count,
        "articulation_vel_iters": articulation.solver_velocity_iteration_count,
        "cube_pos_iters": cube.rigid_props.solver_position_iteration_count,
        "cube_vel_iters": cube.rigid_props.solver_velocity_iteration_count,
        "hand_max_depenetration_velocity": cfg.robot_cfg.spawn.rigid_props.max_depenetration_velocity,
        "cube_max_depenetration_velocity": cube.rigid_props.max_depenetration_velocity,
        "contact_offset": cube.collision_props.contact_offset,
        "rest_offset": cube.collision_props.rest_offset,
        "physics_dt": cfg.sim.dt,
        "self_collision": articulation.enabled_self_collisions,
        "cube_contact": not args_cli.no_cube_contact,
        "collider_type": cfg.robot_cfg.spawn.collider_type,
    }
    print(f"[VEL] label={args_cli.label} actions={args_cli.actions} settings={json.dumps(settings)}")
    torch.manual_seed(args_cli.seed)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    names = cfg.actuated_joint_names
    ids = raw.joint_ids
    lock = [raw.disabled_joint_id]
    if args_cli.disabled_stiffness is not None:
        raw.hand.write_joint_stiffness_to_sim(args_cli.disabled_stiffness, joint_ids=lock)
        raw.hand.write_joint_effort_limit_to_sim(1e3, joint_ids=lock)  # a locking drive, not a DG5F motor
    if args_cli.disabled_armature is not None:
        raw.hand.write_joint_armature_to_sim(args_cli.disabled_armature, joint_ids=lock)
    print(f"[VEL] disabled_joint lock: limits_deg={torch.rad2deg(raw.hand.data.joint_pos_limits[0, raw.disabled_joint_id]).tolist()}"
          f" stiffness={raw.hand.data.joint_stiffness[0, raw.disabled_joint_id].item():g}"
          f" armature={raw.hand.data.joint_armature[0, raw.disabled_joint_id].item():g}"
          f" effort_limit={raw.hand.data.joint_effort_limits[0, raw.disabled_joint_id].item():g}")
    print(f"[VEL] velocity_limit_readback={raw.hand.data.joint_vel_limits[0, ids].tolist()}")

    samples = []  # (q, qdot, episode id) per physics substep
    episode = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
    apply_action = raw._apply_action

    def recorded_apply_action():
        samples.append((raw.hand.data.joint_pos[:, ids].clone(), raw.hand.data.joint_vel[:, ids].clone(),
                        episode.clone()))
        apply_action()

    raw._apply_action = recorded_apply_action
    env.reset()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            if args_cli.actions == "zero":
                action = torch.zeros((raw.num_envs, cfg.action_space), device=raw.device)
            else:
                action = 2 * torch.rand((raw.num_envs, cfg.action_space), device=raw.device) - 1
            _, _, terminated, truncated, _ = env.step(action)
            episode += (terminated | truncated).long()
        samples.append((raw.hand.data.joint_pos[:, ids].clone(), raw.hand.data.joint_vel[:, ids].clone(),
                        episode.clone()))

    q = torch.stack([s[0] for s in samples])       # T x N x 20
    qdot = torch.stack([s[1] for s in samples])
    ep = torch.stack([s[2] for s in samples])
    dt = raw.physics_dt
    # Skip the first substep after a reset (teleported state) and any pair spanning a reset.
    valid = (ep[1:] == ep[:-1])
    valid[0] = False
    fd = (q[1:] - q[:-1]) / dt
    physx = qdot[1:]
    err = (physx - fd).abs()
    mask = valid[..., None].expand_as(err)
    err_masked = torch.where(mask, err, torch.zeros_like(err))
    count = mask.float().sum(dim=(0, 1))
    mean_err = err_masked.sum(dim=(0, 1)) / count
    max_err = err_masked.amax(dim=(0, 1))
    max_physx = torch.where(mask, physx.abs(), torch.zeros_like(err)).amax(dim=(0, 1))
    max_fd = torch.where(mask, fd.abs(), torch.zeros_like(err)).amax(dim=(0, 1))
    # 120 Hz chatter: consecutive PhysX velocities flip sign with a large magnitude.
    flips = (physx[1:] * physx[:-1] < 0) & (physx[1:].abs() > 0.5) & mask[1:] & mask[:-1]
    flip_fraction = flips.float().sum(dim=(0, 1)) / count
    # Peak-to-peak position over the last 0.5 s shows the real motion amplitude.
    tail = q[-int(0.5 / dt):]
    p2p_deg = torch.rad2deg(tail.amax(dim=0) - tail.amin(dim=0)).amax(dim=0)

    total = mask.float().sum()
    settle = int(args_cli.settle_s / dt)
    steady = mask[settle:]
    steady_err = torch.where(steady, err[settle:], torch.zeros_like(err[settle:]))
    steady_physx = torch.where(steady, physx[settle:].abs(), torch.zeros_like(err[settle:]))
    steady_fd = torch.where(steady, fd[settle:].abs(), torch.zeros_like(err[settle:]))
    steady_joint_err = steady_err.sum(dim=(0, 1)) / steady.float().sum(dim=(0, 1))
    worst = torch.argsort(steady_joint_err, descending=True)[:3].tolist()
    print(f"[VEL] label={args_cli.label} STEADY(t>{args_cli.settle_s}s) mean_abs_err="
          f"{(steady_err.sum() / steady.float().sum()).item():.4f} max_abs_err={steady_err.max().item():.4f}"
          f" max_abs_qdot_physx={steady_physx.max().item():.4f} max_abs_qdot_fd={steady_fd.max().item():.4f}"
          f" worst=" + ",".join(f"{names[i]}:{steady_joint_err[i].item():.3f}" for i in worst))
    first_err = err_masked[:settle].amax(dim=(1, 2))
    peak = int(first_err.argmax())
    print(f"[VEL] label={args_cli.label} largest transient error at substep {peak + 1}"
          f" ({(peak + 1) * dt:.4f}s after reset): {first_err[peak].item():.3f} rad/s")
    print(f"[VEL] label={args_cli.label} substeps={q.shape[0]} resets={int(ep[-1].sum())}"
          f" mean_abs_err={(err_masked.sum() / total).item():.4f} max_abs_err={max_err.max().item():.4f} rad/s"
          f" max_abs_qdot_physx={max_physx.max().item():.4f} max_abs_qdot_fd={max_fd.max().item():.4f}")
    disabled = torch.rad2deg(q[:, :, raw.disabled_index].abs())
    print(f"[VEL] label={args_cli.label} disabled_joint |q|_deg max={disabled.max().item():.4f}"
          f" p99={disabled.flatten().quantile(0.99).item():.4f} mean={disabled.mean().item():.4f}"
          f" max|qdot|={qdot[:, :, raw.disabled_index].abs().max().item():.4f}")
    order = torch.argsort(mean_err, descending=True).tolist()
    print("[VEL] joint mean_err max_err max|qdot_physx| max|qdot_fd| chatter_frac p2p_last0.5s_deg")
    for index in order[:6] + [names.index("rj_dg_1_2")]:
        print(f"[VEL] {names[index]} {mean_err[index].item():.4f} {max_err[index].item():.4f}"
              f" {max_physx[index].item():.4f} {max_fd[index].item():.4f}"
              f" {flip_fraction[index].item():.3f} {p2p_deg[index].item():.4f}")
    index = names.index("rj_dg_1_2")
    series = [(round(math.degrees(q[t, 0, index].item()), 4), round(qdot[t, 0, index].item(), 3))
              for t in range(q.shape[0] - 8, q.shape[0])]
    print(f"[VEL] rj_dg_1_2 env0 last substeps (q_deg, qdot_physx)={series}")
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
