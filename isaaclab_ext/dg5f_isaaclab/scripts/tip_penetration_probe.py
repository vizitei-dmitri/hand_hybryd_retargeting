"""How deep do the fingertips go into the cube, and does it depend on speed? Measurement only.

A policy runs deterministically; every control step the pose of each fingertip body, its linear
velocity relative to the cube, its contact flag and the cube pose are recorded. Offline, the
fingertip COLLISION mesh from the URDF (collision/rl_dg_N_tip_c.STL) is placed at the recorded pose
and expressed in the cube frame; the depth of the deepest vertex inside the 60 mm cube is the
penetration. Depth that grows with tip speed points at dynamic penetration (contact_offset 1 mm,
max_depenetration_velocity 0.2 m/s); depth that stays at rest points at the collider (the
convex decomposition) being smaller than the mesh that is rendered.
"""
import argparse, traceback
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', required=True, help='RUN_DIR:ITER')
p.add_argument('--num_envs', type=int, default=64)
p.add_argument('--steps', type=int, default=600)
p.add_argument('--stage', default='M')
p.add_argument('--seed', type=int, default=1234)
p.add_argument('--mode', default='none', help='gait_observation_mode the checkpoint needs')
p.add_argument('--depen', type=float, default=None, help='DIAGNOSTIC: max_depenetration_velocity for hand and cube')
p.add_argument('--solver_pos_iters', type=int, default=None, help='DIAGNOSTIC: solver position iterations for hand and cube')
p.add_argument('--out', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args); simulation_app = app.app

import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
import dg5f_isaaclab.tasks  # noqa: F401

TASK = 'DG5F-Cube-Stream-Direct-v0'


def main():
    run_dir, _, it = args.run.rpartition(':')
    run_dir = Path(run_dir)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.goal_stream_stage = args.stage
    cfg.gait_observation_mode = args.mode
    cfg.seed = args.seed
    if args.depen is not None:
        cfg.robot_cfg.spawn.rigid_props.max_depenetration_velocity = args.depen
        cfg.object_cfg.spawn.rigid_props.max_depenetration_velocity = args.depen
    if args.solver_pos_iters is not None:
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = args.solver_pos_iters
        cfg.object_cfg.spawn.rigid_props.solver_position_iteration_count = args.solver_pos_iters
    cfg.resolve_control_config()
    env_gym = gym.make(TASK, cfg=cfg)
    wrapper = RslRlVecEnvWrapper(env_gym)
    env = env_gym.unwrapped
    runner = OnPolicyRunner(wrapper, load_yaml(str(run_dir / 'params' / 'agent.yaml')), log_dir=None, device=env.device)
    runner.load(str(run_dir / f'model_{it}.pt'))
    policy = runner.get_inference_policy(device=env.device)
    torch.manual_seed(args.seed)
    wrapper.reset()
    obs = wrapper.get_observations()
    tip = torch.tensor(env.tip_ids, device=env.device)
    rec = {k: [] for k in ('tip_pos', 'tip_quat', 'tip_vel', 'cube_pos', 'cube_quat', 'cube_linvel', 'cube_angvel', 'contact', 'done')}
    for _ in range(args.steps):
        obs, _, dones, _ = wrapper.step(policy(obs))
        d = env.hand.data
        rec['tip_pos'].append(d.body_pos_w[:, tip]); rec['tip_quat'].append(d.body_quat_w[:, tip])
        rec['tip_vel'].append(d.body_lin_vel_w[:, tip])
        rec['cube_pos'].append(env.cube.data.root_pos_w); rec['cube_quat'].append(env.cube.data.root_quat_w)
        rec['cube_linvel'].append(env.cube.data.root_lin_vel_w); rec['cube_angvel'].append(env.cube.data.root_ang_vel_w)
        rec['contact'].append(env.tip_in_contact); rec['done'].append(dones.bool())
    out = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    out['cube_half_m'] = np.array(cfg.cube_size_m / 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f'[PROBE] wrote {args.out} steps={args.steps} envs={args.num_envs}', flush=True)
    env_gym.close()


if __name__ == '__main__':
    try:
        with torch.inference_mode():
            main()
    except Exception:
        traceback.print_exc(); raise
    finally:
        simulation_app.close()
