"""Is the fingertip collider as large as the rendered fingertip? Physical approach test, measurement only.

tip_penetration_probe.py found the fingertip mesh 6.8 mm (median) inside the cube while in contact,
independent of speed, and the cached USD carries CollisionAPI/convexDecomposition on an Xform rather
than on the Mesh. A PhysX scene query cannot settle it under the GPU pipeline (it returned nothing, not
even the hand), so this asks the simulation itself: env k moves the cube KINEMATICALLY toward
fingertip k+1 from 100 mm outside the hand, 0.25 mm per control step, face-on along the palm->tip
direction, and records the distance at which that tip's contact sensor first reports force. The
mesh-based touching distance is 30 mm + max over the STL vertices of their projection on the approach
direction. Equal (to ~1 mm contact offset) -> the collider matches the mesh. Smaller -> the difference
is how far the rendered fingertip can sink into the cube before PhysX pushes back.
"""
import argparse, traceback
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--step_mm', type=float, default=0.25)
p.add_argument('--tip_collider', default='convexHull', choices=('convexHull', 'convexDecomposition'))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args); simulation_app = app.app

import numpy as np, torch, trimesh
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply
import dg5f_isaaclab.tasks  # noqa: F401

STL = '/home/yoba/Documents/work/hand_hybryd_retargeting/models/dg5f/assets/dg5f_right/collision/rl_dg_{}_tip_c.STL'


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_from_x_to(n):
    """Unit quaternion (w, x, y, z) rotating +X onto n."""
    x = np.array([1.0, 0.0, 0.0]); n = n / np.linalg.norm(n)
    c = float(np.dot(x, n))
    if c < -0.999999:
        return np.array([0.0, 0.0, 1.0, 0.0])
    axis = np.cross(x, n); q = np.array([1.0 + c, *axis]); return q / np.linalg.norm(q)


def main():
    cfg = parse_env_cfg('DG5F-Cube-Stream-Direct-v0', device=args.device, num_envs=5)
    cfg.max_time_without_tip_contact_s = None   # the cube is deliberately away from the hand
    cfg.max_cube_distance_from_grasp_m = None
    cfg.max_cube_distance_from_palm_m = 10.0
    cfg.min_cube_world_height_m = -10.0
    cfg.episode_length_s = 1e4
    cfg.tip_collider_approximation = args.tip_collider
    # The cube is posed kinematically once per control step; gravity would sag it ~1.4 mm in between.
    cfg.object_cfg.spawn.rigid_props.disable_gravity = True
    env = gym.make('DG5F-Cube-Stream-Direct-v0', cfg=cfg).unwrapped
    env.reset()
    dev = env.device
    zero = torch.zeros((5, env.cfg.action_space), device=dev)
    for _ in range(30):   # settle the hand at its reset pose with the cube still in place
        env.step(zero)
    names = list(env.contact_sensors)
    tips = [env.hand.find_bodies(f'rl_dg_{f}_tip')[0][0] for f in range(1, 6)]
    tip_pos = env.hand.data.body_pos_w[torch.arange(5), torch.tensor(tips)].cpu().numpy()
    tip_quat = env.hand.data.body_quat_w[torch.arange(5), torch.tensor(tips)].cpu().numpy()
    palm = env.palm_pos_w.cpu().numpy()
    approach = tip_pos - palm; approach /= np.linalg.norm(approach, axis=-1, keepdims=True)  # outward
    mesh_touch = np.zeros(5)
    for k in range(5):
        w, x, y, z = tip_quat[k]
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        verts = trimesh.load(STL.format(k + 1)).vertices @ rot.T
        mesh_touch[k] = 0.030 + (verts @ approach[k]).max()
    quats = np.stack([quat_from_x_to(-approach[k]) for k in range(5)])  # a face looks at the tip
    mesh_local = [trimesh.load(STL.format(k + 1)).vertices for k in range(5)]
    depth_at_contact = np.full(5, np.nan); moved_mm = np.full(5, np.nan)
    d = np.full(5, 0.100); contact_at = np.full(5, np.nan)
    zero_vel = torch.zeros((5, 6), device=dev)
    for _ in range(int(0.1 / (args.step_mm / 1000)) + 40):
        pos = tip_pos + approach * d[:, None]
        pose = torch.as_tensor(np.concatenate([pos, quats], -1), dtype=torch.float32, device=dev)
        env.cube.write_root_pose_to_sim(pose); env.cube.write_root_velocity_to_sim(zero_vel)
        env.step(zero)
        forces = np.array([env.contact_sensors[names[k]].data.force_matrix_w[k, 0, 0].norm().item() for k in range(5)])
        newly = np.isnan(contact_at) & (forces > 0.01)
        contact_at[newly] = d[newly]
        for k in np.nonzero(newly)[0]:
            # Mesh depth inside the cube at the moment of first contact, from the CURRENT tip pose
            # (the fingers re-close a little once the cube leaves the grasp, so the pose recorded
            # before the approach is not where the tip is when it is touched).
            tp = env.hand.data.body_pos_w[k, tips[k]].cpu().numpy(); tq = env.hand.data.body_quat_w[k, tips[k]].cpu().numpy()
            cp = env.cube.data.root_pos_w[k].cpu().numpy(); cq = env.cube.data.root_quat_w[k].cpu().numpy()
            world = tp + mesh_local[k] @ quat_matrix(tq).T
            local = (world - cp) @ quat_matrix(cq)
            depth_at_contact[k] = max(0.0, (0.030 - np.abs(local).max(-1)).max())
            moved_mm[k] = 1000 * np.linalg.norm(tp - tip_pos[k])
        moving = np.isnan(contact_at)
        if not moving.any():
            break
        d[moving] -= args.step_mm / 1000
    for k in range(5):
        print(f'[COVER] finger {k + 1}: mesh would touch at {1000 * mesh_touch[k]:.2f} mm, collider touched at '
              f'{1000 * contact_at[k]:.2f} mm -> collider is {1000 * (mesh_touch[k] - contact_at[k]):+.2f} mm '
              f'inside the mesh surface along the approach (stale pose); AT FIRST CONTACT the tip mesh is '
              f'{1000 * depth_at_contact[k]:.2f} mm inside the cube; tip moved {moved_mm[k]:.1f} mm since recorded', flush=True)
    env.close()


if __name__ == '__main__':
    try:
        with torch.inference_mode():
            main()
    except Exception:
        traceback.print_exc(); raise
    finally:
        simulation_app.close()
