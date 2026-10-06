"""Drive the simulated UR10e + DG5F cell along a demo trajectory to check the stack end to end.

The trajectory goes through Ur10Dg5fEnv, so it exercises the same path as a policy:
Robot limits -> MuJoCo backend IK -> physics -> observations and cameras.

Run inside the container, after the `source` lines from step 0 of the guide:

    # live MuJoCo window (on the host first: bash scripts/stack.sh gui-on)
    python3 scripts/ur10_dg5f_trajectory_demo.py --viewer --loops 3

    # headless: tracking table + GIF from the env's own cameras
    MUJOCO_GL=egl python3 scripts/ur10_dg5f_trajectory_demo.py --gif debug_runs/ur10e_dg5f_demo.gif
"""

import argparse
import pathlib
import tempfile
import time

import numpy as np
from lerobot_robot_dg5f.constants import JOINT_NAMES
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ur10_dg5f import Ur10Dg5f, Ur10Dg5fConfig
from lerobot_robot_ur10_dg5f.env import Ur10Dg5fEnv
from lerobot_robot_ur10_dg5f.pose_math import apply_offset, pose_difference


FPS = 10.0

# Closed-hand posture in degrees (inside the SDK limits); rj_dg_5_1 stays 0 in the robot.
GRASP_DEG = {
    "rj_dg_1_1": 20.0, "rj_dg_1_3": 30.0, "rj_dg_1_4": 20.0,
    "rj_dg_2_2": 60.0, "rj_dg_2_3": 50.0, "rj_dg_2_4": 30.0,
    "rj_dg_3_2": 60.0, "rj_dg_3_3": 50.0, "rj_dg_3_4": 30.0,
    "rj_dg_4_2": 60.0, "rj_dg_4_3": 50.0, "rj_dg_4_4": 30.0,
    "rj_dg_5_3": 50.0, "rj_dg_5_4": 30.0,
}
GRASP = np.array([GRASP_DEG.get(joint, 0.0) for joint in JOINT_NAMES])

# Keyframes: (time s, label, TCP offset in the UR base frame [dx dy dz rx ry rz], hand closure 0..1).
# Speeds stay below the robot step limit (10 cm/s, 0.5 rad/s at 10 Hz), so the target is followed.
KEYFRAMES = [
    (0.0, "старт в home", [0, 0, 0, 0, 0, 0], 0.0),
    (2.0, "квадрат: +x", [0.10, 0, 0, 0, 0, 0], 0.0),
    (4.0, "квадрат: +y", [0.10, 0.10, 0, 0, 0, 0], 0.0),
    (6.0, "квадрат: -x", [0, 0.10, 0, 0, 0, 0], 0.0),
    (8.0, "квадрат: -y", [0, 0, 0, 0, 0, 0], 0.0),
    (10.0, "вверх 10 см", [0, 0, 0.10, 0, 0, 0], 0.0),
    (12.0, "поворот +40°", [0, 0, 0.10, 0, 0, np.radians(40)], 0.0),
    (14.0, "поворот назад", [0, 0, 0.10, 0, 0, 0], 0.0),
    (16.0, "вниз в home", [0, 0, 0, 0, 0, 0], 0.0),
    (18.0, "сжать кисть", [0, 0, 0, 0, 0, 0], 1.0),
    (20.0, "разжать кисть", [0, 0, 0, 0, 0, 0], 0.0),
    (21.0, "стоп", [0, 0, 0, 0, 0, 0], 0.0),
]


def keyframe_at(t):
    """Linear interpolation between keyframes: (label, offset, closure)."""
    for (t0, _, off0, c0), (t1, label, off1, c1) in zip(KEYFRAMES, KEYFRAMES[1:]):
        if t <= t1:
            alpha = (t - t0) / (t1 - t0)
            offset = (1 - alpha) * np.asarray(off0, float) + alpha * np.asarray(off1, float)
            return label, offset, (1 - alpha) * c0 + alpha * c1
    _, label, offset, closure = KEYFRAMES[-1]
    return label, np.asarray(offset, float), closure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mjcf", default="/workspace/models/ur10e_dg5f/scene.xml")
    parser.add_argument("--viewer", action="store_true", help="show a live MuJoCo window (needs a display)")
    parser.add_argument("--loops", type=int, default=1, help="repeat the trajectory this many times")
    parser.add_argument("--gif", type=pathlib.Path, help="save cam_front | cam_wrist frames to this GIF")
    args = parser.parse_args()

    cameras = ("cam_front", "cam_wrist") if args.gif else ()
    config = Ur10Dg5fConfig(
        id="demo", calibration_dir=pathlib.Path(tempfile.mkdtemp()), mjcf_path=args.mjcf,
        sim_cameras=cameras, sim_image_hw=(240, 320))
    robot = Ur10Dg5f(config)
    env = Ur10Dg5fEnv(robot, fps=FPS, episode_s=KEYFRAMES[-1][0] + 1.0)

    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(robot.backend.model, robot.backend.data)
        viewer.cam.lookat[:] = [0.0, -0.55, 0.3]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 135.0, -20.0

    frames = []
    lag_pos_mm, lag_rot_deg = 0.0, 0.0
    for loop in range(args.loops):
        observation, _ = env.reset()
        home = robot.last_command.copy()
        print(f"\nпроход {loop + 1}/{args.loops}; home TCP (база UR): {np.round(home[:6], 4)}")
        print(f"{'t, с':>5}  {'фаза':<16} {'цель x/y/z, см':>20} {'факт x/y/z, см':>20} "
              f"{'ошибка, мм':>10} {'поворот, °':>10} {'пальцы цель/факт, °':>20}")
        steps = int(round(KEYFRAMES[-1][0] * FPS))
        for step in range(steps + 1):
            started = time.perf_counter()
            label, offset, closure = keyframe_at(step / FPS)
            target = np.concatenate([apply_offset(home[:6], offset), closure * GRASP])
            observation, _, terminated, truncated, info = env.step(target)
            state = robot.backend.read()
            error = pose_difference(state.tcp, target[:6])
            pos_mm = 1000 * np.linalg.norm(error[:3])
            rot_deg = np.degrees(np.linalg.norm(error[3:]))
            index_flex = JOINT_NAMES.index("rj_dg_2_2")
            hand_err = abs(state.hand_deg[index_flex] - target[6 + index_flex])
            if step % 10 == 0:
                print(f"{step / FPS:>5.1f}  {label:<16} "
                      f"{' '.join(f'{100 * v:6.1f}' for v in offset[:3]):>20} "
                      f"{' '.join(f'{100 * v:6.1f}' for v in (state.tcp[:3] - home[:3])):>20} "
                      f"{pos_mm:>10.2f} {rot_deg:>10.2f} "
                      f"{target[6 + index_flex]:>9.1f}/{state.hand_deg[index_flex]:<9.1f}")
            lag_pos_mm, lag_rot_deg = max(lag_pos_mm, pos_mm), max(lag_rot_deg, rot_deg)
            if args.gif:
                frames.append(np.concatenate([observation["observation.images.cam_front"],
                                              observation["observation.images.cam_wrist"]], axis=1))
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - started)))
            if terminated:
                print("защитная остановка — эпизод прерван")
                break

    print(f"\nнаибольшее отставание в движении: {lag_pos_mm:.2f} мм, {lag_rot_deg:.2f}°")
    print(f"в конце, после 1 с неподвижности: {pos_mm:.2f} мм, {rot_deg:.3f}°, указательный палец {hand_err:.2f}°")
    if args.gif:
        from PIL import Image

        args.gif.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(args.gif, save_all=True, append_images=images[1:], duration=int(1000 / FPS), loop=0)
        print(f"GIF: {args.gif} ({len(images)} кадров, слева cam_front, справа cam_wrist)")
    if viewer is not None:
        viewer.close()
    env.close()


if __name__ == "__main__":
    main()
