import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

mujoco = pytest.importorskip("mujoco")

from lerobot_robot_ur10_dg5f.sim_backend import load_model  # noqa: E402
from lerobot_robot_ur10_dg5f.ur10_dg5f import vector_to_action  # noqa: E402


def drive(robot, target, steps):
    for _ in range(steps):
        robot.send_action(vector_to_action(target))
        robot.wait_next_period(0.1)


def angle_deg(rv_a, rv_b):
    relative = R.from_rotvec(rv_a) * R.from_rotvec(rv_b).inv()
    return np.degrees(np.linalg.norm(relative.as_rotvec()))


def test_gravity_compensation_and_stable_integrator(mjcf_path):
    model = load_model(mjcf_path)
    assert model.ngravcomp > 0
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_tcp_is_reported_in_ur_base_frame(make_sim_robot):
    robot = make_sim_robot()
    backend = robot.backend
    world = backend.data.site_xpos[backend._site]
    tcp = backend.read().tcp
    assert np.allclose(tcp[:3], [-world[0], -world[1], world[2]])


def test_home_is_held_without_drift(make_sim_robot):
    robot = make_sim_robot()
    start = robot.backend.read().tcp
    drive(robot, robot.last_command, 50)
    assert np.linalg.norm(robot.backend.read().tcp[:3] - start[:3]) < 1e-3


@pytest.mark.parametrize("offset", [(0.05, 0, 0), (0, 0, 0.05), (0, -0.05, 0)])
def test_tcp_reaches_5cm_offsets(make_sim_robot, offset):
    robot = make_sim_robot()
    target = robot.last_command.copy()
    target[:3] += offset
    drive(robot, target, 20)
    tcp = robot.backend.read().tcp
    assert np.linalg.norm(tcp[:3] - target[:3]) < 0.5e-3
    assert angle_deg(tcp[3:], target[3:6]) < 0.1
    wrists = robot.backend.data.actuator_force[robot.backend._arm_act][3:]
    assert np.all(np.abs(wrists) < 55.0)  # not chattering against the 56 N*m force range


def test_unreachable_target_stays_finite(make_sim_robot):
    robot = make_sim_robot()
    target = robot.last_command[:6].copy()
    target[0] += 3.0
    robot.backend.command(target, np.zeros(20))
    assert np.all(np.isfinite(robot.backend.data.ctrl))


def test_fingers_flex(make_sim_robot):
    from lerobot_robot_dg5f.constants import JOINT_NAMES

    robot = make_sim_robot()
    target = robot.last_command.copy()
    for joint in ("rj_dg_2_2", "rj_dg_2_3", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_4_2", "rj_dg_4_3"):
        target[6 + JOINT_NAMES.index(joint)] = 30.0
    drive(robot, target, 20)
    hand = robot.backend.read().hand_deg
    # The simulated hand servos are soft (kp = 3 N*m/rad) and stop ~2.4 degrees short.
    assert np.max(np.abs(hand - target[6:])) < 3.0


def test_cameras_render(make_sim_robot):
    try:
        robot = make_sim_robot(sim_cameras=("cam_front",), sim_image_hw=(120, 160))
    except Exception as error:  # no OpenGL: run with MUJOCO_GL=egl
        pytest.skip(f"Rendering unavailable: {error}")
    frame = robot.get_observation()["cam_front"]
    assert frame.shape == (120, 160, 3) and frame.dtype == np.uint8 and frame.mean() > 0
