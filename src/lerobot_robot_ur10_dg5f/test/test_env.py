import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from lerobot.utils.constants import OBS_STATE

from lerobot_robot_ur10_dg5f.env import STATE_DIM, Ur10Dg5fEnv


def test_passes_gymnasium_checker(make_sim_robot):
    env = Ur10Dg5fEnv(make_sim_robot(), fps=10.0, episode_s=2.0)
    check_env(env, skip_render_check=True)


def test_image_keys_follow_lerobot_naming(make_sim_robot):
    try:
        robot = make_sim_robot(sim_cameras=("cam_front",), sim_image_hw=(60, 80))
    except Exception as error:
        pytest.skip(f"Rendering unavailable: {error}")
    observation, _ = Ur10Dg5fEnv(robot).reset()
    assert set(observation) == {OBS_STATE, "observation.images.cam_front"}
    assert observation[OBS_STATE].shape == (STATE_DIM,)


def test_hold_action_keeps_tcp(make_sim_robot):
    env = Ur10Dg5fEnv(make_sim_robot())
    env.reset()
    hold = env.robot.last_command.copy()
    start = env.robot.backend.read().tcp[:3]
    for _ in range(100):
        env.step(hold)
    assert np.linalg.norm(env.robot.backend.read().tcp[:3] - start) < 1e-3


def test_truncates_exactly_at_episode_length(make_sim_robot):
    env = Ur10Dg5fEnv(make_sim_robot(), fps=10.0, episode_s=1.5)
    env.reset()
    hold = env.robot.last_command.copy()
    flags = [env.step(hold)[3] for _ in range(15)]
    assert flags == [False] * 14 + [True]
