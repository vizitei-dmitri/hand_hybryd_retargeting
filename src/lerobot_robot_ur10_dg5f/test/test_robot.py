import numpy as np
import pytest
from lerobot_robot_dg5f.constants import UPPER_LIMITS_DEG

from lerobot_robot_ur10_dg5f.ur10_dg5f import ACTION_KEYS, vector_to_action


def test_feature_counts(make_sim_robot):
    robot = make_sim_robot()
    assert len(robot.action_features) == 26
    assert len(robot.observation_features) == 6 + 6 + 20


def test_large_jump_is_limited_to_one_step(make_sim_robot):
    robot = make_sim_robot()
    previous = robot.last_command[:6].copy()
    target = robot.last_command.copy()
    target[1] += 0.30
    sent = robot.send_action(vector_to_action(target))
    sent_tcp = np.array([sent[key] for key in ACTION_KEYS[:6]])
    assert np.isclose(np.linalg.norm(sent_tcp[:3] - previous[:3]), robot.config.max_step_m)


def test_target_outside_workspace_is_clipped(make_sim_robot):
    robot = make_sim_robot()
    target = robot.last_command.copy()
    target[2] = 0.0  # below the table
    for _ in range(200):
        sent = robot.send_action(vector_to_action(target))
    assert np.isclose(sent["tcp.z"], robot.config.workspace_lo[2])


def test_hand_limits_and_pinky(make_sim_robot):
    robot = make_sim_robot()
    target = robot.last_command.copy()
    target[6:] = 999.0
    sent = robot.send_action(vector_to_action(target))
    hand = np.array([sent[key] for key in ACTION_KEYS[6:]])
    assert hand[16] == 0.0  # rj_dg_5_1
    assert np.allclose(np.delete(hand, 16), np.delete(UPPER_LIMITS_DEG, 16))


def test_nan_is_rejected(make_sim_robot):
    robot = make_sim_robot()
    target = robot.last_command.copy()
    target[0] = np.nan
    with pytest.raises(ValueError):
        robot.send_action(vector_to_action(target))
