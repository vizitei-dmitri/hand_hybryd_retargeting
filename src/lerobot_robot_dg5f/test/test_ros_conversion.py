import numpy as np
import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot_robot_dg5f.constants import JOINT_NAMES
from lerobot_robot_dg5f.ros_bridge_node import trajectory_to_degrees


def test_named_trajectory_is_reordered_and_converted_to_degrees():
    message = JointTrajectory()
    message.joint_names = list(reversed(JOINT_NAMES))
    point = JointTrajectoryPoint()
    point.positions = list(reversed([np.deg2rad(index) for index in range(20)]))
    message.points = [point]

    assert np.allclose(trajectory_to_degrees(message), np.arange(20))


def test_incomplete_trajectory_is_rejected():
    message = JointTrajectory()
    message.joint_names = [JOINT_NAMES[0]]
    point = JointTrajectoryPoint()
    point.positions = [0.0]
    message.points = [point]

    with pytest.raises(ValueError, match="missing"):
        trajectory_to_degrees(message)
