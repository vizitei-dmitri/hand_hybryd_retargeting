import pytest
from lerobot.robots import RobotConfig

from lerobot_robot_ur10_dg5f import Ur10Dg5fConfig


def test_defaults_are_valid_and_registered():
    Ur10Dg5fConfig()
    assert "ur10_dg5f" in RobotConfig.get_known_choices()


@pytest.mark.parametrize("overrides", [
    {"backend": "gazebo"},
    {"hand_backend": "ros"},
    {"workspace_lo": (0.5, 0.35, 0.44)},
    {"home_joints_rad": (0.0,) * 5},
    {"ur_lookahead_s": 0.5},
    {"ur_gain": 50.0},
    {"max_step_m": 0.0},
])
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        Ur10Dg5fConfig(**overrides)
