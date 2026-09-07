import numpy as np

from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.constants import (
    BROKEN_PINKY_INDEX,
    JOINT_NAMES,
    UPPER_LIMITS_DEG,
)
from lerobot_robot_dg5f.dg5f import apply_disabled_joints


def complete_action(value: float) -> dict[str, float]:
    return {f"{joint}.pos": value for joint in JOINT_NAMES}


def test_disabled_pinky_base_is_always_zero():
    positions = np.arange(len(JOINT_NAMES), dtype=np.float64)
    safe = apply_disabled_joints(positions, {"rj_dg_5_1": 0.0})

    assert safe[BROKEN_PINKY_INDEX] == 0.0
    assert np.array_equal(safe[:BROKEN_PINKY_INDEX], positions[:BROKEN_PINKY_INDEX])
    assert np.array_equal(safe[BROKEN_PINKY_INDEX + 1 :], positions[BROKEN_PINKY_INDEX + 1 :])


def test_mock_robot_clips_steps_limits_and_broken_joint():
    robot = Dg5f(
        Dg5fConfig(
            id="test",
            backend="mock",
            control_smoothing=False,
            min_send_step_deg=0.0,
        )
    )
    robot.connect()
    sent = robot.send_action(complete_action(200.0))

    values = np.asarray([sent[f"{joint}.pos"] for joint in JOINT_NAMES])
    assert np.all(values <= UPPER_LIMITS_DEG)
    assert values[BROKEN_PINKY_INDEX] == 0.0
    robot.disconnect()


def test_mock_observation_uses_lerobot_feature_contract():
    robot = Dg5f(
        Dg5fConfig(
            id="test",
            backend="mock",
            control_smoothing=False,
            min_send_step_deg=0.0,
        )
    )
    robot.connect()
    robot.send_action(complete_action(1.0))
    observation = robot.get_observation()

    assert set(observation) == set(robot.observation_features)
    assert observation[f"{JOINT_NAMES[0]}.pos"] == 1.0
    assert observation["rj_dg_5_1.pos"] == 0.0
    diagnostics = robot.get_diagnostics()
    assert diagnostics["transport_connected"] is True
    assert diagnostics["motion_ready"] is True
    assert len(diagnostics["latest_command_deg"]) == 20
    robot.disconnect()


def test_arm_reseed_uses_current_physical_feedback_before_blend():
    robot = Dg5f(
        Dg5fConfig(
            id="test-reseed",
            backend="mock",
            control_smoothing=False,
            min_send_step_deg=0.0,
            startup_blend_s=0.70,
        )
    )
    robot.connect()
    robot.backend._positions = np.full(20, 12.0)
    robot.backend._positions[BROKEN_PINKY_INDEX] = 0.0

    pose = robot.begin_arm_blend_from_feedback(
        max_pose_age_ms=100.0, duration_s=1.0
    )

    assert pose[0] == 12.0
    assert robot.command_shaper.command_pose_deg[0] == 12.0
    assert robot.command_shaper.arm_blend_active
    robot.disconnect()
