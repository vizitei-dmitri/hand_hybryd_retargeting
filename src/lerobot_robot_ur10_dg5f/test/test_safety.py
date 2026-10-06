import numpy as np
from lerobot_robot_dg5f.constants import BROKEN_PINKY_INDEX, UPPER_LIMITS_DEG
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ur10_dg5f.safety import clip_hand, clip_workspace, limit_step


CURRENT = np.r_[0.0, 0.5, 0.5, 0.0, 0.0, 0.0]


def test_long_step_is_shortened_along_its_direction():
    target = CURRENT + np.r_[0.3, 0.0, 0.4, 0.0, 0.0, 0.0]
    out = limit_step(CURRENT, target, 0.01, 0.05)
    step = out[:3] - CURRENT[:3]
    assert np.isclose(np.linalg.norm(step), 0.01)
    assert np.allclose(step / np.linalg.norm(step), [0.6, 0.0, 0.8])


def test_short_step_is_unchanged():
    target = CURRENT + np.r_[0.002, -0.003, 0.001, 0.0, 0.0, 0.01]
    assert np.allclose(limit_step(CURRENT, target, 0.01, 0.05), target)


def test_rotation_step_is_capped():
    target = np.r_[CURRENT[:3], 1.0, 0.0, 0.0]
    out = limit_step(CURRENT, target, 0.01, 0.05)
    assert np.isclose(np.linalg.norm(R.from_rotvec(out[3:]).as_rotvec()), 0.05)
    assert np.allclose(out[3:] / np.linalg.norm(out[3:]), [1.0, 0.0, 0.0])


def test_point_outside_box_lands_on_its_face():
    out = clip_workspace(np.r_[1.0, 0.5, 0.1, 0.1, 0.2, 0.3], (-0.4, 0.35, 0.44), (0.4, 0.9, 0.8))
    assert np.allclose(out, [0.4, 0.5, 0.44, 0.1, 0.2, 0.3])


def test_hand_is_clamped_and_pinky_held():
    out = clip_hand(np.full(20, 999.0), {"rj_dg_5_1": 0.0})
    expected = UPPER_LIMITS_DEG.copy()
    expected[BROKEN_PINKY_INDEX] = 0.0
    assert np.allclose(out, expected)
