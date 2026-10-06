import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ur10_dg5f.pose_math import (
    apply_offset,
    pose_difference,
    rotate_offset,
    rotvec_to_6d,
)


def random_pose(rng):
    return np.r_[rng.normal(size=3), R.random(random_state=rng).as_rotvec()]


def same_rotation(rv_a, rv_b, atol=1e-9):
    # Compare matrices, not rotation vectors: near pi the rotation vector is ambiguous.
    return np.allclose(R.from_rotvec(rv_a).as_matrix(), R.from_rotvec(rv_b).as_matrix(), atol=atol)


def test_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = random_pose(rng), random_pose(rng)
        out = apply_offset(b, pose_difference(a, b))
        assert np.allclose(out[:3], a[:3]) and same_rotation(out[3:], a[3:])


def test_zero_offset_is_identity():
    b = random_pose(np.random.default_rng(1))
    out = apply_offset(b, np.zeros(6))
    assert np.allclose(out[:3], b[:3]) and same_rotation(out[3:], b[3:])


def test_offset_is_in_base_frame():
    base = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]  # tool yawed by 90 degrees
    out = apply_offset(base, np.r_[0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(out[:3], [0.1, 0.0, 0.0])  # moved along base x, not tool x


def test_rotate_offset_matches_conjugation():
    rng = np.random.default_rng(2)
    a = R.random(random_state=rng).as_matrix()
    delta = random_pose(rng)
    rotated = rotate_offset(delta, a)
    expected = a @ R.from_rotvec(delta[3:]).as_matrix() @ a.T
    assert np.allclose(R.from_rotvec(rotated[3:]).as_matrix(), expected)
    assert np.allclose(rotated[:3], a @ delta[:3])


def test_6d_is_continuous_across_pi():
    just_below = rotvec_to_6d([0.0, 0.0, np.pi - 1e-6])
    just_above = rotvec_to_6d([0.0, 0.0, -(np.pi - 1e-6)])  # same rotation from the other side
    assert np.allclose(just_below, just_above, atol=1e-5)
