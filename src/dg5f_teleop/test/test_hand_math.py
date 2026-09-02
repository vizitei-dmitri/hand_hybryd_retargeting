import numpy as np
import pytest

from dg5f_teleop.hand_math import landmarks_to_mano


def open_hand_points() -> np.ndarray:
    points = [np.array([0.0, 0.0, 0.0])]
    points.extend(
        [
            np.array([0.025, 0.025, 0.0]),
            np.array([0.045, 0.042, 0.0]),
            np.array([0.060, 0.057, 0.0]),
            np.array([0.073, 0.068, 0.0]),
        ]
    )
    for x, base_y, lengths in [
        (0.030, 0.065, [0.038, 0.026, 0.020]),
        (0.010, 0.072, [0.043, 0.030, 0.022]),
        (-0.012, 0.067, [0.040, 0.028, 0.021]),
        (-0.032, 0.057, [0.032, 0.023, 0.018]),
    ]:
        position = np.array([x, base_y, 0.0])
        points.append(position.copy())
        for length in lengths:
            position = position + np.array([0.0, length, 0.0])
            points.append(position.copy())
    return np.asarray(points)


def test_landmarks_to_mano_is_finite_and_wrist_relative():
    transformed = landmarks_to_mano(open_hand_points())
    assert transformed.shape == (21, 3)
    assert np.all(np.isfinite(transformed))
    assert np.allclose(transformed[0], 0.0)


def test_landmarks_to_mano_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        landmarks_to_mano(np.zeros((20, 3)))
