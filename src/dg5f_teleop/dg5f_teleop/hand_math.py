"""Pure numerical helpers for converting 21 hand landmarks to MANO frame."""

import numpy as np


# Same operator-to-MANO transform used by the dex-retargeting right-hand
# real-time example. Input is represented as row vectors.
OPERATOR_TO_MANO_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)



def estimate_wrist_frame(points: np.ndarray) -> np.ndarray:
    """Estimate an orthonormal wrist frame from MANO-ordered landmarks.

    The expected order is wrist=0, thumb=1..4, index=5..8,
    middle=9..12, ring=13..16, little=17..20.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Landmarks contain NaN or infinity")

    palm_points = points[[0, 5, 9], :]
    forward = palm_points[0] - palm_points[2]
    centered_palm = palm_points - np.mean(palm_points, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered_palm)

    if singular_values[1] < 1e-8:
        raise ValueError("Palm landmarks are degenerate")

    normal = vh[2, :]
    x_axis = forward - np.dot(forward, normal) * normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Cannot determine palm forward axis")
    x_axis /= x_norm

    z_axis = np.cross(x_axis, normal)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("Cannot determine palm lateral axis")
    z_axis /= z_norm

    # Resolve the normal sign using index relative to middle MCP.
    if np.dot(z_axis, centered_palm[1] - centered_palm[2]) < 0.0:
        normal *= -1.0
        z_axis *= -1.0

    return np.stack([x_axis, normal, z_axis], axis=1)


def landmarks_to_mano(points: np.ndarray) -> np.ndarray:
    """Return wrist-relative landmarks in dex-retargeting MANO convention."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {points.shape}")

    wrist_relative = points - points[0:1, :]
    wrist_frame = estimate_wrist_frame(wrist_relative)
    mano_points = wrist_relative @ wrist_frame @ OPERATOR_TO_MANO_RIGHT

    if not np.all(np.isfinite(mano_points)):
        raise ValueError("MANO transform produced invalid coordinates")
    return mano_points
