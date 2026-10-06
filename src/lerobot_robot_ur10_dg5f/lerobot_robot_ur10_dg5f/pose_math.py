"""UR-style poses ``[x, y, z, rx, ry, rz]``: metres and a rotation vector, UR base frame."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def apply_offset(base, delta):
    """Offset expressed in the UR base frame: p = p_b + dp, R = Exp(dr) @ R_b."""
    base = np.asarray(base, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    rotation = R.from_rotvec(delta[3:6]) * R.from_rotvec(base[3:6])
    return np.concatenate([base[:3] + delta[:3], rotation.as_rotvec()])


def pose_difference(a, b):
    """Inverse of apply_offset: apply_offset(b, pose_difference(a, b)) == a."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rotation = R.from_rotvec(a[3:6]) * R.from_rotvec(b[3:6]).inv()
    return np.concatenate([a[:3] - b[:3], rotation.as_rotvec()])


def rotate_offset(delta, rotation_matrix):
    """Re-express an offset in another frame: dp -> A dp, dr -> A dr (A Exp(dr) A^T = Exp(A dr))."""
    delta = np.asarray(delta, dtype=np.float64)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    return np.concatenate([rotation_matrix @ delta[:3], rotation_matrix @ delta[3:6]])


def rotvec_to_6d(rotvec):
    """First two columns of the rotation matrix, a continuous input for networks."""
    matrix = R.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    return matrix[:, :2].T.reshape(6)
