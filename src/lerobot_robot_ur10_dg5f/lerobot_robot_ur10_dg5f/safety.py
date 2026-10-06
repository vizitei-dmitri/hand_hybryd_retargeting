"""Command limits applied to every action source: policy, residual, operator, replay."""

import numpy as np
from lerobot_robot_dg5f.constants import LOWER_LIMITS_DEG, UPPER_LIMITS_DEG
from lerobot_robot_dg5f.dg5f import apply_disabled_joints

from .pose_math import apply_offset, pose_difference


def clip_workspace(pose, lo_xyz, hi_xyz):
    """Clip the position to a box in the UR base frame; orientation is untouched."""
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3] = np.clip(result[:3], lo_xyz, hi_xyz)
    return result


def limit_step(current, target, max_lin_m, max_ang_rad):
    """Shorten the step current -> target to max_lin_m and max_ang_rad, keeping its direction."""
    delta = pose_difference(target, current)
    linear = np.linalg.norm(delta[:3])
    angular = np.linalg.norm(delta[3:])
    if linear > max_lin_m:
        delta[:3] *= max_lin_m / linear
    if angular > max_ang_rad:
        delta[3:] *= max_ang_rad / angular
    return apply_offset(current, delta)


def clip_hand(hand_deg, disabled_positions_deg):
    """Clamp DG5F targets to the SDK limits and hold mechanically unavailable joints."""
    hand = np.clip(np.asarray(hand_deg, dtype=np.float64), LOWER_LIMITS_DEG, UPPER_LIMITS_DEG)
    return apply_disabled_joints(hand, disabled_positions_deg)
