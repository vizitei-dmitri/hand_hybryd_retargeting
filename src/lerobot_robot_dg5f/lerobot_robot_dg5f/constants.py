"""DG5F joint metadata shared by the LeRobot plugin and ROS bridge."""

import numpy as np


JOINT_NAMES = tuple(
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)

# Tesollo DGSDK uses degrees and the same finger-major order as JOINT_NAMES.
LOWER_LIMITS_DEG = np.asarray(
    [
        -22, -180, -90, -90,
        -24, 0, -90, -90,
        -35, 0, -90, -90,
        -35, 0, -90, -90,
        0, -24, -90, -90,
    ],
    dtype=np.float64,
)
UPPER_LIMITS_DEG = np.asarray(
    [
        51, 0, 90, 90,
        35, 115, 90, 90,
        35, 112, 90, 90,
        24, 109, 90, 90,
        60, 35, 90, 90,
    ],
    dtype=np.float64,
)

BROKEN_PINKY_JOINT = "rj_dg_5_1"
BROKEN_PINKY_INDEX = JOINT_NAMES.index(BROKEN_PINKY_JOINT)

TELEMETRY_FIELDS = ("pos", "vel", "current", "temp")

# DGControl::_checkTemp() stops consuming motion targets at this threshold.
TESOLLO_TEMPERATURE_LIMIT_C = 65.0
