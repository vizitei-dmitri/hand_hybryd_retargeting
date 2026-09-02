"""Shared DG5F naming constants."""

JOINT_NAMES = [
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]

ACTUATOR_NAMES = [f"{name}_ctrl" for name in JOINT_NAMES]

FINGER_TIP_LINK_NAMES = [f"rl_dg_{finger}_tip" for finger in range(1, 6)]
