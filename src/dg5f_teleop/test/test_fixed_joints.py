import numpy as np
import pytest

from dg5f_teleop.constants import JOINT_NAMES
from dg5f_teleop.retarget_node import apply_fixed_joint_positions


def test_first_pinky_joint_is_fixed_without_changing_other_joints():
    target = np.linspace(-0.5, 0.5, len(JOINT_NAMES))
    result = apply_fixed_joint_positions(
        target, JOINT_NAMES, {"rj_dg_5_1": 0.0}
    )
    pinky_base = JOINT_NAMES.index("rj_dg_5_1")

    assert result[pinky_base] == 0.0
    assert np.array_equal(result[:pinky_base], target[:pinky_base])
    assert np.array_equal(result[pinky_base + 1 :], target[pinky_base + 1 :])


def test_unknown_fixed_joint_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        apply_fixed_joint_positions(
            np.zeros(len(JOINT_NAMES)), JOINT_NAMES, {"missing": 0.0}
        )
