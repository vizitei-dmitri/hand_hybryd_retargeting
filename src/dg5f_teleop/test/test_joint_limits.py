import numpy as np
import pytest

from dg5f_teleop.retarget_node import (
    NON_THUMB_DISTAL_FLEXION_JOINTS,
    prevent_distal_hyperextension,
)


class _Optimizer:
    def __init__(self, names):
        self.target_joint_names = names
        self.applied_limits = None
        self.applied_epsilon = None

    def set_joint_limit(self, limits, epsilon=1e-3):
        self.applied_limits = limits.copy()
        self.applied_epsilon = epsilon


class _Retargeting:
    def __init__(self, names):
        self.optimizer = _Optimizer(names)
        self.joint_limits = np.tile([-1.57, 1.57], (len(names), 1))
        self.last_qpos = np.full(len(names), -0.5)


def test_prevent_distal_hyperextension_changes_only_named_distal_joints():
    names = ["rj_dg_1_3", *NON_THUMB_DISTAL_FLEXION_JOINTS, "other_joint"]
    retargeting = _Retargeting(names)

    changed = prevent_distal_hyperextension(retargeting, minimum=0.0)

    assert changed == list(NON_THUMB_DISTAL_FLEXION_JOINTS)
    assert retargeting.joint_limits[0, 0] == pytest.approx(-1.57)
    assert retargeting.joint_limits[-1, 0] == pytest.approx(-1.57)
    assert np.all(retargeting.joint_limits[1:-1, 0] == 0.0)
    assert np.all(retargeting.last_qpos[1:-1] == 0.0)
    assert retargeting.optimizer.applied_epsilon == 0.0


def test_prevent_distal_hyperextension_rejects_invalid_minimum():
    retargeting = _Retargeting(list(NON_THUMB_DISTAL_FLEXION_JOINTS))
    with pytest.raises(ValueError, match="finite"):
        prevent_distal_hyperextension(retargeting, minimum=np.nan)
