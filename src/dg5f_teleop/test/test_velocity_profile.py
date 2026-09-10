from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Pose, PoseArray

from dg5f_teleop import retarget_node


@pytest.mark.parametrize("limit,expected", [(0.0, 1.0), (3.0, 0.06)])
@pytest.mark.parametrize("contact_failure", [False, True])
def test_retarget_velocity_cap_can_be_disabled_without_removing_fault_map(monkeypatch, limit, expected, contact_failure):
    monkeypatch.setattr(retarget_node, "landmarks_to_mano", lambda points: points)
    monkeypatch.setattr(retarget_node, "reference_for", lambda model, points: points)
    outputs = []
    def publish_contact(*_):
        if contact_failure:
            raise ValueError("synthetic observer failure")
    node = SimpleNamespace(
        _now_seconds=lambda: 1.02,
        _retargeting=SimpleNamespace(retarget=lambda _: np.ones(20)),
        _output_indices=np.arange(20), _hybrid=False,
        _last_output=np.zeros(20), _last_command_time=1.0,
        get_parameter=lambda _: SimpleNamespace(value=limit),
        _fixed_positions={"rj_dg_5_1": 0.0},
        _set_tracking=lambda _: None,
        _publish_command=lambda source, output: outputs.append(output),
        _publish_contact=publish_contact,
        _warn_throttled=lambda _: None,
    )
    message = PoseArray(poses=[Pose() for _ in range(21)])
    retarget_node.RetargetNode._on_landmarks(node, message)
    assert outputs[0][0] == pytest.approx(expected)
    assert outputs[0][16] == 0.0
