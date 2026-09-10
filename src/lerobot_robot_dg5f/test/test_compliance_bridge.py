"""ROS boundary/event tests with fake objects: never open a hardware socket."""

import json
from types import SimpleNamespace

import numpy as np
from std_msgs.msg import Float64MultiArray

from lerobot_robot_dg5f.ros_bridge_node import Dg5fLeRobotBridge


def test_source_freshness_nan_and_replays_cannot_refresh_diagnostic_contact():
    clock = [100.0]
    node = SimpleNamespace(
        _contact_samples={},
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock[0] * 1e9)),
        get_parameter=lambda _: SimpleNamespace(value=0.15),
        _now_seconds=lambda: clock[0], _warn_throttled=lambda _: None,
        _contact_kinematics=SimpleNamespace(pair_gradients=lambda _: np.zeros((7, 20))),
    )
    message = Float64MultiArray(data=[100.0] + [0.02] * 7)
    Dg5fLeRobotBridge._on_contact(node, message, hybrid=False)
    assert Dg5fLeRobotBridge._contact_for_diagnostics(node, np.zeros(20), 100.0)
    clock[0] += 0.10
    Dg5fLeRobotBridge._on_contact(node, message, hybrid=False)
    assert node._contact_samples["proximity"][1] == 100.0
    assert not Dg5fLeRobotBridge._contact_for_diagnostics(node, np.zeros(20), 100.16)
    clock[0] = 100.2
    Dg5fLeRobotBridge._on_contact(node, message, hybrid=False)
    assert not node._contact_samples
    message.data = [100.2] + [float("nan")] * 7
    Dg5fLeRobotBridge._on_contact(node, message, hybrid=False)
    assert not node._contact_samples


def test_event_latch_filters_contact_chatter_but_not_actual_frame_state():
    events = []
    node = SimpleNamespace(_load_event_latches={}, _event=lambda name, **details: events.append(name))
    for i in range(20):
        Dg5fLeRobotBridge._load_event_transition(node, "COMPLIANCE", i % 2 == 0, i * 0.02)
    assert events == ["COMPLIANCE_ACTIVE"]
    for now in [0.4, 0.5, 0.61]:
        Dg5fLeRobotBridge._load_event_transition(node, "COMPLIANCE", False, now)
    assert events == ["COMPLIANCE_ACTIVE", "COMPLIANCE_RELEASED"]
    Dg5fLeRobotBridge._load_event_transition(node, "COMPLIANCE", True, 0.62)
    assert events[-1] == "COMPLIANCE_ACTIVE"


def test_event_payload_is_strict_json_even_without_contact():
    messages = []
    node = SimpleNamespace(_events_pub=SimpleNamespace(publish=messages.append))
    Dg5fLeRobotBridge._event(node, "COMPLIANCE_ACTIVE", pair_distances_m=np.full(7, np.nan),
                            joint_tracking_scale=np.ones(20), sample=np.float64(float("inf")))
    def reject(value):
        raise AssertionError(f"Non-standard JSON: {value}")
    decoded = json.loads(messages[0].data, parse_constant=reject)
    assert decoded["pair_distances_m"] == [None] * 7
    assert decoded["sample"] is None
    assert decoded["joint_tracking_scale"] == [1] * 20
