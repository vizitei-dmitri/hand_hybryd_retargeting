"""Five targeted regressions from the physical-test report; no hardware."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from lerobot_robot_dg5f.debug_recorder import _parse_diagnostics
from lerobot_robot_dg5f.debug_recording import DebugRunWriter
from lerobot_robot_dg5f.ros_bridge_node import Dg5fLeRobotBridge
from test_robot_contact import guard, test_near_robot_rising_current_and_reference_boundary_limit_closing as assert_thumb_index_boundary


def signals(pair, *, measured=0.027, effective=0.027, desired=0.02, joint=13, gradient=-0.001):
    robot = np.full((3, 7), 0.1)
    robot[:, pair] = [measured, effective, desired]
    jac = np.zeros((7, 20))
    jac[pair, joint] = gradient
    human = np.full(7, 0.1)
    human[pair] = 0.0245
    return dict(robot_pair_distances_m=robot, pair_gradients_m_deg=jac,
                pair_distances_m=human, hybrid_weights=np.zeros(5), contact_stamp_s=1)


def test_thumb_index_25mm_boundary_unchanged():
    # Reuse the established boundary, rising-current and no-auto-retract cases.
    assert_thumb_index_boundary()
    assert guard().compliance.thumb_index_contact_distance_mm == 25
    assert guard().compliance.thumb_index_slowdown_start_mm == 35


def test_geometry_relief_cannot_bypass_joint_global_hard_or_stall():
    effective, measured, desired = np.zeros(20), np.zeros(20), np.zeros(20)
    effective[13], measured[13], desired[13] = 83.9, 72.8, 90
    geometry = signals(5, desired=0.04, gradient=0.001)  # geometrically opening middle-ring
    current = np.zeros(20)
    current[13], current[9] = 638, 430  # actual reported ring error + total >1 A
    g = guard()
    decision = g.update(current_ma=current, measured_deg=measured, effective_deg=effective,
                        desired_deg=desired, now=1, **geometry)
    assert decision.diagnostics["global_current_scale"] == 0
    assert decision.diagnostics["joint_tracking_scale"][13] == 0
    assert decision.target_deg[13] == effective[13]
    assert g.update(current_ma=current, measured_deg=measured, effective_deg=effective,
                    desired_deg=desired, now=1.05, **geometry).trip
    current[9], current[13] = 0, 700
    assert guard().update(current_ma=current, measured_deg=measured, effective_deg=effective,
        desired_deg=desired, now=1, **geometry).target_deg[13] == effective[13]
    current[13] = 500
    g = guard()
    for i in range(35):
        decision = g.update(current_ma=current, measured_deg=measured, effective_deg=effective,
                            desired_deg=desired, now=1 + i * 0.02, **geometry)
    assert decision.stall_trip and not decision.trip


def test_sustained_external_load_yields_gradually_and_operator_relief_is_immediate():
    g = guard()
    current, measured, effective, desired = [np.zeros(20) for _ in range(4)]
    current[4], measured[4], effective[4], desired[4] = 300, 50, 56, 80
    # Warm velocity history without sending any synthetic motion.
    for now in [0.96, 0.98]:
        g.update(current_ma=current, measured_deg=measured, effective_deg=effective,
                 desired_deg=effective, now=now)
    yielded = []
    for i in range(150):
        now = 1 + i * 0.02
        previous = effective[4]
        result = g.update(current_ma=current, measured_deg=measured, effective_deg=effective,
                          desired_deg=desired, now=now)
        # Existing direct profile: max 5 degrees per accepted command.
        effective += np.clip(result.target_deg - effective, -5, 5)
        if result.diagnostics["yield_active"]:
            yielded.append(now)
            assert 0 < previous - effective[4] <= 5 * 0.02 + 1e-8
            assert result.diagnostics["yield_delta_deg"][4] < 0
            assert result.diagnostics["joint_tracking_scale"][4] == 0
        assert effective[4] >= measured[4]
        assert not result.trip and not result.stall_trip
    assert yielded and yielded[0] >= 0.98 + g.compliance.yield_hold_s
    assert effective[4] - measured[4] == pytest.approx(2, abs=0.02)
    desired[4] = 20
    result = g.update(current_ma=current, measured_deg=measured, effective_deg=effective,
                      desired_deg=desired, now=4)
    assert result.target_deg[4] == 20
    assert not result.diagnostics["yield_active"]


def test_adjacent_human_and_robot_proximity_at_low_current_stays_fast():
    g = guard()
    geometry = signals(5)
    geometry["robot_pair_distances_m"][:, 4:] = [[0.02] * 3, [0.02] * 3, [0.012] * 3]
    geometry["pair_distances_m"][4:] = 0.0245
    for pair, joint in zip([4, 5, 6], [6, 10, 14]):
        geometry["pair_gradients_m_deg"][pair, joint] = -0.001
    for i in range(20):
        result = g.update(current_ma=np.full(20, 10), measured_deg=np.full(20, 9),
            effective_deg=np.full(20, 10), desired_deg=np.full(20, 40), now=1 + i * 0.02, **geometry)
        assert not result.active
        np.testing.assert_array_equal(result.target_deg, np.full(20, 40))
        np.testing.assert_array_equal(result.diagnostics["joint_tracking_scale"], np.ones(20))


def test_thumb_ring_moderate_load_no_zero_scale_and_all_pairs_reach_timeline(tmp_path):
    g = guard()
    geometry = signals(2, measured=0.033, effective=0.033, desired=0.026, joint=14)
    geometry["hybrid_weights"] = np.array([1, 0, 0, 1, 0])
    current, measured, effective, desired = [np.zeros(20) for _ in range(4)]
    measured[14], effective[14], desired[14] = 55, 60, 80
    scales = []
    for i in range(40):
        current[14] = [180, 200, 220][i % 3]
        result = g.update(current_ma=current, measured_deg=measured, effective_deg=effective,
                          desired_deg=desired, now=1 + i * 0.02, **geometry)
        scales.append(result.diagnostics["joint_tracking_scale"][14])
        assert result.target_deg[14] > effective[14]
        assert not result.diagnostics["yield_active"]
    assert min(scales) >= 0.5
    assert scales[0] == 1 and scales[1] > scales[-1]  # short SOFT attack, no instant stop
    message = DiagnosticArray(status=[DiagnosticStatus(name="dg5f_lerobot_bridge", values=[
        KeyValue(key=k, value=Dg5fLeRobotBridge._diagnostic_text(v)) for k, v in result.diagnostics.items()])])
    writer = DebugRunWriter(tmp_path, {}, arm_events_from_bridge=True)
    writer.record(_parse_diagnostics(message))
    writer.finalize()
    with (writer.run_dir / "timeline.csv").open() as stream:
        row = next(csv.DictReader(stream))
    for pair in ("thumb-index", "thumb-middle", "thumb-ring", "thumb-little", "index-middle", "middle-ring", "ring-little"):
        for prefix in ("robot_measured_pair_distances_mm", "robot_effective_pair_distances_mm",
                       "robot_desired_pair_distances_mm", "pair_tracking_scale"):
            assert f"{prefix}_{pair}" in row
    assert float(row["robot_measured_pair_distances_mm_thumb-ring"]) == 33
    assert float(row["pair_tracking_scale_thumb-ring"]) >= 0.5
    # Distinct pair transitions share event names, not their latch state.
    events = []
    node = SimpleNamespace(_load_event_latches={}, _event=lambda name, **data: events.append((name, data)))
    for i in range(3):
        for pair in ("thumb-ring", "middle-ring"):
            Dg5fLeRobotBridge._load_event_transition(node, "ROBOT_CONTACT_LIMIT", True, 1 + i * .02,
                                                    _event_key=pair, pair=pair, tracking_scale=0.8)
    assert len(events) == 2 and events[0][1]["pair"] != events[1][1]["pair"]
    current.fill(0)
    for i in range(12):
        result = g.update(current_ma=current, measured_deg=measured, effective_deg=effective,
                          desired_deg=desired, now=1.8 + i * .02, **geometry)
    assert result.diagnostics["joint_tracking_scale"][14] == 1
