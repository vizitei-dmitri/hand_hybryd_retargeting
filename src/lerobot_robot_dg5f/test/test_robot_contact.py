"""Four focused robot-space contact regressions. No SDK or physical backend."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import String

from lerobot_robot_dg5f.experimental_current_guard import AdaptiveCurrentGuard, ComplianceConfig
from lerobot_robot_dg5f.contact_kinematics import ContactKinematics
from lerobot_robot_dg5f.debug_recording import DebugRunWriter
from lerobot_robot_dg5f.debug_recorder import Dg5fDebugRecorder
from lerobot_robot_dg5f.ros_bridge_node import Dg5fLeRobotBridge


def guard():
    return AdaptiveCurrentGuard(20, soft_ma=350, hard_ma=650, trip_ma=850,
        total_soft_ma=600, total_hard_ma=850, total_trip_ma=1050,
        trip_hold_s=0.04, release_tau_s=0.2, nominal_step_deg=5,
        compliance=ComplianceConfig())


def frame(g, *, measured_mm=28, effective_mm=28, desired_mm=7, current=0, now=1):
    effective = np.zeros(20)
    effective[6] = 60
    measured = effective.copy()
    measured[6] += effective_mm - measured_mm
    desired = effective.copy()
    desired[6] += effective_mm - desired_mm
    currents = np.zeros(20)
    currents[6] = current
    robot_distances = np.full((3, 7), 0.1)
    robot_distances[:, 0] = np.array([measured_mm, effective_mm, desired_mm]) / 1000
    gradients = np.zeros((7, 20))
    gradients[0, 6] = -0.001
    human = np.full(7, 0.1)
    human[0] = 0.026
    def distances(q):
        result = np.full(7, 0.1)
        result[0] = effective_mm / 1000 - (q[6] - 60) * 0.001
        return result
    return g.update(current_ma=currents, measured_deg=measured, effective_deg=effective,
        desired_deg=desired, now=now, hybrid_weights=np.array([1, 1, 0, 0, 0]),
        pair_distances_m=human, contact_stamp_s=now, pair_gradients_m_deg=gradients,
        robot_pair_distances_m=robot_distances, robot_distance_fn=distances)


def test_human_pinch_far_robot_stays_fast_and_fk_reuses_same_model():
    result = frame(guard(), measured_mm=85, effective_mm=70, desired_mm=7, current=30)
    assert result.diagnostics["thumb_index_human_contact_weight"] == 1
    assert result.diagnostics["thumb_index_contact_tracking_scale"] == 1
    assert result.target_deg[6] == 123  # unchanged desired; normal shaper follows
    assert not result.active
    # Bridge computes all three poses using its existing URDF evaluator, even
    # without human side-channel packets. No second kinematic model is built.
    fk = ContactKinematics(Path(__file__).parents[3] / "models/dg5f/urdf/dg5f_right.urdf")
    node = SimpleNamespace(_contact_samples={}, _contact_kinematics=fk,
        get_parameter=lambda _: SimpleNamespace(value=0.15), _warn_throttled=lambda _: None)
    poses = [np.zeros(20), np.full(20, 10), np.full(20, 20)]
    signals = Dg5fLeRobotBridge._contact_for_diagnostics(node, poses[1], 1, measured=poses[0], desired=poses[2])
    for index, pose in enumerate(poses):
        tips, _ = fk.tips_and_jacobians(pose)
        assert signals["robot_pair_distances_m"][index, 0] == pytest.approx(np.linalg.norm(tips[1] - tips[0]))


def test_near_robot_rising_current_and_reference_boundary_limit_closing():
    g = guard()
    results = [frame(g, current=current, now=1 + i * 0.02)
               for i, current in enumerate([50, 100, 150, 200, 250, 300])]
    assert results[-1].target_deg[6] - 60 < 0.5
    assert results[-1].target_deg[6] < results[0].target_deg[6]
    assert results[-1].diagnostics["joint_current_slope_ma_s"][6] > 0
    assert not results[-1].trip
    # No current is required to stop pushing a 25 mm reference toward 7 mm.
    at_boundary = frame(guard(), measured_mm=25, effective_mm=25)
    assert at_boundary.target_deg[6] == 60
    # Already deeper than the boundary: hold, never auto-retract toward 25 mm.
    already_inside = frame(guard(), measured_mm=25, effective_mm=24)
    assert already_inside.target_deg[6] == 60
    # Large closing step from outside the slowdown region cannot jump through
    # the reference boundary (gradient here models a fast distance change).
    gradients = np.zeros((7, 20))
    gradients[0, 6] = -0.004
    pose = np.zeros(20)
    target = pose.copy()
    target[6] = 20
    distances = np.full((3, 7), 0.1)
    distances[:, 0] = [0.04, 0.04, 0.005]
    crossing = guard().update(current_ma=pose, measured_deg=pose, effective_deg=pose,
        desired_deg=target, now=1, pair_gradients_m_deg=gradients, robot_pair_distances_m=distances)
    assert crossing.target_deg[6] <= 3.75  # 40 - 3.75*4 = 25 mm


def test_relief_is_fast_and_contact_transitions_reach_debug(tmp_path):
    g = guard()
    closing = frame(g, measured_mm=25, effective_mm=25, current=500)
    # Actual actuator relief too: a geometric opening alone must NOT bypass
    # a hard-current freeze when effective==measured (zero actuator error).
    relief = frame(g, measured_mm=26, effective_mm=25, desired_mm=45, current=700, now=1.02)
    assert relief.target_deg[6] == 40
    assert relief.diagnostics["joint_tracking_scale"][6] == 1
    assert relief.diagnostics["thumb_index_contact_tracking_scale"] == 1
    assert not relief.diagnostics["robot_contact_limit_active"]
    writer = DebugRunWriter(tmp_path, {}, arm_events_from_bridge=True, run_stamp="robot-contact")
    recorder = SimpleNamespace(writer=writer)
    bridge = SimpleNamespace(_load_event_latches={})
    bridge._event = lambda name, **details: Dg5fDebugRecorder._on_event(recorder, String(data=json.dumps({"event": name, **details})))
    for now, result in [(1, closing), (1.02, closing), (1.04, relief), (1.25, relief)]:
        details = {key: value for key, value in result.diagnostics.items() if key.startswith("thumb_index_")}
        Dg5fLeRobotBridge._load_event_transition(bridge, "ROBOT_CONTACT_LIMIT",
            result.diagnostics["robot_contact_limit_active"], now, **details)
        writer.record(details)
    writer.finalize()
    assert writer.event_counts["ROBOT_CONTACT_LIMIT_ACTIVE"] == 1
    assert writer.event_counts["ROBOT_CONTACT_LIMIT_RELEASED"] == 1
    with (writer.run_dir / "timeline.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["thumb_index_measured_tip_distance_mm"]) == 25
    assert float(row["thumb_index_desired_tip_distance_mm"]) == 7


def test_robot_contact_does_not_bypass_original_emergency_trip():
    g = guard()
    assert not frame(g, current=900).trip
    # Even opening cannot cancel sustained absolute overcurrent protection.
    assert frame(g, desired_mm=45, current=900, now=1.05).trip
    assert g.trip_ma == 850 and g.total_trip_ma == 1050 and g.trip_hold_s == 0.04
