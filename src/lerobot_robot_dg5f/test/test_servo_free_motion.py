"""Regression for FREE fingers inheriting the legacy 15 deg/s contact resume."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import String

from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.current_guard import AdaptiveCurrentGuard, ComplianceConfig
from lerobot_robot_dg5f.object_contact import ObjectContactConfig, PerFingerObjectContact
from lerobot_robot_dg5f.servo_controller import GuardedServoCommand
from lerobot_robot_dg5f.debug_recording import DebugRunWriter
from lerobot_robot_dg5f.debug_recorder import Dg5fDebugRecorder


def make_guard():
    return AdaptiveCurrentGuard(20, soft_ma=350, hard_ma=650, trip_ma=850,
        total_soft_ma=600, total_hard_ma=850, total_trip_ma=1050,
        trip_hold_s=.04, release_tau_s=.2, nominal_step_deg=2,
        compliance=ComplianceConfig(), object_contact=ObjectContactConfig())


def advance(robot, guard, target, measured, currents, now):
    decisions = []
    def safety(proposal):
        d = guard.update(current_ma=currents, measured_deg=measured,
                         effective_deg=robot.command_shaper.command_pose_deg,
                         desired_deg=proposal, operator_target_deg=target, now=now,
                         trajectory_braking=robot.command_shaper.telemetry["trajectory_braking"])
        decisions.append(d)
        return GuardedServoCommand(d.target_deg, d.diagnostics['physical_lower_deg'],
                                   d.diagnostics['physical_upper_deg'])
    robot.servo_tick(target, guard=safety, now=now)
    return decisions[0]


@pytest.mark.parametrize('finger,joint', [('index', 6), ('middle', 10), ('ring', 14), ('little', 18)])
@pytest.mark.parametrize('current', [5, 8, 20])
@pytest.mark.parametrize('released_offset', [False, True])
def test_free_progress_uses_full_servo_step_for_every_finger(finger, joint, current, released_offset):
    robot = Dg5f(Dg5fConfig(control_mode='servo'))
    robot.connect()
    guard = make_guard()
    target = np.zeros(20)
    target[joint] = 66
    currents = np.zeros(20)
    currents[joint] = current
    state = guard._object_contact.fingers[finger]
    if released_offset:
        state.reason = 'OPERATOR_OPENING'
        state.resume_offset = np.full(len(guard._object_contact.chains[finger]), 50.)
        guard._object_contact.last_time = 0
        guard._object_contact.last_desired = target.copy()  # Stationary human target.
    try:
        for tick in range(1, 21):
            previous = robot.command_shaper.command_pose_deg.copy()
            measured = previous.copy()
            measured[joint] -= 3  # Normal lag, progressing +2 deg every tick.
            decision = advance(robot, guard, target, measured, currents, tick / 60)
            d = decision.diagnostics
            assert d['joint_tracking_scale'][joint] == pytest.approx(1)
            expected_step = min(tick * .2, 2)
            assert robot.command_shaper.command_pose_deg[joint] - previous[joint] == pytest.approx(expected_step)
            assert robot.command_shaper.effective_command()[joint] - previous[joint] == pytest.approx(expected_step)
            assert d['joint_limiting_reason'][joint] == 'NONE'
            assert not d['joint_object_contact_evidence'][joint]
            assert state.state == 'FREE' and state.resume_offset is None
            assert robot.command_shaper.telemetry['physical_limit_reason'][joint] == 'NONE'
        assert d['object_contact_window_ready']
        assert d['joint_measured_motion_deg'][joint] > 0
        assert robot.config.servo_rate_hz == 60
        assert robot.config.servo_max_velocity_deg_s == 120
    finally:
        robot.disconnect()


def test_logged_free_sample_is_legacy_resume_ramp_not_current_or_lead():
    dt = .017180593000375666
    effective = np.zeros(20); effective[6] = -1.9887658842889016
    measured = effective.copy(); measured[6] = -3.700000047683716
    desired = effective.copy(); desired[6] += 2
    operator = effective.copy(); operator[6] = 66.07918119572957
    current = np.zeros(20); current[6] = 8
    outputs = []
    for ramp in (True, False):
        contact = PerFingerObjectContact(ObjectContactConfig())
        contact.last_time = 0
        contact.last_desired = operator.copy()
        contact.fingers['index'].reason = 'OPERATOR_OPENING'
        contact.fingers['index'].resume_offset = np.array([0., 50., 0.])
        decision = contact.update(desired_deg=operator, effective_deg=effective,
            measured_deg=measured, current_ma=current, slope_ma_s=np.zeros(20),
            soft_target_deg=desired, now=dt, apply_resume_ramp=ramp)
        outputs.append(decision.target_deg[6])
    assert (outputs[0] - effective[6]) / 2 == pytest.approx(.1288544475028175)
    assert outputs[1] - effective[6] == pytest.approx(2)


@pytest.mark.parametrize('finger,joint', [('index', 6), ('middle', 10), ('ring', 14), ('little', 18)])
def test_real_contact_holds_then_release_restores_free_servo(finger, joint):
    robot = Dg5f(Dg5fConfig(control_mode='servo'))
    robot.connect()
    initial = np.zeros(20); initial[joint] = 10
    robot.command_shaper.reset(initial, now=0)
    guard = make_guard()
    target = initial.copy(); target[joint] = 66
    measured = np.zeros(20)
    currents = np.zeros(20); currents[joint] = 500
    events = []
    try:
        for tick in range(1, 31):
            previous = robot.command_shaper.command_pose_deg.copy()
            d = advance(robot, guard, target, measured, currents, tick / 60)
            events.extend(d.object_contact_events)
            assert abs(robot.command_shaper.effective_command()[joint] - measured[joint]) <= d.diagnostics['joint_lead_budget_deg'][joint] + 1e-8
        assert guard._object_contact.fingers[finger].state == 'CONTACT_HOLD'
        assert d.diagnostics['joint_tracking_scale'][joint] < 1
        assert d.target_deg[joint] <= previous[joint]  # HOLD/yield, no added closure.
        assert any(e['event'] == 'OBJECT_CONTACT_LATCHED' for e in events)
        assert 'OBJECT_CONTACT_HOLD' in d.diagnostics['joint_limiting_reason'][joint]
        target[joint] = -10  # Explicit opening releases the actual contact.
        currents[joint] = 8
        d = advance(robot, guard, target, measured, currents, 31 / 60)
        assert any(e['event'] == 'OBJECT_CONTACT_RELEASED' for e in d.object_contact_events)
        assert guard._object_contact.fingers[finger].resume_offset is None
        target[joint] = 66
        for tick in range(32, 152):
            previous = robot.command_shaper.command_pose_deg.copy()
            previous_v = robot.command_shaper.velocity_deg_s[joint]
            measured = previous.copy(); measured[joint] -= 3
            d = advance(robot, guard, target, measured, currents, tick / 60)
            step = robot.command_shaper.command_pose_deg[joint] - previous[joint]
            assert abs(step) <= 2 + 1e-8
            if step < -1e-8:
                assert robot.command_shaper.telemetry['trajectory_braking'][joint]
                assert abs(robot.command_shaper.velocity_deg_s[joint]) < abs(previous_v)
            assert guard._object_contact.fingers[finger].state == 'FREE'
        assert d.diagnostics['joint_tracking_scale'][joint] > .99
        assert robot.command_shaper.command_pose_deg[joint] == pytest.approx(66)
    finally:
        robot.disconnect()


def test_scale_decomposition_is_saved_by_existing_servo_recorder(tmp_path):
    guard = make_guard()
    q = np.zeros(20)
    d = guard.update(current_ma=q, measured_deg=q - 3, effective_deg=q,
                     desired_deg=q + 2, operator_target_deg=q + 66, now=1)
    payload = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.diagnostics.items()}
    writer = DebugRunWriter(tmp_path, {}, run_stamp='scale_factors')
    Dg5fDebugRecorder._on_servo(SimpleNamespace(writer=writer), String(data=json.dumps(payload)))
    writer.finalize()
    saved = json.loads((writer.run_dir / 'servo_ticks.jsonl').read_text())
    for key in ('joint_raw_current_scale', 'joint_current_scale', 'joint_slope_scale',
                'joint_contact_scale', 'joint_raw_adaptive_scale', 'joint_adaptive_scale',
                'joint_combined_scale', 'joint_lead_step_scale', 'joint_compliance_gain',
                'joint_object_contact_gain', 'joint_guard_gain', 'joint_tracking_scale',
                'joint_limiting_reason', 'joint_progress_ratio', 'joint_measured_motion_deg',
                'joint_command_error_deg', 'object_contact_resume_offset_deg'):
        assert saved[key] == payload[key]
