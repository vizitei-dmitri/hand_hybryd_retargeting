"""Trajectory acceleration precedes safety; feedback never drives q_cmd/v_cmd."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import String

from lerobot_robot_dg5f.servo_controller import ServoController, GuardedServoCommand
from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.debug_recorder import Dg5fDebugRecorder
from lerobot_robot_dg5f.debug_recording import DebugRunWriter


def controller(accel=720):
    s = ServoController(np.full(20, -180.), np.full(20, 180.),
                        disabled_positions_deg={16: 0}, max_acceleration_deg_s2=accel)
    s.reset(np.zeros(20), now=0)
    return s


def step(s, target, now):
    proposal = s.propose(np.full(20, target), now)
    command = s.constrain_guarded_output(proposal)
    physical = s.physical_output(command)
    s.accept_output(command, physical_deg=physical)
    return s.command_pose_deg[6], s.velocity_deg_s[6]


@pytest.mark.parametrize('accel', [480, 720, 960])
@pytest.mark.parametrize('target', [70, -70, .01])
def test_step_accelerates_brakes_and_settles_without_overshoot(accel, target):
    s = controller(accel)
    previous_q = previous_v = 0
    velocities = []
    for tick in range(1, 151):
        q, v = step(s, target, tick/60)
        assert abs(v) <= 120 + 1e-8
        assert abs(v - previous_v) <= accel/60 + 1e-7
        assert min(0, target) - 1e-9 <= q <= max(0, target) + 1e-9
        assert (q - previous_q) * target >= -1e-9
        assert s.command_pose_deg[16] == s.velocity_deg_s[16] == 0
        velocities.append(abs(v))
        previous_q, previous_v = q, v
    assert q == pytest.approx(target, abs=1e-8)
    assert v == 0
    if abs(target) > 1:
        assert max(velocities) == pytest.approx(120)
        assert 0 < velocities[0] < 120
        assert any(0 < b < a for a, b in zip(velocities, velocities[1:]))


def test_reversal_brakes_through_zero_before_opening():
    s = controller()
    for tick in range(1, 13):
        step(s, 70, tick/60)
    assert s.velocity_deg_s[6] == pytest.approx(120)
    previous = 120
    velocities = []
    for tick in range(13, 101):
        _, v = step(s, -70, tick/60)
        assert abs(v - previous) <= 12 + 1e-7
        velocities.append(v)
        previous = v
    assert velocities[0] == pytest.approx(108)
    assert 0 in velocities
    assert min(velocities) == pytest.approx(-120)
    assert s.command_pose_deg[6] == pytest.approx(-70, abs=1e-8)


def test_jitter_uses_elapsed_time_and_gap_never_catches_up():
    s = controller()
    now = 0
    old_v = old_q = 0
    for dt in [1/60, .012, .022, .014, .021, .018, .013]:
        now += dt
        q, v = step(s, 170, now)
        assert v == pytest.approx(old_v + 720*dt)
        assert q - old_q == pytest.approx(v*dt)
        assert s.telemetry['limiter_dt'] == pytest.approx(dt)
        old_v, old_q = v, q
    q, v = step(s, 170, now+3)
    assert q-old_q <= 2 + 1e-8
    assert s.telemetry['dt'] == pytest.approx(3)
    assert s.telemetry['limiter_dt'] == 1/60


def test_hold_and_reset_clear_velocity_resume_starts_from_command():
    s = controller()
    for tick in range(1, 16):
        step(s, 70, tick/60)
    held = s.command_pose_deg.copy()
    physical = s.effective_command()
    s.hold()
    np.testing.assert_array_equal(s.command_pose_deg, held)
    np.testing.assert_array_equal(s.effective_command(), physical)
    assert not np.any(s.velocity_deg_s)
    q, v = step(s, -70, 20)
    assert q == pytest.approx(held[6] - .2)
    assert v == pytest.approx(-12)
    s.reset(np.full(20, 10))
    assert not np.any(s.velocity_deg_s)


def test_jitter_including_braking_never_oscillates():
    s = controller()
    rng = np.random.default_rng(42)
    now = old_q = old_v = 0
    for _ in range(250):
        now += rng.uniform(.011, .026)
        q, v = step(s, 70, now)
        dt = s.telemetry['limiter_dt']
        assert abs(v - old_v) <= 720*dt + 1e-6
        assert old_q - 1e-9 <= q <= 70
        old_q, old_v = q, v
    assert q == pytest.approx(70, abs=1e-8)
    assert v == 0


def test_target_jumps_inside_stopping_distance_clamps_without_oscillation():
    s = controller()
    for tick in range(1, 13):
        step(s, 70, tick/60)
    close = s.command_pose_deg[6] + .05
    q, v = step(s, close, 13/60)
    assert q == close and v == 0
    assert s.telemetry['target_clamped'][6]
    for tick in range(14, 25):
        assert step(s, close, tick/60) == (close, 0)


@pytest.mark.parametrize('fault', ['guard_hold', 'physical_lead', 'reject'])
def test_safety_overrides_acceleration_and_failed_send_never_commits(monkeypatch, fault):
    robot = Dg5f(Dg5fConfig(control_mode='servo'))
    robot.connect()
    s = robot.command_shaper
    try:
        for tick in range(1, 13):
            robot.servo_tick(np.full(20, 70), guard=lambda q:q, now=tick/60)
        before = s.command_pose_deg.copy()
        assert s.velocity_deg_s[6] == pytest.approx(120)
        def guard(proposal):
            if fault == 'guard_hold':
                return before.copy()
            if fault == 'physical_lead':
                return GuardedServoCommand(proposal, before-2, before-2)
            return proposal
        if fault == 'reject':
            def reject(q):
                raise RuntimeError('rejected')
            monkeypatch.setattr(robot.backend, 'send_positions', reject)
            with pytest.raises(RuntimeError, match='rejected'):
                robot.servo_tick(np.full(20, -70), guard=guard, now=13/60)
            np.testing.assert_array_equal(s.command_pose_deg, before)
            assert s.velocity_deg_s[6] == pytest.approx(120)
        else:
            robot.servo_tick(np.full(20, 70), guard=guard, now=13/60)
            assert s.velocity_deg_s[6] == 0
            expected = before[6] if fault == 'guard_hold' else before[6]-2
            assert s.effective_command()[6] == pytest.approx(expected)
            assert s.telemetry['safety_velocity_override'][6]
    finally:
        robot.disconnect()


def test_acceleration_fields_survive_unmodified_recorder(tmp_path):
    s = controller()
    step(s, 70, 1/60)
    writer = DebugRunWriter(tmp_path, {}, run_stamp='accel')
    Dg5fDebugRecorder._on_servo(SimpleNamespace(writer=writer), String(data=json.dumps(s.telemetry)))
    writer.finalize()
    tick = json.loads((writer.run_dir/'servo_ticks.jsonl').read_text())
    assert tick['desired_velocity_deg_s'][6] == 120
    assert tick['commanded_velocity_deg_s'][6] == pytest.approx(12)
    assert tick['acceleration_deg_s2'][6] == pytest.approx(720)
    assert tick['acceleration_limited'][6]
    assert tick['max_acceleration_deg_s2'] == 720


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_acceleration_configuration_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        Dg5fConfig(servo_max_acceleration_deg_s2=value)
