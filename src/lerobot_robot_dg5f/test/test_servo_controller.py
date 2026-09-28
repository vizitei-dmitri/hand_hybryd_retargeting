import numpy as np
import pytest

from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.constants import LOWER_LIMITS_DEG, UPPER_LIMITS_DEG
from lerobot_robot_dg5f.servo_controller import ServoController


def controller():
    result = ServoController(LOWER_LIMITS_DEG, UPPER_LIMITS_DEG,
                             disabled_positions_deg={16: 0.0}, max_velocity_deg_s=30.0)
    result.reset(np.zeros(20), now=0)
    return result


def tick(servo, target, now):
    command = servo.propose(np.full(20, target), now)
    servo.accept_output(servo.constrain_guarded_output(command))
    return servo.effective_command()


def test_persistent_command_does_not_follow_stalled_feedback_and_reverses_immediately():
    servo = controller()
    for index in range(60):
        tick(servo, 90, (index + 1) / 60)
        servo.observe(np.zeros(20))  # Motor never moves.
    assert servo.effective_command()[6] == pytest.approx(30)
    assert tick(servo, -90, 61 / 60)[6] == pytest.approx(29.5)
    assert servo.telemetry['max_step_deg'] <= 0.5 + 1e-12


def test_small_steps_limits_fixed_joint_and_stalled_scheduler():
    servo = controller()
    assert tick(servo, 0.01, 1)[6] == pytest.approx(0.01)
    command = tick(servo, 999, 10)  # No nine-second catch-up jump.
    assert command[6] == pytest.approx(0.51)
    assert command[16] == 0
    assert np.all(command <= UPPER_LIMITS_DEG)
    assert np.all(command >= LOWER_LIMITS_DEG)
    assert servo.telemetry['dt'] == 9
    assert servo.telemetry['limiter_dt'] == 1 / 60
    assert servo.telemetry['actual_rate_hz'] == 1 / 9
    previous = command.copy()
    command = tick(servo, 999, 10.001)
    assert np.max(command - previous) == pytest.approx(0.03)


def test_tracking_warning_requires_same_joint_continuously_and_rearms_after_clear():
    servo = controller()
    servo.reset(np.full(20, 20), now=0)
    target = servo.effective_command()
    def observe(now, joint):
        servo.propose(target, now)
        measured = target.copy()
        measured[joint] -= 11
        return servo.observe(measured)
    assert observe(1.0, 6) == []
    assert observe(1.29, 6) == []
    assert observe(1.30, 7) == []  # A different joint must start its own timer.
    assert observe(1.59, 7) == []
    assert observe(1.60, 7) == [7]
    assert observe(1.70, 7) == []
    assert observe(1.80, 6) == []
    assert observe(2.10, 6) == [6]
    servo.hold(now=3)
    servo.propose(target, 4)
    assert servo.observe(np.zeros(20)) == []


def test_guard_runs_after_limiter_and_commits_only_accepted_output(monkeypatch):
    robot = Dg5f(Dg5fConfig(control_mode='servo', servo_max_velocity_deg_s=30.0))
    robot.connect()
    calls = []
    monkeypatch.setattr(robot.backend, 'send_positions', lambda q: calls.append(q.copy()))
    def freeze(q):
        assert q[6] == 0.5
        return np.zeros(20)
    robot.servo_tick(np.full(20, 90), guard=freeze, now=1)
    assert robot.command_shaper.effective_command()[6] == 0
    assert len(calls) == 1
    robot.servo_tick(np.full(20, 90), guard=lambda q: None, now=2)
    assert len(calls) == 1
    def reject(q):
        raise RuntimeError('SDK rejected')
    monkeypatch.setattr(robot.backend, 'send_positions', reject)
    with pytest.raises(RuntimeError, match='SDK rejected'):
        robot.servo_tick(np.full(20, 90), guard=lambda q: q, now=3)
    assert robot.command_shaper.effective_command()[6] == 0
    robot.disconnect()


def test_stationary_pose_still_streams_all_twenty_joints(monkeypatch):
    robot = Dg5f(Dg5fConfig(control_mode='servo', servo_max_velocity_deg_s=30.0))
    robot.connect()
    calls = []
    monkeypatch.setattr(robot.backend, 'send_positions', lambda q: calls.append(q.copy()))
    for index in range(12):
        robot.servo_tick(np.zeros(20), guard=lambda q: q, now=index / 60)
    assert len(calls) == 12
    assert all(q.shape == (20,) for q in calls)
    robot.disconnect()


@pytest.mark.parametrize('target', [np.zeros(19), np.full(20, np.nan), np.full(20, np.inf)])
def test_invalid_target_does_not_change_command(target):
    servo = controller()
    before = servo.effective_command()
    with pytest.raises(ValueError):
        servo.propose(target, now=1)
    np.testing.assert_array_equal(servo.effective_command(), before)


@pytest.mark.parametrize('config', [dict(control_mode='typo'), dict(servo_rate_hz=0),
                                    dict(servo_max_velocity_deg_s=float('nan'))])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        Dg5fConfig(**config)
