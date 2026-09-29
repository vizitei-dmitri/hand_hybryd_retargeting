"""Regression for the 2026-09-28 hardware runaway after contact release."""
import numpy as np
import pytest

from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.constants import LOWER_LIMITS_DEG, UPPER_LIMITS_DEG
from lerobot_robot_dg5f.current_guard import AdaptiveCurrentGuard, ComplianceConfig
from lerobot_robot_dg5f.object_contact import ObjectContactConfig


def make_guard():
    return AdaptiveCurrentGuard(20, soft_ma=350, hard_ma=650, trip_ma=850,
        total_soft_ma=600, total_hard_ma=850, total_trip_ma=1050,
        trip_hold_s=0.04, release_tau_s=0.2, nominal_step_deg=2,
        compliance=ComplianceConfig(), object_contact=ObjectContactConfig())


def make_robot(initial=None, speed=120):
    robot = Dg5f(Dg5fConfig(control_mode='servo', servo_max_velocity_deg_s=speed))
    robot.connect()
    robot.command_shaper.reset(np.zeros(20) if initial is None else initial, now=0)
    return robot


def advance(robot, guard, target, now, *, measured=None, current=None, correct_intent=True):
    previous = robot.command_shaper.effective_command()
    measured = previous.copy() if measured is None else measured
    current = np.zeros(20) if current is None else current
    decisions = []
    def safety(proposed):
        decision = guard.update(
            desired_deg=proposed, effective_deg=previous, measured_deg=measured,
            current_ma=current, now=now,
            operator_target_deg=target if correct_intent else None,
            trajectory_braking=robot.command_shaper.telemetry["trajectory_braking"] if correct_intent else None)
        assert not decision.trip and not decision.stall_trip
        decisions.append(decision)
        return decision.target_deg
    robot.servo_tick(target, guard=safety, now=now)
    return robot.command_shaper.effective_command(), decisions[-1]


def released_contact_offset(guard, initial):
    # Reachable state after OBJECT_CONTACT_RELEASED / OPERATOR_OPENING.
    # The hardware bundle latched q6 with >9 degrees of tracking error.
    finger = guard._object_contact.fingers['index']
    finger.reason = 'OPERATOR_OPENING'
    finger.resume_offset = np.array([0., 5., 0.])
    guard._object_contact.last_time = 0
    guard._object_contact.last_desired = initial.copy()
    guard._object_contact.last_desired[6] += 1


def test_reproduces_old_wrong_stage_then_corrects_target_plus70_command_minus50():
    initial = np.zeros(20); initial[6] = -50
    target = initial.copy(); target[6] = 70
    old_robot, old_guard = make_robot(initial, 30), make_guard()
    released_contact_offset(old_guard, initial)
    old, _ = advance(old_robot, old_guard, target, 1/60, correct_intent=False)
    assert old_robot.command_shaper.telemetry['q_proposed'][6] == -49.8
    assert old[6] == -50.5  # Downstream legacy offset can bypass trajectory acceleration.
    old_robot.disconnect()

    robot, guard = make_robot(initial, 30), make_guard()
    released_contact_offset(guard, initial)
    for tick in range(1, 81):
        previous = robot.command_shaper.effective_command()
        command, decision = advance(robot, guard, target, tick/60)
        assert command[6] >= previous[6]
        assert abs(target[6]-command[6]) <= abs(target[6]-previous[6])
        assert decision.diagnostics['post_contact_cmd'][6] >= previous[6]
    assert command[6] > -50
    robot.disconnect()


@pytest.mark.parametrize('joint', range(20))
def test_every_joint_reversal_cross_zero_and_rapid_targets(joint):
    robot, guard = make_robot(), make_guard()
    target = np.zeros(20)
    # All targets respect the unchanged per-joint limits. Fixed joint stays zero.
    lo, hi = LOWER_LIMITS_DEG[joint], UPPER_LIMITS_DEG[joint]
    if joint == 16:
        lo = hi = 0
    values = ([min(hi, 40)]*25 + [max(lo, -30)]*45
              + [min(hi, 70)]*55 + [lo, hi, 0, hi, lo]*10)
    for tick, value in enumerate(values, 1):
        target[joint] = value
        previous = robot.command_shaper.effective_command()
        command, _ = advance(robot, guard, target, tick/60)
        braking = np.asarray(robot.command_shaper.telemetry['trajectory_braking'])
        assert np.all(((target - previous) * (command - previous) >= -1e-8) | braking)
        assert np.all((np.abs(target-command) <= np.abs(target-previous) + 1e-8) | braking)
        assert np.max(np.abs(command-previous)) <= 2 + 1e-8
    robot.disconnect()


def test_contact_yield_uses_real_intent_and_cannot_self_release_or_ratchet():
    initial = np.zeros(20); initial[6] = 50
    target = initial.copy(); target[6] = 70
    measured = initial.copy(); measured[6] = 40
    current = np.zeros(20); current[6] = 300
    robot, guard = make_robot(initial), make_guard()
    held = False
    first_floor = None
    for tick in range(1, 91):
        previous = robot.command_shaper.effective_command()
        # Even if measured drifts backward after latch, yield has a fixed floor.
        if held:
            measured[6] = min(measured[6], previous[6] - 4)
        command, decision = advance(robot, guard, target, tick/60, measured=measured, current=current)
        diag = decision.diagnostics
        if diag['object_contact_state'][1] == 'CONTACT_HOLD':
            held = True
            floor = diag['contact_anchor_measured'][6] + diag['contact_preload_deg'][1]
            first_floor = floor if first_floor is None else first_floor
            assert command[6] >= first_floor - 1e-8
        if held:
            assert diag['object_contact_state'][1] == 'CONTACT_HOLD'
        if command[6] < previous[6] - 1e-8:
            assert diag['command_direction_reason'][6] == 'CONTACT_PRELOAD_YIELD'
            assert previous[6] - command[6] <= 15/60 + 1e-8
    assert held
    target[6] = -30
    command, decision = advance(robot, guard, target, 91/60, measured=measured, current=current)
    assert decision.diagnostics['object_contact_state'][1] == 'FREE'
    target[6] = 70
    previous = command.copy()
    previous_velocity = robot.command_shaper.velocity_deg_s[6]
    command, _ = advance(robot, guard, target, 92/60, measured=command, current=np.zeros(20))
    # The prior opening may still brake outward for a few ticks; it must be
    # planned deceleration, never a stale offset inventing a retreat.
    if command[6] < previous[6]:
        assert robot.command_shaper.telemetry['trajectory_braking'][6]
        assert abs(robot.command_shaper.velocity_deg_s[6]) < abs(previous_velocity)
        assert previous_velocity < 0
    robot.disconnect()


def test_free_offset_cannot_overshoot_a_nearby_operator_target():
    initial = np.zeros(20); initial[6] = -50
    target = initial.copy(); target[6] = -49
    robot, guard = make_robot(initial), make_guard()
    released_contact_offset(guard, initial)
    command, decision = advance(robot, guard, target, 1/60)
    assert -50 <= command[6] <= -49
    assert command[6] == pytest.approx(-49.8)
    assert guard._object_contact.fingers['index'].resume_offset is None
    assert decision.diagnostics['command_direction_reason'][6] == 'NONE'
    robot.disconnect()


@pytest.mark.parametrize('speed,step', [(60,1), (120,2), (180,3)])
def test_requested_hardware_comparison_speeds(speed, step):
    robot = make_robot(speed=speed)
    for tick in range(1, 21):
        previous = robot.command_shaper.effective_command()
        robot.servo_tick(np.full(20,70), guard=lambda q:q, now=tick/60)
    assert robot.command_shaper.effective_command()[6] - previous[6] == pytest.approx(step)
    assert robot.config.servo_rate_hz == 60
    robot.disconnect()


def test_default_speed_is120_without_changing_legacy_speed():
    config = Dg5fConfig()
    assert config.control_mode == 'legacy'
    assert config.servo_max_velocity_deg_s == 120
    assert config.max_speed_deg_s == 30


def test_blocked_offset_must_not_disable_existing_stall_protection():
    guard=make_guard()
    effective=np.zeros(20); effective[6]=50
    measured=effective.copy(); measured[6]=40
    target=effective.copy(); target[6]=70
    current=np.zeros(20); current[6]=400
    finger=guard._object_contact.fingers['index']
    finger.reason='OPERATOR_OPENING'
    finger.resume_offset=np.array([0.,50.,0.])
    finger.released_at=1e6  # Prevent a new latch; test an outstanding FREE offset.
    for tick in range(40):
        proposed=effective.copy(); proposed[6]+=2
        decision=guard.update(current_ma=current,measured_deg=measured,
                              effective_deg=effective,desired_deg=proposed,
                              operator_target_deg=target,now=tick/60)
        assert decision.target_deg[6]==50
    assert decision.stall_trip
