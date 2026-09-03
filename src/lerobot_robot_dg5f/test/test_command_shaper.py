import numpy as np
import pytest

from lerobot_robot_dg5f.command_shaper import PositionCommandShaper
from lerobot_robot_dg5f.constants import BROKEN_PINKY_INDEX


def make_shaper(**overrides):
    settings = {
        "disabled_positions_deg": {BROKEN_PINKY_INDEX: 0.0},
        "max_speed_deg_s": 30.0,
        "max_accel_deg_s2": 60.0,
        "response_time_s": 0.15,
        "filter_tau_s": 0.05,
        "target_deadband_deg": 0.20,
        "min_send_step_deg": 0.20,
        "max_dt_s": 0.05,
    }
    settings.update(overrides)
    return PositionCommandShaper(
        np.full(20, -1000.0),
        np.full(20, 1000.0),
        **settings,
    )


def test_first_update_does_not_jump_to_target():
    shaper = make_shaper()
    shaper.reset(np.full(20, 10.0), now=0.0)

    step = shaper.step(np.full(20, 50.0), now=0.02)

    assert np.all(step.command_deg[:BROKEN_PINKY_INDEX] < 50.0)
    assert step.command_deg[0] == pytest.approx(10.024)


def test_velocity_and_acceleration_are_bounded():
    shaper = make_shaper(filter_tau_s=0.0, target_deadband_deg=0.0)
    shaper.reset(np.zeros(20), now=0.0)
    previous_velocity = np.zeros(20)

    for tick in range(1, 101):
        step = shaper.step(np.full(20, 500.0), now=tick * 0.02)
        assert np.max(np.abs(step.velocity_deg_s)) <= 30.0 + 1e-9
        acceleration = np.abs(step.velocity_deg_s - previous_velocity) / 0.02
        assert np.max(acceleration) <= 60.0 + 1e-8
        previous_velocity = step.velocity_deg_s


def test_target_is_not_overshot():
    shaper = make_shaper(
        max_speed_deg_s=1000.0,
        max_accel_deg_s2=100000.0,
        response_time_s=0.001,
        filter_tau_s=0.0,
        target_deadband_deg=0.0,
    )
    shaper.reset(np.zeros(20), now=0.0)

    step = shaper.step(np.ones(20), now=0.05)

    assert np.all(step.command_deg <= 1.0)
    assert step.command_deg[0] == pytest.approx(1.0)


def test_disabled_joint_is_fixed_in_every_state():
    shaper = make_shaper(
        filter_tau_s=0.0,
        target_deadband_deg=0.0,
        min_send_step_deg=0.0,
    )
    initial = np.full(20, 30.0)
    shaper.reset(initial, now=0.0)

    for tick, target_value in enumerate((500.0, -500.0, 42.0), start=1):
        step = shaper.step(np.full(20, target_value), now=tick * 0.05)
        assert step.command_deg[BROKEN_PINKY_INDEX] == 0.0
        assert step.output_deg[BROKEN_PINKY_INDEX] == 0.0
        assert step.velocity_deg_s[BROKEN_PINKY_INDEX] == 0.0
        assert shaper.filtered_target_deg[BROKEN_PINKY_INDEX] == 0.0
        shaper.accept_output(step.output_deg)
        assert shaper.effective_command()[BROKEN_PINKY_INDEX] == 0.0


def test_joint_limits_and_non_finite_validation():
    shaper = PositionCommandShaper(
        np.full(20, -10.0),
        np.full(20, 20.0),
        smoothing=False,
        min_send_step_deg=0.0,
    )
    shaper.reset(np.zeros(20), now=0.0)
    step = shaper.step(np.linspace(-100.0, 100.0, 20), now=0.02)
    assert np.all(step.command_deg >= -10.0)
    assert np.all(step.command_deg <= 20.0)

    bad = np.zeros(20)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        shaper.step(bad, now=0.04)


def test_deadband_rejects_small_target_jitter():
    shaper = make_shaper(filter_tau_s=0.0, target_deadband_deg=0.20)
    shaper.reset(np.zeros(20), now=0.0)

    step = shaper.step(np.full(20, 0.19), now=0.05)

    assert np.array_equal(step.command_deg, np.zeros(20))
    assert not step.should_send


def test_small_updates_accumulate_before_send():
    shaper = make_shaper(
        smoothing=False,
        min_send_step_deg=0.20,
        disabled_positions_deg={},
    )
    shaper.reset(np.zeros(20), now=0.0)

    first = shaper.step(np.full(20, 0.10), now=0.02)
    second = shaper.step(np.full(20, 0.19), now=0.04)
    third = shaper.step(np.full(20, 0.21), now=0.06)

    assert not first.should_send
    assert not second.should_send
    assert third.should_send
    assert np.allclose(third.output_deg, 0.21)


def test_hold_returns_to_last_accepted_setpoint_without_sending():
    shaper = make_shaper(
        filter_tau_s=0.0,
        target_deadband_deg=0.0,
        min_send_step_deg=10.0,
    )
    shaper.reset(np.zeros(20), now=0.0)
    step = shaper.step(np.full(20, 100.0), now=0.05)
    assert np.any(step.command_deg != 0.0)
    assert not step.should_send

    shaper.hold(now=0.06)

    assert np.array_equal(shaper.command_pose_deg, shaper.effective_command())
    assert np.array_equal(shaper.velocity_deg_s, np.zeros(20))
