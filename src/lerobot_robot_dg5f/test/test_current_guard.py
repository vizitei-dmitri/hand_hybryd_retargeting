import numpy as np

from lerobot_robot_dg5f.current_guard import AdaptiveCurrentGuard


def make_guard(**overrides):
    params = dict(
        joint_count=20,
        soft_ma=350.0,
        hard_ma=650.0,
        trip_ma=850.0,
        total_soft_ma=800.0,
        total_hard_ma=1200.0,
        total_trip_ma=1400.0,
        trip_hold_s=0.06,
        release_tau_s=0.20,
        nominal_step_deg=5.0,
    )
    params.update(overrides)
    return AdaptiveCurrentGuard(**params)


def test_below_soft_threshold_does_not_change_normal_motion():
    guard = make_guard()
    desired = np.full(20, 40.0)
    decision = guard.update(
        current_ma=np.full(20, 10.0),
        measured_deg=np.zeros(20),
        effective_deg=np.full(20, 10.0),
        desired_deg=desired,
        now=1.0,
    )
    np.testing.assert_allclose(decision.target_deg, desired)
    assert not decision.active
    assert decision.min_scale == 1.0


def test_soft_current_progressively_limits_only_load_increasing_joint():
    guard = make_guard(total_soft_ma=5000.0, total_hard_ma=6000.0, total_trip_ma=7000.0)
    current = np.zeros(20)
    current[3] = 500.0  # halfway between 350 and 650 => scale 0.5
    measured = np.zeros(20)
    effective = np.full(20, 10.0)
    desired = np.full(20, 30.0)
    decision = guard.update(
        current_ma=current,
        measured_deg=measured,
        effective_deg=effective,
        desired_deg=desired,
        now=1.0,
    )
    assert decision.target_deg[3] == 12.5  # 10 + 5 deg nominal step * 0.5
    assert decision.target_deg[2] == 30.0
    assert decision.active
    assert decision.limited_mask[3]


def test_relief_motion_is_not_slowed_even_when_joint_is_loaded():
    guard = make_guard(total_soft_ma=5000.0, total_hard_ma=6000.0, total_trip_ma=7000.0)
    current = np.zeros(20)
    current[4] = 600.0
    measured = np.zeros(20)
    effective = np.zeros(20)
    effective[4] = 20.0
    desired = np.zeros(20)  # operator asks to return exactly toward measured pose
    decision = guard.update(
        current_ma=current,
        measured_deg=measured,
        effective_deg=effective,
        desired_deg=desired,
        now=1.0,
    )
    assert decision.target_deg[4] == 0.0


def test_hard_current_walks_command_back_toward_measured_pose():
    guard = make_guard(total_soft_ma=5000.0, total_hard_ma=6000.0, total_trip_ma=7000.0)
    current = np.zeros(20)
    current[6] = 700.0
    measured = np.zeros(20)
    effective = np.zeros(20)
    effective[6] = 25.0
    desired = np.zeros(20)
    desired[6] = 50.0
    decision = guard.update(
        current_ma=current,
        measured_deg=measured,
        effective_deg=effective,
        desired_deg=desired,
        now=1.0,
    )
    assert decision.target_deg[6] == measured[6]
    assert decision.active


def test_trip_requires_sustained_current_not_one_sample_spike():
    guard = make_guard(total_soft_ma=5000.0, total_hard_ma=6000.0, total_trip_ma=7000.0)
    current = np.zeros(20)
    current[2] = 900.0
    args = dict(
        current_ma=current,
        measured_deg=np.zeros(20),
        effective_deg=np.full(20, 10.0),
        desired_deg=np.full(20, 30.0),
    )
    first = guard.update(now=1.00, **args)
    second = guard.update(now=1.04, **args)
    third = guard.update(now=1.07, **args)
    assert not first.trip
    assert not second.trip
    assert third.trip
