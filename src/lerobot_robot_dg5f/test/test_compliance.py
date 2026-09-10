"""Offline contact/load/relief scenarios and final command-envelope regression."""

from pathlib import Path

import numpy as np
import pytest

from lerobot_robot_dg5f.command_shaper import PositionCommandShaper
from lerobot_robot_dg5f.constants import JOINT_NAMES, LOWER_LIMITS_DEG, UPPER_LIMITS_DEG
from lerobot_robot_dg5f.contact_kinematics import ContactKinematics
from lerobot_robot_dg5f.current_guard import AdaptiveCurrentGuard, ComplianceConfig


def make_guard(**config):
    return AdaptiveCurrentGuard(
        20, soft_ma=350, hard_ma=650, trip_ma=850,
        total_soft_ma=600, total_hard_ma=850, total_trip_ma=1050,
        trip_hold_s=0.04, release_tau_s=0.2, nominal_step_deg=5,
        compliance=ComplianceConfig(**config),
    )


def contact(now=1.0, pair=0):
    distances = np.full(7, 0.1)
    distances[pair] = 0.010
    gradients = np.zeros((7, 20))
    if pair == 0:
        gradients[pair, [2, 6]] = -0.001
    else:
        gradients[pair, [10, 14]] = -0.001
    robot = np.tile(distances, (3, 1))
    if pair == 0:
        robot[:, 0] = [0.027, 0.027, 0.007]
    return dict(pair_distances_m=distances, pair_gradients_m_deg=gradients,
                hybrid_weights=np.array([0.8, 0.8, 0, 0, 0]), contact_stamp_s=now,
                robot_pair_distances_m=robot)


def update(guard, current=0.0, effective=10.0, measured=9.0, desired=80.0, now=1.0, **extra):
    currents = np.zeros(20)
    currents[6] = current
    return guard.update(current_ma=currents, effective_deg=np.full(20, effective),
                        measured_deg=np.full(20, measured), desired_deg=np.full(20, desired), now=now, **extra)


def test_free_motion_is_unchanged():
    guard = make_guard()
    for n in range(20):
        decision = update(guard, current=30, now=1 + n * 0.02)
        np.testing.assert_array_equal(decision.target_deg, np.full(20, 80))
        assert not decision.active


def test_human_contact_alone_does_not_reduce_closing_step():
    signals = contact()
    signals["robot_pair_distances_m"][:, 0] = [0.085, 0.070, 0.007]
    decision = update(make_guard(), **signals)
    assert decision.target_deg[6] == 80
    assert decision.diagnostics["joint_tracking_scale"][6] == 1
    assert decision.target_deg[18] == 80
    assert decision.target_deg[4] == 80  # spread joint not geometrically closing


def test_contact_and_rising_current_decrease_convergence_before_soft_threshold():
    guard = make_guard()
    steps = []
    for n, current in enumerate([50, 100, 150, 200, 250, 300]):
        now = 1 + n * 0.02
        result = update(guard, current, now=now, **contact(now))
        steps.append(result.target_deg[6] - 10)
        assert not np.any(result.limited_mask[16:20])
    assert steps[-1] < steps[0] / 3
    assert result.diagnostics["joint_current_slope_ma_s"][6] > 0
    assert not result.trip


def test_loaded_contact_accepts_large_vr_error_without_retreat():
    guard = make_guard()
    effective = 60.0
    for n in range(20):
        now = 1 + n * 0.02
        decision = update(guard, 450, effective, 59, 80, now=now, **contact(now))
        assert effective <= decision.target_deg[6] <= 61
        effective = decision.target_deg[6]
    assert effective < 65  # Never insists on eventually reaching the blocked 80°.


def test_relief_crossing_measured_pose_bypasses_contact_and_current_slowdown():
    result = update(make_guard(), 700, 60, 59, 20, **contact())
    assert result.target_deg[6] == 20
    assert result.diagnostics["joint_tracking_scale"][6] == 1
    assert not result.limited_mask[6]


def test_current_release_is_gradual():
    guard = make_guard()
    first = update(guard, 700)
    scales = [first.diagnostics["joint_tracking_scale"][6]]
    for n in range(1, 10):
        decision = update(guard, 20, now=1 + n * 0.02)
        scales.append(decision.diagnostics["joint_tracking_scale"][6])
    assert scales[0] == 0
    assert 0 < scales[1] < 0.2
    assert all(a <= b for a, b in zip(scales, scales[1:]))
    assert scales[-1] < 1


def test_disappearing_contact_does_not_cause_catchup_jump():
    guard = make_guard()
    shaper = PositionCommandShaper(LOWER_LIMITS_DEG, UPPER_LIMITS_DEG,
                                   smoothing=False, max_direct_step_deg=5, startup_blend_s=0,
                                   min_send_step_deg=0, disabled_positions_deg={16: 0})
    effective = np.zeros(20)
    effective[6] = 60
    shaper.reset(effective, now=1)
    measured, desired = effective.copy(), effective.copy()
    measured[6], desired[6] = 59, 90
    current = np.zeros(20)
    current[6] = 450
    for n in range(30):
        now = 1 + n * 0.02
        extra = contact(now) if n < 3 else {}
        if n >= 3:
            current[6] = 0
        decision = guard.update(current_ma=current, measured_deg=measured,
                                effective_deg=effective, desired_deg=desired, now=now, **extra)
        step = shaper.step(decision.target_deg, now=now)
        assert 0 <= step.output_deg[6] - effective[6] <= 5.000001
        shaper.accept_output(step.output_deg)
        effective = shaper.effective_command()
        assert effective[16] == 0
    assert effective[6] > 60


def test_thumb_index_load_does_not_limit_little_finger():
    decision = update(make_guard(), current=300, **contact())
    assert decision.limited_mask[2] and decision.limited_mask[6]
    assert not np.any(decision.limited_mask[16:20])
    assert decision.diagnostics["limited_fingers"] == ["thumb", "index"]
    assert decision.diagnostics["contact_limited_pairs"] == ["thumb-index"]


def test_spread_only_limited_if_it_reduces_distance():
    data = contact()
    data["pair_gradients_m_deg"][0, 4] = 0.001
    decision = update(make_guard(), 300, **data)
    assert not decision.limited_mask[4]
    data["pair_gradients_m_deg"][0, 4] = -0.001
    assert update(make_guard(), 300, **data).limited_mask[4]


def test_adjacent_proximity_without_load_does_not_limit_motion():
    data = contact(pair=5)  # middle-ring; not part of original Hybrid weights
    data["hybrid_weights"] = np.zeros(5)
    decision = update(make_guard(), **data)
    assert not decision.limited_mask[10] and not decision.limited_mask[14]
    assert not decision.limited_mask[2]


def test_short_current_spike_slowdown_without_emergency_trip():
    guard = make_guard()
    assert update(guard, 900).active
    assert not update(guard, 900, now=1.02).trip
    assert not update(guard, 0, now=1.03).trip


def test_sustained_current_still_trips_at_original_envelope():
    guard = make_guard()
    assert not update(guard, 900).trip
    assert update(guard, 900, now=1.041).trip


@pytest.mark.parametrize("bad", [None, np.full(7, np.nan), np.zeros(6)])
def test_missing_or_nan_contact_keeps_current_protection(bad):
    result = update(make_guard(), 700, pair_distances_m=bad,
                    pair_gradients_m_deg=np.ones((7, 20)), contact_stamp_s=1)
    assert not result.diagnostics["contact_signal_valid"]
    assert result.target_deg[6] == 10
    assert result.active


def test_trend_rejects_single_moderate_noisy_sample():
    guard = make_guard()
    for n, current in enumerate([100, 100, 100, 300, 100, 100, 100]):
        result = update(guard, current, now=1 + n * 0.02)
        assert result.diagnostics["joint_current_slope_ma_s"][6] == pytest.approx(0)
        assert not result.active


def test_total_current_growth_detected_across_multiple_fingers():
    guard = make_guard()
    for n, current in enumerate([5, 8, 12, 18, 24, 29]):
        result = guard.update(current_ma=np.full(20, current), measured_deg=np.zeros(20),
                              effective_deg=np.ones(20), desired_deg=np.full(20, 30), now=1 + n * 0.02)
    assert result.total_current_ma < 600
    assert result.diagnostics["total_current_slope_ma_s"] > 1200
    assert result.diagnostics["global_slope_scale"] < 1
    assert result.active and not result.trip


def test_sustained_stall_has_distinct_trip_and_relief_clears_it():
    guard = make_guard()
    for n in range(32):
        result = update(guard, 500, 60, 40, 80, now=1 + n * 0.02)
    assert result.stall_trip and not result.trip
    relief = update(guard, 500, 60, 40, 20, now=1.65)
    assert not relief.stall_trip


def test_urdf_jacobian_matches_finite_difference_and_pinocchio():
    import pinocchio as pin
    path = Path(__file__).parents[3] / "models/dg5f/urdf/dg5f_right.urdf"
    fk = ContactKinematics(path)
    q = np.array([10, -80, 20, 25, 0, 30, 35, 40, 0, 30, 35, 40,
                  0, 30, 35, 40, 0, 0, 35, 40], dtype=float)
    tips, jac = fk.tips_and_jacobians(q)
    gradients = fk.pair_gradients(q)
    from dg5f_teleop.contact_signals import PAIRS
    for joint in range(20):
        shifted = q.copy()
        shifted[joint] += 1e-4
        new_tips, _ = fk.tips_and_jacobians(shifted)
        np.testing.assert_allclose((new_tips - tips) / 1e-4, jac[:, :, joint], atol=2e-8)
        old_d = [np.linalg.norm(tips[b] - tips[a]) for a, b in PAIRS]
        new_d = [np.linalg.norm(new_tips[b] - new_tips[a]) for a, b in PAIRS]
        np.testing.assert_allclose((np.array(new_d) - old_d) / 1e-4, gradients[:, joint], atol=2e-8)
    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    pin_q = pin.neutral(model)
    for index, name in enumerate(JOINT_NAMES):
        pin_q[model.joints[model.getJointId(name)].idx_q] = np.deg2rad(q[index])
    pin.framesForwardKinematics(model, data, pin_q)
    for finger in range(5):
        pose = data.oMf[model.getFrameId(f"rl_dg_{finger + 1}_tip")]
        np.testing.assert_allclose(tips[finger], pose.translation, atol=1e-10)
