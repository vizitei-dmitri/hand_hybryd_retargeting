"""Six offline physical-contact scenarios, including both logged load regressions."""

import numpy as np
import pytest

from lerobot_robot_dg5f.command_shaper import PositionCommandShaper
from lerobot_robot_dg5f.constants import BROKEN_PINKY_INDEX, FINGER_FLEXION_JOINTS, JOINT_NAMES
from lerobot_robot_dg5f.current_guard import AdaptiveCurrentGuard, ComplianceConfig
from lerobot_robot_dg5f.debug_recording import DebugRunWriter
from lerobot_robot_dg5f.object_contact import ObjectContactConfig


def guard(with_object=True):
    return AdaptiveCurrentGuard(
        20, soft_ma=350, hard_ma=650, trip_ma=850,
        total_soft_ma=600, total_hard_ma=850, total_trip_ma=1050,
        trip_hold_s=0.04, release_tau_s=0.20, nominal_step_deg=4,
        compliance=ComplianceConfig(),
        object_contact=ObjectContactConfig() if with_object else None,
    )


def poses(joint=6, current=200, error=9):
    measured = np.zeros(20)
    measured[joint] = 69
    effective = measured.copy()
    effective[joint] += error
    desired = effective.copy()
    desired[joint] = 100
    currents = np.zeros(20)
    currents[joint] = current
    return dict(measured_deg=measured, effective_deg=effective,
                desired_deg=desired, current_ma=currents, hybrid_weights=np.zeros(5))


def shaper_at(effective):
    shaper = PositionCommandShaper(np.full(20, -180), np.full(20, 180),
                                   smoothing=False, min_send_step_deg=0, startup_blend_s=0,
                                   disabled_positions_deg={BROKEN_PINKY_INDEX: 0})
    shaper.reset(effective, now=0)
    return shaper


def send(shaper, result, now):
    effective = shaper.effective_command()
    lower, upper = np.full(20, -np.inf), np.full(20, np.inf)
    mask = result.limited_mask
    lower[mask] = np.minimum(effective[mask], result.target_deg[mask])
    upper[mask] = np.maximum(effective[mask], result.target_deg[mask])
    step = shaper.step(result.target_deg, now=now, command_bounds=(lower, upper))
    shaper.accept_output(step.output_deg)
    return step.output_deg


def settle(current=200, joint=6, error=9):
    controller, data = guard(), poses(joint, current, error)
    # Replay an already blocked setpoint. At 533 mA the existing guard also
    # freezes a further close request, while leaving the logged 9-degree error.
    data["desired_deg"][joint] = data["effective_deg"][joint] + (2 if current > 350 else 0)
    shaper = shaper_at(data["effective_deg"])
    events = []
    for tick in range(71):
        now = tick * 0.02
        result = controller.update(**data, now=now)
        if tick == 0 and current == 533:
            assert result.min_scale == 0
        events.extend(result.object_contact_events)
        old = data["effective_deg"].copy()
        data["effective_deg"] = send(shaper, result, now)
        assert old[joint] - 0.300001 <= data["effective_deg"][joint] <= old[joint]
    return controller, data, result, events, now


def test_free_low_current_matches_existing_path_and_direct_step():
    old, new = guard(False), guard()
    data = poses(current=20, error=0)
    shaper = shaper_at(data["effective_deg"])
    data["desired_deg"][:] = 100
    for tick in range(20):
        now = tick * 0.02
        before = old.update(**data, now=now)
        after = new.update(**data, now=now)
        np.testing.assert_array_equal(after.target_deg, before.target_deg)
        np.testing.assert_array_equal(after.limited_mask, before.limited_mask)
        assert after.min_scale == before.min_scale
        assert after.diagnostics["object_contact_state"] == ["FREE"] * 5
        assert not after.object_contact_events
        output = send(shaper, after, now)
        assert np.max(np.abs(output - data["effective_deg"])) <= 4
        assert output[BROKEN_PINKY_INDEX] == 0
        data["effective_deg"], data["measured_deg"] = output, output.copy()


def test_rising_current_with_real_progress_does_not_latch_on_velocity_noise():
    controller = guard()
    data = poses(error=4)
    for tick in range(30):
        data["desired_deg"][6] = 85 + tick
        data["effective_deg"][6] = 73 + tick
        data["measured_deg"][6] = 69 + tick - (9 if tick == 12 else 0)
        data["current_ma"][6] = min(400, 100 + tick * 20)
        result = controller.update(**data, now=tick * 0.02)
        assert result.diagnostics["object_contact_state"][1] == "FREE"
        assert not result.object_contact_events


def test_sustained_poor_progress_latches_and_yields_logged_middle_preload(tmp_path):
    for current, preload in ((200, 1.5), (300, 1.0), (533, 0.5)):
        controller, data, result, events, now = settle(current, joint=10)
        assert result.diagnostics["object_contact_state"][2] == "CONTACT_HOLD"
        assert data["effective_deg"][10] == pytest.approx(69 + preload)
        assert result.diagnostics["contact_preload_deg"][2] == pytest.approx(preload)
        assert [e["event"] for e in events] == ["OBJECT_CONTACT_PENDING", "OBJECT_CONTACT_LATCHED"]
        assert events[-1]["contact_anchor_effective"][1] == 78
        assert events[-1]["contact_anchor_measured"][1] == 69
        assert not result.trip
    # Recorded values must survive CSV serialization with states/reasons intact.
    writer = DebugRunWriter(tmp_path, manifest={}, run_stamp="object_contact", defer_archive=True)
    snapshot = {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in result.diagnostics.items()}
    row = writer.record(snapshot)
    for event in events:
        details = dict(event)
        writer.add_event(details.pop("event"), **details)
    writer.finalize()
    assert row["object_contact_state_middle"] == "CONTACT_HOLD"
    assert row["contact_anchor_measured_10"] == 69
    assert "rj_dg_3_3" in row["object_contact_limited_joints"]
    # The original 1245 mA total trip remains authoritative even after latch.
    data["current_ma"][0] = 400
    data["current_ma"][4] = 312
    assert controller.update(**data, now=now + 0.02).total_current_ma == 1245
    assert controller.update(**data, now=now + 0.061).trip


def test_one_blocked_index_joint_limits_chain_and_preserves_emergency_trip():
    controller, data, result, _, now = settle(350, joint=6, error=9.5)
    chain = [JOINT_NAMES.index(name) for name in FINGER_FLEXION_JOINTS["index"]]
    assert chain == [5, 6, 7]
    data["desired_deg"][chain] += 20
    data["desired_deg"][4] += 10  # lateral motion is independent
    result = controller.update(**data, now=now + 0.02)
    assert np.all(result.limited_mask[chain])
    assert np.all(result.target_deg[chain] <= data["effective_deg"][chain] + 1)
    assert not result.limited_mask[4]
    assert result.target_deg[4] == data["desired_deg"][4]
    assert result.diagnostics["object_contact_active"] == [False, True, False, False, False]
    # Logged index overload: 778 mA on _3, 1118 mA total. Local relief is
    # permitted, but cannot cancel or delay the unchanged emergency trip.
    data["effective_deg"][6] = 78.5
    data["current_ma"][6], data["current_ma"][10] = 778, 340
    result = controller.update(**data, now=now + 0.04)
    assert result.total_current_ma == 1118 and result.max_current_ma == 778
    assert 78.2 - 1e-8 <= result.target_deg[6] < 78.5
    assert not result.trip
    assert controller.update(**data, now=now + 0.081).trip


def test_post_contact_closing_gain_and_smooth_release_when_object_disappears():
    controller, data, _, _, now = settle()
    start = data["effective_deg"][6]
    data["desired_deg"][6] += 20
    data["measured_deg"][6] += 1  # allow a little physical closure without excess preload
    result = controller.update(**data, now=now + 0.02)
    assert result.target_deg[6] - start == pytest.approx(20 * 0.05)
    assert result.diagnostics["post_contact_gain"][1] == 0.05
    data["effective_deg"] = result.target_deg.copy()
    data["current_ma"].fill(0)
    released = []
    for tick in range(2, 25):
        data["measured_deg"] = data["effective_deg"].copy()
        result = controller.update(**data, now=now + tick * 0.02)
        released.extend(result.object_contact_events)
        assert result.target_deg[6] - data["effective_deg"][6] <= 0.300001
        data["effective_deg"] = result.target_deg.copy()
    assert result.diagnostics["object_contact_state"][1] == "FREE"
    assert [e["reason"] for e in released] == ["LOAD_RELEASED"]
    assert start + 1 < data["effective_deg"][6] < data["desired_deg"][6]


def test_opening_has_unit_gain_releases_latch_without_catchup_or_chatter():
    controller, data, _, _, now = settle()
    data["desired_deg"][6] -= 1
    partial = controller.update(**data, now=now + 0.02)
    assert partial.target_deg[6] == pytest.approx(data["effective_deg"][6] - 1)
    data["effective_deg"] = partial.target_deg.copy()
    data["desired_deg"][6] += 1
    reclose = controller.update(**data, now=now + 0.04)
    assert reclose.target_deg[6] == pytest.approx(data["effective_deg"][6] + 0.05)
    assert reclose.diagnostics["object_contact_state"][1] == "CONTACT_HOLD"
    data["effective_deg"] = reclose.target_deg.copy()
    now += 0.04
    events = []
    for tick in range(1, 5):
        data["desired_deg"][6] -= 1
        result = controller.update(**data, now=now + tick * 0.02)
        assert result.target_deg[6] == pytest.approx(data["effective_deg"][6] - 1)
        events.extend(result.object_contact_events)
        data["effective_deg"] = result.target_deg.copy()
    assert result.diagnostics["object_contact_state"][1] == "FREE"
    assert [e["reason"] for e in events] == ["OPERATOR_OPENING"]
    result = controller.update(**data, now=now + 0.10)  # Holding the opened hand must not reclose it.
    assert result.target_deg[6] == pytest.approx(data["effective_deg"][6])
    assert not result.object_contact_events
