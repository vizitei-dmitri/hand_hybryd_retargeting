"""Five focused temporal-contact scenarios using the physical bridge profile."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from lerobot_robot_dg5f.object_contact import ObjectContactConfig, PerFingerObjectContact


def detector():
    path = Path(__file__).parents[1] / "config/bridge.params.yaml"
    params = yaml.safe_load(path.read_text())["dg5f_lerobot_bridge"]["ros__parameters"]
    config = ObjectContactConfig(**{
        name: params[f"object_contact_{name}"] for name in vars(ObjectContactConfig())
    })
    assert config == ObjectContactConfig()
    return PerFingerObjectContact(config)


def sample(control, now, *, current=300, desired=78.4, effective=78, measured=69, slope=0):
    arrays = [np.zeros(20) for _ in range(4)]
    currents, commands, positions, targets = arrays
    currents[6], commands[6], positions[6], targets[6] = current, effective, measured, desired
    slopes = np.zeros(20)
    slopes[6] = slope
    return control.update(current_ma=currents, effective_deg=commands, measured_deg=positions,
                          desired_deg=targets, soft_target_deg=targets, slope_ma_s=slopes, now=now)


def warm(control, desired=78.4):
    for tick in range(12):
        result = sample(control, tick * 0.01, current=0, desired=desired)
        assert result.diagnostics["object_contact_state"][1] == "FREE"


def test_slow_free_index_with_small_jitter_does_not_latch():
    for current in (240, 300):
        control = detector()
        for tick in range(100):
            position = 74.5 + tick * 0.05
            jitter = (0, 0.1, -0.1, 0.2, -0.2)[tick % 5]
            result = sample(control, tick * 0.02, current=current, slope=2000,
                            measured=position, effective=position + 3.5,
                            desired=position + 5.5 + jitter)
            assert result.diagnostics["object_contact_state"][1] == "FREE"
            assert not result.transitions


def test_short_evidence_dropouts_preserve_pending_hold():
    control = detector()
    warm(control)
    result = sample(control, 0.12)
    events = list(result.transitions)
    finger = control.fingers["index"]
    assert finger.pending_since == pytest.approx(0.12)
    # Each gap is shorter than 40 ms, including small retarget jitter.
    for now, current, jitter in (
        (0.13, 200, -0.2), (0.14, 200, -0.1), (0.15, 300, 0),
        (0.16, 300, 0), (0.17, 200, -0.1), (0.18, 200, -0.2), (0.19, 300, 0),
        (0.20, 200, 0), (0.209, 200, 0),
    ):
        result = sample(control, now, current=current, desired=78.4 + jitter)
        events.extend(result.transitions)
        assert finger.state == "CONTACT_PENDING"
        assert finger.pending_since == pytest.approx(0.12)
    # Do not latch using stale evidence, even after the 80 ms hold elapsed.
    assert finger.last_evidence_time == pytest.approx(0.19)
    result = sample(control, 0.21)
    events.extend(result.transitions)
    assert finger.state == "CONTACT_HOLD"
    assert [event["event"] for event in events] == ["OBJECT_CONTACT_PENDING", "OBJECT_CONTACT_LATCHED"]


def test_evidence_dropout_over_grace_clears_pending():
    control = detector()
    warm(control)
    sample(control, 0.12)
    for now in (0.13, 0.14, 0.15, 0.16):
        result = sample(control, now, current=200)
        assert result.diagnostics["object_contact_state"][1] == "CONTACT_PENDING"
        assert not result.transitions
    result = sample(control, 0.161, current=200)
    assert control.fingers["index"].state == "FREE"
    assert control.fingers["index"].pending_since is None
    assert control.fingers["index"].last_evidence_time is None
    assert [event["reason"] for event in result.transitions] == ["EVIDENCE_CLEARED"]
    sample(control, 0.17)
    assert control.fingers["index"].pending_since == pytest.approx(0.17)


def test_high_load_stall_latches_in_10ms_without_ordinary_alignment():
    for current in (647, 650):
        control = detector()
        # Human desired is behind effective: ordinary closing alignment is false.
        # It remains ahead of measured, with no opening motion in the window.
        warm(control, desired=77)
        first = sample(control, 0.12, current=current, desired=77)
        assert first.diagnostics["object_contact_state"][1] == "CONTACT_PENDING"
        assert first.diagnostics["finger_progress_ratio"][1] == pytest.approx(0)
        before_hold = sample(control, 0.129, current=current, desired=76.85)
        assert before_hold.diagnostics["object_contact_state"][1] == "CONTACT_PENDING"
        latched = sample(control, 0.13, current=current, desired=77)
        assert latched.diagnostics["object_contact_state"][1] == "CONTACT_HOLD"
        assert [event["reason"] for event in latched.transitions] == ["HIGH_LOAD_LOW_PROGRESS"]
        assert np.all(latched.limited_mask[[5, 6, 7]])


def test_clear_opening_cancels_pending_and_releases_contact():
    control = detector()
    warm(control)
    sample(control, 0.12)
    # One outlier must not cancel pending; sustained opening must cancel it
    # immediately once recognized, without waiting for evidence grace.
    sample(control, 0.13, desired=77.8)
    sample(control, 0.14, desired=78.4)
    assert control.fingers["index"].state == "CONTACT_PENDING"
    opening = sample(control, 0.15, desired=78.0)
    released = sample(control, 0.16, desired=77.6)
    assert control.fingers["index"].state == "FREE"
    assert [event["reason"] for event in opening.transitions + released.transitions] == ["OPERATOR_OPENING"]

    control = detector()
    warm(control)
    sample(control, 0.12, current=650)
    latched = sample(control, 0.13, current=650)
    assert control.fingers["index"].state == "CONTACT_HOLD"
    effective = latched.target_deg[6]
    events = []
    for tick in range(1, 7):
        result = sample(control, 0.13 + tick * 0.01,
                        desired=78.4 - tick * 0.5, effective=effective)
        assert result.target_deg[6] == pytest.approx(effective - 0.5)
        assert result.diagnostics["post_contact_gain"][1] == 1
        effective = result.target_deg[6]
        events.extend(result.transitions)
    assert control.fingers["index"].state == "FREE"
    assert [event["reason"] for event in events] == ["OPERATOR_OPENING"]
