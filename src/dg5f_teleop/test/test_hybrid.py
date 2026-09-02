import pytest

from dg5f_teleop.retarget_node import smooth_contact_weight


def test_contact_weight_is_zero_far_and_maximum_near():
    assert smooth_contact_weight(0.060, 0.055, 0.025, 0.8) == 0.0
    assert smooth_contact_weight(0.020, 0.055, 0.025, 0.8) == 0.8


def test_contact_weight_changes_smoothly_inside_activation_band():
    far = smooth_contact_weight(0.050, 0.055, 0.025, 0.8)
    middle = smooth_contact_weight(0.040, 0.055, 0.025, 0.8)
    near = smooth_contact_weight(0.030, 0.055, 0.025, 0.8)

    assert 0.0 < far < middle < near < 0.8


def test_contact_weight_rejects_reversed_thresholds():
    with pytest.raises(ValueError, match="contact_start"):
        smooth_contact_weight(0.03, 0.02, 0.04, 0.8)
