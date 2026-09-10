"""Contact side channel observes landmarks without changing retarget output."""

from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseArray

from dg5f_teleop.contact_signals import (
    TIP_INDICES, PAIR_NAMES, decode_contact_packet, fingertip_distances, proximity_weights,
)
from dg5f_teleop.retarget_node import RetargetNode


def points():
    result = np.zeros((21, 3))
    result[list(TIP_INDICES), 0] = [0.0, 0.12, 0.20, 0.21, 0.30]
    return result


def test_safety_sees_middle_ring_even_with_no_thumb_contact():
    distances = fingertip_distances(points())
    weights = proximity_weights(distances)
    assert PAIR_NAMES[5] == "middle-ring"
    assert distances[5] == pytest.approx(0.01)
    assert weights[5] == 1
    np.testing.assert_array_equal(weights[:4], np.zeros(4))


@pytest.mark.parametrize("hybrid", [False, True])
def test_publisher_preserves_landmarks_output_and_original_hybrid_weights(hybrid):
    params = dict(hybrid_contact_start=0.055, hybrid_contact_full=0.025, hybrid_max_blend=0.8)
    safety, contacts = [], []
    node = SimpleNamespace(
        _hybrid=hybrid,
        _safety_proximity_pub=SimpleNamespace(publish=safety.append),
        _hybrid_contact_pub=SimpleNamespace(publish=contacts.append),
        get_parameter=lambda key: SimpleNamespace(value=params[key]),
        _last_output=np.arange(20.0),
    )
    node._hybrid_weights = lambda p: RetargetNode._hybrid_weights(node, p)
    source = PoseArray()
    source.header.stamp.sec = 100
    source.header.stamp.nanosec = 250000000
    landmarks = points()
    original = landmarks.copy()
    output = node._last_output.copy()
    expected_weights = node._hybrid_weights(landmarks)
    RetargetNode._publish_contact(node, source, landmarks)
    packet = decode_contact_packet(safety[0].data)
    assert packet[0] == 100.25
    np.testing.assert_allclose(packet[1:], fingertip_distances(landmarks))
    assert "middle-ring" in safety[0].layout.dim[0].label
    if hybrid:
        decoded = decode_contact_packet(contacts[0].data, hybrid=True)
        np.testing.assert_array_equal(decoded[1:6], expected_weights)
        np.testing.assert_array_equal(decoded[6:], packet[1:5])
    else:
        assert not contacts
    np.testing.assert_array_equal(landmarks, original)
    np.testing.assert_array_equal(node._last_output, output)


@pytest.mark.parametrize("data,hybrid", [
    ([1] + [float("nan")] * 7, False), ([1] + [-1] * 7, False),
    ([0] + [0.1] * 7, False), ([1] + [0.1] * 6, False),
    ([1] + [1.1] * 5 + [0.01] * 4, True),
])
def test_bad_contact_packet_is_rejected(data, hybrid):
    with pytest.raises(ValueError):
        decode_contact_packet(data, hybrid=hybrid)
