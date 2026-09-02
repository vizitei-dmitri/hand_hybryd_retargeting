import pytest
from geometry_msgs.msg import Point
from vr_haptic_msgs.msg import ManoLandmarks

from dg5f_unity_teleop.conversion import mano_to_pose_array


def _message(count: int) -> ManoLandmarks:
    message = ManoLandmarks()
    message.header.frame_id = "quest_origin"
    for index in range(count):
        message.landmarks.append(
            Point(x=float(index), y=float(index + 1), z=float(index + 2))
        )
    return message


def test_converts_all_21_points_and_preserves_header():
    output = mano_to_pose_array(_message(21))
    assert output.header.frame_id == "quest_origin"
    assert len(output.poses) == 21
    assert output.poses[8].position.x == 8.0
    assert output.poses[8].position.y == 9.0
    assert output.poses[8].position.z == 10.0
    assert all(pose.orientation.w == 1.0 for pose in output.poses)


@pytest.mark.parametrize("count", [0, 20, 22])
def test_rejects_wrong_landmark_count(count):
    with pytest.raises(ValueError, match="expected 21"):
        mano_to_pose_array(_message(count))
