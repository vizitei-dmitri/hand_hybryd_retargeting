"""Message conversion for the Unity teleoperation hand publisher."""

from geometry_msgs.msg import Pose, PoseArray
from vr_haptic_msgs.msg import ManoLandmarks


EXPECTED_LANDMARKS = 21


def mano_to_pose_array(message: ManoLandmarks) -> PoseArray:
    """Convert the project's 21-point Mano message without changing its frame."""
    if len(message.landmarks) != EXPECTED_LANDMARKS:
        raise ValueError(
            f"expected {EXPECTED_LANDMARKS} Mano landmarks, "
            f"received {len(message.landmarks)}"
        )

    output = PoseArray()
    output.header = message.header
    for point in message.landmarks:
        pose = Pose()
        pose.position.x = point.x
        pose.position.y = point.y
        pose.position.z = point.z
        pose.orientation.w = 1.0
        output.poses.append(pose)
    return output
