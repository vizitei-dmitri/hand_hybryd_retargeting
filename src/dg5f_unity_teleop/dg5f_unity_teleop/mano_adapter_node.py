"""Adapt Unity ManoLandmarks messages to the internal 21-Pose interface."""

from typing import Optional

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from vr_haptic_msgs.msg import ManoLandmarks

from dg5f_unity_teleop.conversion import mano_to_pose_array


class ManoAdapterNode(Node):
    """Bridge `/quest/hand_pose` into `/hands/right/landmarks`."""

    def __init__(self) -> None:
        super().__init__("unity_mano_adapter")
        self.declare_parameter("input_topic", "/quest/hand_pose")
        self.declare_parameter("output_topic", "/hands/right/landmarks")
        self.declare_parameter("reliability", "reliable")

        reliability_name = str(self.get_parameter("reliability").value).lower()
        reliability = (
            QoSReliabilityPolicy.BEST_EFFORT
            if reliability_name == "best_effort"
            else QoSReliabilityPolicy.RELIABLE
        )
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=reliability,
        )

        self._publisher = self.create_publisher(
            PoseArray,
            str(self.get_parameter("output_topic").value),
            qos,
        )
        self._subscription = self.create_subscription(
            ManoLandmarks,
            str(self.get_parameter("input_topic").value),
            self._on_landmarks,
            qos,
        )
        self._last_warning_ns = -10**18
        self.get_logger().info(
            "%s -> %s (ManoLandmarks[21] -> PoseArray[21])"
            % (
                self.get_parameter("input_topic").value,
                self.get_parameter("output_topic").value,
            )
        )

    def _on_landmarks(self, message: ManoLandmarks) -> None:
        try:
            output = mano_to_pose_array(message)
        except ValueError as error:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_warning_ns >= 1_000_000_000:
                self.get_logger().warning(str(error))
                self._last_warning_ns = now_ns
            return
        self._publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ManoAdapterNode] = None
    try:
        node = ManoAdapterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
