"""Synthetic Unity ManoLandmarks source for a V2 integration test."""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from vr_haptic_msgs.msg import ManoLandmarks


class FakeManoNode(Node):
    """Publish an animated, anatomically ordered 21-point right hand."""

    def __init__(self) -> None:
        super().__init__("fake_unity_mano")
        self.declare_parameter("topic", "/quest/hand_pose")
        self.declare_parameter("rate_hz", 30.0)
        self._publisher = self.create_publisher(
            ManoLandmarks,
            str(self.get_parameter("topic").value),
            1,
        )
        self._phase = 0.0
        rate_hz = float(self.get_parameter("rate_hz").value)
        self._timer = self.create_timer(1.0 / rate_hz, self._publish)

    @staticmethod
    def _finger(base_x: float, base_y: float, lengths, curl: float):
        points = []
        position = np.array([base_x, base_y, 0.0], dtype=np.float64)
        points.append(position.copy())
        angle = 0.0
        for segment_index, length in enumerate(lengths):
            angle += curl * (0.35 + 0.15 * segment_index)
            position = position + np.array(
                [0.0, length * math.cos(angle), -length * math.sin(angle)]
            )
            points.append(position.copy())
        return points

    def _make_points(self) -> np.ndarray:
        curl = 0.55 + 0.45 * math.sin(self._phase)
        points = [np.array([0.0, 0.0, 0.0])]
        points.extend(
            [
                np.array([0.025, 0.025, 0.0]),
                np.array([0.045, 0.042, -0.005 * curl]),
                np.array([0.060, 0.057, -0.012 * curl]),
                np.array([0.073, 0.068, -0.022 * curl]),
            ]
        )
        for base_x, base_y, lengths in [
            (0.030, 0.065, [0.038, 0.026, 0.020]),
            (0.010, 0.072, [0.043, 0.030, 0.022]),
            (-0.012, 0.067, [0.040, 0.028, 0.021]),
            (-0.032, 0.057, [0.032, 0.023, 0.018]),
        ]:
            points.extend(self._finger(base_x, base_y, lengths, curl))
        return np.asarray(points)

    def _publish(self) -> None:
        message = ManoLandmarks()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "quest_origin"
        for xyz in self._make_points():
            point = Point()
            point.x = float(xyz[0])
            point.y = float(xyz[1])
            point.z = float(xyz[2])
            message.landmarks.append(point)
        self._publisher.publish(message)
        self._phase += 0.04


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeManoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
