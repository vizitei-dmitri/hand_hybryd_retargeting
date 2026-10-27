"""Publish reconstruction arrays or live result JSON; ROS imports are optional."""

from dataclasses import replace
import json
import math
import time
from typing import Mapping, Optional, Union

import numpy as np

from .data import (
    JsonlFollower, Reconstruction, ReplaySchedule, ResultFrame, SKELETON_EDGES,
    axis_rotation, dumps,
)

try:
    import rclpy
    from geometry_msgs.msg import Point, Pose, PoseArray
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, String
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError as error:
    rclpy = None
    Node = object
    ROS_IMPORT_ERROR = error
else:
    ROS_IMPORT_ERROR = None


def require_ros() -> None:
    if ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS 2 dependencies are unavailable. Source /opt/ros/humble/setup.bash "
            "and the colcon overlay, or use the kinect-ros Docker profile. "
            "The standalone kinect_mano CLI does not require ROS."
        ) from ROS_IMPORT_ERROR


class KinectManoNode(Node):
    """Transport only: fitting and all model dependencies belong to the core."""

    def __init__(self, **kwargs) -> None:
        require_ros()
        kwargs.setdefault("namespace", "/kinect_mano")
        super().__init__("tracker", **kwargs)
        defaults = {
            "mode": "replay", "reconstruction": "", "quality_json": "",
            "source_axes": "optical", "frame_id": "kinect_mano_world",
            "speed": 1.0, "loop": False, "poll_hz": 200.0,
            "publish_cloud": False, "marker_lifetime_s": 0.5,
            "live_jsonl": "",
            "max_live_age_s": 0.35,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.mode = str(self.get_parameter("mode").value)
        if self.mode not in ("replay", "live"):
            raise ValueError("mode must be 'replay' or 'live'")
        self.source_axes = str(self.get_parameter("source_axes").value)
        axis_rotation(self.source_axes)
        self.frame_id = str(self.get_parameter("frame_id").value)
        if not self.frame_id or self.frame_id.startswith("/"):
            raise ValueError("frame_id must be nonempty and must not start with '/'")
        self.marker_lifetime = float(self.get_parameter("marker_lifetime_s").value)
        if not math.isfinite(self.marker_lifetime) or self.marker_lifetime < 0:
            raise ValueError("marker_lifetime_s must be finite and nonnegative")
        self.max_live_age_s = float(self.get_parameter("max_live_age_s").value)
        if not math.isfinite(self.max_live_age_s) or self.max_live_age_s <= 0:
            raise ValueError("max_live_age_s must be finite and positive")
        self._landmarks = self.create_publisher(PoseArray, "landmarks", 1)
        self._markers = self.create_publisher(MarkerArray, "markers", 1)
        self._mano_pub = self.create_publisher(String, "mano", 1)
        self._quality = self.create_publisher(String, "quality", 1)
        self._cloud = (
            self.create_publisher(PointCloud2, "cloud", 1)
            if self.get_parameter("publish_cloud").value else None
        )
        self._camera_clouds = {
            topic: self.create_publisher(PointCloud2, topic, 1)
            for topic in ("cloud_A", "cloud_B")
        } if self._cloud is not None else {}
        self._source = None
        self._schedule = None
        self._timer = None
        self._subscription = None
        if self.mode == "replay":
            path = str(self.get_parameter("reconstruction").value)
            if not path:
                raise ValueError("replay requires reconstruction:=/path/to/reconstruction.npz")
            self._source = Reconstruction(path, str(self.get_parameter("quality_json").value))
            if self._source.metadata.get("handedness", "right") != "right":
                raise ValueError("The right-hand landmark interface requires a right MANO model")
            self._schedule = ReplaySchedule(
                self._source.timestamps, float(self.get_parameter("speed").value),
                bool(self.get_parameter("loop").value),
            )
            poll_hz = float(self.get_parameter("poll_hz").value)
            if not math.isfinite(poll_hz) or poll_hz <= 0:
                raise ValueError("poll_hz must be finite and positive")
            self._timer = self.create_timer(1. / poll_hz, self._replay_tick)
            self.get_logger().info(
                f"Replaying {len(self._source.timestamps)} frames at recorded pacing; "
                "message stamps use current ROS time, source timestamps remain in JSON."
            )
        else:
            live_jsonl = str(self.get_parameter("live_jsonl").value)
            if live_jsonl:
                self._follower = JsonlFollower(live_jsonl)
                self._timer = self.create_timer(0.01, self._live_tick)
                self.get_logger().info(f"Following live JSONL results: {live_jsonl}")
            else:
                self._subscription = self.create_subscription(
                    String, "input/result", self._on_live_result, 1,
                )
                self.get_logger().info("Listening for live result JSON on input/result")

    def _live_tick(self) -> None:
        try:
            lines = self._follower.poll()
        except (OSError, ValueError) as error:
            self.get_logger().warning(f"Cannot read live JSONL: {error}")
            return
        # When the producer outpaces ROS, publish the newest complete valid
        # record in this chunk, rather than replaying delayed control landmarks.
        for line in reversed(lines):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("live result must be a JSON object")
                frame = ResultFrame.from_mapping(value)
                self.publish_result(frame)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                self.get_logger().warning(f"Rejected live JSONL result: {error}")
                continue
            break

    def _replay_tick(self) -> None:
        index = self._schedule.poll(time.monotonic())
        if index is not None:
            self.publish_result(self._source.frame(index))
        if self._schedule.finished:
            self._timer.cancel()
            self.get_logger().info("Replay complete; no further landmarks will be published.")

    def _on_live_result(self, message) -> None:
        try:
            value = json.loads(message.data)
            if not isinstance(value, dict):
                raise ValueError("live result must be a JSON object")
            self.publish_result(value)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            self.get_logger().warning(f"Rejected live result: {error}")

    def publish_result(self, result: Union[ResultFrame, Mapping]) -> None:
        """In-process live adapter: call on the ROS executor thread with one fit.

        The mapping schema is the NPZ schema with a scalar timestamp_s and no
        leading frame dimension. No Torch objects or model code are imported.
        """
        if isinstance(result, ResultFrame):
            # Apply the same validation to direct callers as to the JSON bridge.
            result = ResultFrame.from_mapping(vars(result))
        else:
            result = ResultFrame.from_mapping(result)
        if result.metadata.get("handedness", "right") != "right":
            raise ValueError("The right-hand landmark interface requires a right MANO model")
        source_valid = result.valid
        if self.mode == "live":
            quality = dict(result.quality) if isinstance(result.quality, Mapping) else {
                "source_quality": result.quality,
            }
            received = quality.get("host_received_perf_counter_s")
            reason = "missing_host_received_perf_counter_s"
            age = None
            if isinstance(received, (float, int)) and not isinstance(received, bool):
                age = time.perf_counter() - received
                if not math.isfinite(age):
                    reason = "invalid_host_received_perf_counter_s"
                elif age < 0:
                    reason = "future_or_incompatible_host_clock"
                elif age > self.max_live_age_s:
                    reason = "stale"
                else:
                    reason = "fresh"
            quality["live_freshness"] = {
                "accepted": reason == "fresh", "reason": reason,
                "age_s": age, "max_age_s": self.max_live_age_s,
            }
            result = replace(result, valid=result.valid and reason == "fresh", quality=quality)
        frame = result.transformed(self.source_axes)
        header = Header()
        header.frame_id = self.frame_id
        header.stamp = self.get_clock().now().to_msg()
        stamp_s = header.stamp.sec + header.stamp.nanosec * 1e-9
        metadata = {
            "schema_version": 1, "frame_id": self.frame_id,
            "stamp_s": stamp_s, "source_axes": self.source_axes,
            "output_axes": "ros", "mode": self.mode,
            "offline_reconstruction": self.mode == "replay",
            "source_valid": source_valid,
        }
        parameters = frame.parameters()
        parameters.update(metadata)
        self._mano_pub.publish(String(data=dumps(parameters)))
        self._quality.publish(String(data=dumps({
            **metadata, "frame_index": frame.frame_index, "valid": frame.valid,
            "source_timestamp_s": frame.timestamp_s, "quality": frame.quality,
        })))
        if frame.valid:
            poses = PoseArray(header=header)
            for xyz in frame.joints_m:
                pose = Pose()
                pose.position = self._point(xyz)
                pose.orientation.w = 1.0
                poses.poses.append(pose)
            self._landmarks.publish(poses)
        # Invalid frames clear old geometry and never produce control landmarks.
        self._markers.publish(self._make_markers(frame, header))
        if self._cloud is not None:
            points = frame.cloud_m if frame.valid and frame.cloud_m is not None else []
            self._cloud.publish(self._make_cloud(points, header))
            for topic, aliases in (("cloud_A", ("camera_A", "synthetic_0")),
                                   ("cloud_B", ("camera_B", "synthetic_1"))):
                points = next((frame.clouds_m[key] for key in aliases
                               if key in frame.clouds_m), []) if frame.valid else []
                self._camera_clouds[topic].publish(self._make_cloud(points, header))

    @staticmethod
    def _point(xyz):
        return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

    def _make_markers(self, frame: ResultFrame, header):
        markers = []
        kinds = (Marker.TRIANGLE_LIST, Marker.LINE_LIST, Marker.SPHERE_LIST)
        for marker_id, kind in enumerate(kinds):
            marker = Marker()
            marker.header = header
            marker.ns = "mano"
            marker.id = marker_id
            marker.type = kind
            marker.action = Marker.ADD if frame.valid else Marker.DELETE
            marker.pose.orientation.w = 1.0
            lifetime_ns = round(self.marker_lifetime * 1e9)
            marker.lifetime.sec, marker.lifetime.nanosec = divmod(lifetime_ns, 1_000_000_000)
            marker.color.r, marker.color.g, marker.color.b = (0.25, 0.65, 0.95)
            marker.color.a = 0.65 if marker_id == 0 else 1.0
            if marker_id == 0:
                marker.scale.x = marker.scale.y = marker.scale.z = 1.0
                points = frame.vertices_m[frame.faces.reshape(-1)]
            elif marker_id == 1:
                marker.scale.x = 0.003
                marker.color.r, marker.color.g, marker.color.b = (1.0, 0.7, 0.1)
                points = frame.joints_m[np.asarray(SKELETON_EDGES).reshape(-1)]
            else:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.006
                marker.color.r, marker.color.g, marker.color.b = (0.95, 0.25, 0.25)
                points = frame.joints_m
            if frame.valid:
                marker.points = [self._point(xyz) for xyz in points]
            markers.append(marker)
        return MarkerArray(markers=markers)

    @staticmethod
    def _make_cloud(points, header):
        # Build standard XYZ PointCloud2 directly to avoid another ROS helper dependency.
        xyz = np.asarray(points, dtype="<f4").reshape(-1, 3)
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
        message = PointCloud2()
        message.header = header
        message.height, message.width = 1, len(xyz)
        message.fields = [
            PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
            for name, offset in (("x", 0), ("y", 4), ("z", 8))
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = 12 * len(xyz)
        message.is_dense = True
        message.data = xyz.tobytes()
        return message


def main(args=None) -> None:
    require_ros()
    rclpy.init(args=args)
    node: Optional[KinectManoNode] = None
    try:
        node = KinectManoNode()
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
