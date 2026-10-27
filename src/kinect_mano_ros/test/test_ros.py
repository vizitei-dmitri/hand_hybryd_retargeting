"""Real ROS pub/sub and launch-policy checks; skipped on a non-ROS host."""

import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
import uuid

import numpy as np

from kinect_mano_ros.data import dumps
from kinect_mano_ros.node import KinectManoNode, rclpy
from test_ros_data import arrays, live_frame

if rclpy is not None:
    from geometry_msgs.msg import PoseArray
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray


@unittest.skipIf(rclpy is None, "ROS 2 not installed")
class TransportTests(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.temporary = tempfile.TemporaryDirectory()
        self.namespace = "/test_kinect_" + uuid.uuid4().hex[:8]
        self.observer = Node("observer", namespace=self.namespace)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.observer)
        self.tracker = None
        self.messages = {key: [] for key in (
            "landmarks", "mano", "quality", "markers", "cloud", "cloud_A", "cloud_B",
        )}
        self.subscriptions = [
            self.observer.create_subscription(kind, topic, self.messages[topic].append, 10)
            for topic, kind in (
                ("landmarks", PoseArray), ("mano", String), ("quality", String),
                ("markers", MarkerArray), ("cloud", PointCloud2),
                ("cloud_A", PointCloud2), ("cloud_B", PointCloud2),
            )
        ]

    def tearDown(self):
        if self.tracker is not None:
            self.executor.remove_node(self.tracker)
            self.tracker.destroy_node()
        self.executor.remove_node(self.observer)
        self.observer.destroy_node()
        self.executor.shutdown()
        rclpy.shutdown()
        self.temporary.cleanup()

    def spin_for(self, duration):
        until = time.monotonic() + duration
        while time.monotonic() < until:
            self.executor.spin_once(timeout_sec=.01)

    def make_node(self, **params):
        self.tracker = KinectManoNode(
            namespace=self.namespace,
            parameter_overrides=[Parameter(name, value=value) for name, value in params.items()],
        )
        self.executor.add_node(self.tracker)
        if self.tracker._timer is not None:
            self.tracker._timer.cancel()
        self.spin_for(.25)  # Establish discovery before publishing the first frame.
        if self.tracker._timer is not None:
            self.tracker._timer.reset()

    def test_replay_pacing_stamps_geometry_and_invalid_frames(self):
        data = arrays()
        data["valid"][1] = False
        data["joints_m"][1] = np.nan
        data["cloud_m"] = np.tile([[[1., 2., 3.], [np.nan, 0., 0.]]], (4, 1, 1))
        data["cloud_camera_ids"] = np.array(["camera_A", "synthetic_1"])
        data["clouds_per_camera_m"] = np.tile(
            [[[[1., 2., 3.]], [[-2., 4., 1.]]]], (4, 1, 1, 1),
        )
        path = Path(self.temporary.name) / "reconstruction.npz"
        np.savez(path, **data)
        self.make_node(reconstruction=str(path), publish_cloud=True)
        self.spin_for(.08)
        self.assertEqual(len(self.messages["mano"]), 1, "replay must not burst future frames")
        self.spin_for(.6)
        self.assertEqual(len(self.messages["mano"]), 4)
        self.assertEqual(len(self.messages["landmarks"]), 3)
        self.assertEqual(len(self.messages["quality"]), 4)
        pose = self.messages["landmarks"][0]
        self.assertEqual(pose.header.frame_id, "kinect_mano_world")
        self.assertLess(abs(pose.header.stamp.sec - time.time()), 5.)
        point = pose.poses[0].position
        np.testing.assert_allclose([point.x, point.y, point.z], [3., -1., -2.])
        payload = json.loads(self.messages["mano"][0].data)
        self.assertEqual(payload["source_timestamp_s"], 10_000_000_000.)
        self.assertTrue(payload["offline_reconstruction"])
        self.assertEqual(len(payload["theta"]), 45)
        self.assertEqual(len(payload["beta"]), 10)
        first_markers = self.messages["markers"][0].markers
        self.assertEqual([m.type for m in first_markers],
                         [Marker.TRIANGLE_LIST, Marker.LINE_LIST, Marker.SPHERE_LIST])
        self.assertEqual([len(m.points) for m in first_markers], [3, 40, 21])
        self.assertTrue(all(m.action == Marker.DELETE
                            for m in self.messages["markers"][1].markers))
        cloud = self.messages["cloud"][0]
        self.assertEqual((cloud.width, cloud.point_step), (1, 12))
        np.testing.assert_allclose(np.frombuffer(bytes(cloud.data), dtype="<f4"), [3., -1., -2.])
        self.assertEqual(self.messages["cloud"][1].width, 0)
        cloud_a = self.messages["cloud_A"][0]
        cloud_b = self.messages["cloud_B"][0]
        np.testing.assert_allclose(np.frombuffer(bytes(cloud_a.data), dtype="<f4"),
                                   [3., -1., -2.])
        np.testing.assert_allclose(np.frombuffer(bytes(cloud_b.data), dtype="<f4"),
                                   [1., 2., -4.])
        self.assertTrue(self.tracker._timer.is_canceled())

    def test_live_json_topic(self):
        self.make_node(mode="live")
        publisher = self.observer.create_publisher(String, "input/result", 1)
        self.spin_for(.2)
        publisher.publish(String(data=dumps(live_frame())))
        self.spin_for(.15)
        self.assertEqual(len(self.messages["landmarks"]), 1)
        self.assertFalse(json.loads(self.messages["mano"][0].data)["offline_reconstruction"])
        publisher.publish(String(data="{}"))
        self.spin_for(.1)
        self.assertEqual(len(self.messages["landmarks"]), 1)

    def test_live_jsonl_partial_line(self):
        path = Path(self.temporary.name) / "stream.jsonl"
        self.make_node(mode="live", live_jsonl=str(path))
        serialized = dumps(live_frame()).encode()
        path.write_bytes(serialized[:100])
        self.spin_for(.08)
        self.assertEqual(len(self.messages["mano"]), 0)
        with path.open("ab") as stream:
            stream.write(serialized[100:] + b"\n")
        self.spin_for(.12)
        self.assertEqual(len(self.messages["landmarks"]), 1)
        frame = live_frame()
        frame["valid"] = False
        with path.open("ab") as stream:
            stream.write(dumps(frame).encode() + b"\n")
        self.spin_for(.12)
        self.assertEqual(len(self.messages["mano"]), 2)
        self.assertEqual(len(self.messages["landmarks"]), 1)
        frame = live_frame()
        frame["metadata"] = {"handedness": "left"}
        with path.open("ab") as stream:
            stream.write(dumps(frame).encode() + b"\n")
        self.spin_for(.08)
        self.assertEqual(len(self.messages["landmarks"]), 1)

    def test_live_freshness_and_per_camera_clouds(self):
        self.make_node(mode="live", publish_cloud=True)
        frame = live_frame()
        frame["clouds_m"] = {"synthetic_0": [[1., 2., 3.]], "camera_B": [[2., 3., 4.]]}
        self.tracker.publish_result(frame)
        self.spin_for(.05)
        self.assertEqual(len(self.messages["landmarks"]), 1)
        self.assertEqual(self.messages["cloud_A"][-1].width, 1)
        self.assertEqual(self.messages["cloud_B"][-1].width, 1)
        for provenance, reason in (
            (time.perf_counter() - 5., "stale"),
            (time.perf_counter() + 500., "future_or_incompatible_host_clock"),
            (None, "missing_host_received_perf_counter_s"),
        ):
            frame = live_frame()
            frame["quality"]["host_received_perf_counter_s"] = provenance
            self.tracker.publish_result(frame)
            self.spin_for(.05)
            diagnostic = json.loads(self.messages["mano"][-1].data)
            self.assertFalse(diagnostic["valid"])
            self.assertTrue(diagnostic["source_valid"])
            self.assertEqual(diagnostic["quality"]["live_freshness"]["reason"], reason)
            self.assertEqual(len(self.messages["landmarks"]), 1)
            self.assertEqual(self.messages["cloud_A"][-1].width, 0)


@unittest.skipIf(rclpy is None, "ROS 2 launch not installed")
class LaunchPolicyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "launch" / "hand_source.launch.py"
        spec = importlib.util.spec_from_file_location("hand_source", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def context(self, source, **overrides):
        from launch import LaunchContext

        context = LaunchContext()
        context.launch_configurations.update(
            source=source, start_lerobot="false", lerobot_backend="mock", **overrides,
        )
        return context

    def test_kinect_reuses_original_launch_with_robot_disabled(self):
        actions = self.module._selected_pipeline(self.context("kinect"))
        self.assertEqual(len(actions), 2)
        kinect_arguments = dict(actions[0].launch_arguments)
        self.assertEqual(kinect_arguments["landmarks_topic"], "/hands/right/landmarks")
        arguments = dict(actions[1].launch_arguments)
        for key in ("start_endpoint", "start_adapter", "start_lerobot", "fake_mano"):
            self.assertEqual(arguments[key], "false")
        self.assertEqual(arguments["lerobot_backend"], "mock")
        self.assertEqual(arguments["lerobot_auto_enable"], "false")

    def test_quest_reuses_endpoint_adapter_and_disables_robot(self):
        actions = self.module._selected_pipeline(self.context("quest"))
        self.assertEqual(len(actions), 1)
        arguments = dict(actions[0].launch_arguments)
        self.assertEqual(arguments["start_endpoint"], "true")
        self.assertEqual(arguments["start_adapter"], "true")
        self.assertEqual(arguments["start_lerobot"], "false")

    def test_physical_startup_overrides_are_rejected(self):
        context = self.context("kinect")
        context.launch_configurations["start_lerobot"] = "true"
        with self.assertRaises(ValueError):
            self.module._selected_pipeline(context)
        context = self.context("quest")
        context.launch_configurations["lerobot_backend"] = "tesollo"
        with self.assertRaises(ValueError):
            self.module._selected_pipeline(context)


if __name__ == "__main__":
    unittest.main()
