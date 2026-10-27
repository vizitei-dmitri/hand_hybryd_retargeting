"""ROS array contract, coordinate and stream tests; no ROS installation needed."""

import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from kinect_mano_ros.data import (
    JsonlFollower, OPTICAL_TO_ROS, Reconstruction, ReplaySchedule, ResultFrame,
    axis_rotation, dumps,
)


def arrays(count=4):
    joints = np.zeros((count, 21, 3))
    joints[:, :, :] = [1., 2., 3.]
    vertices = np.zeros((count, 778, 3))
    vertices[:, :3, :] = [[1., 2., 3.], [1.01, 2., 3.], [1., 2.01, 3.]]
    return dict(
        timestamps_s=10_000_000_000. + np.arange(count) * .15,
        joints_m=joints, vertices_m=vertices, faces=np.array([[0, 1, 2]], dtype=np.int32),
        theta=np.zeros((count, 45)), beta=np.zeros(10),
        global_orient=np.zeros((count, 3)), translation_m=np.ones((count, 3)),
        valid=np.ones(count, dtype=bool),
    )


def live_frame():
    data = arrays(1)
    result = {k: v if k in ("faces", "beta") else v[0]
              for k, v in data.items() if k != "timestamps_s"}
    return dict(result, timestamp_s=100., quality={
        "loss": .1, "host_received_perf_counter_s": time.perf_counter(),
    })


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "reconstruction.npz"

    def save(self, **overrides):
        data = arrays()
        data.update(overrides)
        np.savez(self.path, **data)
        return Reconstruction(str(self.path))

    def test_axes_and_global_orientation_are_consistent(self):
        frame = ResultFrame.from_mapping(live_frame()).transformed("optical")
        np.testing.assert_allclose(frame.joints_m[0], [3., -1., -2.])
        np.testing.assert_allclose(frame.translation_m, [1., -1., -1.])
        np.testing.assert_allclose(Rotation.from_rotvec(frame.global_orient).as_matrix(),
                                   OPTICAL_TO_ROS, atol=1e-12)
        np.testing.assert_allclose(frame.beta, np.zeros(10))
        self.assertAlmostEqual(np.linalg.det(axis_rotation("optical")), 1.)
        original = live_frame()
        original["global_orient"] = [.1, -.2, .3]
        output = ResultFrame.from_mapping(original).transformed("optical")
        np.testing.assert_allclose(
            Rotation.from_rotvec(output.global_orient).as_matrix(),
            OPTICAL_TO_ROS @ Rotation.from_rotvec(original["global_orient"]).as_matrix(),
        )

    def test_ros_axes_are_identity(self):
        frame = ResultFrame.from_mapping(live_frame())
        np.testing.assert_array_equal(frame.transformed("ros").joints_m, frame.joints_m)
        with self.assertRaises(ValueError):
            axis_rotation("unknown")

    def test_load_and_quality_sidecar(self):
        quality = {"frames": [{"loss": i} for i in range(4)], "summary": {"ok": True}}
        self.path.with_suffix(".quality.json").write_text(json.dumps(quality))
        source = self.save()
        self.assertEqual(source.frame(2).quality, {"loss": 2})
        self.assertEqual(source.frame(3).frame_index, 3)

    def test_embedded_quality_and_explicit_missing_path(self):
        source = self.save(quality_json=np.array('{"summary":true}'))
        self.assertEqual(source.frame(0).quality, {"summary": True})
        with self.assertRaises(FileNotFoundError):
            Reconstruction(str(self.path), str(self.path.parent / "missing.json"))

    def test_invalid_nan_fits_are_diagnostics_only(self):
        data = arrays()
        data["valid"][1] = False
        data["joints_m"][1] = np.nan
        source = self.save(**data)
        self.assertFalse(source.frame(1).valid)
        frame = source.frame(1)
        message = json.loads(dumps(dict(frame.parameters(), diagnostic=np.inf)))
        self.assertIsNone(message["diagnostic"])
        self.assertNotIn("NaN", dumps(frame.joints_m))

    def test_rejects_corrupt_arrays_before_replay(self):
        for changes in (
            {"joints_m": np.zeros((4, 20, 3))},
            {"theta": np.full((4, 45), np.nan)},
            {"timestamps_s": [3., 1., 2., 4.]},
            {"timestamps_s": [0., np.nan, 2., 4.]},
            {"faces": np.array([[0, 1, 778]])},
            {"faces": np.array([[0., 1., 2.]])},
            {"beta": np.zeros((4, 10))},
            {"valid": np.array([1, 2, 1, 1])},
            {"cloud_m": np.zeros((3, 10, 3))},
        ):
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(ValueError):
                    self.save(**changes)

    def test_missing_array_and_object_arrays_are_rejected(self):
        data = arrays()
        del data["theta"]
        np.savez(self.path, **data)
        with self.assertRaises(ValueError):
            Reconstruction(str(self.path))
        data["theta"] = np.full((4, 45), object())
        np.savez(self.path, **data)
        with self.assertRaises(ValueError):
            Reconstruction(str(self.path))

    def test_quality_count_validation(self):
        self.path.parent.joinpath("quality.json").write_text('[{},{}]')
        with self.assertRaises(ValueError):
            self.save()

    def test_optional_cloud_filters_nonfinite_points(self):
        frame = live_frame()
        frame["cloud_m"] = [[1., 2., 3.], [np.nan, 0., 1.]]
        result = ResultFrame.from_mapping(frame).transformed("optical")
        np.testing.assert_array_equal(result.cloud_m, [[3., -1., -2.]])

    def test_live_empty_cloud_serialization(self):
        frame = dict(live_frame(), cloud_m=[])
        self.assertEqual(ResultFrame.from_mapping(frame).cloud_m.shape, (0, 3))

    def test_per_camera_npz_clouds_and_axes(self):
        clouds = np.full((4, 2, 3, 3), np.nan)
        clouds[:, 0, 0] = [1., 2., 3.]
        clouds[:, 1, :2] = [[2., 3., 4.], [3., 4., 5.]]
        source = self.save(cloud_camera_ids=np.array(["camera_A", "synthetic_1"]),
                           clouds_per_camera_m=clouds)
        frame = source.frame(1).transformed("optical")
        np.testing.assert_array_equal(frame.clouds_m["camera_A"], [[3., -1., -2.]])
        np.testing.assert_array_equal(frame.clouds_m["synthetic_1"],
                                      [[4., -2., -3.], [5., -3., -4.]])

    def test_per_camera_npz_contract_rejects_missing_ids_or_bad_shapes(self):
        for changes in (
            {"cloud_camera_ids": np.array(["camera_A"])},
            {"cloud_camera_ids": np.array(["camera_A"]),
             "clouds_per_camera_m": np.zeros((4, 2, 3, 3))},
            {"cloud_camera_ids": np.array(["camera_A", "camera_A"]),
             "clouds_per_camera_m": np.zeros((4, 2, 3, 3))},
        ):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                self.save(**changes)

    def test_npz_with_no_observed_cameras(self):
        # np.array([]) from the core exporter has a floating dtype when C=0.
        source = self.save(cloud_camera_ids=np.array([]),
                           clouds_per_camera_m=np.empty((4, 0, 0, 3)))
        self.assertEqual(source.frame(0).clouds_m, {})

    def test_node_import_is_optional(self):
        import kinect_mano_ros.node as node

        if node.rclpy is None:
            with self.assertRaisesRegex(RuntimeError, "ROS 2 dependencies"):
                node.KinectManoNode()

    def test_model_metadata_survives_replay_and_live(self):
        source = self.save(metadata_json=np.array('{"model_type":"numerical_fixture"}'))
        self.assertEqual(source.frame(0).parameters()["metadata"]["model_type"],
                         "numerical_fixture")
        frame = dict(live_frame(), model_type="numerical_fixture")
        self.assertEqual(ResultFrame.from_mapping(frame).transformed("optical").metadata,
                         {"model_type": "numerical_fixture"})


class ScheduleTests(unittest.TestCase):
    def test_pacing_ignores_future_epoch_and_obeys_speed(self):
        schedule = ReplaySchedule(np.array([1.e12, 1.e12 + 1., 1.e12 + 3.]), speed=2.)
        self.assertEqual(schedule.poll(40.), 0)
        self.assertIsNone(schedule.poll(40.49))
        self.assertEqual(schedule.poll(40.5), 1)
        self.assertIsNone(schedule.poll(41.49))
        self.assertEqual(schedule.poll(41.5), 2)
        self.assertTrue(schedule.finished)
        self.assertIsNone(schedule.poll(1.e20))

    def test_loop_has_a_final_frame_interval(self):
        schedule = ReplaySchedule(np.array([8., 8.1, 8.2]), loop=True)
        self.assertEqual(schedule.poll(0.), 0)
        self.assertEqual(schedule.poll(.1), 1)
        self.assertEqual(schedule.poll(.2), 2)
        self.assertIsNone(schedule.poll(.29))
        self.assertEqual(schedule.poll(.3), 0)

    def test_single_frame_loop_and_duplicate_timestamps(self):
        schedule = ReplaySchedule(np.array([4.]), loop=True)
        self.assertEqual(schedule.poll(1.), 0)
        self.assertIsNone(schedule.poll(1.01))
        self.assertEqual(schedule.poll(1.04), 0)
        schedule = ReplaySchedule(np.array([4., 4., 4.]))
        self.assertEqual([schedule.poll(1.) for _ in range(3)], [0, 1, 2])

    def test_invalid_timing_inputs(self):
        for speed in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                ReplaySchedule(np.array([1., 2.]), speed=speed)


class FollowerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "live.jsonl"

    def test_missing_file_and_partial_lines(self):
        follower = JsonlFollower(str(self.path))
        self.assertEqual(follower.poll(), [])
        self.path.write_bytes(b'{"a":1}\n{"b":')
        self.assertEqual(follower.poll(), [b'{"a":1}'])
        self.assertEqual(follower.poll(), [])
        with self.path.open("ab") as stream:
            stream.write(b'2}\n')
        self.assertEqual(follower.poll(), [b'{"b":2}'])

    def test_file_replacement_and_truncation(self):
        self.path.write_bytes(b'{"a":1111111111}\n')
        follower = JsonlFollower(str(self.path))
        follower.poll()
        self.path.write_bytes(b'{}\n')
        self.assertEqual(follower.poll(), [b'{}'])
        replacement = self.path.with_suffix(".new")
        replacement.write_bytes(b'{"new":true}\n')
        replacement.replace(self.path)
        self.assertEqual(follower.poll(), [b'{"new":true}'])

    def test_unbounded_partial_line_is_rejected(self):
        self.path.write_bytes(b'12345')
        follower = JsonlFollower(str(self.path), max_line_bytes=4)
        with self.assertRaises(ValueError):
            follower.poll()


if __name__ == "__main__":
    unittest.main()
