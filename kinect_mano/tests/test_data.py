"""Deterministic recording, registered-geometry, and pyk4a adapter checks."""

from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from kinect_mano.capture import AzureKinectCamera, DualKinectCapture, record_dual_cameras
from kinect_mano.data import (CameraCalibration, CameraFrame, JointEstimate, MissingCalibrationError,
                              load_recording, match_frames_by_timestamp, save_recording)
from kinect_mano.geometry import (frame_point_cloud, project_points, transform_points,
                                  unproject_depth, unproject_pixels, valid_depth_mask)


def calibration(camera_id="A", width=4, height=3):
    transform = np.eye(4)
    transform[:3, 3] = [1, -2, 0.1]
    return CameraCalibration(camera_id, [[2, 0, 1], [0, 4, 1], [0, 0, 1]], [], transform, width, height)


def frame(timestamp=1.0, camera_id="A", **kwargs):
    return CameraFrame(camera_id, timestamp, kwargs.pop("color", None),
                       kwargs.pop("depth_m", np.ones((3, 4), np.float32)), **kwargs)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "camera.npz"

    def test_roundtrip_optional_arrays_and_no_pickle(self):
        color = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        depth = np.ones((3, 4), np.float32)
        depth[0, 0], depth[0, 1] = 0, np.nan
        a = frame(1, color=color, depth_m=depth, mask=depth > 0,
                  metadata={"depth_timestamp_usec": 1000000, "color_timestamp_usec": 999840})
        b = frame(1.1)
        save_recording(self.path, iter([a, b]), calibration(), metadata={"clock": "device"})
        recording = load_recording(self.path)
        self.assertEqual(recording.metadata, {"clock": "device"})
        self.assertEqual(recording.frames[0].metadata, a.metadata)
        np.testing.assert_array_equal(recording.frames[0].color, color)
        np.testing.assert_array_equal(recording.frames[0].depth_m, depth)
        np.testing.assert_array_equal(recording.frames[0].mask, a.mask)
        self.assertIsNone(recording.frames[1].color)
        self.assertIsNone(recording.frames[1].mask)
        np.testing.assert_array_equal(recording.calibration.T_world_from_camera, calibration().T_world_from_camera)
        with np.load(self.path, allow_pickle=False) as archive:
            self.assertTrue(all(archive[key].dtype.kind != "O" for key in archive.files))

    def test_empty_recording_roundtrip_and_refuse_overwrite(self):
        save_recording(self.path, [], calibration())
        self.assertEqual(load_recording(self.path).frames, [])
        with self.assertRaises(FileExistsError):
            save_recording(self.path, [frame()], calibration())

    def test_missing_calibration_is_explicit(self):
        np.savez(self.path, depth_m=np.ones((1, 3, 4)))
        with self.assertRaises(MissingCalibrationError):
            load_recording(self.path)
        with self.assertRaises(MissingCalibrationError):
            CameraCalibration("A", np.eye(3), [], None, 4, 3)

    def test_calibration_only_requires_explicit_override(self):
        marker = {"extrinsics_status": "uncalibrated_local_only"}
        local = calibration()
        local.T_world_from_camera = np.eye(4)
        a = frame(mask=np.ones((3, 4), bool), metadata=marker)
        save_recording(self.path, [a], local, metadata=marker)
        with self.assertRaisesRegex(MissingCalibrationError, "local-only"):
            load_recording(self.path)
        board = load_recording(self.path, allow_uncalibrated=True)
        with self.assertRaises(MissingCalibrationError):
            frame_point_cloud(board.frames[0], board.calibration)
        self.assertEqual(frame_point_cloud(board.frames[0], board.calibration, world=False).shape, (12, 3))
        rig = load_recording(self.path, calibration=calibration())
        self.assertEqual(rig.metadata["extrinsics_status"], "calibrated")
        self.assertEqual(rig.frames[0].metadata["extrinsics_status"], "calibrated")
        self.assertEqual(frame_point_cloud(rig.frames[0], rig.calibration).shape, (12, 3))

    def test_calibration_only_cannot_lose_status_on_save(self):
        a = frame(metadata={"extrinsics_status": "uncalibrated_local_only"})
        with self.assertRaises(MissingCalibrationError):
            save_recording(self.path, [a], calibration())

    def test_override_rejects_wrong_mode_lens_id_and_resolution(self):
        original = CameraCalibration("A", [[600, 0, 2], [0, 600, 1], [0, 0, 1]],
                                      [.1, -.01, 0, 0, 0], np.eye(4), 4, 3)
        save_recording(self.path, [frame()], original)
        for field in ("camera_id", "width", "intrinsics", "distortion"):
            wrong = CameraCalibration.from_dict(original.to_dict())
            if field == "camera_id":
                wrong.camera_id = "other"
            elif field == "width":
                wrong.width = 8
            elif field == "intrinsics":
                wrong.intrinsics[0, 0] *= 1.2
            else:
                wrong.distortion[0] += .1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "override"):
                load_recording(self.path, calibration=wrong)
        verified = CameraCalibration.from_dict(original.to_dict())
        verified.T_world_from_camera[:3, 3] = [.2, .01, -.03]
        loaded = load_recording(self.path, calibration=verified)
        np.testing.assert_allclose(loaded.calibration.T_world_from_camera, verified.T_world_from_camera)

    def test_wrong_ids_sizes_and_timestamps_are_rejected(self):
        for frames in ([frame(camera_id="B")], [frame(depth_m=np.ones((2, 2)))],
                       [frame(1), frame(1)], [frame(2), frame(1)]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                save_recording(self.path, frames, calibration())

    def test_malformed_recording_presence_is_rejected(self):
        save_recording(self.path, [frame()], calibration())
        with np.load(self.path, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays["color_valid"] = np.array([True])
        other = self.path.with_name("bad.npz")
        np.savez(other, **arrays)
        with self.assertRaisesRegex(ValueError, "array is missing"):
            load_recording(other)

    def test_timestamp_matching_with_offset_drops_and_no_reuse(self):
        a = [frame(t) for t in [1.0, 1.03, 1.06, 1.09]]
        b = [frame(t, "B") for t in [11.001, 11.061, 11.2]]
        pairs = match_frames_by_timestamp(a, b, tolerance_s=.003, offset_b_s=-10)
        self.assertEqual([(x.timestamp_s, y.timestamp_s) for x, y in pairs], [(1.0, 11.001), (1.06, 11.061)])
        self.assertEqual(len({id(y) for _, y in pairs}), len(pairs))
        self.assertEqual(match_frames_by_timestamp(a, b, tolerance_s=.003), [])
        with self.assertRaises(ValueError):
            match_frames_by_timestamp(a, b, tolerance_s=-1)
        with self.assertRaises(ValueError):
            match_frames_by_timestamp(a[::-1], b, tolerance_s=1)

    def test_joint_invalid_coordinates_never_have_weight(self):
        joints = JointEstimate(np.ones((21, 3)), np.ones(21), np.zeros(21, bool), False)
        self.assertTrue(np.isnan(joints.joints_m).all())
        self.assertFalse(joints.confidence.any())
        with self.assertRaises(ValueError):
            JointEstimate.invalid().__class__(np.ones((21, 3)), np.zeros(21), np.ones(21, bool), True)


class GeometryTests(unittest.TestCase):
    def test_analytic_pinhole_and_world_transform(self):
        pixels = np.array([[1, 1], [3, 2], [0, 0]])
        depths = np.array([1, 2, .5])
        expected = np.array([[0, 0, 1], [2, .5, 2], [-.25, -.125, .5]])
        camera = unproject_pixels(pixels, depths, calibration(), world=False)
        np.testing.assert_allclose(camera, expected)
        world = unproject_pixels(pixels, depths, calibration())
        np.testing.assert_allclose(world, expected + [1, -2, .1])
        np.testing.assert_allclose(project_points(world, calibration()), pixels)

    def test_mask_range_stride_and_invalid_pixels(self):
        depth = np.array([[0, 1, np.nan, 2], [1, .1, 3, 1], [1, 1, 1, 1]])
        mask = np.ones((3, 4), bool)
        mask[2, 0] = False
        points, uv = unproject_depth(depth, calibration(), mask, min_depth_m=.5,
                                      max_depth_m=2, stride=2, return_pixels=True)
        np.testing.assert_array_equal(uv, [[2, 2]])
        self.assertEqual(points.shape, (1, 3))
        invalid = unproject_pixels([[-1, 0], [4, 0], [0, np.nan], [0, 0]], [1, 1, 1, 0], calibration())
        self.assertTrue(np.isnan(invalid).all())
        self.assertFalse(valid_depth_mask(depth)[0, 0])

    def test_distorted_rgb_pixels_are_undistorted(self):
        camera = CameraCalibration("C", [[500, 0, 320], [0, 510, 240], [0, 0, 1]],
                                   [.18, -.06, .002, -.001, .01], np.eye(4), 640, 480)
        expected = np.array([[-.3, -.2, 1], [.1, .15, .7], [.4, .1, 1.5]])
        pixels = project_points(expected, camera)
        points = unproject_pixels(pixels, expected[:, 2], camera)
        np.testing.assert_allclose(points, expected, atol=1e-7)
        no_distortion = calibration("C", 640, 480)
        no_distortion.intrinsics = camera.intrinsics
        no_distortion.T_world_from_camera = np.eye(4)
        self.assertGreater(np.linalg.norm(unproject_pixels(pixels, expected[:, 2], no_distortion) - expected), .001)

    def test_cloud_requires_explicit_mask_and_camera_identity(self):
        with self.assertRaisesRegex(ValueError, "hand mask"):
            frame_point_cloud(frame(), calibration())
        points = frame_point_cloud(frame(), calibration(), require_mask=False)
        self.assertEqual(len(points), 12)
        with self.assertRaises(ValueError):
            frame_point_cloud(frame(camera_id="B"), calibration())
        with self.assertRaises(MissingCalibrationError):
            unproject_depth(np.ones((3, 4)), None)
        with self.assertRaises(ValueError):
            transform_points(np.zeros((3, 3)), np.diag([2, 1, 1, 1]))


def fake_pyk4a():
    """Small SDK-shaped test double; raw depth intentionally has another grid."""
    events, devices = [], []
    module = types.ModuleType("pyk4a")
    for name, values in {"ColorResolution": ["RES_720P"], "ImageFormat": ["COLOR_BGRA32"],
                         "DepthMode": ["NFOV_UNBINNED"], "FPS": ["FPS_30"],
                         "WiredSyncMode": ["MASTER", "SUBORDINATE", "STANDALONE"],
                         "CalibrationType": ["COLOR"]}.items():
        setattr(module, name, types.SimpleNamespace(**{value: value for value in values}))
    module.Config = lambda **kwargs: types.SimpleNamespace(**kwargs)

    class Factory:
        def get_camera_matrix(self, kind):
            if kind != "COLOR":
                raise AssertionError("must use color calibration")
            return np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]])

        def get_distortion_coefficients(self, kind):
            if kind != "COLOR":
                raise AssertionError("must use color distortion")
            return np.array([.1, -.01, 0, 0, 0])

    class Device:
        def __init__(self, config, device_id):
            self.config, self.device_id = config, device_id
            self.is_running = self.opened = False
            self.serial = "serial-" + str(device_id)
            self.sync_jack_status = (True, True)
            self.calibration, self.calibration_raw = Factory(), '{"factory":true}'
            self.sequence = 0
            devices.append(self)

        def open(self):
            self.opened = True
            events.append(("open", self.device_id))

        def start(self):
            self.is_running = True
            events.append(("start", self.device_id))

        def stop(self):
            self.is_running = self.opened = False
            events.append(("stop", self.device_id))

        def close(self):
            self.opened = False
            events.append(("close", self.device_id))

        def get_capture(self, timeout):
            self.sequence += 1
            bgra = np.empty((720, 1280, 4), np.uint8)
            bgra[:] = [10, 20, 30, 255]
            depth_timestamp = 1000000 + self.sequence * 33333 + self.config.depth_delay_off_color_usec
            return types.SimpleNamespace(color=bgra, depth=np.full((2, 2), 9999, np.uint16),
                transformed_depth=np.full((720, 1280), 750, np.uint16),
                depth_timestamp_usec=depth_timestamp, color_timestamp_usec=depth_timestamp - 160,
                depth_system_timestamp_nsec=9000000000, color_system_timestamp_nsec=8999840000)

    module.PyK4A = Device
    return module, events, devices


class CaptureTests(unittest.TestCase):
    def test_adapter_uses_registered_depth_color_calibration_and_device_time(self):
        sdk, events, devices = fake_pyk4a()
        with patch.dict("sys.modules", {"pyk4a": sdk}):
            with AzureKinectCamera("A", T_world_from_camera=np.eye(4)) as camera:
                observation = camera.read()
                np.testing.assert_array_equal(observation.color[0, 0], [30, 20, 10])
                np.testing.assert_allclose(observation.depth_m, .75)
                self.assertEqual(observation.depth_m.shape, (720, 1280))
                self.assertAlmostEqual(observation.timestamp_s, 1.033333)
                self.assertEqual(observation.metadata["color_timestamp_usec"], 1033173)
                self.assertEqual(observation.metadata["depth_system_timestamp_nsec"], 9000000000)
                self.assertEqual(camera.calibration.distortion[0], .1)
                self.assertTrue(devices[0].config.synchronized_images_only)
                self.assertEqual(devices[0].exposure, 8000)
        self.assertEqual(events[-1], ("stop", 0))

    def test_subordinate_starts_first_and_wired_settings_are_passed(self):
        sdk, events, devices = fake_pyk4a()
        a = AzureKinectCamera("A", 0, T_world_from_camera=np.eye(4), sync_mode="master")
        b = AzureKinectCamera("B", 1, T_world_from_camera=np.eye(4), sync_mode="subordinate",
                              depth_delay_off_color_usec=160, subordinate_delay_off_master_usec=20)
        with patch.dict("sys.modules", {"pyk4a": sdk}), DualKinectCapture(a, b) as capture:
            pair = capture.read_pair(tolerance_s=.001)
            self.assertAlmostEqual(pair[0].metadata["pair_delta_s"], .00016)
            self.assertEqual([event for event in events if event[0] == "start"], [("start", 1), ("start", 0)])
            self.assertEqual(devices[0].config.wired_sync_mode, "SUBORDINATE")
            self.assertEqual(devices[0].config.subordinate_delay_off_master_usec, 20)
        self.assertFalse(any(device.opened for device in devices))

    def test_two_standalone_cameras_can_be_recorded(self):
        sdk, events, _ = fake_pyk4a()
        a = AzureKinectCamera("A", 0, T_world_from_camera=np.eye(4))
        b = AzureKinectCamera("B", 1, T_world_from_camera=np.eye(4))
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"pyk4a": sdk}):
            paths = record_dual_cameras(a, b, directory, frame_count=1, tolerance_s=.005)
            self.assertEqual(len(paths), 2)
            for path in paths:
                recorded = load_recording(path)
                self.assertEqual(recorded.metadata["wired_sync_mode"], "standalone")
                self.assertEqual(recorded.metadata["timestamp_domain"], "device")
                self.assertEqual(len(recorded.frames), 1)

    def test_shutdown_closes_if_stop_leaves_handle_open(self):
        events = []
        device = types.SimpleNamespace(is_running=True, opened=True)

        def stop():
            events.append("stop")
            device.is_running = False

        def close():
            events.append("close")
            device.opened = False

        device.stop, device.close = stop, close
        camera = AzureKinectCamera("A", T_world_from_camera=np.eye(4))
        camera._device = device
        camera.stop()
        camera.stop()
        self.assertEqual(events, ["stop", "close"])
        self.assertIsNone(camera._device)

    def test_pairing_discards_older_capture_and_reports_actual_delta(self):
        reads_a, reads_b = iter([frame(1), frame(1.033)]), iter([frame(1.034, "B")])
        a = types.SimpleNamespace(sync_mode="master", device_id=0, camera_id="A", fps=30,
            exposure_usec=8000, calibration_only=False, read=lambda _: next(reads_a))
        b = types.SimpleNamespace(sync_mode="subordinate", device_id=1, camera_id="B", fps=30,
            exposure_usec=8000, calibration_only=False, read=lambda _: next(reads_b))
        first, second = DualKinectCapture(a, b).read_pair(tolerance_s=.002)
        self.assertAlmostEqual(first.timestamp_s, 1.033)
        self.assertEqual(second.metadata["pair_skipped_frames"], 1)
        self.assertAlmostEqual(second.metadata["pair_delta_s"], .001)

    def test_calibration_only_recording_can_bootstrap_board_workflow(self):
        sdk, _, _ = fake_pyk4a()
        a = AzureKinectCamera("A", 0, calibration_only=True, sync_mode="master")
        b = AzureKinectCamera("B", 1, calibration_only=True, sync_mode="subordinate", depth_delay_off_color_usec=160)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"pyk4a": sdk}):
            paths = record_dual_cameras(a, b, directory, frame_count=1, tolerance_s=.001)
            for path in paths:
                with self.assertRaises(MissingCalibrationError):
                    load_recording(path)
                board = load_recording(path, allow_uncalibrated=True)
                self.assertEqual(board.metadata["extrinsics_status"], "uncalibrated_local_only")
                np.testing.assert_array_equal(board.calibration.T_world_from_camera, np.eye(4))
                self.assertEqual(board.metadata["depth_registration"], "color")

    def test_missing_extrinsics_serial_failure_and_no_sync_match(self):
        with self.assertRaises(MissingCalibrationError):
            AzureKinectCamera("A")
        sdk, _, devices = fake_pyk4a()
        with patch.dict("sys.modules", {"pyk4a": sdk}):
            camera = AzureKinectCamera("A", T_world_from_camera=np.eye(4), expected_serial="wrong")
            with self.assertRaisesRegex(ValueError, "serial"):
                camera.start()
            self.assertFalse(devices[0].opened)
        a = types.SimpleNamespace(sync_mode="master", device_id=0, camera_id="A", fps=30,
                                  exposure_usec=8000, calibration_only=False, read=lambda _: frame(1))
        b = types.SimpleNamespace(sync_mode="subordinate", device_id=1, camera_id="B", fps=30,
                                  exposure_usec=8000, calibration_only=False, read=lambda _: frame(2, "B"))
        with self.assertRaises(TimeoutError):
            DualKinectCapture(a, b).read_pair(tolerance_s=.001, max_skips=2)


if __name__ == "__main__":
    unittest.main()
