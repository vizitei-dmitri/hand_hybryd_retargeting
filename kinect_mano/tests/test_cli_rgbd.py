"""Hardware-free CLI integration using independently constructed numeric fixtures.

The checkerboards are perspective-rendered planar textures; the depth images are
analytic planes and the saved 21 points are arbitrary metric test coordinates.
These fixtures contain no MANO assets, human hand measurements, detector output,
or claims of anatomical/calibration accuracy. Their supplied masks test transport
and exclusion only; real grasp masks remain the caller's segmentation task.
"""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from kinect_mano.calibration import load_calibrations, save_calibrations
from kinect_mano.cli import build_parser, main
from kinect_mano.data import (CameraCalibration, CameraFrame, JointEstimate,
                              MissingCalibrationError, load_recording, save_recording)
from kinect_mano.initializers import save_joint_estimates
from kinect_mano.io import load_observations


_LOCAL_ONLY = "uncalibrated_local_only"
_FIXTURE = "independent_synthetic_cli_fixture_not_MANO"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _board_recordings(directory, *, poses_count=4):
    """Render known board poses into two calibrated pinhole RGB pixel grids.

    The rendering uses outer board vertices and a texture homography, never the
    corner detector or the calibration solver being tested. Supersampling reduces
    pixel aliasing. A common 180-degree detector ordering on these plain boards
    is harmless to the relative-camera solve; poses avoid orientation ambiguity.
    """
    columns, rows, square = 7, 5, .03
    width, height = 800, 600
    intrinsics = np.array([[900., 0, 400], [0, 900., 300], [0, 0, 1]])
    distortion = np.zeros(5)
    relative = np.eye(4)
    relative[:3, :3] = cv2.Rodrigues(np.array([.01, -.04, .005]))[0]
    relative[:3, 3] = [-.10, .005, .01]
    texture = np.zeros((600, 800, 3), np.uint8)
    for row in range(rows + 1):
        for column in range(columns + 1):
            texture[row * 100:(row + 1) * 100, column * 100:(column + 1) * 100] = (
                255 if (row + column) % 2 else 0)
    outer = np.array([[-square, -square, 0], [columns * square, -square, 0],
                      [columns * square, rows * square, 0], [-square, rows * square, 0]], np.float32)
    source = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], np.float32)
    poses = [([-.20, .15, -.02], [-.09, -.06, .8]),
             ([.18, -.18, .04], [-.05, -.08, .95]),
             ([-.10, -.10, -.06], [-.08, -.04, .75]),
             ([.25, .20, .08], [-.1, -.08, .9])]
    streams = [[], []]
    for index, (rvec, tvec) in enumerate(poses[:poses_count]):
        rotation_a = cv2.Rodrigues(np.asarray(rvec))[0]
        translation_a = np.asarray(tvec)
        rotation_b = relative[:3, :3] @ rotation_a
        translation_b = relative[:3, :3] @ translation_a + relative[:3, 3]
        for camera, (rotation, translation) in enumerate(((rotation_a, translation_a),
                                                          (rotation_b, translation_b))):
            destination = cv2.projectPoints(outer, cv2.Rodrigues(rotation)[0], translation,
                                            intrinsics, distortion)[0].reshape(-1, 2)
            homography = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
            image = cv2.warpPerspective(texture, np.diag([2, 2, 1]) @ homography,
                                       (2 * width, 2 * height), flags=cv2.INTER_LINEAR,
                                       borderValue=(205, 205, 205))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            camera_id = ("camera_A", "camera_B")[camera]
            timestamp = 1 + index * .1 + (10.001 if camera else 0)
            streams[camera].append(CameraFrame(camera_id, timestamp, image,
                np.zeros((height, width), np.float32),
                metadata={"extrinsics_status": _LOCAL_ONLY, "fixture": _FIXTURE}))
    paths = []
    for camera_id, frames in zip(("camera_A", "camera_B"), streams):
        calibration = CameraCalibration(camera_id, intrinsics, distortion, np.eye(4), width, height)
        path = directory / (camera_id + ".npz")
        save_recording(path, frames, calibration, metadata={"extrinsics_status": _LOCAL_ONLY,
            "fixture": _FIXTURE, "serial": "synthetic-" + camera_id})
        paths.append(path)
    return SimpleNamespace(a=paths[0], b=paths[1], columns=columns, rows=rows,
                           square=square, relative=relative, pairs=poses_count)


def _rgbd_recordings(directory, *, masks=True):
    """Two pixel-shifted analytic plane patches represent the same world points.

    At Z=.8 m and fx=40, a two-pixel shift corresponds to the independently
    chosen 4-cm camera baseline. Distinct clocks and dropped frames exercise
    timestamp matching. Local-only inputs require the CLI's verified rig override.
    """
    intrinsics = np.array([[40., 0, 3.5], [0, 40., 2.5], [0, 0, 1]])
    rig = []
    paths = []
    for index, camera_id in enumerate(("camera_A", "camera_B")):
        transform = np.eye(4)
        transform[0, 3] = index * .04
        rig.append(CameraCalibration(camera_id, intrinsics, [], transform, 8, 6))
        local = CameraCalibration(camera_id, intrinsics, [], np.eye(4), 8, 6)
        mask = np.zeros((6, 8), bool)
        mask[1:5, 2 - index * 2:6 - index * 2] = True
        # With masks, valid background at a different Z proves mask exclusion.
        # Without masks, invalid background leaves a static 16-point plane patch.
        depth = np.full((6, 8), 1.2 if masks else 0, np.float32)
        depth[mask] = .8
        timestamps = [1., 1.04, 1.08] if index == 0 else [11.001, 11.081, 11.20]
        frames = [CameraFrame(camera_id, timestamp, np.full((6, 8, 3), 64 + index, np.uint8),
            depth.copy(), mask.copy() if masks else None,
            {"extrinsics_status": _LOCAL_ONLY, "fixture": _FIXTURE}) for timestamp in timestamps]
        path = directory / (camera_id + ".npz")
        save_recording(path, frames, local, metadata={"extrinsics_status": _LOCAL_ONLY, "fixture": _FIXTURE})
        paths.append(path)
    rig_path = directory / "rig.json"
    save_calibrations(rig_path, rig, metadata={"fixture": _FIXTURE, "method": "analytic test transforms"})
    estimates = []
    for index in range(2):
        joints = np.column_stack((np.linspace(-.015, .015, 21) + .001 * index,
                                  np.linspace(-.01, .01, 21), np.full(21, .8)))
        validity = np.ones(21, bool)
        validity[20] = bool(index)
        estimates.append(JointEstimate(joints, np.linspace(.5, .9, 21), validity, True))
    joint_path = directory / "joints.npz"
    save_joint_estimates(joint_path, [1., 1.08], estimates)
    # Independent analytic expected world coordinates, without using geometry.py.
    v, u = np.mgrid[1:5, 2:6]
    expected_cloud = np.column_stack(((u.ravel() - 3.5) * .8 / 40,
                                      (v.ravel() - 2.5) * .8 / 40, np.full(16, .8)))
    return SimpleNamespace(a=paths[0], b=paths[1], rig_path=rig_path, rig=rig,
                           joints=joint_path, estimates=estimates, expected_cloud=expected_cloud)


class CLIRGBDTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def invoke(self, *arguments, expected_code=0, missing_sdk=False):
        """Run the actual CLI parser/dispatch, protecting every offline hardware path."""
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        # Either prohibit hardware start entirely, or test the real start method
        # with pyk4a imports blocked before any SDK/device code can run.
        guard = (patch.dict("sys.modules", {"pyk4a": None}) if missing_sdk else
                 patch("kinect_mano.capture.AzureKinectCamera.start",
                       side_effect=AssertionError("CLI unexpectedly tried to open Kinect hardware")))
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with guard:
                try:
                    main([str(argument) for argument in arguments])
                except SystemExit as error:
                    code = error.code
        self.assertEqual(code, expected_code, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        if expected_code:
            self.assertEqual(stdout.getvalue(), "")
            return stderr.getvalue()
        self.assertEqual(stderr.getvalue(), "")
        return json.loads(stdout.getvalue())

    def paired_arguments(self, command, session, output):
        return [command, "--camera-a", session.a, "--camera-b", session.b,
                "--offset-b", -10, "--tolerance", .005, "--output", output]

    def test_capture_parser_exposes_standalone_and_wired_configuration(self):
        parser = build_parser()
        for sync in ("standalone", "wired"):
            with self.subTest(sync=sync):
                args = parser.parse_args(["capture", "--output", str(self.root / "capture"),
                    "--calibration-only", "--sync", sync, "--frames", "3",
                    "--device-a", "2", "--device-b", "4", "--offset-b", "-10"])
                self.assertEqual(args.sync, sync)
                self.assertTrue(args.calibration_only)
                self.assertEqual((args.frames, args.device_a, args.device_b, args.offset_b), (3, 2, 4, -10))
        error = self.invoke("capture", "--output", self.root / "invalid", "--sync", "invalid",
                            expected_code=2)
        self.assertIn("invalid choice", error)
        self.assertFalse((self.root / "invalid").exists())

    def test_capture_requires_calibration_before_hardware(self):
        output = self.root / "capture"
        error = self.invoke("capture", "--output", output, expected_code=2)
        self.assertIn("Provide --rig", error)
        self.assertFalse(output.exists())
        error = self.invoke("capture", "--output", output, "--rig", self.root / "missing-rig.json",
                            expected_code=2)
        self.assertIn("calibration file does not exist", error)
        self.assertFalse(output.exists())

    def test_capture_rejects_zero_frames_and_conflicting_calibration_modes(self):
        output = self.root / "capture"
        error = self.invoke("capture", "--output", output, "--calibration-only", "--frames", 0,
                            expected_code=2)
        self.assertIn("frames must be positive", error)
        session = _rgbd_recordings(self.root)
        error = self.invoke("capture", "--output", output, "--rig", session.rig_path,
                            "--calibration-only", expected_code=2)
        self.assertIn("Choose --calibration-only or --rig", error)
        self.assertFalse(output.exists())

    def test_capture_reports_missing_optional_sdk_without_opening_devices(self):
        # A missing import is deterministic even on a workstation with pyk4a.
        # No stub device or fake successful capture is substituted.
        error = self.invoke("capture", "--calibration-only", "--frames", 1,
            "--output", self.root / "capture", expected_code=2, missing_sdk=True)
        self.assertIn("requires optional pyk4a", error)
        self.assertFalse((self.root / "capture").exists())

    def test_calibrate_detects_rendered_boards_and_saves_verified_rig(self):
        session = _board_recordings(self.root)
        with self.assertRaises(MissingCalibrationError):
            load_recording(session.a)
        output = self.root / "solved-rig.json"
        arguments = self.paired_arguments("calibrate", session, output) + [
            "--columns", session.columns, "--rows", session.rows, "--square-size", session.square,
            "--min-pairs", 3, "--max-rms", 1.]
        result = self.invoke(*arguments)
        payload = json.loads(output.read_text())
        rig = load_calibrations(output)
        self.assertEqual(set(rig), {"camera_A", "camera_B"})
        self.assertEqual(result["num_pairs"], session.pairs)
        self.assertLess(result["rms_px"], 1.)
        self.assertEqual(result["translation_unit"], "m")
        self.assertEqual(result["world_frame"], "camera_A_color")
        self.assertEqual(payload["metadata"]["camera_a_sha256"], _sha256(session.a))
        self.assertEqual(payload["metadata"]["camera_b_sha256"], _sha256(session.b))
        self.assertEqual(payload["metadata"]["serials"]["camera_B"], "synthetic-camera_B")
        np.testing.assert_allclose(rig["camera_A"].T_world_from_camera, np.eye(4), atol=1e-10)
        # Rasterized corner locations carry subpixel error; these bounds test
        # metric scale and transform direction without claiming physical accuracy.
        np.testing.assert_allclose(rig["camera_B"].T_world_from_camera,
                                   np.linalg.inv(session.relative), atol=.002)
        loaded = load_recording(session.b, calibration=rig["camera_B"])
        self.assertEqual(loaded.frames[0].metadata["extrinsics_status"], "calibrated")
        before = output.read_bytes()
        error = self.invoke(*arguments, expected_code=2)
        self.assertIn("Refusing to overwrite rig", error)
        self.assertEqual(output.read_bytes(), before)

    def test_calibrate_insufficient_real_detections_does_not_write_rig(self):
        session = _board_recordings(self.root, poses_count=2)
        output = self.root / "insufficient-rig.json"
        error = self.invoke(*self.paired_arguments("calibrate", session, output),
            "--columns", session.columns, "--rows", session.rows, "--square-size", session.square,
            "--min-pairs", 3, expected_code=2)
        self.assertIn("got 2", error)
        self.assertFalse(output.exists())

    def test_prepare_saved_masks_and_joints_exports_two_world_space_views(self):
        session = _rgbd_recordings(self.root)
        output = self.root / "prepared" / "observations.npz"
        result = self.invoke(*self.paired_arguments("prepare", session, output),
            "--rig", session.rig_path, "--joints", session.joints, "--points-per-camera", 64)
        self.assertEqual((result["matched_pairs"], result["unmatched_a"], result["unmatched_b"]), (2, 1, 1))
        observations, truth, metadata = load_observations(output)
        self.assertEqual(truth, {})
        self.assertEqual(metadata["camera_ids"], ["camera_A", "camera_B"])
        self.assertEqual(metadata["rig_sha256"], _sha256(session.rig_path))
        self.assertFalse(metadata["polygon_roi"])
        np.testing.assert_allclose([obs.timestamp_s for obs in observations], [1, 1.08])
        for index, observation in enumerate(observations):
            expected = session.estimates[index]
            # Fitting observations can use finite placeholders for missing joints;
            # they must retain zero weight and preserve all measured coordinates.
            np.testing.assert_allclose(observation.joints_m[expected.valid_mask], expected.joints_m[expected.valid_mask])
            np.testing.assert_allclose(observation.confidence, expected.confidence)
            self.assertFalse(observation.confidence[~expected.valid_mask].any())
            self.assertEqual(len(observation.views), 2)
            for view, camera in zip(observation.views, session.rig):
                self.assertEqual(view.camera_id, camera.camera_id)
                np.testing.assert_allclose(view.camera_center_m, camera.T_world_from_camera[:3, 3])
                np.testing.assert_allclose(view.points_m, session.expected_cloud, atol=1e-7)
                self.assertTrue(np.isfinite(view.points_m).all())
        synchronization = json.loads(Path(str(output) + ".sync.json").read_text())
        np.testing.assert_allclose([pair["timestamp_b_s"] for pair in synchronization], [11.001, 11.081])
        np.testing.assert_allclose([pair["delta_s"] for pair in synchronization], [.001, .001], atol=1e-12)
        with np.load(output, allow_pickle=False) as archive:
            self.assertTrue(all(archive[key].dtype.kind != "O" for key in archive.files))
            np.testing.assert_array_equal(archive["cloud_0_count"], [16, 16])
            np.testing.assert_array_equal(archive["cloud_1_count"], [16, 16])
        # Applying a rig must not rewrite or relabel original local-only inputs.
        self.assertEqual(load_recording(session.a, allow_uncalibrated=True).metadata["extrinsics_status"], _LOCAL_ONLY)

    def test_prepare_requires_supplied_masks_for_precomputed_joints(self):
        session = _rgbd_recordings(self.root, masks=False)
        output = self.root / "observations.npz"
        error = self.invoke(*self.paired_arguments("prepare", session, output),
            "--rig", session.rig_path, "--joints", session.joints, expected_code=2)
        self.assertIn("requires a hand mask per camera", error)
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output) + ".sync.json").exists())

    def test_prepare_rejects_wrong_factory_intrinsics_before_export(self):
        session = _rgbd_recordings(self.root)
        session.rig[1].intrinsics[0, 0] *= 1.25
        save_calibrations(session.rig_path, session.rig, metadata={"fixture": _FIXTURE})
        output = self.root / "observations.npz"
        error = self.invoke(*self.paired_arguments("prepare", session, output),
            "--rig", session.rig_path, "--joints", session.joints, expected_code=2)
        self.assertIn("RGB intrinsics differ", error)
        self.assertFalse(output.exists())

    def test_check_calibration_exports_consistency_json_and_headless_png(self):
        session = _rgbd_recordings(self.root, masks=False)
        output = self.root / "alignment"
        result = self.invoke(*self.paired_arguments("check-calibration", session, output),
                             "--rig", session.rig_path, "--points-per-camera", 64)
        report = json.loads((output / "alignment.json").read_text())
        self.assertEqual(report, result)
        self.assertEqual(report["points"], [16, 16])
        self.assertLess(max(report["nearest_neighbor_median_mm"]), .001)
        self.assertLess(max(report["nearest_neighbor_p90_mm"]), .001)
        self.assertIn("not independent calibration accuracy", report["note"])
        image_path = output / "alignment.png"
        self.assertEqual(image_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image)
        self.assertGreater(image.size, 10000)
        self.assertGreater(float(image.std()), 1)

    def test_check_calibration_reports_displaced_clouds(self):
        session = _rgbd_recordings(self.root, masks=False)
        session.rig[1].T_world_from_camera[0, 3] += .1
        save_calibrations(session.rig_path, session.rig, metadata={"fixture": _FIXTURE})
        report = self.invoke(*self.paired_arguments("check-calibration", session, self.root / "bad-alignment"),
                             "--rig", session.rig_path, "--points-per-camera", 64)
        self.assertGreater(min(report["nearest_neighbor_median_mm"]), 50.)

    def test_check_calibration_rejects_unmatched_clock_domains(self):
        session = _rgbd_recordings(self.root, masks=False)
        output = self.root / "no-pairs"
        # B timestamps are 10 seconds ahead; intentionally omit the measured offset.
        error = self.invoke("check-calibration", "--camera-a", session.a, "--camera-b", session.b,
                            "--rig", session.rig_path, "--output", output, expected_code=2)
        self.assertIn("no suitable synchronized pair", error)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
