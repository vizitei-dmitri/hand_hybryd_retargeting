"""Calibrate deterministic projected boards; no physical accuracy claim or MANO assets."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from kinect_mano.calibration import (ChessboardSpec, calibrate_chessboard, calibrate_stereo_from_points,
                                     detect_chessboard, load_calibration, load_calibrations,
                                     save_calibration, save_calibrations)
from kinect_mano.data import CameraCalibration, CameraFrame, MissingCalibrationError, validate_transform


def projected_boards():
    board = ChessboardSpec(9, 6, .025)
    intrinsics_a = np.array([[750., 0, 640], [0, 755, 360], [0, 0, 1]])
    intrinsics_b = np.array([[765., 0, 636], [0, 760, 364], [0, 0, 1]])
    distortion_a = np.array([.05, -.01, .001, -.002, 0])
    distortion_b = np.array([.03, -.005, -.001, .001, 0])
    rotation = cv2.Rodrigues(np.array([.02, -.1, .04]))[0]
    translation = np.array([-.2, .015, .025])
    relative = np.eye(4)
    relative[:3, :3], relative[:3, 3] = rotation, translation
    anchor = np.eye(4)
    anchor[:3, :3] = cv2.Rodrigues(np.array([.2, -.05, .03]))[0]
    anchor[:3, 3] = [.4, -.1, .2]
    objects, pixels_a, pixels_b = [], [], []
    for i in range(8):
        obj = board.object_points()
        rvec = np.array([.05 + .12 * np.sin(i), -.12 + .06 * i, -.08 + .02 * i])
        tvec = np.array([-.12 + .015 * i, -.08 + .008 * i, .85 + .06 * i])
        r_b = rotation @ cv2.Rodrigues(rvec)[0]
        t_b = rotation @ tvec + translation
        objects.append(obj)
        pixels_a.append(cv2.projectPoints(obj, rvec, tvec, intrinsics_a, distortion_a)[0].reshape(-1, 2))
        pixels_b.append(cv2.projectPoints(obj, cv2.Rodrigues(r_b)[0], t_b,
                                        intrinsics_b, distortion_b)[0].reshape(-1, 2))
    kwargs = dict(camera_id_a="A", camera_id_b="B", intrinsics_a=intrinsics_a, distortion_a=distortion_a,
                  intrinsics_b=intrinsics_b, distortion_b=distortion_b, image_size=(1280, 720),
                  T_world_from_camera_a=anchor)
    return objects, pixels_a, pixels_b, kwargs, relative


class CalibrationTests(unittest.TestCase):
    def test_stereo_recovers_metric_baseline_and_correct_inverse(self):
        objects, pixels_a, pixels_b, kwargs, relative = projected_boards()
        result = calibrate_stereo_from_points(objects, pixels_a, pixels_b, **kwargs)
        self.assertEqual(result.num_pairs, 8)
        self.assertLess(result.rms_px, .001)
        np.testing.assert_allclose(result.T_camera_b_from_camera_a, relative, atol=1e-6)
        expected = kwargs["T_world_from_camera_a"] @ np.linalg.inv(relative)
        np.testing.assert_allclose(result.calibration_b.T_world_from_camera, expected, atol=1e-6)
        np.testing.assert_allclose(result.calibration_a.T_world_from_camera, kwargs["T_world_from_camera_a"])
        self.assertEqual(result.metadata()["translation_unit"], "m")
        np.testing.assert_array_equal(result.calibration_b.intrinsics, kwargs["intrinsics_b"])

    def test_bad_correspondences_fail_residual_threshold(self):
        objects, pixels_a, pixels_b, kwargs, _ = projected_boards()
        rng = np.random.default_rng(12)
        pixels_b = [points[rng.permutation(len(points))] for points in pixels_b]
        with self.assertRaisesRegex(ValueError, "RMS"):
            calibrate_stereo_from_points(objects, pixels_a, pixels_b, **kwargs, max_rms_px=.2)

    def test_insufficient_degenerate_and_missing_anchor_are_rejected(self):
        objects, pixels_a, pixels_b, kwargs, _ = projected_boards()
        with self.assertRaisesRegex(ValueError, "paired"):
            calibrate_stereo_from_points(objects[:2], pixels_a[:2], pixels_b[:2], **kwargs)
        objects[0][:, 1:] = 0
        with self.assertRaisesRegex(ValueError, "plane"):
            calibrate_stereo_from_points(objects, pixels_a, pixels_b, **kwargs)
        kwargs["T_world_from_camera_a"] = None
        with self.assertRaises(MissingCalibrationError):
            calibrate_stereo_from_points(objects, pixels_a, pixels_b, **kwargs)

    def test_extrinsic_validation_rejects_scale_shear_reflection_and_last_row(self):
        matrices = [np.diag([2., 1, 1, 1]), np.diag([-1., 1, 1, 1]), np.zeros((4, 4)), np.ones((3, 3))]
        shear = np.eye(4)
        shear[0, 1] = .1
        matrices.append(shear)
        nan = np.eye(4)
        nan[0, 3] = np.nan
        matrices.append(nan)
        for matrix in matrices:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                validate_transform(matrix)

    def test_chessboard_object_points_have_measured_spacing(self):
        points = ChessboardSpec(7, 5, .03).object_points()
        self.assertEqual(points.shape, (35, 3))
        np.testing.assert_allclose(points[1] - points[0], [.03, 0, 0], atol=1e-8)
        np.testing.assert_allclose(points[7] - points[0], [0, .03, 0], atol=1e-8)
        with self.assertRaises(ValueError):
            ChessboardSpec(7, 5, 0)

    def test_detect_rendered_rgb_chessboard_and_missing_board(self):
        board = ChessboardSpec(7, 5, .03)
        image = np.full((400, 480, 3), 255, np.uint8)
        for row in range(6):
            for col in range(8):
                if (row + col) % 2 == 0:
                    image[80 + row * 40:80 + (row + 1) * 40,
                          80 + col * 40:80 + (col + 1) * 40] = 0
        corners = detect_chessboard(image, board)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (35, 2))
        # Detector may choose either physical orientation on a plain checkerboard.
        self.assertAlmostEqual(float(np.linalg.norm(corners[1] - corners[0])), 40, delta=.2)
        self.assertIsNone(detect_chessboard(np.full_like(image, 255), board))

    def test_chessboard_pairing_does_not_pair_by_frame_index(self):
        image = np.zeros((20, 20, 3), np.uint8)
        a = [CameraFrame("A", 1, image, np.ones((20, 20)))]
        b = [CameraFrame("B", 2, image, np.ones((20, 20)))]
        with self.assertRaisesRegex(ValueError, "synchronized"):
            calibrate_chessboard(a, b, ChessboardSpec(7, 5, .03), intrinsics_a=np.eye(3),
                distortion_a=[], intrinsics_b=np.eye(3), distortion_b=[],
                T_world_from_camera_a=np.eye(4), tolerance_s=.005)

    def test_single_and_rig_json_roundtrip_and_missing_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            camera = CameraCalibration("A", np.eye(3), [], np.eye(4), 4, 3)
            save_calibration(path, camera)
            loaded = load_calibration(path)
            np.testing.assert_array_equal(loaded.T_world_from_camera, np.eye(4))
            save_calibrations(path, [camera], metadata={"method": "synthetic test"})
            self.assertEqual(set(load_calibrations(path)), {"A"})
            with self.assertRaises(ValueError):
                save_calibrations(path, [camera, camera])
            path.unlink()
            with self.assertRaises(MissingCalibrationError):
                load_calibration(path)
            with self.assertRaises(MissingCalibrationError):
                CameraCalibration.from_dict({"intrinsics": np.eye(3)})


if __name__ == "__main__":
    unittest.main()
