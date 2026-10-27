"""Synthetic pinhole depth and precomputed-joint tests; no model downloads."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from kinect_mano.data import CameraCalibration, CameraFrame, JointEstimate, MissingCalibrationError
from kinect_mano.geometry import polygon_hand_mask
from kinect_mano.initializers import (MediaPipeInitializer, PrecomputedJointsInitializer,
    fuse_joint_estimates, joints_from_landmarks_depth, load_joint_estimates, save_joint_estimates)


def inputs():
    transform = np.eye(4)
    transform[:3, 3] = [.1, -.2, .3]
    calibration = CameraCalibration("A", [[20, 0, 16], [0, 20, 12], [0, 0, 1]], [], transform, 32, 24)
    frame = CameraFrame("A", 1, None, np.full((24, 32), .6, np.float32))
    pixels = np.array([(7 + i % 7 * 3, 6 + i // 7 * 4) for i in range(21)], dtype=float)
    return frame, calibration, pixels


class InitializerTests(unittest.TestCase):
    def test_metric_lifting_uses_measured_z_and_world_extrinsics(self):
        frame, calibration, pixels = inputs()
        scores = np.linspace(.2, 1, 21)
        estimate = joints_from_landmarks_depth(frame, calibration, pixels, scores, patch_radius=0)
        expected_camera = np.column_stack(((pixels[:, 0] - 16) * .6 / 20,
                                          (pixels[:, 1] - 12) * .6 / 20, np.full(21, .6)))
        np.testing.assert_allclose(estimate.joints_m, expected_camera + [.1, -.2, .3], atol=1e-7)
        np.testing.assert_allclose(estimate.confidence, scores * .5)
        self.assertTrue(estimate.valid)
        self.assertTrue(estimate.valid_mask.all())

    def test_missing_outside_and_masked_landmarks_stay_invalid(self):
        frame, calibration, pixels = inputs()
        frame.depth_m[int(pixels[0, 1]), int(pixels[0, 0])] = 0
        pixels[1] = [-1, 5]
        pixels[2] = [np.nan, 4]
        frame.mask = np.ones_like(frame.depth_m, bool)
        frame.mask[int(pixels[3, 1]), int(pixels[3, 0])] = False
        estimate = joints_from_landmarks_depth(frame, calibration, pixels, patch_radius=0)
        self.assertFalse(estimate.valid)  # wrist is required even if other points exist
        self.assertFalse(estimate.valid_mask[:4].any())
        self.assertTrue(np.isnan(estimate.joints_m[:4]).all())
        self.assertFalse(estimate.confidence[:4].any())
        self.assertEqual(estimate.valid_mask.sum(), 17)

    def test_object_mask_excludes_samples_and_depth_discontinuity_rejected(self):
        frame, calibration, pixels = inputs()
        object_mask = np.zeros_like(frame.depth_m, bool)
        object_mask[int(pixels[0, 1]), int(pixels[0, 0])] = True
        estimate = joints_from_landmarks_depth(frame, calibration, pixels, patch_radius=0, object_mask=object_mask)
        self.assertFalse(estimate.valid_mask[0])
        x, y = pixels[1].astype(int)
        frame.depth_m[y - 1:y + 2, x - 1:x + 2] = [[.6, .6, 1], [.6, 1, 1], [.6, 1, 1]]
        estimate = joints_from_landmarks_depth(frame, calibration, pixels, patch_radius=1)
        self.assertFalse(estimate.valid_mask[1])

    def test_calibration_only_cannot_initialize_world_joints(self):
        frame, calibration, pixels = inputs()
        frame.metadata["extrinsics_status"] = "uncalibrated_local_only"
        with self.assertRaises(MissingCalibrationError):
            joints_from_landmarks_depth(frame, calibration, pixels)
        with self.assertRaises(MissingCalibrationError):
            joints_from_landmarks_depth(frame, None, pixels)

    def test_fusion_respects_weights_and_rejects_disagreement(self):
        a = JointEstimate(np.full((21, 3), .1), np.full(21, .5), np.ones(21, bool), True)
        b = JointEstimate(np.full((21, 3), .12), np.ones(21), np.ones(21, bool), True)
        fused = fuse_joint_estimates([a, b])
        np.testing.assert_allclose(fused.joints_m, (.1 * .5 + .12) / 1.5)
        self.assertTrue(fused.valid)
        b.joints_m[2] += 1
        fused = fuse_joint_estimates([a, b])
        self.assertFalse(fused.valid_mask[2])
        self.assertEqual(fused.confidence[2], 0)
        self.assertTrue(np.isnan(fused.joints_m[2]).all())
        self.assertFalse(fuse_joint_estimates([]).valid)

    def test_polygon_roi_includes_landmarks_and_can_remove_known_object(self):
        points = np.array([[3, 3], [12, 3], [12, 12], [3, 12]])
        objects = np.zeros((16, 16), bool)
        objects[7:9, 7:9] = True
        mask = polygon_hand_mask(points, (16, 16), padding_px=0, object_mask=objects)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[3, 3])
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[7, 7])
        # Entirely offscreen predictions must not become a visible edge polygon.
        self.assertFalse(polygon_hand_mask(points + [100, 100], (16, 16), padding_px=0).any())
        self.assertFalse(polygon_hand_mask(points[:2], (16, 16)).any())

    def test_precomputed_world_joints_timestamp_lookup_and_copy(self):
        frame, calibration, pixels = inputs()
        estimate = joints_from_landmarks_depth(frame, calibration, pixels, patch_radius=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joints.npz"
            save_joint_estimates(path, [11, 11.1], [estimate, JointEstimate.invalid()])
            times, estimates = load_joint_estimates(path)
            np.testing.assert_array_equal(times, [11, 11.1])
            self.assertTrue(estimates[0].valid)
            self.assertFalse(estimates[1].valid)
            initializer = PrecomputedJointsInitializer(path, offset_s=-10, tolerance_s=.005)
            value = initializer.estimate_joints(frame, calibration)
            self.assertTrue(value.valid)
            value.joints_m[:] = 123
            np.testing.assert_allclose(initializer.estimate_joints(1).joints_m, estimate.joints_m)
            self.assertFalse(initializer.estimate_joints(1.05).valid)
            self.assertFalse(initializer.estimate_joints(1.1).valid)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["coordinate_frame"].item(), "world")
                self.assertEqual(archive["units"].item(), "m")
            with self.assertRaises(FileExistsError):
                save_joint_estimates(path, [1], [estimate])

    def test_precomputed_requires_units_frame_and_increasing_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joints.npz"
            np.savez(path, joints_m=np.ones((21, 3)))
            with self.assertRaisesRegex(ValueError, "missing"):
                load_joint_estimates(path)
            with self.assertRaises(ValueError):
                save_joint_estimates(path, [2, 1], [JointEstimate.invalid()] * 2)
            empty = Path(directory) / "empty.npz"
            save_joint_estimates(empty, [], [])
            self.assertFalse(PrecomputedJointsInitializer(empty).estimate_joints(1).valid)

    def test_mediapipe_missing_model_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-provided.task"
            with self.assertRaisesRegex(FileNotFoundError, "no model is downloaded"):
                MediaPipeInitializer(path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
