"""Local 2D detection + measured RGB-aligned depth, or saved metric landmarks.

No initializer downloads models. MediaPipe's predicted z/world landmarks are
deliberately unused: scale and world position come from registered Kinect depth
and explicit calibration. A sampled depth is a visible surface, not a measured
joint centre, so its confidence is reduced. Object occlusion remains ambiguous
unless the caller supplies trustworthy segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import CameraCalibration, CameraFrame, JointEstimate, MissingCalibrationError
from .geometry import polygon_hand_mask, unproject_pixels, valid_depth_mask


def joints_from_landmarks_depth(frame: CameraFrame, calibration: CameraCalibration,
                                landmarks_px: np.ndarray, confidence: np.ndarray | None = None, *,
                                patch_radius: int = 2, min_depth_m: float = 0.1,
                                max_depth_m: float = 3.0, max_depth_spread_m: float = 0.02,
                                surface_confidence: float = 0.5, min_valid_joints: int = 6,
                                require_wrist: bool = True,
                                object_mask: np.ndarray | None = None) -> JointEstimate:
    """Lift 21 RGB pixel landmarks with robust local *measured* axial depth.

    Samples respect frame.mask and optional object_mask. A patch crossing a depth
    discontinuity is rejected when its 90%-10% quantile spread exceeds the bound.
    Missing/out-of-image points remain NaN/zero-confidence. Inputs use the usual
    wrist, thumb, index, middle, ring, little 21-landmark order.
    """
    if not isinstance(calibration, CameraCalibration):
        raise MissingCalibrationError("joint lifting requires calibrated color pixels and world extrinsics")
    if frame.metadata.get("extrinsics_status") == "uncalibrated_local_only":
        raise MissingCalibrationError("calibration-only frame requires an explicit rig calibration override before joint lifting")
    if (frame.camera_id != calibration.camera_id
            or frame.depth_m.shape != (calibration.height, calibration.width)):
        raise ValueError("frame camera ID/resolution differs from color calibration")
    pixels = np.asarray(landmarks_px, dtype=np.float64)
    scores = np.ones(21) if confidence is None else np.asarray(confidence, dtype=np.float64)
    if pixels.shape != (21, 2) or scores.shape != (21,):
        raise ValueError("landmarks_px/confidence must have shape [21,2]/[21]")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("2D confidence must be finite in [0,1]")
    if isinstance(patch_radius, bool) or int(patch_radius) != patch_radius or patch_radius < 0:
        raise ValueError("patch_radius must be a nonnegative integer")
    if not np.isfinite(max_depth_spread_m) or max_depth_spread_m <= 0:
        raise ValueError("max_depth_spread_m must be positive")
    if not np.isfinite(surface_confidence) or not 0 < surface_confidence <= 1:
        raise ValueError("surface_confidence must be in (0,1]")
    if int(min_valid_joints) != min_valid_joints or not 1 <= min_valid_joints <= 21:
        raise ValueError("min_valid_joints must be an integer in [1,21]")
    valid_depth = valid_depth_mask(frame.depth_m, frame.mask, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    if object_mask is not None:
        objects = np.asarray(object_mask)
        if objects.shape != frame.depth_m.shape or not np.isin(objects, [0, 1]).all():
            raise ValueError("object_mask must be binary and match the RGB/depth grid")
        valid_depth &= ~objects.astype(bool)
    depths, output_scores = np.full(21, np.nan), np.zeros(21)
    height, width = frame.depth_m.shape
    radius = int(patch_radius)
    for joint, (u, v) in enumerate(pixels):
        if (not np.isfinite([u, v]).all() or scores[joint] <= 0
                or not (0 <= u < width and 0 <= v < height)):
            continue
        # Clamp rounding at the final image pixel, while rejecting genuinely
        # off-image continuous coordinates above.
        x, y = min(int(np.floor(u + 0.5)), width - 1), min(int(np.floor(v + 0.5)), height - 1)
        y0, y1, x0, x1 = max(y - radius, 0), min(y + radius + 1, height), max(x - radius, 0), min(x + radius + 1, width)
        patch_valid = valid_depth[y0:y1, x0:x1]
        samples = frame.depth_m[y0:y1, x0:x1][patch_valid]
        if not len(samples):
            continue
        spread = float(np.quantile(samples, 0.9) - np.quantile(samples, 0.1))
        if spread > max_depth_spread_m:
            continue
        depths[joint] = np.median(samples)
        support = float(patch_valid.mean())
        output_scores[joint] = scores[joint] * surface_confidence * support * np.exp(-spread / max_depth_spread_m)
    joints = unproject_pixels(pixels, depths, calibration)
    valid = np.isfinite(joints).all(axis=1) & (output_scores > 0)
    usable = valid.sum() >= min_valid_joints and (valid[0] or not require_wrist)
    return JointEstimate(joints, output_scores, valid, bool(usable))


def fuse_joint_estimates(estimates: Sequence[JointEstimate], *, min_valid_joints: int = 6,
                         max_disagreement_m: float = 0.06, require_wrist: bool = True) -> JointEstimate:
    """Confidence-average common-world estimates; conflicting observations are invalid.

    Incomplete per-view estimates may contribute observed joints even when their
    own overall ``valid`` is false. A pair of inconsistent metric joints is
    rejected instead of silently averaging incompatible hands or calibrations.
    """
    if int(min_valid_joints) != min_valid_joints or not 1 <= min_valid_joints <= 21:
        raise ValueError("min_valid_joints must be in [1,21]")
    if not np.isfinite(max_disagreement_m) or max_disagreement_m <= 0:
        raise ValueError("max_disagreement_m must be positive")
    joints, scores, validity = np.full((21, 3), np.nan), np.zeros(21), np.zeros(21, bool)
    for index in range(21):
        sources = [estimate for estimate in estimates if estimate.valid_mask[index]]
        if not sources:
            continue
        xyz = np.array([estimate.joints_m[index] for estimate in sources])
        weights = np.array([estimate.confidence[index] for estimate in sources])
        if np.max(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)) > max_disagreement_m:
            continue
        joints[index] = np.average(xyz, axis=0, weights=weights)
        # Conservative fusion: extra views improve geometry, not claimed detector
        # certainty. This remains a surface-derived initialization confidence.
        scores[index] = np.average(weights, weights=weights)
        validity[index] = True
    usable = validity.sum() >= min_valid_joints and (validity[0] or not require_wrist)
    return JointEstimate(joints, scores, validity, bool(usable))


class MediaPipeInitializer:
    """Optional MediaPipe Tasks IMAGE detector using an explicitly local .task model.

    ``estimate_joints`` stores ``last_landmarks_px`` and ``last_mask`` for the CLI.
    ``last_mask`` is only a polygon ROI; object/hand segmentation remains the
    caller's responsibility. Existing frame masks constrain depth and the ROI.
    Handedness is the model's label on the provided RGB; mirror conventions must
    be verified for the capture setup. No detector depth output is used.
    """

    def __init__(self, model_path: str | Path, *, handedness: str | None = None,
                 min_detection_confidence: float = 0.5, padding_px: int = 5,
                 **depth_options: Any):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"local MediaPipe Hand Landmarker model is required: {path}; no model is downloaded")
        if handedness is not None and handedness.lower() not in ("left", "right"):
            raise ValueError("handedness must be Left, Right or None")
        if not np.isfinite(min_detection_confidence) or not 0 < min_detection_confidence <= 1:
            raise ValueError("min_detection_confidence must be in (0,1]")
        if int(padding_px) != padding_px or padding_px < 0:
            raise ValueError("padding_px must be a nonnegative integer")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            raise ImportError("MediaPipe initialization requires the optional mediapipe package and a local .task model") from error
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path.resolve())),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence)
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.handedness = handedness.lower() if handedness else None
        self.padding_px, self.depth_options = int(padding_px), depth_options
        self.last_landmarks_px: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None

    def estimate_joints(self, frame: CameraFrame, calibration: CameraCalibration,
                        previous_state: Any = None) -> JointEstimate:
        """Detect 2D landmarks and lift real depth; previous_state is currently unused."""
        self.last_landmarks_px = self.last_mask = None
        if not isinstance(calibration, CameraCalibration):
            raise MissingCalibrationError("MediaPipe RGB-D initialization requires camera calibration")
        if (frame.camera_id != calibration.camera_id
                or frame.depth_m.shape != (calibration.height, calibration.width)):
            raise ValueError("frame camera ID/resolution differs from calibration")
        if frame.color is None:
            return JointEstimate.invalid()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame.color))
        result = self._landmarker.detect(image)
        candidates = []
        for index, categories in enumerate(result.handedness):
            category = categories[0] if categories else None
            if category is not None and (self.handedness is None or category.category_name.lower() == self.handedness):
                candidates.append((category.score, index))
        if not candidates:
            return JointEstimate.invalid()
        index = max(candidates)[1]
        height, width = frame.depth_m.shape
        pixels = np.array([(landmark.x * width, landmark.y * height) for landmark in result.hand_landmarks[index]])
        if pixels.shape != (21, 2):
            raise ValueError("MediaPipe returned an unexpected hand landmark count")
        self.last_landmarks_px = pixels
        self.last_mask = polygon_hand_mask(pixels, (height, width), padding_px=self.padding_px)
        if frame.mask is not None:
            self.last_mask &= frame.mask
        return joints_from_landmarks_depth(frame, calibration, pixels, **self.depth_options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> MediaPipeInitializer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def save_joint_estimates(path: str | Path, timestamps_s: Sequence[float],
                         estimates: Sequence[JointEstimate]) -> Path:
    """Write world-metre precomputed joints without pickle. Existing files are refused."""
    times = np.asarray(timestamps_s, dtype=np.float64)
    if (times.shape != (len(estimates),) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("joint timestamps must be finite, strictly increasing, and match estimates")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, schema_version=np.array(1), coordinate_frame=np.array("world"),
            units=np.array("m"), timestamps_s=times,
            joints_m=np.array([e.joints_m for e in estimates]).reshape(-1, 21, 3),
            confidence=np.array([e.confidence for e in estimates]).reshape(-1, 21),
            valid_mask=np.array([e.valid_mask for e in estimates], dtype=bool).reshape(-1, 21),
            valid=np.array([e.valid for e in estimates], dtype=bool))
    return path


def load_joint_estimates(path: str | Path) -> tuple[np.ndarray, list[JointEstimate]]:
    """Read precomputed 21-landmark world-metre estimates with explicit frame/units."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema_version", "coordinate_frame", "units", "timestamps_s", "joints_m", "confidence", "valid_mask", "valid"}
        if not required.issubset(archive.files):
            raise ValueError("precomputed joint NPZ is missing: " + ", ".join(sorted(required - set(archive.files))))
        if (archive["schema_version"].item() != 1 or archive["coordinate_frame"].item() != "world"
                or archive["units"].item() != "m"):
            raise ValueError("precomputed joints require schema v1, coordinate_frame='world', units='m'")
        times = np.array(archive["timestamps_s"], dtype=np.float64)
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("joint timestamps must be finite and strictly increasing")
        for key, expected in (("joints_m", (len(times), 21, 3)), ("confidence", (len(times), 21)),
                              ("valid_mask", (len(times), 21)), ("valid", (len(times),))):
            if archive[key].shape != expected:
                raise ValueError(key + " has an invalid shape")
        if archive["valid_mask"].dtype != np.bool_ or archive["valid"].dtype != np.bool_:
            raise ValueError("valid_mask and valid must be boolean arrays")
        estimates = [JointEstimate(archive["joints_m"][i], archive["confidence"][i],
                                   archive["valid_mask"][i], bool(archive["valid"][i])) for i in range(len(times))]
    return times, estimates


class PrecomputedJointsInitializer:
    """Nearest timestamp lookup of saved world-metre joints; no index-based fallback."""

    def __init__(self, path: str | Path, *, tolerance_s: float = 0.01, offset_s: float = 0.0):
        if not np.isfinite(tolerance_s) or tolerance_s < 0 or not np.isfinite(offset_s):
            raise ValueError("tolerance_s must be nonnegative and offset_s finite")
        self.timestamps_s, self.estimates = load_joint_estimates(path)
        self.tolerance_s, self.offset_s = float(tolerance_s), float(offset_s)

    def estimate_joints(self, frame: CameraFrame | float, calibration: CameraCalibration | None = None,
                        previous_state: Any = None) -> JointEstimate:
        """Saved timestamps + offset_s are compared with the frame timestamp."""
        timestamp = frame.timestamp_s if isinstance(frame, CameraFrame) else float(frame)
        if not np.isfinite(timestamp):
            raise ValueError("query timestamp must be finite")
        times = self.timestamps_s + self.offset_s
        if not len(times):
            return JointEstimate.invalid()
        insertion = int(np.searchsorted(times, timestamp))
        candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(times)]
        best = min(candidates, key=lambda i: (abs(times[i] - timestamp), i))
        if abs(times[best] - timestamp) > self.tolerance_s + 1e-12:
            return JointEstimate.invalid()
        estimate = self.estimates[best]
        return JointEstimate(estimate.joints_m, estimate.confidence, estimate.valid_mask, estimate.valid)
