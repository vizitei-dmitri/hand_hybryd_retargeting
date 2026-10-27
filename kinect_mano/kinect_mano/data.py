"""Validated RGB-D contracts and portable, per-camera NPZ recordings.

All distances are metres. Images and depth share the *color* pixel grid; camera
axes are x right, y down, z forward. ``T_world_from_camera`` acts on column
vectors. No implicit identity extrinsic or inferred depth scaling is used.

Recording schema v1 uses numeric/Unicode arrays only (``allow_pickle=False``):
``timestamps_s[N]``, ``depth_m[N,H,W]``, optional ``color[N,H,W,3]`` and
``mask[N,H,W]`` with per-frame presence flags, and JSON calibration/metadata.
NPZ is an in-memory interchange format; use bounded chunks for long sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class MissingCalibrationError(ValueError):
    """Required camera intrinsics or world extrinsics were not supplied."""


def validate_transform(matrix: Any, *, atol: float = 1e-5) -> np.ndarray:
    """Return a copy of a finite, proper rigid 4x4 transform, or raise."""
    if matrix is None:
        raise MissingCalibrationError("T_world_from_camera is required; calibrate the cameras first")
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("extrinsic transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=atol, rtol=0):
        raise ValueError("extrinsic transform must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0):
        raise ValueError("extrinsic rotation must be orthonormal (no scale or shear)")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0):
        raise ValueError("extrinsic rotation must have determinant +1 (no reflection)")
    return transform.copy()


@dataclass
class CameraCalibration:
    """Calibration of the RGB-aligned depth grid; extrinsic translation is metres."""

    camera_id: str
    intrinsics: np.ndarray
    distortion: np.ndarray
    T_world_from_camera: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id must be a nonempty string")
        if self.intrinsics is None:
            raise MissingCalibrationError("color camera intrinsics are required")
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64).copy()
        if self.intrinsics.shape != (3, 3) or not np.isfinite(self.intrinsics).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        if (self.intrinsics[0, 0] <= 0 or self.intrinsics[1, 1] <= 0
                or not np.allclose(self.intrinsics[2], [0, 0, 1], atol=1e-8)
                or not np.isclose(self.intrinsics[1, 0], 0)):
            raise ValueError("intrinsics must have positive focal lengths and pinhole form")
        if self.distortion is None:
            raise MissingCalibrationError("distortion is required; explicitly supply [] for rectified pixels")
        distortion = np.asarray(self.distortion, dtype=np.float64)
        if distortion.ndim > 2 or (distortion.ndim == 2 and 1 not in distortion.shape):
            raise ValueError("distortion must be a vector")
        self.distortion = distortion.reshape(-1).copy()
        if self.distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise ValueError("distortion must contain 0, 4, 5, 8, 12 or 14 finite OpenCV coefficients")
        self.T_world_from_camera = validate_transform(self.T_world_from_camera)
        if (isinstance(self.width, bool) or isinstance(self.height, bool)
                or int(self.width) != self.width or int(self.height) != self.height
                or self.width <= 0 or self.height <= 0):
            raise ValueError("width and height must be positive integers")
        self.width, self.height = int(self.width), int(self.height)

    def to_dict(self) -> dict[str, Any]:
        return {"camera_id": self.camera_id, "intrinsics": self.intrinsics.tolist(),
                "distortion": self.distortion.tolist(),
                "T_world_from_camera": self.T_world_from_camera.tolist(),
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CameraCalibration:
        required = {"camera_id", "intrinsics", "distortion", "T_world_from_camera", "width", "height"}
        missing = required.difference(value)
        if missing:
            raise MissingCalibrationError("missing calibration fields: " + ", ".join(sorted(missing)))
        return cls(**{key: value[key] for key in required})


@dataclass
class CameraFrame:
    """One color-aligned depth observation; zero/NaN depth means unavailable.

    ``timestamp_s`` is the depth device timestamp, before cross-camera clock
    correction. Optional metadata preserves RGB/depth/host timestamps separately.
    ``color`` is uint8 RGB, never BGR. Masks are boolean on the same pixel grid.
    """

    camera_id: str
    timestamp_s: float
    color: np.ndarray | None
    depth_m: np.ndarray
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id must be a nonempty string")
        self.timestamp_s = float(self.timestamp_s)
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
        if self.depth_m.ndim != 2 or min(self.depth_m.shape) < 1:
            raise ValueError("depth_m must have shape [H,W]")
        if np.isinf(self.depth_m).any() or (self.depth_m < 0).any():
            raise ValueError("depth_m must be nonnegative metres, with zero/NaN for missing depth")
        if self.color is not None:
            self.color = np.asarray(self.color)
            if self.color.shape != (*self.depth_m.shape, 3) or self.color.dtype != np.uint8:
                raise ValueError("color must be uint8 RGB [H,W,3], aligned with depth_m")
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.shape != self.depth_m.shape:
                raise ValueError("mask must have the same [H,W] shape as depth_m")
            if not np.isin(mask, [0, 1, False, True]).all():
                raise ValueError("mask must be boolean or binary 0/1")
            self.mask = mask.astype(bool, copy=False)
        if not isinstance(self.metadata, dict):
            raise ValueError("frame metadata must be a JSON object")


@dataclass
class JointEstimate:
    """21 landmarks in world metres, wrist then thumb/index/middle/ring/pinky.

    Missing landmarks have NaN coordinates, zero confidence, and a false mask.
    ``valid`` is a caller's overall initialization decision, not a claim that all
    21 joints were observed. A valid estimate must have at least one valid joint.
    """

    joints_m: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    valid: bool

    def __post_init__(self) -> None:
        self.joints_m = np.asarray(self.joints_m, dtype=np.float64).copy()
        self.confidence = np.asarray(self.confidence, dtype=np.float64).copy()
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool).copy()
        if self.joints_m.shape != (21, 3) or self.confidence.shape != (21,) or self.valid_mask.shape != (21,):
            raise ValueError("joints_m/confidence/valid_mask must have shapes [21,3]/[21]/[21]")
        if (not np.isfinite(self.confidence).all() or (self.confidence < 0).any()
                or (self.confidence > 1).any()):
            raise ValueError("joint confidence must be finite and in [0,1]")
        if (not np.isfinite(self.joints_m[self.valid_mask]).all()
                or (self.confidence[self.valid_mask] <= 0).any()):
            raise ValueError("valid joints require finite metric coordinates and positive confidence")
        self.joints_m[~self.valid_mask] = np.nan
        self.confidence[~self.valid_mask] = 0
        self.valid = bool(self.valid)
        if self.valid and not self.valid_mask.any():
            raise ValueError("a valid estimate needs at least one valid joint")

    @classmethod
    def invalid(cls) -> JointEstimate:
        return cls(np.full((21, 3), np.nan), np.zeros(21), np.zeros(21, bool), False)


@dataclass
class CameraRecording:
    calibration: CameraCalibration
    frames: list[CameraFrame]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, CameraCalibration):
            raise MissingCalibrationError("a recording requires camera calibration including world extrinsics")
        self.frames = list(self.frames)
        _check_frames(self.frames, self.calibration)
        if not isinstance(self.metadata, dict):
            raise ValueError("recording metadata must be a JSON object")
        status = self.metadata.get("extrinsics_status", "calibrated")
        if status not in ("calibrated", "uncalibrated_local_only"):
            raise ValueError("unknown recording extrinsics_status")
        if any(f.metadata.get("extrinsics_status") == "uncalibrated_local_only" for f in self.frames):
            if status != "uncalibrated_local_only":
                raise MissingCalibrationError("calibration-only frames require extrinsics_status='uncalibrated_local_only'")


def _check_frames(frames: Sequence[CameraFrame], calibration: CameraCalibration) -> None:
    for frame in frames:
        if frame.camera_id != calibration.camera_id:
            raise ValueError("frame camera_id differs from recording calibration")
        if frame.depth_m.shape != (calibration.height, calibration.width):
            raise ValueError("frame resolution differs from color camera calibration")
    times = np.asarray([frame.timestamp_s for frame in frames])
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("recording timestamps must be strictly increasing")


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True)


def save_recording(path: str | Path, frames: Iterable[CameraFrame],
                   calibration: CameraCalibration, *, metadata: dict[str, Any] | None = None) -> Path:
    """Write one camera's NPZ. Input depth must already be registered and in metres.

    Parent directories are created; the exact path is used without appending a
    suffix. Existing paths are refused to protect source recordings. Frames are
    materialized in memory; split long acquisitions into bounded recordings.
    """
    recording = CameraRecording(calibration, list(frames), {} if metadata is None else metadata)
    shape = (len(recording.frames), calibration.height, calibration.width)
    arrays: dict[str, Any] = {
        "schema_version": np.array(1, dtype=np.int64),
        "camera_id": np.array(calibration.camera_id),
        "timestamps_s": np.array([f.timestamp_s for f in recording.frames], dtype=np.float64),
        "depth_m": (np.stack([f.depth_m for f in recording.frames]) if recording.frames
                    else np.empty(shape, dtype=np.float32)),
        "calibration_json": np.array(_json(calibration.to_dict())),
        "metadata_json": np.array(_json(recording.metadata)),
        "frame_metadata_json": np.array([_json(f.metadata) for f in recording.frames], dtype=str),
    }
    for key in ("color", "mask"):
        present = np.array([getattr(f, key) is not None for f in recording.frames], dtype=bool)
        arrays[key + "_valid"] = present
        if present.any():
            size = (*shape, 3) if key == "color" else shape
            values = np.zeros(size, dtype=np.uint8 if key == "color" else bool)
            for index, frame in enumerate(recording.frames):
                if present[index]:
                    values[index] = getattr(frame, key)
            arrays[key] = values
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def load_recording(path: str | Path, *, calibration: CameraCalibration | None = None,
                    allow_uncalibrated: bool = False) -> CameraRecording:
    """Load schema-v1 NPZ without pickle, optionally applying an explicit rig.

    Calibration-only board recordings carry local identity anchors and are
    rejected unless ``allow_uncalibrated=True`` or a solved ``calibration=`` is
    supplied. The latter must match the recorded RGB intrinsics, distortion,
    camera ID and resolution; it replaces calibration and marks frames calibrated.
    """
    override_supplied = calibration is not None
    with np.load(path, allow_pickle=False) as archive:
        if "calibration_json" not in archive:
            raise MissingCalibrationError("NPZ recording has no calibration_json; supply calibrated data")
        if "schema_version" not in archive or int(archive["schema_version"].item()) != 1:
            raise ValueError("unsupported recording schema_version (expected 1)")
        required = {"camera_id", "timestamps_s", "depth_m", "metadata_json", "frame_metadata_json",
                    "color_valid", "mask_valid"}
        if not required.issubset(archive.files):
            raise ValueError("recording is missing required arrays: " + ", ".join(sorted(required - set(archive.files))))
        recorded_calibration = CameraCalibration.from_dict(json.loads(str(archive["calibration_json"].item())))
        if str(archive["camera_id"].item()) != recorded_calibration.camera_id:
            raise ValueError("NPZ camera_id differs from calibration")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("recording metadata must be a JSON object")
        if calibration is not None:
            if not isinstance(calibration, CameraCalibration):
                raise MissingCalibrationError("calibration override must be a CameraCalibration")
            if (calibration.camera_id != recorded_calibration.camera_id
                    or (calibration.width, calibration.height) != (recorded_calibration.width, recorded_calibration.height)):
                raise ValueError("calibration override camera ID/resolution differs from recording")
            if not np.allclose(calibration.intrinsics, recorded_calibration.intrinsics, rtol=1e-4, atol=1e-3):
                raise ValueError("calibration override RGB intrinsics differ from recorded factory intrinsics; recalibrate in the recorded camera mode")
            if (calibration.distortion.shape != recorded_calibration.distortion.shape
                    or not np.allclose(calibration.distortion, recorded_calibration.distortion, rtol=1e-4, atol=1e-5)):
                raise ValueError("calibration override RGB distortion differs from recorded factory distortion; use the matching lens calibration")
            metadata["recorded_extrinsics_status"] = metadata.get("extrinsics_status", "calibrated")
            metadata["extrinsics_status"] = "calibrated"
            metadata["extrinsics_source"] = "explicit_calibration_override"
        else:
            calibration = recorded_calibration
        times, depth = archive["timestamps_s"], archive["depth_m"]
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("timestamps_s must be a finite [N] vector")
        size = (len(times), calibration.height, calibration.width)
        if depth.shape != size:
            raise ValueError("recording depth_m resolution/count differs from calibration/timestamps")
        if archive["frame_metadata_json"].shape != (len(times),):
            raise ValueError("frame metadata count differs from timestamps")
        optional = {}
        for key in ("color", "mask"):
            presence = archive[key + "_valid"]
            if presence.shape != (len(times),) or presence.dtype != np.bool_:
                raise ValueError(key + "_valid must be a boolean [N] vector")
            expected = (*size, 3) if key == "color" else size
            values = archive[key] if key in archive else None
            if presence.any() and values is None:
                raise ValueError(key + "_valid indicates data, but the array is missing")
            if values is not None and values.shape != expected:
                raise ValueError(key + " array has incorrect resolution/count")
            optional[key] = (presence, values)
        frames = []
        for index, timestamp in enumerate(times):
            values = {key: (array[index].copy() if present[index] else None)
                      for key, (present, array) in optional.items()}
            frame_metadata = json.loads(str(archive["frame_metadata_json"][index]))
            if override_supplied:
                frame_metadata["extrinsics_status"] = "calibrated"
                frame_metadata["extrinsics_source"] = "explicit_calibration_override"
            frames.append(CameraFrame(calibration.camera_id, float(timestamp), values["color"],
                                      depth[index].copy(), values["mask"], frame_metadata))
        result = CameraRecording(calibration, frames, metadata)
        if metadata.get("extrinsics_status") == "uncalibrated_local_only" and not allow_uncalibrated:
            raise MissingCalibrationError("recording has uncalibrated local-only extrinsics; solve the rig and supply calibration=, or use allow_uncalibrated=True only for board calibration")
        return result


def match_frames_by_timestamp(frames_a: Sequence[CameraFrame], frames_b: Sequence[CameraFrame],
                              *, tolerance_s: float, offset_b_s: float = 0.0
                              ) -> list[tuple[CameraFrame, CameraFrame]]:
    """Greedy nearest, chronological, one-to-one matching within a tolerance.

    Compare ``a.timestamp_s`` with ``b.timestamp_s + offset_b_s``. Caller must
    establish a common clock (wired sync or measured offset); this does not infer
    hardware synchronization. Original timestamps are preserved. Unmatched frames
    are dropped. Ties use the earlier B frame. Both streams must be increasing.
    """
    if not np.isfinite(tolerance_s) or tolerance_s < 0 or not np.isfinite(offset_b_s):
        raise ValueError("tolerance_s must be nonnegative and offset_b_s finite")
    times_a = np.array([frame.timestamp_s for frame in frames_a])
    times_b = np.array([frame.timestamp_s + offset_b_s for frame in frames_b])
    if np.any(np.diff(times_a) <= 0) or np.any(np.diff(times_b) <= 0):
        raise ValueError("both timestamp streams must be strictly increasing")
    pairs, cursor = [], 0
    for index, timestamp in enumerate(times_a):
        if cursor >= len(times_b):
            break
        insertion = max(cursor, int(np.searchsorted(times_b, timestamp)))
        candidates = [j for j in (insertion - 1, insertion) if cursor <= j < len(times_b)]
        if not candidates:
            continue
        best = min(candidates, key=lambda j: (abs(times_b[j] - timestamp), j))
        if abs(times_b[best] - timestamp) <= tolerance_s + 1e-12:
            pairs.append((frames_a[index], frames_b[best]))
            cursor = best + 1
    return pairs
