"""Paired RGB chessboard calibration and explicit world-from-color extrinsics.

OpenCV stereoCalibrate returns p_B = R @ p_A + t; consequently
T_world_from_B = T_world_from_A @ inverse(T_B_from_A). Square size and
translation use metres. The caller explicitly chooses the world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import CameraCalibration, CameraFrame, MissingCalibrationError, match_frames_by_timestamp, validate_transform
from .geometry import _cv2


@dataclass(frozen=True)
class ChessboardSpec:
    """Number of *inner* corners and measured square edge length in metres.

    Both cameras must use the same physical corner ordering. Plain checkerboards
    have an orientation ambiguity; use a marked corner and consistent board
    orientation during paired capture, or provide verified point correspondences.
    """

    columns: int
    rows: int
    square_size_m: float

    def __post_init__(self) -> None:
        if (int(self.columns) != self.columns or int(self.rows) != self.rows
                or self.columns < 2 or self.rows < 2):
            raise ValueError("a chessboard needs at least 2x2 inner corners")
        if not np.isfinite(self.square_size_m) or self.square_size_m <= 0:
            raise ValueError("square_size_m must be positive metres")

    def object_points(self) -> np.ndarray:
        grid = np.zeros((self.columns * self.rows, 3), dtype=np.float32)
        grid[:, :2] = np.mgrid[:self.columns, :self.rows].T.reshape(-1, 2) * self.square_size_m
        return grid


@dataclass
class StereoCalibrationResult:
    calibration_a: CameraCalibration
    calibration_b: CameraCalibration
    T_camera_b_from_camera_a: np.ndarray
    rms_px: float
    num_pairs: int

    def metadata(self) -> dict[str, Any]:
        return {"method": "cv2.stereoCalibrate/CALIB_FIX_INTRINSIC", "rms_px": self.rms_px,
                "num_pairs": self.num_pairs, "translation_unit": "m",
                "coordinate_frame": "color", "T_camera_b_from_camera_a": self.T_camera_b_from_camera_a.tolist()}


def save_calibration(path: str | Path, calibration: CameraCalibration) -> Path:
    """Save one explicit camera calibration as JSON."""
    if not isinstance(calibration, CameraCalibration):
        raise MissingCalibrationError("camera calibration is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2, allow_nan=False) + "\n")
    return path


def load_calibration(path: str | Path) -> CameraCalibration:
    path = Path(path)
    if not path.is_file():
        raise MissingCalibrationError(f"calibration file does not exist: {path}")
    return CameraCalibration.from_dict(json.loads(path.read_text()))


def save_calibrations(path: str | Path, calibrations: Sequence[CameraCalibration], *,
                      metadata: dict[str, Any] | None = None) -> Path:
    """Save a multi-camera rig, including optional calibration provenance/residuals."""
    ids = [camera.camera_id for camera in calibrations]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("calibrations must contain unique nonempty camera IDs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "cameras": {c.camera_id: c.to_dict() for c in calibrations},
               "metadata": {} if metadata is None else metadata}
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path


def load_calibrations(path: str | Path) -> dict[str, CameraCalibration]:
    path = Path(path)
    if not path.is_file():
        raise MissingCalibrationError(f"rig calibration file does not exist: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not payload.get("cameras"):
        raise MissingCalibrationError("expected schema-v1 rig calibration with nonempty cameras")
    cameras = {key: CameraCalibration.from_dict(value) for key, value in payload["cameras"].items()}
    if any(key != camera.camera_id for key, camera in cameras.items()):
        raise ValueError("rig camera key differs from embedded camera_id")
    return cameras


def detect_chessboard(color_rgb: np.ndarray, board: ChessboardSpec) -> np.ndarray | None:
    """Find subpixel [N,2] chessboard corners in an RGB uint8 image."""
    color = np.asarray(color_rgb)
    if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        raise ValueError("chessboard detection requires uint8 RGB [H,W,3]")
    cv2 = _cv2()
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    pattern = (board.columns, board.rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        found, corners = cv2.findChessboardCorners(gray, pattern,
                                                  flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
    return corners.reshape(-1, 2) if found else None


def calibrate_stereo_from_points(object_points: Sequence[np.ndarray],
                                  image_points_a: Sequence[np.ndarray], image_points_b: Sequence[np.ndarray], *,
                                  camera_id_a: str, camera_id_b: str,
                                  intrinsics_a: np.ndarray, distortion_a: np.ndarray,
                                  intrinsics_b: np.ndarray, distortion_b: np.ndarray,
                                  image_size: tuple[int, int], T_world_from_camera_a: np.ndarray,
                                  min_pairs: int = 3, max_rms_px: float = 2.0) -> StereoCalibrationResult:
    """Solve a paired board stereo rig with fixed supplied RGB intrinsics.

    Object points are [N,3] metres, image points [N,2] pixels, one array per
    synchronized board pose (N may vary, e.g. shared ChArUco IDs). Only complete
    common correspondences should be passed. Use several tilted board poses over
    the field of view. RMS is a fitting residual, not independent accuracy proof.
    ``image_size`` is (width,height); both RGB streams must have this resolution.
    """
    if camera_id_a == camera_id_b:
        raise ValueError("stereo calibration needs two different camera IDs")
    anchor = validate_transform(T_world_from_camera_a)
    if len(image_size) != 2:
        raise ValueError("image_size must be (width,height)")
    a = CameraCalibration(camera_id_a, intrinsics_a, distortion_a, anchor, *image_size)
    # Validate B intrinsics using the explicitly chosen anchor only as a temporary
    # validation context. B's returned transform is exclusively the stereo solve.
    b_intrinsics = CameraCalibration(camera_id_b, intrinsics_b, distortion_b, anchor, *image_size)
    count = len(object_points)
    if isinstance(min_pairs, bool) or int(min_pairs) != min_pairs or min_pairs < 3:
        raise ValueError("min_pairs must be an integer >= 3")
    if count < min_pairs or len(image_points_a) != count or len(image_points_b) != count:
        raise ValueError(f"need at least {min_pairs} paired board observations with matching counts")
    if not np.isfinite(max_rms_px) or max_rms_px <= 0:
        raise ValueError("max_rms_px must be finite and positive")
    objects, pixels_a, pixels_b = [], [], []
    for obj, pa, pb in zip(object_points, image_points_a, image_points_b):
        obj, pa, pb = (np.asarray(value, dtype=np.float32) for value in (obj, pa, pb))
        if (obj.ndim != 2 or obj.shape[1] != 3 or len(obj) < 6
                or pa.shape != (len(obj), 2) or pb.shape != pa.shape
                or not all(np.isfinite(value).all() for value in (obj, pa, pb))):
            raise ValueError("each pair needs >=6 finite matched object [N,3] and image [N,2] points")
        if np.linalg.matrix_rank(obj - obj.mean(axis=0)) < 2:
            raise ValueError("board points must span a plane, not a line")
        objects.append(obj)
        pixels_a.append(pa)
        pixels_b.append(pb)
    cv2 = _cv2()
    rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
        objects, pixels_a, pixels_b, a.intrinsics.copy(), a.distortion.copy(),
        b_intrinsics.intrinsics.copy(), b_intrinsics.distortion.copy(), (a.width, a.height),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9))
    if not np.isfinite(rms) or rms > max_rms_px:
        raise ValueError(f"stereo calibration RMS {rms:.4g}px exceeds {max_rms_px:g}px; verify paired corners and board units")
    relative = np.eye(4)
    relative[:3, :3], relative[:3, 3] = rotation, translation.reshape(3)
    relative = validate_transform(relative)
    b = CameraCalibration(camera_id_b, intrinsics_b, distortion_b,
                          anchor @ np.linalg.inv(relative), a.width, a.height)
    return StereoCalibrationResult(a, b, relative, float(rms), count)


def calibrate_chessboard(frames_a: Sequence[CameraFrame], frames_b: Sequence[CameraFrame],
                         board: ChessboardSpec, *, intrinsics_a: np.ndarray, distortion_a: np.ndarray,
                         intrinsics_b: np.ndarray, distortion_b: np.ndarray,
                         T_world_from_camera_a: np.ndarray, tolerance_s: float, offset_b_s: float = 0.0,
                         min_pairs: int = 3, max_rms_px: float = 2.0) -> StereoCalibrationResult:
    """Timestamp-match RGB frames, detect a chessboard in both, and stereo-calibrate."""
    pairs = match_frames_by_timestamp(frames_a, frames_b, tolerance_s=tolerance_s, offset_b_s=offset_b_s)
    objects, points_a, points_b = [], [], []
    size, camera_ids = None, None
    for frame_a, frame_b in pairs:
        if frame_a.color is None or frame_b.color is None:
            continue
        if frame_a.color.shape != frame_b.color.shape:
            raise ValueError("stereo chessboard RGB image resolutions must match")
        current_size = (frame_a.color.shape[1], frame_a.color.shape[0])
        ids = (frame_a.camera_id, frame_b.camera_id)
        if size is not None and (current_size != size or ids != camera_ids):
            raise ValueError("camera IDs and RGB resolution must stay constant during calibration")
        size, camera_ids = current_size, ids
        corners_a = detect_chessboard(frame_a.color, board)
        corners_b = detect_chessboard(frame_b.color, board)
        if corners_a is not None and corners_b is not None:
            objects.append(board.object_points())
            points_a.append(corners_a)
            points_b.append(corners_b)
    if size is None or len(objects) < min_pairs:
        raise ValueError(f"need at least {min_pairs} synchronized RGB pairs with chessboard corners in both cameras; got {len(objects)}")
    return calibrate_stereo_from_points(objects, points_a, points_b,
        camera_id_a=camera_ids[0], camera_id_b=camera_ids[1], intrinsics_a=intrinsics_a,
        distortion_a=distortion_a, intrinsics_b=intrinsics_b, distortion_b=distortion_b,
        image_size=size, T_world_from_camera_a=T_world_from_camera_a,
        min_pairs=min_pairs, max_rms_px=max_rms_px)
