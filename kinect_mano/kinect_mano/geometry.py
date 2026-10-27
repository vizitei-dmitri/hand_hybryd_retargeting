"""Geometry on the registered RGB pixel grid, in metres and camera/world axes.

Registration is not rectification: transformed Kinect depth still uses distorted
color pixels. Nonzero distortion therefore requires OpenCV's undistortPoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import CameraCalibration, CameraFrame, MissingCalibrationError, validate_transform


def _calibration(calibration: CameraCalibration | None) -> CameraCalibration:
    if not isinstance(calibration, CameraCalibration):
        raise MissingCalibrationError("color calibration and T_world_from_camera are required")
    return calibration


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise ImportError("OpenCV is required for lens distortion and polygon masks") from error
    return cv2


def transform_points(points_m: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a rigid column-vector transform to row-stored [...,3] points."""
    transform = validate_transform(transform)
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points must have shape [...,3]")
    return points @ transform[:3, :3].T + transform[:3, 3]


def unproject_pixels(pixels_uv: np.ndarray, depth_m: np.ndarray,
                     calibration: CameraCalibration, *, world: bool = True) -> np.ndarray:
    """Unproject [N,2] pixels and their measured axial Z depths [N].

    Invalid depths/pixels produce NaN points. Coordinates need not be integers.
    Distortion is inverted before applying Z. Depth is axial, not ray length.
    """
    calibration = _calibration(calibration)
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depth.shape != (len(pixels),):
        raise ValueError("pixels_uv and depth_m must have shapes [N,2] and [N]")
    points = np.full((len(pixels), 3), np.nan)
    valid = (np.isfinite(pixels).all(axis=1) & np.isfinite(depth) & (depth > 0)
             & (pixels[:, 0] >= 0) & (pixels[:, 0] < calibration.width)
             & (pixels[:, 1] >= 0) & (pixels[:, 1] < calibration.height))
    if not valid.any():
        return points
    if np.any(calibration.distortion):
        rays_xy = _cv2().undistortPoints(pixels[valid, None, :], calibration.intrinsics,
                                        calibration.distortion).reshape(-1, 2)
        rays = np.column_stack((rays_xy, np.ones(len(rays_xy))))
    else:
        rays = np.column_stack((pixels[valid], np.ones(valid.sum()))) @ np.linalg.inv(calibration.intrinsics).T
    points[valid] = rays * depth[valid, None]
    return transform_points(points, calibration.T_world_from_camera) if world else points


def project_points(points_m: np.ndarray, calibration: CameraCalibration,
                   *, world: bool = True) -> np.ndarray:
    """Project [N,3] points into distorted RGB pixels; behind-camera points are NaN."""
    calibration = _calibration(calibration)
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must have shape [N,3]")
    if world:
        points = transform_points(points, np.linalg.inv(calibration.T_world_from_camera))
    pixels = np.full((len(points), 2), np.nan)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    if valid.any():
        if np.any(calibration.distortion):
            pixels[valid] = _cv2().projectPoints(points[valid], np.zeros(3), np.zeros(3),
                                                calibration.intrinsics, calibration.distortion)[0].reshape(-1, 2)
        else:
            homogeneous = points[valid] @ calibration.intrinsics.T
            pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels


def valid_depth_mask(depth_m: np.ndarray, mask: np.ndarray | None = None, *,
                     min_depth_m: float = 0.0, max_depth_m: float = np.inf) -> np.ndarray:
    """Intersect finite positive Z depth, inclusive metric range, and an optional mask."""
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth_m must have shape [H,W]")
    if not np.isfinite(min_depth_m) or min_depth_m < 0 or np.isnan(max_depth_m) or max_depth_m <= min_depth_m:
        raise ValueError("depth range must satisfy 0 <= min_depth_m < max_depth_m")
    valid = np.isfinite(depth) & (depth > 0) & (depth >= min_depth_m) & (depth <= max_depth_m)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != depth.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("mask must be binary and match the depth grid")
        valid &= mask.astype(bool)
    return valid


def unproject_depth(depth_m: np.ndarray, calibration: CameraCalibration, mask: np.ndarray | None = None,
                    *, world: bool = True, min_depth_m: float = 0.0, max_depth_m: float = np.inf,
                    stride: int = 1, return_pixels: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return valid [N,3] points in row-major order, optionally with [N,2] UV pixels."""
    calibration = _calibration(calibration)
    depth = np.asarray(depth_m)
    if depth.shape != (calibration.height, calibration.width):
        raise ValueError("registered depth resolution differs from color calibration")
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError("stride must be a positive integer")
    valid = valid_depth_mask(depth, mask, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    sampled = np.zeros_like(valid)
    sampled[::int(stride), ::int(stride)] = True
    v, u = np.nonzero(valid & sampled)
    pixels = np.column_stack((u, v))
    points = unproject_pixels(pixels, depth[v, u], calibration, world=world)
    return (points, pixels) if return_pixels else points


def frame_point_cloud(frame: CameraFrame, calibration: CameraCalibration, *,
                      require_mask: bool = True, **kwargs: Any) -> np.ndarray:
    """Extract a frame's point cloud; fitting requires an explicit hand mask by default."""
    calibration = _calibration(calibration)
    if frame.camera_id != calibration.camera_id:
        raise ValueError("frame and calibration camera IDs differ")
    if kwargs.get("world", True) and frame.metadata.get("extrinsics_status") == "uncalibrated_local_only":
        raise MissingCalibrationError("calibration-only frame has no rig extrinsics; load it with an explicit calibration override")
    if require_mask and frame.mask is None:
        raise ValueError("hand mask required; supply segmentation or an explicit polygon ROI")
    return unproject_depth(frame.depth_m, calibration, frame.mask, **kwargs)


def polygon_hand_mask(landmarks_px: np.ndarray, image_shape: tuple[int, int], *,
                      padding_px: int = 5, valid_mask: np.ndarray | None = None,
                      object_mask: np.ndarray | None = None) -> np.ndarray:
    """Convex polygon ROI around 2D hand landmarks, optionally excluding object pixels.

    This is a hand *ROI*, not semantic segmentation: it includes gaps between
    fingers, occluded objects, and background inside the hull. Grasping sequences
    need a supplied hand/object segmentation mask for trustworthy surface fitting.
    No depth clustering is used as a substitute for segmentation.
    """
    points = np.asarray(landmarks_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("landmarks_px must have shape [N,2]")
    if (len(image_shape) != 2 or any(int(v) != v or v < 1 for v in image_shape)
            or int(padding_px) != padding_px or padding_px < 0):
        raise ValueError("image shape must be positive and padding_px a nonnegative integer")
    image_shape = tuple(int(v) for v in image_shape)
    valid = np.isfinite(points).all(axis=1)
    if valid_mask is not None:
        validity = np.asarray(valid_mask, dtype=bool)
        if validity.shape != (len(points),):
            raise ValueError("valid_mask must have shape [N]")
        valid &= validity
    output = np.zeros(image_shape, dtype=np.uint8)
    if valid.sum() >= 3:
        cv2 = _cv2()
        # Keep offscreen points offscreen; clipping each vertex would create a
        # false visible hand when the entire prediction lies outside the image.
        bounded = np.clip(points[valid], -(2**20), 2**20)
        hull = cv2.convexHull(np.rint(bounded).astype(np.int32))
        if cv2.contourArea(hull) > 0:
            cv2.fillConvexPoly(output, hull, 1)
            if padding_px:
                size = int(padding_px) * 2 + 1
                output = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    result = output.astype(bool)
    if object_mask is not None:
        objects = np.asarray(object_mask)
        if objects.shape != image_shape or not np.isin(objects, [0, 1]).all():
            raise ValueError("object_mask must be binary and match image_shape")
        result &= ~objects.astype(bool)
    return result
