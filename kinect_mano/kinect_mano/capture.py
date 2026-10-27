"""Optional pyk4a acquisition of real Azure Kinect RGB-aligned depth.

Primary API sources:
https://github.com/etiennedub/pyk4a/blob/master/pyk4a/capture.py
https://github.com/etiennedub/pyk4a/blob/master/pyk4a/calibration.py
https://github.com/etiennedub/pyk4a/blob/master/pyk4a/config.py
https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/multi-camera-sync

Requires the Azure Kinect Sensor SDK, optional pyk4a, two USB connections and a
physical sync cable for wired mode. Subordinates start before the master. Manual
RGB exposure is set once. Stagger depth timing by >=160us for overlapping views.
Device timestamps are preserved; a measured clock offset can be supplied when
pairing. Host perf-counter timestamps are recorded as diagnostics, never silently
substituted for device time. This module downloads no SDKs, models or firmware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import CameraCalibration, CameraFrame, MissingCalibrationError, save_recording, validate_transform


_COLOR_SIZES = {"RES_720P": (1280, 720), "RES_1080P": (1920, 1080),
                "RES_1440P": (2560, 1440), "RES_1536P": (2048, 1536),
                "RES_2160P": (3840, 2160), "RES_3072P": (4096, 3072)}


class AzureKinectCamera:
    """One camera, with explicit rig extrinsics or explicit calibration-only mode.

    In ``calibration_only=True`` the identity is *only* the camera's local frame,
    marked ``uncalibrated_local_only`` on frames/recordings. It is never accepted
    as calibrated multi-camera data by the default recording loader.
    """

    def __init__(self, camera_id: str, device_id: int = 0, *,
                 T_world_from_camera: np.ndarray | None = None, calibration_only: bool = False,
                 sync_mode: str = "standalone", color_resolution: str = "RES_720P",
                 depth_mode: str = "NFOV_UNBINNED", fps: int = 30, exposure_usec: int = 8000,
                 depth_delay_off_color_usec: int = 0, subordinate_delay_off_master_usec: int = 0,
                 expected_serial: str | None = None):
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("camera_id must be nonempty")
        if isinstance(device_id, bool) or int(device_id) != device_id or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        if calibration_only and T_world_from_camera is not None:
            raise ValueError("choose explicit rig extrinsics or calibration_only=True, not both")
        if not calibration_only and T_world_from_camera is None:
            raise MissingCalibrationError("rig T_world_from_camera is required; use calibration_only=True explicitly to acquire board data")
        self.T_world_from_camera = np.eye(4) if calibration_only else validate_transform(T_world_from_camera)
        if sync_mode.lower() not in ("master", "subordinate", "standalone"):
            raise ValueError("sync_mode must be master, subordinate or standalone")
        if color_resolution not in _COLOR_SIZES:
            raise ValueError("unsupported enabled color_resolution: " + color_resolution)
        if depth_mode not in ("NFOV_2X2BINNED", "NFOV_UNBINNED", "WFOV_2X2BINNED", "WFOV_UNBINNED"):
            raise ValueError("an active depth_mode is required for registered depth")
        if fps not in (5, 15, 30):
            raise ValueError("fps must be 5, 15 or 30")
        if depth_mode == "WFOV_UNBINNED" and fps == 30:
            raise ValueError("WFOV_UNBINNED does not support 30 FPS")
        if color_resolution == "RES_3072P" and fps == 30:
            raise ValueError("RES_3072P does not support 30 FPS")
        if isinstance(exposure_usec, bool) or int(exposure_usec) != exposure_usec or exposure_usec <= 0:
            raise ValueError("manual exposure_usec must be a positive integer")
        for name, delay in (("depth_delay_off_color_usec", depth_delay_off_color_usec),
                            ("subordinate_delay_off_master_usec", subordinate_delay_off_master_usec)):
            if isinstance(delay, bool) or int(delay) != delay:
                raise ValueError(name + " must be an integer")
        if subordinate_delay_off_master_usec < 0 or subordinate_delay_off_master_usec >= 1e6 / fps:
            raise ValueError("subordinate delay must be nonnegative and shorter than a frame period")
        if sync_mode.lower() != "subordinate" and subordinate_delay_off_master_usec:
            raise ValueError("subordinate delay applies only to a subordinate camera")
        if abs(depth_delay_off_color_usec) >= 1e6 / fps:
            raise ValueError("depth delay must be shorter than a frame period")
        self.camera_id, self.device_id = camera_id, int(device_id)
        self.calibration_only, self.sync_mode = bool(calibration_only), sync_mode.lower()
        self.color_resolution, self.depth_mode, self.fps = color_resolution, depth_mode, fps
        self.exposure_usec = int(exposure_usec)
        self.depth_delay_off_color_usec = int(depth_delay_off_color_usec)
        self.subordinate_delay_off_master_usec = int(subordinate_delay_off_master_usec)
        self.expected_serial = expected_serial
        self._device: Any = None
        self._calibration: CameraCalibration | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def calibration(self) -> CameraCalibration:
        if self._calibration is None:
            raise MissingCalibrationError("start the camera before accessing factory RGB calibration")
        return self._calibration

    @property
    def metadata(self) -> dict[str, Any]:
        if self._device is None:
            raise RuntimeError("start the camera before accessing capture metadata")
        return dict(self._metadata)

    def start(self) -> AzureKinectCamera:
        if self._device is not None:
            return self
        try:
            import pyk4a
        except ImportError as error:
            raise ImportError("Azure Kinect capture requires optional pyk4a and the installed Azure Kinect Sensor SDK") from error
        config = pyk4a.Config(color_resolution=getattr(pyk4a.ColorResolution, self.color_resolution),
            color_format=pyk4a.ImageFormat.COLOR_BGRA32, depth_mode=getattr(pyk4a.DepthMode, self.depth_mode),
            camera_fps=getattr(pyk4a.FPS, "FPS_" + str(self.fps)), synchronized_images_only=True,
            depth_delay_off_color_usec=self.depth_delay_off_color_usec,
            wired_sync_mode=getattr(pyk4a.WiredSyncMode, self.sync_mode.upper()),
            subordinate_delay_off_master_usec=self.subordinate_delay_off_master_usec)
        device = pyk4a.PyK4A(config=config, device_id=self.device_id)
        self._device = device
        try:
            device.open()
            if self.expected_serial is not None and device.serial != self.expected_serial:
                raise ValueError(f"camera {self.camera_id}: expected serial {self.expected_serial}, got {device.serial}")
            if self.sync_mode != "standalone":
                jack_in, jack_out = device.sync_jack_status
                if (self.sync_mode == "master" and not jack_out) or (self.sync_mode == "subordinate" and not jack_in):
                    raise RuntimeError("wired sync cable is missing for " + self.sync_mode)
            device.exposure = self.exposure_usec
            device.start()
            factory = device.calibration
            width, height = _COLOR_SIZES[self.color_resolution]
            self._calibration = CameraCalibration(self.camera_id,
                factory.get_camera_matrix(pyk4a.CalibrationType.COLOR),
                factory.get_distortion_coefficients(pyk4a.CalibrationType.COLOR),
                self.T_world_from_camera, width, height)
            self._metadata = {"backend": "pyk4a", "serial": device.serial, "device_id": self.device_id,
                "depth_unit": "m", "depth_registration": "color", "color_order": "RGB",
                "timestamp_domain": "device", "timestamp_source": "depth_timestamp_usec",
                "extrinsics_status": "uncalibrated_local_only" if self.calibration_only else "calibrated",
                "wired_sync_mode": self.sync_mode, "color_resolution": self.color_resolution,
                "depth_mode": self.depth_mode, "fps": self.fps, "exposure_usec": self.exposure_usec,
                "depth_delay_off_color_usec": self.depth_delay_off_color_usec,
                "subordinate_delay_off_master_usec": self.subordinate_delay_off_master_usec,
                "factory_calibration_raw": device.calibration_raw}
        except BaseException:
            self.stop()
            raise
        return self

    def read(self, timeout_ms: int = 1000) -> CameraFrame:
        """Read transformed_depth (mm -> m), BGRA -> RGB, and raw SDK timestamps."""
        if self._device is None:
            raise RuntimeError("camera is not started")
        if isinstance(timeout_ms, bool) or int(timeout_ms) != timeout_ms or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        capture = self._device.get_capture(timeout=int(timeout_ms))
        color, depth_mm = capture.color, capture.transformed_depth
        if color is None or depth_mm is None:
            raise RuntimeError("capture lacks synchronized color/transformed_depth")
        color, depth_mm = np.asarray(color), np.asarray(depth_mm)
        if color.dtype != np.uint8 or color.shape != (self.calibration.height, self.calibration.width, 4):
            raise ValueError("SDK returned unexpected BGRA color format/resolution")
        if depth_mm.shape != color.shape[:2]:
            raise ValueError("transformed_depth is not registered to the color resolution")
        depth_timestamp = int(capture.depth_timestamp_usec)
        color_timestamp = int(capture.color_timestamp_usec)
        if min(depth_timestamp, color_timestamp) < 0:
            raise ValueError("SDK returned a negative device timestamp")
        metadata = {"extrinsics_status": self._metadata["extrinsics_status"],
                    "timestamp_domain": "device", "depth_timestamp_usec": depth_timestamp,
                    "color_timestamp_usec": color_timestamp,
                    "depth_system_timestamp_nsec": int(capture.depth_system_timestamp_nsec),
                    "color_system_timestamp_nsec": int(capture.color_system_timestamp_nsec)}
        return CameraFrame(self.camera_id, depth_timestamp * 1e-6, color[..., [2, 1, 0]].copy(),
                           depth_mm.astype(np.float32) * np.float32(0.001), metadata=metadata)

    def stop(self) -> None:
        device, self._device = self._device, None
        if device is not None:
            try:
                if device.is_running:
                    device.stop()
            finally:
                if device.opened:
                    device.close()

    def __enter__(self) -> AzureKinectCamera:
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()


class DualKinectCapture:
    """Capture standalone pairs or start wired devices subordinate-first.

    Standalone devices have independent clocks: pairing requires a measured
    offset, and nearest timestamps do not make them hardware synchronized.
    """

    def __init__(self, master: AzureKinectCamera, subordinate: AzureKinectCamera):
        modes = (master.sync_mode, subordinate.sync_mode)
        if modes not in (("master", "subordinate"), ("standalone", "standalone")):
            raise ValueError("dual capture needs master+subordinate or two standalone cameras")
        if master.device_id == subordinate.device_id or master.camera_id == subordinate.camera_id:
            raise ValueError("dual capture needs different device and camera IDs")
        if master.fps != subordinate.fps:
            raise ValueError("paired cameras must use matching FPS")
        if modes[0] == "master" and master.exposure_usec != subordinate.exposure_usec:
            raise ValueError("wired cameras must use matching manual RGB exposure")
        if master.calibration_only != subordinate.calibration_only:
            raise ValueError("both cameras must use the same explicit calibration-only/rig mode")
        self.master, self.subordinate = master, subordinate

    def start(self) -> DualKinectCapture:
        try:
            self.subordinate.start()
            self.master.start()
        except BaseException:
            self.stop()
            raise
        return self

    def read_pair(self, *, tolerance_s: float, offset_b_s: float = 0.0,
                   timeout_ms: int = 1000, max_skips: int = 100) -> tuple[CameraFrame, CameraFrame]:
        """Drop earlier frames until depth-device timestamps agree within tolerance."""
        if not np.isfinite(tolerance_s) or tolerance_s < 0 or not np.isfinite(offset_b_s):
            raise ValueError("tolerance_s must be nonnegative and offset_b_s finite")
        if int(max_skips) != max_skips or max_skips < 0:
            raise ValueError("max_skips must be a nonnegative integer")
        a, b = self.master.read(timeout_ms), self.subordinate.read(timeout_ms)
        for skipped in range(max_skips + 1):
            delta = b.timestamp_s + offset_b_s - a.timestamp_s
            if abs(delta) <= tolerance_s + 1e-12:
                for frame in (a, b):
                    frame.metadata["pair_delta_s"] = delta
                    frame.metadata["pair_skipped_frames"] = skipped
                return a, b
            if skipped < max_skips:
                if delta > 0:
                    a = self.master.read(timeout_ms)
                else:
                    b = self.subordinate.read(timeout_ms)
        raise TimeoutError(f"no synchronized pair within {tolerance_s:g}s after {max_skips} skipped frames; verify cables and clock offset")

    def stop(self) -> None:
        try:
            self.master.stop()
        finally:
            self.subordinate.stop()

    def __enter__(self) -> DualKinectCapture:
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()


def record_dual_cameras(master: AzureKinectCamera, subordinate: AzureKinectCamera,
                         output_dir: str | Path, *, frame_count: int, tolerance_s: float,
                         offset_b_s: float = 0.0, timeout_ms: int = 1000) -> tuple[Path, Path]:
    """Acquire a bounded chunk into camera_A.npz / camera_B.npz and stop devices.

    Configure subordinate ``depth_delay_off_color_usec=160`` (master=0) for
    overlapping active depth views, or choose measured offsets appropriate to
    your sensor modes. A per-camera expected_serial prevents device-index swaps.
    Do not request unbounded sessions: frames are held in memory before saving.
    """
    if isinstance(frame_count, bool) or int(frame_count) != frame_count or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    output_dir = Path(output_dir)
    paths = output_dir / "camera_A.npz", output_dir / "camera_B.npz"
    if any(path.exists() for path in paths):
        raise FileExistsError("output recording already exists; choose a new session directory")
    frames_a, frames_b = [], []
    with DualKinectCapture(master, subordinate) as pair:
        for _ in range(int(frame_count)):
            a, b = pair.read_pair(tolerance_s=tolerance_s, offset_b_s=offset_b_s, timeout_ms=timeout_ms)
            frames_a.append(a)
            frames_b.append(b)
        common = {"pair_tolerance_s": tolerance_s, "offset_b_s": offset_b_s}
        save_recording(paths[0], frames_a, master.calibration, metadata={**master.metadata, **common})
        save_recording(paths[1], frames_b, subordinate.calibration, metadata={**subordinate.metadata, **common})
    return paths
