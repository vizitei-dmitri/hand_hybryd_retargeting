"""Validate the core's public arrays without importing ROS, Torch or MANO."""

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial.transform import Rotation


OPTICAL_TO_ROS = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
SKELETON_EDGES = tuple(
    edge
    for start in (1, 5, 9, 13, 17)
    for edge in ((0, start), (start, start + 1), (start + 1, start + 2),
                 (start + 2, start + 3))
)


def axis_rotation(source_axes: str) -> np.ndarray:
    """Map optical (right, down, forward) to ROS (forward, left, up)."""
    if source_axes == "optical":
        return OPTICAL_TO_ROS.copy()
    if source_axes == "ros":
        return np.eye(3)
    raise ValueError("source_axes must be 'optical' or 'ros'")


def _array(value: Any, name: str, shape: tuple) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {result.shape}")
    return result


def _faces(value: Any) -> np.ndarray:
    faces = np.asarray(value)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must be an integer [K,3] array")
    if faces.size and (faces.min() < 0 or faces.max() >= 778):
        raise ValueError("faces index outside the 778 MANO vertices")
    return faces.astype(np.int32, copy=False)


def _valid(value: Any) -> bool:
    if np.ndim(value) != 0 or value not in (True, False, 0, 1):
        raise ValueError("valid must be a boolean or 0/1")
    return bool(value)


def _cloud(value: Any, name: str) -> np.ndarray:
    cloud = np.asarray(value, dtype=np.float64)
    if cloud.shape == (0,):
        cloud = cloud.reshape(0, 3)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N,3]")
    return cloud[np.all(np.isfinite(cloud), axis=1)]


def json_safe(value: Any) -> Any:
    """Keep diagnostics valid JSON even for invalid fits containing NaNs."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value: Any) -> str:
    return json.dumps(json_safe(value), allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class ResultFrame:
    timestamp_s: float
    joints_m: np.ndarray
    vertices_m: np.ndarray
    faces: np.ndarray
    theta: np.ndarray
    beta: np.ndarray
    global_orient: np.ndarray
    translation_m: np.ndarray
    valid: bool
    quality: Any = field(default_factory=dict)
    cloud_m: Optional[np.ndarray] = None
    frame_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    clouds_m: Mapping[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResultFrame":
        """One live result; geometry and global pose are in source axes/metres."""
        timestamp = float(value["timestamp_s"])
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        valid = _valid(value["valid"])
        arrays = {
            name: _array(value[name], name, shape)
            for name, shape in (
                ("joints_m", (21, 3)), ("vertices_m", (778, 3)),
                ("theta", (45,)), ("beta", (10,)),
                ("global_orient", (3,)), ("translation_m", (3,)),
            )
        }
        for name, array in arrays.items():
            if valid and not np.all(np.isfinite(array)):
                raise ValueError(f"valid frame contains non-finite {name}")
        cloud = value.get("cloud_m")
        if cloud is not None:
            cloud = _cloud(cloud, "cloud_m")
        clouds = value.get("clouds_m", {})
        if not isinstance(clouds, Mapping):
            raise ValueError("clouds_m must be a camera_id -> [N,3] mapping")
        clouds = {str(camera): _cloud(points, f"clouds_m[{camera}]")
                  for camera, points in clouds.items()}
        metadata = dict(value.get("metadata") or {})
        for name in ("model_type", "handedness", "status"):
            if name in value:
                metadata[name] = value[name]
        return cls(
            timestamp_s=timestamp, faces=_faces(value["faces"]), valid=valid,
            quality=value.get("quality", {}), cloud_m=cloud,
            frame_index=int(value.get("frame_index", 0)), metadata=metadata,
            clouds_m=clouds, **arrays,
        )

    def transformed(self, source_axes: str) -> "ResultFrame":
        """Rotate geometry AND the model-to-world global pose consistently."""
        rotation = axis_rotation(source_axes)
        orient = self.global_orient
        if np.all(np.isfinite(orient)):
            orient = (Rotation.from_matrix(rotation) * Rotation.from_rotvec(orient)).as_rotvec()
        return ResultFrame(
            timestamp_s=self.timestamp_s,
            joints_m=self.joints_m @ rotation.T,
            vertices_m=self.vertices_m @ rotation.T,
            faces=self.faces, theta=self.theta, beta=self.beta,
            global_orient=orient, translation_m=self.translation_m @ rotation.T,
            valid=self.valid, quality=self.quality,
            cloud_m=None if self.cloud_m is None else self.cloud_m @ rotation.T,
            frame_index=self.frame_index,
            metadata=self.metadata,
            clouds_m={camera: points @ rotation.T for camera, points in self.clouds_m.items()},
        )

    def parameters(self) -> dict:
        return {
            "frame_index": self.frame_index, "source_timestamp_s": self.timestamp_s,
            "valid": self.valid, "theta": self.theta, "beta": self.beta,
            "global_orient": self.global_orient, "translation_m": self.translation_m,
            "quality": self.quality,
            "metadata": self.metadata,
        }


class Reconstruction:
    """Eagerly validate an NPZ (never allow_pickle) before publishing any frame."""

    def __init__(self, path: str, quality_path: str = "") -> None:
        source = Path(path).expanduser()
        with np.load(source, allow_pickle=False) as archive:
            names = (
                "timestamps_s", "joints_m", "vertices_m", "faces", "theta", "beta",
                "global_orient", "translation_m", "valid",
            )
            missing = set(names) - set(archive.files)
            if missing:
                raise ValueError(f"reconstruction missing arrays: {sorted(missing)}")
            self.arrays = {name: archive[name] for name in names}
            if "cloud_m" in archive.files:
                self.arrays["cloud_m"] = archive["cloud_m"]
            embedded_quality = (
                json.loads(str(archive["quality_json"].item()))
                if "quality_json" in archive.files else {}
            )
            self.metadata = (
                json.loads(str(archive["metadata_json"].item()))
                if "metadata_json" in archive.files else {}
            )
            self.camera_ids = []
            self.camera_clouds = None
            cloud_keys = {"cloud_camera_ids", "clouds_per_camera_m"}
            if cloud_keys.intersection(archive.files):
                if not cloud_keys.issubset(archive.files):
                    raise ValueError(
                        "cloud_camera_ids and clouds_per_camera_m must be supplied together"
                    )
                ids = archive["cloud_camera_ids"]
                if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "US"):
                    raise ValueError("cloud_camera_ids must be a string [C] array")
                self.camera_ids = ids.astype(str).tolist()
                if (len(set(self.camera_ids)) != len(self.camera_ids)
                        or any(not x for x in self.camera_ids)):
                    raise ValueError("cloud_camera_ids must be unique nonempty camera names")
                self.camera_clouds = archive["clouds_per_camera_m"]
        self.timestamps = np.asarray(self.arrays["timestamps_s"], dtype=np.float64)
        if self.timestamps.ndim != 1 or not len(self.timestamps):
            raise ValueError("timestamps_s must be a nonempty [F] array")
        if not np.all(np.isfinite(self.timestamps)) or np.any(np.diff(self.timestamps) < 0):
            raise ValueError("timestamps_s must be finite and nondecreasing")
        count = len(self.timestamps)
        if self.camera_clouds is not None:
            shape = self.camera_clouds.shape
            if (len(shape) != 4 or shape[0] != count or shape[1] != len(self.camera_ids)
                    or shape[3] != 3):
                raise ValueError("clouds_per_camera_m must have shape [F,C,N,3]")
        for name, shape in (
            ("joints_m", (count, 21, 3)), ("vertices_m", (count, 778, 3)),
            ("theta", (count, 45)), ("beta", (10,)),
            ("global_orient", (count, 3)), ("translation_m", (count, 3)),
            ("valid", (count,)),
        ):
            if self.arrays[name].shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {self.arrays[name].shape}")
        self.arrays["faces"] = _faces(self.arrays["faces"])
        if "cloud_m" in self.arrays:
            cloud = self.arrays["cloud_m"]
            if cloud.ndim != 3 or cloud.shape[0] != count or cloud.shape[2] != 3:
                raise ValueError("cloud_m must have shape [F,N,3]")
        candidates = (
            [Path(quality_path).expanduser()] if quality_path else
            [source.with_suffix(".quality.json"), source.parent / "quality.json"]
        )
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if quality_path and selected is None:
            raise FileNotFoundError(quality_path)
        self.quality = json.loads(selected.read_text()) if selected else embedded_quality
        self.per_frame_quality = None
        if isinstance(self.quality, list):
            self.per_frame_quality = self.quality
        elif isinstance(self.quality, dict):
            for key in ("frames", "per_frame"):
                if isinstance(self.quality.get(key), list):
                    self.per_frame_quality = self.quality[key]
                    break
        if self.per_frame_quality is not None and len(self.per_frame_quality) != count:
            raise ValueError("quality frame count differs from reconstruction")
        # Fail before any ROS output, including when an invalid flag is malformed.
        for index in range(count):
            self.frame(index)

    def frame(self, index: int) -> ResultFrame:
        quality = self.quality
        if self.per_frame_quality is not None:
            quality = self.per_frame_quality[index]
        value = {
            name: array if name in ("faces", "beta") else array[index]
            for name, array in self.arrays.items() if name != "timestamps_s"
        }
        value.update(timestamp_s=self.timestamps[index], quality=quality, frame_index=index,
                     metadata=self.metadata)
        if self.camera_clouds is not None:
            value["clouds_m"] = {camera: self.camera_clouds[index, camera_index]
                                 for camera_index, camera in enumerate(self.camera_ids)}
        return ResultFrame.from_mapping(value)


class ReplaySchedule:
    """Monotonic elapsed time drives playback; acquisition epochs are irrelevant."""

    def __init__(self, timestamps: np.ndarray, speed: float = 1., loop: bool = False):
        times = np.asarray(timestamps, dtype=np.float64)
        if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
            raise ValueError("timestamps must be a finite, nonempty vector")
        if np.any(np.diff(times) < 0):
            raise ValueError("timestamps must be nondecreasing")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        self.offsets = (times - times[0]) / speed
        positive = np.diff(self.offsets)
        positive = positive[positive > 0]
        interval = float(np.median(positive)) if len(positive) else 1. / 30. / speed
        self.period = float(self.offsets[-1]) + interval
        self.loop = loop
        self.index = 0
        self.epoch = None
        self.finished = False

    def poll(self, monotonic_s: float) -> Optional[int]:
        """Return at most one due frame per tick, avoiding a backlog burst."""
        if self.finished:
            return None
        if self.epoch is None:
            self.epoch = monotonic_s
        if monotonic_s - self.epoch + 1e-9 < self.offsets[self.index]:
            return None
        index = self.index
        self.index += 1
        if self.index == len(self.offsets):
            if self.loop:
                # A slow publisher cannot accumulate whole cycles of backlog.
                self.epoch = max(self.epoch + self.period, monotonic_s)
                self.index = 0
            else:
                self.finished = True
        return index


class JsonlFollower:
    """Read complete appended UTF-8 JSON lines, including file replacement/truncation.

    Missing files are allowed while online capture starts. Each poll has bounded
    I/O and leaves a partial final line buffered until the producer finishes it.
    """

    def __init__(self, path: str, max_line_bytes: int = 16 * 1024 * 1024):
        self.path = Path(path).expanduser()
        self.max_line_bytes = max_line_bytes
        self._identity = None
        self._position = 0
        self._pending = b""

    def poll(self) -> list:
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            return []
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if identity != self._identity or stat.st_size < self._position:
                self._identity = identity
                self._position = 0
                self._pending = b""
            stream.seek(self._position)
            chunk = stream.read(1024 * 1024)
            self._position = stream.tell()
        lines = (self._pending + chunk).split(b"\n")
        self._pending = lines.pop()
        if any(len(line) > self.max_line_bytes for line in lines + [self._pending]):
            self._pending = b""
            raise ValueError("live JSONL line exceeds max_line_bytes")
        return [line for line in lines if line.strip()]
