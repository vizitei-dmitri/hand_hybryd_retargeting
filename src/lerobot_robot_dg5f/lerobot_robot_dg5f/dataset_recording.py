"""Passive sampling and official LeRobotDataset episode storage (no ROS/SDK)."""

from __future__ import annotations

import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from .constants import BROKEN_PINKY_INDEX, BROKEN_PINKY_JOINT, JOINT_NAMES


VECTOR_KEYS = (
    "observation.state", "action", "observation.velocity", "observation.current",
    "observation.temperature", "expert.raw_action",
)
RUNTIME_KEYS = (
    "backend", "startup_blend_s", "max_direct_step_deg", "tracking_grace_s",
    "tracking_resume_blend_s", "control_smoothing", "command_profile",
    "current_guard_enabled", "current_guard_soft_ma", "current_guard_hard_ma",
    "current_guard_trip_ma", "current_guard_total_soft_ma",
    "current_guard_total_hard_ma", "current_guard_total_trip_ma",
)
BOOL_KEYS = ("armed", "tracking_ok", "motion_ready", "current_guard_active")


def dataset_class():
    # Local-only, including when the official loader encounters incomplete files.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # The Robot plugin may have imported these libraries before the CLI ran.
    import huggingface_hub.constants
    import datasets.config
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    datasets.config.HF_DATASETS_OFFLINE = True
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def feature_schema(camera_shapes=None):
    result = {key: {"dtype": "float32", "shape": (20,), "names": list(JOINT_NAMES)}
              for key in VECTOR_KEYS}
    for name in (*BOOL_KEYS, "current_guard_min_scale"):
        result[f"diagnostic.{name}"] = {"dtype": "float32", "shape": (1,), "names": None}
    result["diagnostic.disarm_reason"] = {"dtype": "string", "shape": (1,), "names": None}
    for name in ("monotonic_s", "wall_time_s"):
        result[f"recording.{name}"] = {"dtype": "float64", "shape": (1,), "names": None}
    result["recording.segment_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    for name, shape in (camera_shapes or {}).items():
        result[f"observation.images.{name}"] = {
            "dtype": "image", "shape": tuple(shape), "names": ["height", "width", "channels"],
        }
    return result


def vector(values):
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (20,) or not np.all(np.isfinite(result)):
        raise ValueError("Expected exactly 20 finite joint values")
    return result.copy()


def ordered_degrees(names, values):
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("Invalid or duplicate joint names")
    by_name = dict(zip(names, values))
    if any(name not in by_name for name in JOINT_NAMES):
        raise ValueError("Incomplete DG5F joint names")
    return vector(np.rad2deg([by_name[name] for name in JOINT_NAMES]))


def decode_diagnostics(message):
    """Decode the existing coherent bridge snapshot; don't merge old fields."""
    for status in message.status:
        if status.name != "dg5f_lerobot_bridge":
            continue
        level = status.level
        result = {"diagnostic_level": level[0] if isinstance(level, bytes) else int(level)}
        for item in status.values:
            text = item.value
            if item.key.startswith("measured_") and item.key in {
                "measured_pos", "measured_vel", "measured_current", "measured_temp"
            }:
                result[item.key] = vector([float(x) for x in text.split(",")])
            elif text.lower() in ("true", "false"):
                result[item.key] = text.lower() == "true"
            else:
                try:
                    result[item.key] = float(text)
                except ValueError:
                    result[item.key] = text
        return result
    raise ValueError("No dg5f_lerobot_bridge diagnostic status")


class SampleBuffer:
    """Latest-value sampler; all ages/watchdogs use monotonic receive times."""

    def __init__(self, max_age_ms=100.0, cameras=(), clock=time.monotonic):
        if not np.isfinite(max_age_ms) or max_age_ms <= 0:
            raise ValueError("required_data_max_age_ms must be finite and positive")
        self.max_age = max_age_ms / 1000.0
        self.cameras = tuple(cameras)
        self.clock = clock
        self.samples = {}
        self.tracking_restored_at = float("inf")
        self.was_tracking = False

    def put(self, key, value, source_age_s=0.0):
        now = self.clock()
        if not np.isfinite(source_age_s) or source_age_s < 0:
            raise ValueError("Invalid source timestamp age")
        if key == "tracking":
            previous = self.samples.get(key)
            if value and (not self.was_tracking or previous is None
                          or now - previous[1] > self.max_age):
                self.tracking_restored_at = now
            self.was_tracking = bool(value)
        self.samples[key] = (value, now - source_age_s)

    def invalidate(self, key):
        self.samples.pop(key, None)
        if key == "tracking":
            self.was_tracking = False

    def snapshot(self):
        """Return (frame, skip reason, fatal). Fail closed on missing fields."""
        now = self.clock()
        diag_sample = self.samples.get("diagnostics")
        if diag_sample:
            diag, diag_at = diag_sample
            if now - diag_at <= self.max_age:
                for name in ("transport_connected", "motion_ready", "temperature_safe", "armed"):
                    if diag.get(name) is False:
                        return None, f"{name}=false ({diag.get('disarm_reason', 'UNKNOWN')})", True
                if diag.get("diagnostic_level", 0) >= 2 or diag.get("recovery_required") is True:
                    return None, "Hardware diagnostic fault / recovery required", True
        required = ("diagnostics", "effective", "raw", "tracking",
                    *(f"image:{name}" for name in self.cameras))
        for key in required:
            if key not in self.samples:
                return None, f"Waiting for {key}", False
            _, at = self.samples[key]
            if not 0 <= now - at <= self.max_age:
                if key == "tracking":
                    self.was_tracking = False
                return None, f"Stale {key}", False
        diag, diag_at = self.samples["diagnostics"]
        for name in ("transport_connected", "motion_ready", "temperature_safe", "armed", "telemetry_valid"):
            if diag.get(name) is not True:
                return None, f"Missing/invalid diagnostic {name}", False
        if not self.samples["tracking"][0] or diag.get("tracking_ok") is not True:
            self.was_tracking = False
            return None, "Tracking lost; imitation frames skipped", False
        if diag.get("tracking_grace_active") or diag.get("tracking_resume_pending"):
            return None, "Bridge is holding/waiting for fresh tracking command", False
        for name in ("last_position_sample_age_ms", "last_telemetry_age_ms",
                     "last_command_age_ms", "last_tracking_age_ms"):
            age = float(diag.get(name, float("inf"))) / 1000.0 + now - diag_at
            if not np.isfinite(age) or not 0 <= age <= self.max_age:
                return None, f"Stale source {name}", False
        # Don't reuse a pre-loss target or accepted command after tracking returns.
        if any(self.samples[key][1] < self.tracking_restored_at for key in ("raw", "effective")):
            return None, "Waiting for post-tracking raw/effective command", False
        try:
            result = {key: vector(diag.get(source, [])) for key, source in (
                ("observation.state", "measured_pos"), ("observation.velocity", "measured_vel"),
                ("observation.current", "measured_current"), ("observation.temperature", "measured_temp"),
            )}
            result["action"] = vector(self.samples["effective"][0])
            result["expert.raw_action"] = vector(self.samples["raw"][0])
            if abs(float(result["action"][BROKEN_PINKY_INDEX])) > 1e-5:
                return None, "Effective rj_dg_5_1 is not zero; refusing incorrect training action", True
            # Only remove harmless rad/deg roundoff, never fabricate a different trajectory.
            result["action"][BROKEN_PINKY_INDEX] = 0.0
            for name in BOOL_KEYS:
                if not isinstance(diag.get(name), bool):
                    raise ValueError(f"Missing boolean diagnostic {name}")
                result[f"diagnostic.{name}"] = np.array([diag[name]], dtype=np.float32)
            scale = float(diag["current_guard_min_scale"])
            if not np.isfinite(scale) or not 0 <= scale <= 1:
                raise ValueError("Invalid current guard scale")
            result["diagnostic.current_guard_min_scale"] = np.array([scale], dtype=np.float32)
            result["diagnostic.disarm_reason"] = str(diag["disarm_reason"])
            result["recording.monotonic_s"] = np.array([now], dtype=np.float64)
            result["recording.wall_time_s"] = np.array([time.time()], dtype=np.float64)
            for key in self.cameras:
                result[f"observation.images.{key}"] = self.samples[f"image:{key}"][0].copy()
            return result, "", False
        except (ValueError, KeyError, TypeError) as error:
            return None, str(error), False


class EpisodeRecorder:
    """Single-writer episodes. Each finish seals its parquet files immediately."""

    def __init__(self, repo_id, root, task, samples, fps=30, resume=False, project_dir="/workspace"):
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*(/[A-Za-z0-9_-][A-Za-z0-9_.-]*)?", repo_id):
            raise ValueError("repo-id must be a local name or namespace/name (no '..' paths)")
        if not isinstance(fps, int) or not 1 <= fps <= 200 or not task.strip():
            raise ValueError("fps must be an integer in 1..200 and task must be non-empty")
        self.repo_id, self.task, self.fps, self.samples = repo_id, task, fps, samples
        self.path = Path(root).expanduser().resolve() / repo_id
        if self.path.exists() and not resume:
            raise FileExistsError(f"{self.path} already exists; choose a new repo-id or use --resume")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = (self.path.parent / f".{self.path.name}.recorder.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise RuntimeError(f"Another recorder owns {self.path}") from None
        self.dataset = None
        self.state, self.reason = "IDLE", ""
        self.frames = self.saved_frames = self.episodes = self.skipped = 0
        self.segment = 0
        self._gap = False
        self._last_frame_time = None
        self._closed = False
        self._project_dir = project_dir
        self.manifest = None
        if resume:
            try:
                self._load_existing()
            except Exception:
                self._lock.close()
                raise

    def _load_existing(self):
        manifest_path = self.path / "meta/dg5f_recording.json"
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["repo_id"] != self.repo_id:
            raise ValueError("Dataset repo_id mismatch")
        self.dataset = dataset_class()(repo_id=self.repo_id, root=self.path, download_videos=False)
        if self.dataset.fps != self.fps:
            raise ValueError(f"Existing dataset fps is {self.dataset.fps}, not {self.fps}")
        self.episodes = self.dataset.num_episodes
        self.saved_frames = self.dataset.num_frames

    def _write_manifest(self):
        path = self.path / "meta/dg5f_recording.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)

    def _runtime_metadata(self):
        diag = self.samples.samples["diagnostics"][0]
        def git(*args):
            try:
                return subprocess.check_output(
                    ["git", "-C", self._project_dir, *args], text=True, timeout=3,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                return "unavailable"
        return {
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "runtime_configuration": {key: diag.get(key) for key in RUNTIME_KEYS},
        }

    def start(self):
        if self.state != "IDLE":
            raise RuntimeError(f"Episode is {self.state}; finish or discard it first")
        frame, reason, _ = self.samples.snapshot()
        if frame is None:
            raise RuntimeError(f"Cannot start: {reason}")
        shapes = {key: frame[f"observation.images.{key}"].shape for key in self.samples.cameras}
        schema = feature_schema(shapes)
        if self.dataset is None:
            if self.manifest is not None:
                self._load_existing()
            else:
                self.dataset = dataset_class().create(
                    repo_id=self.repo_id, root=self.path, fps=self.fps, robot_type="dg5f",
                    features=schema, use_videos=False,
                )
                self.manifest = {
                    "repo_id": self.repo_id, "robot_type": "dg5f", "hand": "right", "joint_count": 20,
                    "joint_names": list(JOINT_NAMES), "position_unit": "degree",
                    "velocity_unit": "degree/s", "current_unit": "mA", "temperature_unit": "C",
                    "disabled_joints": {BROKEN_PINKY_JOINT: 0.0},
                    "lerobot_version": importlib.metadata.version("lerobot"),
                    "required_data_max_age_ms": self.samples.max_age * 1000,
                    "action_source": "/dg5f/lerobot/commanded_joint_states",
                    "measured_source": "/dg5f/lerobot/diagnostics:measured_*",
                    "camera_topics": getattr(self.samples, "camera_topics", {}),
                    "episodes": [],
                    **self._runtime_metadata(),
                }
                self._write_manifest()
        for key, feature in schema.items():
            actual = self.dataset.features.get(key)
            if actual is None or tuple(actual["shape"]) != feature["shape"] or actual["dtype"] != feature["dtype"]:
                raise ValueError(f"Feature mismatch: {key}; use a new dataset")
            if key in VECTOR_KEYS and tuple(actual.get("names", ())) != JOINT_NAMES:
                raise ValueError(f"Joint order mismatch: {key}; use a new dataset")
        actual_cameras = set(self.dataset.meta.camera_keys)
        if actual_cameras != {f"observation.images.{key}" for key in shapes}:
            raise ValueError("Camera set differs from existing dataset")
        if self.dataset.episode_buffer is None:
            # Official resume constructor leaves it None until the first add_frame.
            # Explicitly initialize it so an empty resumed episode is discardable.
            self.dataset.clear_episode_buffer(delete_images=False)
        self._episode_metadata = self._runtime_metadata()
        self.state, self.reason = "RECORDING", ""
        self.frames = self.skipped = self.segment = 0
        self._gap = False
        self._last_frame_time = None

    def tick(self):
        if self.state != "RECORDING":
            return False
        frame, reason, fatal = self.samples.snapshot()
        if frame is None:
            self.reason = reason
            self.skipped += 1
            self._gap = self.frames > 0
            if fatal:
                self.state = "PAUSED"
            return False
        now = self.samples.clock()
        # An executor/disk stall can miss timer calls without any explicit skipped tick.
        if self._last_frame_time is not None and now - self._last_frame_time > 2.5 / self.fps:
            self._gap = True
        if self._gap:
            self.segment += 1
            self._gap = False
        frame["recording.segment_index"] = np.array([self.segment], dtype=np.int64)
        frame["task"] = self.task
        try:
            self.dataset.add_frame(frame)
        except Exception as error:
            self.state, self.reason = "ERROR", f"add_frame failed: {error}"
            return False
        self.frames += 1
        self._last_frame_time = now
        self.reason = ""
        return True

    def finish(self, interrupted=False):
        if self.state not in ("RECORDING", "PAUSED") or not self.frames:
            raise RuntimeError("No savable episode; use discard for an empty/error episode")
        previous_state = self.state
        try:
            # save_episode mutates its argument in 0.4.4. Keep the good buffer on failure.
            self.dataset.save_episode(episode_data=copy.deepcopy(self.dataset.episode_buffer))
            self.dataset.clear_episode_buffer()
            self.dataset.finalize()
        except Exception as error:
            self.state, self.reason = "ERROR", f"save/finalize failed: {error}; stop and inspect dataset"
            raise RuntimeError(self.reason) from error
        self.manifest["episodes"].append({
            "episode_index": self.episodes, "frames": self.frames, "task": self.task,
            "interrupted": interrupted, "end_state": previous_state, "end_reason": self.reason,
            "skipped_ticks": self.skipped, "segments": self.segment + 1,
            **self._episode_metadata,
        })
        self.episodes += 1
        self.saved_frames += self.frames
        self.dataset = None  # Reopen officially next time; never overwrite sealed parquet.
        self.state, self.reason, self.frames = "IDLE", "", 0
        self._write_manifest()

    def discard(self):
        if self.state == "IDLE":
            raise RuntimeError("No active episode")
        if self.state == "ERROR":
            raise RuntimeError("Storage error: stop recorder and inspect files; no automatic deletion")
        self.dataset.clear_episode_buffer()
        self.state, self.reason, self.frames = "IDLE", "", 0

    def status(self):
        _, preflight, _ = self.samples.snapshot()
        return {"state": self.state, "reason": self.reason, "preflight": preflight or "OK",
                "episodes": self.episodes, "saved_frames": self.saved_frames,
                "episode_frames": self.frames, "skipped_ticks": self.skipped,
                "fps": self.fps, "path": str(self.path)}

    def close(self):
        if self._closed:
            return
        try:
            if self.state in ("RECORDING", "PAUSED") and self.frames:
                self.finish(interrupted=True)
        finally:
            try:
                if self.dataset is not None:
                    self.dataset.finalize()
            finally:
                self._closed = True
                self._lock.close()
