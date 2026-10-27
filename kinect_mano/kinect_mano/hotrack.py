"""Generic NPZ/process joint initializer; official HOTrack inference is not wired.

The worker is a user-supplied executable in its own environment. This module
imports neither torch nor the official research repository. All public arrays
use world metres and wrist, thumb, index, middle, ring, pinky landmark order.
See ``docs/HOTRACK.md`` for the audited upstream requirements and wire protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from zipfile import BadZipFile

import numpy as np

from .data import JointEstimate


logger = logging.getLogger(__name__)
OFFICIAL_INFERENCE_IMPLEMENTED = False
NPZ_SCHEMA_VERSION = 1


class HOTrackProcessError(RuntimeError):
    """A configured external worker failed or did not produce a valid result."""


def _require_file(path: str | Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing or not a regular file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def _real_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values, not {array.dtype}")
    return array.astype(np.float64, copy=True)


def _bool_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "biu"
            or not np.isin(array, [0, 1]).all()):
        raise ValueError(f"{name} must be boolean/binary with shape {shape}")
    return array.astype(bool, copy=True)


def _frame_index(value: Any) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu" or int(array) < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    return int(array)


def _estimate(arrays: Mapping[str, Any]) -> JointEstimate:
    if "joints_m" not in arrays:
        raise ValueError("joint NPZ requires 'joints_m' with shape (21, 3), in world metres")
    joints = _real_array(arrays["joints_m"], "joints_m")
    if joints.shape != (21, 3):
        raise ValueError("joints_m must have shape (21, 3)")
    confidence = _real_array(arrays.get("confidence", np.ones(21)), "confidence")
    if (confidence.shape != (21,) or not np.isfinite(confidence).all()
            or ((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("confidence must have shape (21,) and finite values in [0, 1]")
    mask = _bool_array(arrays.get("valid_mask", confidence > 0), (21,), "valid_mask")
    if np.isinf(joints).any() or not np.isfinite(joints[mask]).all():
        raise ValueError("joints_m must be finite at valid joints; infinity is never allowed")
    if (confidence[mask] <= 0).any():
        raise ValueError("valid joints require positive confidence")
    valid = bool(_bool_array(arrays.get("valid", np.array(mask.any())), (), "valid"))
    return JointEstimate(joints_m=joints, confidence=confidence, valid_mask=mask, valid=valid)


def _read_estimates(path: Path) -> tuple[list[JointEstimate], bool]:
    """Validate every frame before returning a cached precomputed sequence."""
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("expected an NPZ archive, not an NPY array")
        with archive:
            # Read only protocol fields; upstream pickle-based .npy/.pkl is unsupported.
            names = ("schema_version", "joints_m", "confidence", "valid_mask", "valid")
            arrays = {name: archive[name] for name in names if name in archive}
        if "schema_version" in arrays:
            version = np.asarray(arrays.pop("schema_version"))
            if version.shape != () or version.dtype.kind not in "iu" or int(version) != NPZ_SCHEMA_VERSION:
                raise ValueError(f"schema_version must be integer {NPZ_SCHEMA_VERSION}")
        if "joints_m" not in arrays:
            raise ValueError("joint NPZ requires 'joints_m'")
        joints = arrays["joints_m"]
        if joints.shape == (21, 3):
            return [_estimate(arrays)], False
        if joints.ndim != 3 or joints.shape[1:] != (21, 3) or joints.shape[0] < 1:
            raise ValueError("joints_m must have shape (21, 3) or (T, 21, 3), T >= 1")
        count = joints.shape[0]
        shapes = {"confidence": (count, 21), "valid_mask": (count, 21), "valid": (count,)}
        for name, shape in shapes.items():
            if name in arrays and arrays[name].shape != shape:
                raise ValueError(f"sequence {name} must have shape {shape}; no implicit broadcast")
        return [_estimate({name: value[index] for name, value in arrays.items()})
                for index in range(count)], True
    except (OSError, ValueError, TypeError, EOFError, BadZipFile) as exc:
        raise ValueError(f"Invalid joint NPZ {path}: {exc}") from exc


def load_joint_estimate(path: str | Path, *, frame_index: int | None = None) -> JointEstimate:
    """Load a single estimate, or explicitly select a row of a sequence NPZ."""
    estimates, sequence = _read_estimates(_require_file(path, "Precomputed joints NPZ"))
    index = _select_frame(estimates, sequence, frame_index)
    estimate = estimates[index]
    return JointEstimate(estimate.joints_m, estimate.confidence, estimate.valid_mask, estimate.valid)


def _select_frame(estimates: list[JointEstimate], sequence: bool, index: Any) -> int:
    if not sequence:
        if index is not None:
            _frame_index(index)
        return 0
    if index is None:
        raise ValueError("a sequence joint NPZ requires observation.frame_index (or a mapping key)")
    index = _frame_index(index)
    if index >= len(estimates):
        raise IndexError(f"frame_index {index} outside precomputed sequence of {len(estimates)} frames")
    return index


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _previous_estimate(previous_state: Any) -> JointEstimate:
    joints = _field(previous_state, "joints_m")
    if joints is None:
        return _estimate({"joints_m": previous_state})
    arrays = {"joints_m": joints}
    for name in ("confidence", "valid_mask", "valid"):
        value = _field(previous_state, name)
        if value is not None:
            arrays[name] = value
    return _estimate(arrays)


class HOTrackInitializer:
    """Read precomputed joints or invoke an explicitly configured NPZ worker.

    Exactly one of ``joints_npz`` and ``command`` is required. ``command`` is an
    argv sequence containing ``{input}`` and ``{output}`` placeholders; optional
    ``{weights}`` refers to a checked, nonempty local ``weights_path`` file.
    Weights are checked for presence only; the user's worker validates their
    format/architecture. No resources are downloaded or model APIs invented.

    ``estimate_joints`` accepts a cloud ndarray [N,3] or a mapping/object with
    ``points_m`` and optional ``frame_index``. Precomputed mode needs only the
    index for sequences; ``None`` is sufficient for a single-frame NPZ.
    """

    def __init__(
        self, *, joints_npz: str | Path | None = None,
        command: Sequence[str] | None = None, weights_path: str | Path | None = None,
        timeout_s: float = 120.0, cwd: str | Path | None = None,
    ) -> None:
        if (joints_npz is None) == (command is None):
            raise ValueError("configure exactly one of joints_npz or command")
        if isinstance(timeout_s, bool) or not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.timeout_s = float(timeout_s)
        self.cwd = None if cwd is None else Path(cwd).expanduser().resolve()
        if self.cwd is not None and not self.cwd.is_dir():
            raise FileNotFoundError(f"Worker cwd is missing or not a directory: {self.cwd}")
        self.command: tuple[str, ...] | None = None
        self.weights_path: Path | None = None
        self._estimates: list[JointEstimate] = []
        self._sequence = False
        if joints_npz is not None:
            if weights_path is not None:
                raise ValueError("weights_path is only applicable with command")
            path = _require_file(joints_npz, "Precomputed joints NPZ")
            self._estimates, self._sequence = _read_estimates(path)
            mode = "precomputed joints NPZ"
        else:
            if (isinstance(command, (str, bytes)) or not isinstance(command, Sequence)
                    or not command or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in command)):
                raise ValueError("command must be a nonempty argv sequence of nonempty strings")
            if not all(any(token in arg for arg in command) for token in ("{input}", "{output}")):
                raise ValueError("command must contain both {input} and {output} placeholders")
            if weights_path is not None:
                self.weights_path = _require_file(weights_path, "HOTrack/worker weights")
            if any("{weights}" in arg for arg in command) and self.weights_path is None:
                raise ValueError("command uses {weights} but weights_path was not supplied")
            self.command = tuple(command)
            mode = "user-provided external NPZ process"
        logger.info("Generic joint initializer configured: %s. Official HOTrack inference integration is not implemented.", mode)

    def estimate_joints(self, observation: Any, previous_state: Any = None) -> JointEstimate:
        """Return validated joints; never substitute previous joints after failure.

        Invalid observations/results raise ValueError; worker startup, timeout,
        nonzero exit, missing output, or malformed output raise HOTrackProcessError.
        A worker can report a normal detection failure via ``valid=False``.
        """
        index = _field(observation, "frame_index")
        if self.command is None:
            estimate = self._estimates[_select_frame(self._estimates, self._sequence, index)]
            return JointEstimate(estimate.joints_m, estimate.confidence, estimate.valid_mask, estimate.valid)
        points = observation if isinstance(observation, np.ndarray) else _field(observation, "points_m")
        if points is None:
            raise ValueError("observation must be an ndarray or provide points_m in world metres")
        points = _real_array(points, "points_m")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("points_m must be finite, nonempty, and have shape (N, 3)")
        arrays = {"schema_version": np.array(NPZ_SCHEMA_VERSION, dtype=np.int64), "points_m": points}
        if index is not None:
            arrays["frame_index"] = np.array(_frame_index(index), dtype=np.int64)
        if previous_state is not None:
            previous = _previous_estimate(previous_state)
            arrays.update(previous_joints_m=previous.joints_m, previous_confidence=previous.confidence,
                          previous_valid_mask=previous.valid_mask, previous_valid=np.array(previous.valid))
        if self.weights_path is not None:
            _require_file(self.weights_path, "HOTrack/worker weights")
        with tempfile.TemporaryDirectory(prefix="kinect-mano-initializer-") as directory:
            input_path, output_path = Path(directory) / "input.npz", Path(directory) / "output.npz"
            np.savez(input_path, **arrays)
            replacements = {"{input}": str(input_path), "{output}": str(output_path)}
            if self.weights_path is not None:
                replacements["{weights}"] = str(self.weights_path)
            argv = list(self.command)
            for token, value in replacements.items():
                argv = [arg.replace(token, value) for arg in argv]
            # Worker logs go to temporary files: large model logs do not fill RAM.
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                try:
                    result = subprocess.run(argv, cwd=self.cwd, stdin=subprocess.DEVNULL,
                                            stdout=stdout, stderr=stderr, timeout=self.timeout_s,
                                            shell=False, check=False)
                except subprocess.TimeoutExpired as exc:
                    raise HOTrackProcessError(f"Initializer worker timed out after {self.timeout_s:g}s") from exc
                except OSError as exc:
                    raise HOTrackProcessError(f"Cannot start initializer worker {argv[0]!r}: {exc}") from exc
                if result.returncode:
                    stderr.seek(0, 2)
                    stderr.seek(max(0, stderr.tell() - 4096))
                    detail = stderr.read().decode("utf-8", errors="replace").strip()
                    raise HOTrackProcessError(f"Initializer worker exited with code {result.returncode}: {detail}")
            try:
                path = _require_file(output_path, "Initializer worker output NPZ")
                estimates, sequence = _read_estimates(path)
                if sequence:
                    raise ValueError("worker output must contain a single frame, not a sequence")
                return estimates[0]
            except (OSError, ValueError) as exc:
                raise HOTrackProcessError(f"Invalid initializer worker output: {exc}") from exc
