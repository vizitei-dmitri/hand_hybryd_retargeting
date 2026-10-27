"""Device-independent RGB-D preprocessing and recoverable sequence fitting."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import threading
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .data import match_frames_by_timestamp
from .fitting import FitState, Observation
from .geometry import frame_point_cloud
from .initializers import fuse_joint_estimates
from .io import json_safe, write_json
from .surface import SurfaceView


def filtered_cloud(points, max_points, *, seed=0):
    points = np.asarray(points)
    points = points[np.isfinite(points).all(1)]
    if len(points) > max_points * 4:
        points = points[np.random.default_rng(seed).choice(len(points), max_points * 4, replace=False)]
    if len(points) > 16:
        distances = cKDTree(points).query(points, k=min(8, len(points)))[0][:, 1:].mean(1)
        median = np.median(distances)
        spread = np.median(np.abs(distances - median))
        points = points[distances <= median + max(6 * spread, .002)]
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    return points


class RGBDProcessor:
    def __init__(self, calibrations, *, detector=None, precomputed=None, allow_polygon_roi=False, points_per_camera=2048):
        if detector is None and precomputed is None:
            raise ValueError("A local RGB-D initializer or precomputed metric joints are required")
        self.calibrations = calibrations
        self.detector, self.precomputed = detector, precomputed
        self.allow_polygon_roi = allow_polygon_roi
        self.points_per_camera = points_per_camera

    def process_pair(self, frame_a, frame_b, previous_state=None):
        start = perf_counter()
        estimates, views = [], []
        segmentation = []
        for frame in (frame_a, frame_b):
            calibration = self.calibrations[frame.camera_id]
            if self.detector is not None:
                estimate = self.detector.estimate_joints(frame, calibration, previous_state)
                estimates.append(estimate)
                if frame.mask is None and self.detector.last_mask is not None:
                    if not self.allow_polygon_roi:
                        raise ValueError("No semantic hand mask. Supply saved masks or explicitly --allow-polygon-roi for free-hand experiments (objects are not segmented)")
                    # Do not mutate the stored input frame when generating an ROI.
                    from .data import CameraFrame
                    frame = CameraFrame(frame.camera_id, frame.timestamp_s, frame.color, frame.depth_m, self.detector.last_mask, frame.metadata)
                    segmentation.append("polygon_roi_not_semantic")
                else:
                    segmentation.append("provided_mask" if frame.mask is not None else "no_hand")
            elif frame.mask is None:
                raise ValueError("Precomputed-joint RGB-D processing requires a hand mask per camera")
            if frame.mask is None:
                continue
            points = frame_point_cloud(frame, calibration, min_depth_m=.1, max_depth_m=2.5)
            # Reject distant background inside an imperfect polygon ROI.
            if estimates and estimates[-1].valid:
                valid_joints = estimates[-1].joints_m[estimates[-1].valid_mask]
                if len(valid_joints):
                    distance = cKDTree(valid_joints).query(points)[0]
                    points = points[distance < .07]
            points = filtered_cloud(points, self.points_per_camera)
            views.append(SurfaceView(points, calibration.T_world_from_camera[:3, 3], frame.camera_id))
        if self.precomputed is not None:
            estimate = self.precomputed.estimate_joints(frame_a.timestamp_s, previous_state=previous_state)
        else:
            estimate = fuse_joint_estimates(estimates)
        observation = Observation(frame_a.timestamp_s, estimate.joints_m, estimate.confidence, tuple(views),valid=estimate.valid)
        observation.preprocessing_ms = (perf_counter() - start) * 1000
        observation.segmentation = segmentation
        return observation

    def process_recordings(self, recording_a, recording_b, *, tolerance_s=.005, offset_b_s=0.):
        pairs = match_frames_by_timestamp(recording_a.frames, recording_b.frames, tolerance_s=tolerance_s, offset_b_s=offset_b_s)
        if not pairs:
            raise ValueError("No synchronized frame pairs within tolerance; verify clock offset")
        observations = [self.process_pair(a, b) for a, b in pairs]
        sync = [dict(timestamp_a_s=a.timestamp_s, timestamp_b_s=b.timestamp_s, delta_s=b.timestamp_s+offset_b_s-a.timestamp_s) for a, b in pairs]
        return observations, sync


def fit_sequence(fitter, observations, output, *, fingerprint, resume=False, emit=None):
    """Persist completed states per frame; validate inputs before resume."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "checkpoint.jsonl"
    manifest = output / "checkpoint_manifest.json"
    states = []
    if resume:
        if not progress.exists() or not manifest.exists():
            raise ValueError("Resume requires checkpoint.jsonl and checkpoint_manifest.json")
        if json.loads(manifest.read_text()) != json_safe(fingerprint):
            raise ValueError("Checkpoint input/model/configuration mismatch; start a new output directory")
        complete_lines = []
        for line in progress.read_text().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                break  # Interrupted last write is not a completed frame.
            for name in ("theta", "beta", "global_orient", "translation_m", "joints_m", "vertices_m"):
                value[name] = np.array(value[name])
            states.append(FitState(**value))
            complete_lines.append(line)
        if len(states) > len(observations) or any(s.timestamp_s != observations[i].timestamp_s for i, s in enumerate(states)):
            raise ValueError("Checkpoint sequence does not match observations")
        progress.write_text("\n".join(complete_lines) + ("\n" if complete_lines else ""))
        for state in states[-2:]:
            if state.valid:
                fitter.history.append(state)
            else:
                fitter.reset()
    elif progress.exists() or (output / "reconstruction.npz").exists():
        raise FileExistsError("Output already contains a run; use --resume or a new directory")
    else:
        write_json(manifest, fingerprint)
    with progress.open("a", encoding="utf-8") as stream:
        for index, observation in enumerate(observations[len(states):], start=len(states)):
            state = fitter.fit(observation)
            state.quality["preprocessing_ms"] = getattr(observation, "preprocessing_ms", 0.)
            state.quality["segmentation"] = getattr(observation, "segmentation", [])
            states.append(state)
            stream.write(json.dumps(json_safe(asdict(state)), allow_nan=False) + "\n")
            stream.flush()
            if emit:
                emit(index, state, observation)
    return states


class LatestPairSource:
    """Capture independently from inference, with a single overwriteable slot."""

    def __init__(self, capture, *, tolerance_s, offset_b_s=0.):
        self.capture, self.tolerance_s, self.offset_b_s = capture, tolerance_s, offset_b_s
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.latest = None
        self.error = None
        self.sequence = 0
        self.thread = threading.Thread(target=self._run, name="kinect-latest-pair", daemon=True)

    def _run(self):
        try:
            while not self.stop_event.is_set():
                pair = self.capture.read_pair(tolerance_s=self.tolerance_s, offset_b_s=self.offset_b_s, timeout_ms=1000)
                captured_at = perf_counter()
                with self.condition:
                    self.sequence += 1
                    self.latest = (self.sequence, pair, captured_at)
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def start(self):
        self.thread.start()
        return self

    def next(self, previous_sequence=0, timeout_s=3.):
        with self.condition:
            ready = self.condition.wait_for(lambda: self.error is not None or (self.latest is not None and self.latest[0] > previous_sequence), timeout=timeout_s)
            if self.error:
                raise RuntimeError(f"Kinect acquisition failed: {self.error}") from self.error
            if not ready:
                raise TimeoutError("Timed out waiting for a fresh Kinect frame pair")
            return self.latest

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.)
