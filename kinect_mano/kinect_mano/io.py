"""Portable non-pickle observations and parameter/geometry reconstruction export."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .fitting import Observation
from .surface import SurfaceView


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_observations(path, observations, *, truth=None, metadata=None):
    if not observations:
        raise ValueError("No observations to save")
    ids = sorted({view.camera_id for obs in observations for view in obs.views})
    arrays = dict(timestamps_s=np.array([o.timestamp_s for o in observations]),
                  joints_m=np.stack([o.joints_m for o in observations]), confidence=np.stack([o.confidence for o in observations]),
                  valid=np.array([o.valid for o in observations]),
                  metadata_json=np.array(json.dumps(dict(metadata or {}, camera_ids=ids), allow_nan=False)))
    arrays["preprocessing_ms"] = np.array([getattr(o,"preprocessing_ms",0.) for o in observations])
    arrays["segmentation_json"] = np.array([json.dumps(getattr(o,"segmentation",[])) for o in observations])
    for index, camera_id in enumerate(ids):
        views = [next((v for v in o.views if v.camera_id == camera_id), None) for o in observations]
        count = np.array([len(v.points_m) if v else 0 for v in views])
        points = np.zeros((len(observations), int(count.max()), 3), dtype=np.float32)
        centers = np.zeros((len(observations), 3))
        for frame, view in enumerate(views):
            if view is not None:
                points[frame, :count[frame]] = view.points_m
                centers[frame] = view.camera_center_m
        arrays[f"cloud_{index}_m"] = points
        arrays[f"cloud_{index}_count"] = count
        arrays[f"camera_{index}_center_m"] = centers
    for key, value in (truth or {}).items():
        arrays[f"truth_{key}"] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_observations(path):
    with np.load(path, allow_pickle=False) as archive:
        times, joints, confidence = (archive[key] for key in ("timestamps_s", "joints_m", "confidence"))
        if times.ndim != 1 or len(times) == 0 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
            raise ValueError("Observations require nonempty increasing finite timestamps_s")
        if joints.shape != (len(times), 21, 3) or confidence.shape != (len(times), 21):
            raise ValueError("Observation sequence dimensions do not match")
        if "valid" in archive:
            valid_values=archive["valid"]
            if valid_values.shape != (len(times),) or valid_values.dtype.kind not in "biu" or not np.isin(valid_values,[0,1]).all():
                raise ValueError("Observation valid must be boolean/binary [F]")
        metadata = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive else {}
        observations = []
        for frame, timestamp in enumerate(times):
            views = []
            for index, camera_id in enumerate(metadata.get("camera_ids", [])):
                count = int(archive[f"cloud_{index}_count"][frame])
                points = archive[f"cloud_{index}_m"]
                if points.ndim != 3 or points.shape[0] != len(times) or points.shape[2] != 3 or not 0 <= count <= points.shape[1]:
                    raise ValueError("Invalid point cloud/count shape")
                views.append(SurfaceView(points[frame, :count], archive[f"camera_{index}_center_m"][frame], camera_id))
            observation = Observation(float(timestamp), joints[frame], confidence[frame], tuple(views), str(frame),valid=bool(archive["valid"][frame]) if "valid" in archive else True)
            observation.preprocessing_ms = float(archive["preprocessing_ms"][frame]) if "preprocessing_ms" in archive else 0.
            observation.segmentation = json.loads(str(archive["segmentation_json"][frame])) if "segmentation_json" in archive else []
            observations.append(observation)
        truth = {name[6:]: archive[name].copy() for name in archive.files if name.startswith("truth_")}
    return observations, truth, metadata


def state_mapping(state, faces, *, frame_index=0, observation=None, metadata=None):
    result = asdict(state)
    result.update(faces=faces, frame_index=frame_index,metadata=metadata or {})
    if observation is not None:
        result["clouds_m"] = {view.camera_id: view.points_m for view in observation.views}
        result["cloud_m"] = np.concatenate([view.points_m for view in observation.views]) if observation.views else np.empty((0, 3))
    return json_safe(result)


def save_reconstruction(directory, states, model, config, *, observations=None, manifest=None, overwrite=False):
    if not states:
        raise ValueError("No reconstruction states")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reconstruction.npz"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Use a new output directory; refusing overwrite: {path}")
    quality = [state.quality | {"status": state.status} for state in states]
    arrays = {name: np.stack([getattr(state, name) for state in states]) for name in ("theta", "global_orient", "translation_m", "joints_m", "vertices_m")}
    arrays.update(timestamps_s=np.array([s.timestamp_s for s in states]), beta=states[0].beta,
                  valid=np.array([s.valid for s in states]), faces=model.faces.cpu().numpy(),
                  quality_json=np.array(json.dumps(json_safe(quality), allow_nan=False)),
                  metadata_json=np.array(json.dumps(dict(model.metadata, model_sha256=model.sha256, coordinate_axes="optical", joint_order="wrist,thumb,index,middle,ring,little", translation="wrist_position_m", pose="45_local_axis_angle_flat_hand; separate_global_axis_angle"))))
    if observations is not None:
        max_points = max((sum(len(v.points_m) for v in o.views) for o in observations), default=0)
        clouds = np.full((len(states), max_points, 3), np.nan, dtype=np.float32)
        for f, obs in enumerate(observations):
            points = np.concatenate([v.points_m for v in obs.views]) if obs.views else np.empty((0, 3))
            clouds[f, :len(points)] = points
        arrays["cloud_m"] = clouds
        camera_ids = sorted({v.camera_id for o in observations for v in o.views})
        max_per_camera = max((len(v.points_m) for o in observations for v in o.views),default=0)
        per_camera = np.full((len(states),len(camera_ids),max_per_camera,3),np.nan,dtype=np.float32)
        for frame,obs in enumerate(observations):
            for view in obs.views:
                per_camera[frame,camera_ids.index(view.camera_id),:len(view.points_m)] = view.points_m
        arrays["cloud_camera_ids"] = np.array(camera_ids)
        arrays["clouds_per_camera_m"] = per_camera
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory,prefix=".reconstruction-",suffix=".npz",delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(directory / "quality.json", quality)
    write_json(directory / "config.json", asdict(config))
    write_json(directory / "manifest.json", dict(manifest or {}, model=model.metadata, model_sha256=model.sha256, synthetic_test_model=model.model_type != "MANO"))
    return path
