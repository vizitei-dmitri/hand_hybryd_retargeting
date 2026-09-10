"""Validate local training data by reopening it with the official LeRobot API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constants import BROKEN_PINKY_INDEX, JOINT_NAMES
from .dataset_recording import VECTOR_KEYS, dataset_class, feature_schema


def check_dataset(path, pinky_tolerance=1e-5):
    path = Path(path).resolve()
    if not np.isfinite(pinky_tolerance) or pinky_tolerance < 0:
        raise ValueError("pinky_tolerance must be finite and nonnegative")
    if not (path / "meta/info.json").is_file():
        raise ValueError(f"Not a local LeRobotDataset: {path} (meta/info.json missing)")
    manifest = json.loads((path / "meta/dg5f_recording.json").read_text())
    info = json.loads((path / "meta/info.json").read_text())
    if info.get("total_episodes", 0) < 1:
        raise ValueError("Dataset has no saved episodes; record and finish a nonempty episode first")
    dataset = dataset_class()(repo_id=manifest["repo_id"], root=path, download_videos=False)
    try:
        for key, expected in feature_schema().items():
            feature = dataset.features.get(key)
            if feature is None or tuple(feature["shape"]) != expected["shape"] or feature["dtype"] != expected["dtype"]:
                raise ValueError(f"Invalid feature {key}: {feature}; expected {expected}")
            if key in VECTOR_KEYS and tuple(feature["names"]) != JOINT_NAMES:
                raise ValueError(f"Wrong joint order in {key}")
        if dataset.meta.robot_type != "dg5f":
            raise ValueError("Wrong robot_type")
        ranges = {key: [np.full(20, np.inf), np.full(20, -np.inf)]
                  for key in ("observation.state", "action")}
        episode_counts = {}
        segments = {}
        sample = None
        # __getitem__ also exercises official image decoding and task lookup.
        for index in range(len(dataset)):
            frame = dataset[index]
            if sample is None:
                sample = {key: np.asarray(frame[key]).tolist() for key in VECTOR_KEYS}
                sample["task"] = frame["task"]
            for key, feature in dataset.features.items():
                if feature["dtype"] == "string":
                    continue
                value = np.asarray(frame[key])
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"NaN/Inf at frame {index}, {key}")
                if key in VECTOR_KEYS and value.shape != (20,):
                    raise ValueError(f"Wrong shape at frame {index}, {key}: {value.shape}")
                if feature["dtype"] == "image":
                    h, w, c = feature["shape"]
                    if value.shape != (c, h, w):
                        raise ValueError(f"Wrong decoded image shape at frame {index}: {key} {value.shape}")
            if abs(float(frame["action"][BROKEN_PINKY_INDEX])) > pinky_tolerance:
                raise ValueError(f"Nonzero rj_dg_5_1 effective action at frame {index}")
            for key, (minimum, maximum) in ranges.items():
                values = np.asarray(frame[key])
                np.minimum(minimum, values, out=minimum)
                np.maximum(maximum, values, out=maximum)
            episode = int(frame["episode_index"].item())
            expected_index = episode_counts.get(episode, 0)
            if int(frame["frame_index"].item()) != expected_index:
                raise ValueError(f"Non-contiguous frame_index in episode {episode}")
            if abs(float(frame["timestamp"].item()) - expected_index / dataset.fps) > 1e-4:
                raise ValueError(f"Incorrect nominal timestamp at frame {index}")
            episode_counts[episode] = expected_index + 1
            segments.setdefault(episode, set()).add(int(frame["recording.segment_index"].item()))
        if len(episode_counts) != dataset.num_episodes or len(dataset) != info["total_frames"]:
            raise ValueError("Metadata/data episode or frame count mismatch")
        if sorted(episode_counts) != list(range(dataset.num_episodes)):
            raise ValueError("Non-contiguous episode indices")
        for episode in range(dataset.num_episodes):
            metadata = dataset.meta.episodes[episode]
            if episode_counts[episode] != metadata["length"]:
                raise ValueError(f"Incorrect episode length metadata: {episode}")
        return {
            "path": str(path), "repo_id": manifest["repo_id"], "episodes": dataset.num_episodes,
            "frames": len(dataset), "fps": dataset.fps, "features": dataset.features,
            "tasks": dataset.meta.tasks.index.tolist(), "cameras": dataset.meta.camera_keys,
            "joint_ranges_degree": {key: {"min": low.tolist(), "max": high.tolist()}
                                    for key, (low, high) in ranges.items()},
            "segments_per_episode": {key: len(value) for key, value in segments.items()},
            "interrupted_episodes": [item["episode_index"] for item in manifest["episodes"]
                                     if item.get("interrupted")],
            "first_frame": sample, "finite": True, "disabled_joint_zero": True,
        }
    finally:
        dataset.finalize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Local dataset directory or repo-id under --root")
    parser.add_argument("--root", default="/workspace/lerobot_datasets")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.dataset).expanduser()
    if not path.is_dir() and not path.is_absolute():
        path = Path(args.root) / path
    try:
        report = check_dataset(path)
    except Exception as error:
        print(f"DATASET CHECK FAILED: {error}\nStop the recorder before checking. No files were deleted.")
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    for key in ("path", "episodes", "frames", "fps", "tasks", "cameras"):
        print(f"{key}: {report[key]}")
    for key, feature in report["features"].items():
        print(f"{key}: {list(feature['shape'])} {feature['dtype']}")
    print("Joint ranges (degrees): state min/max | action min/max")
    ranges = report["joint_ranges_degree"]
    for index, joint in enumerate(JOINT_NAMES):
        state, action = ranges["observation.state"], ranges["action"]
        print(f"  {joint}: {state['min'][index]:8.3f} {state['max'][index]:8.3f} | "
              f"{action['min'][index]:8.3f} {action['max'][index]:8.3f}")
    print("segments_per_episode:", report["segments_per_episode"])
    print("interrupted_episodes:", report["interrupted_episodes"])
    print("OK: all numeric values finite; state/action 20D; rj_dg_5_1 action zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
