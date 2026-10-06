"""Hand synergies: PCA over demonstrated DG5F postures."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lerobot_robot_dg5f.constants import JOINT_NAMES


@dataclass
class HandSynergies:
    mean_deg: np.ndarray                  # (20,)
    components: np.ndarray                # (k, 20), orthonormal rows
    explained_variance_ratio: np.ndarray  # (k,)

    @property
    def k(self) -> int:
        return self.components.shape[0]

    @classmethod
    def fit(cls, hand_deg, k: int) -> "HandSynergies":
        data = np.asarray(hand_deg, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(JOINT_NAMES):
            raise ValueError(f"Expected an (N, {len(JOINT_NAMES)}) array of hand postures")
        if not 1 <= k <= min(data.shape):
            raise ValueError("k must be between 1 and min(N, 20)")
        mean = data.mean(axis=0)
        _, singular, vt = np.linalg.svd(data - mean, full_matrices=False)
        variance = singular**2
        total = variance.sum() or 1.0
        return cls(mean, vt[:k].copy(), variance[:k] / total)

    def to_joint_offset(self, dz) -> np.ndarray:
        """Synergy coordinates (k,) -> joint offset in degrees (20,)."""
        return self.components.T @ np.asarray(dz, dtype=np.float64)

    def project_offset(self, dq) -> np.ndarray:
        """Joint offset in degrees (20,) -> synergy coordinates (k,)."""
        return self.components @ np.asarray(dq, dtype=np.float64)

    def save(self, path) -> None:
        np.savez(path, mean_deg=self.mean_deg, components=self.components,
                 explained_variance_ratio=self.explained_variance_ratio)

    @classmethod
    def load(cls, path) -> "HandSynergies":
        with np.load(path) as data:
            return cls(data["mean_deg"], data["components"], data["explained_variance_ratio"])


def hand_actions_from_dataset(root) -> np.ndarray:
    """DG5F joint targets (N, 20) from the action column of a local LeRobot v3 dataset."""
    import pandas as pd

    root = Path(root)
    names = json.loads((root / "meta" / "info.json").read_text())["features"]["action"]["names"]
    # The DG5F recorder names actions "rj_dg_1_1"; the Robot API names them "rj_dg_1_1.pos".
    index = {name.removesuffix(".pos"): i for i, name in enumerate(names)}
    columns = [index[joint] for joint in JOINT_NAMES]
    files = sorted((root / "data").rglob("*.parquet"))
    actions = np.stack(pd.concat(pd.read_parquet(f, columns=["action"]) for f in files)["action"].to_numpy())
    return actions[:, columns]
