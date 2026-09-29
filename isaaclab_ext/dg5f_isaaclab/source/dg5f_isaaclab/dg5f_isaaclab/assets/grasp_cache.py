"""Validated fingertip-grasp cache: the reset distribution for the precision-grasp task.

Written by scripts/fingertip_grasp_search.py as a plain .npz (no pickle). Each entry is one
settled grasp that survived releasing the cube on the fingertips, so the entries are states,
not a policy: Chen et al. (CoRL 2021) obtain the same thing from a separate lifting policy,
and the 2026 in-hand works call it a grasp cache. The configured cfg grasp is a power grasp
and cannot serve this purpose, see logs/fingertip_grasp/.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from collections.abc import Sequence

import numpy as np

DEFAULT_GRASP_CACHE_PATH = Path(__file__).parent / "data/fingertip_grasp_cache.npz"
REQUIRED_ARRAYS = ("q", "q_cmd", "cube_pos", "cube_quat", "joint_names")
# PhysX settles a joint slightly outside its limit under contact. The same scale is already
# documented elsewhere in this project (cfg disabled_joint_lock_armature: "up to 0.065 deg" in
# random rollouts; grasp_hold_sanity uses a 0.25 deg tolerance for the locked joint), so anything
# inside this band is accepted and then clamped, while a real ordering/unit error still fails.
JOINT_LIMIT_TOLERANCE_RAD = math.radians(0.25)


@dataclass(frozen=True)
class GraspCache:
    """Palm-frame cube poses and 20-DOF joint states, in URDF joint order."""

    source: str
    sha256: str
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray      # (n, 20) rad
    joint_command: np.ndarray  # (n, 20) rad
    cube_pos: np.ndarray       # (n, 3) m, palm frame
    cube_quat: np.ndarray      # (n, 4) wxyz, palm frame

    def __len__(self) -> int:
        return int(self.joint_pos.shape[0])


def load_grasp_cache(
    path: str | Path,
    expected_joint_names: Sequence[str],
    joint_limits: Sequence[tuple[float, float]] | None = None,
) -> GraspCache:
    """Validate array presence, shapes, joint order, finiteness, limits and quaternion norms."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    # allow_pickle stays False: the cache must be plain arrays.
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_ARRAYS if name not in data]
        if missing:
            raise ValueError(f"Grasp cache {path} is missing arrays: {missing}")
        arrays = {name: data[name] for name in REQUIRED_ARRAYS}

    names = tuple(str(value) for value in arrays["joint_names"])
    if names != tuple(expected_joint_names):
        raise ValueError(f"Grasp cache joint order does not match the URDF order: {names}")
    count = arrays["q"].shape[0]
    if count == 0:
        raise ValueError(f"Grasp cache {path} is empty")
    shapes = {"q": (count, len(names)), "q_cmd": (count, len(names)),
              "cube_pos": (count, 3), "cube_quat": (count, 4)}
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Grasp cache array {name} has shape {arrays[name].shape}, expected {shape}")
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"Grasp cache array {name} has non-finite values")

    norms = np.linalg.norm(arrays["cube_quat"], axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Grasp cache cube_quat is not unit-norm")
    if joint_limits is not None:
        lower = np.array([limit[0] for limit in joint_limits], dtype=arrays["q"].dtype)
        upper = np.array([limit[1] for limit in joint_limits], dtype=arrays["q"].dtype)
        for name in ("q", "q_cmd"):
            excess = np.maximum(lower - arrays[name], arrays[name] - upper).max()
            if excess > JOINT_LIMIT_TOLERANCE_RAD:
                raise ValueError(
                    f"Grasp cache array {name} leaves the URDF joint limits by "
                    f"{math.degrees(float(excess)):.3f} deg, above the "
                    f"{math.degrees(JOINT_LIMIT_TOLERANCE_RAD):.2f} deg settling tolerance")
            # Clamped so a reset target is strictly legal regardless of how it settled.
            arrays[name] = np.clip(arrays[name], lower, upper)

    return GraspCache(
        source=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        joint_names=names,
        joint_pos=np.ascontiguousarray(arrays["q"], dtype=np.float32),
        joint_command=np.ascontiguousarray(arrays["q_cmd"], dtype=np.float32),
        cube_pos=np.ascontiguousarray(arrays["cube_pos"], dtype=np.float32),
        cube_quat=np.ascontiguousarray(arrays["cube_quat"], dtype=np.float32),
    )
