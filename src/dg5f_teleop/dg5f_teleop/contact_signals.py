"""Read-only contact wire contract and MANO fingertip geometry; no retargeting."""

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")
TIP_INDICES = (4, 8, 12, 16, 20)
PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4))
PAIR_NAMES = tuple(f"{FINGERS[a]}-{FINGERS[b]}" for a, b in PAIRS)
HYBRID_LAYOUT = "v1:stamp_ros_s|weights[thumb,index,middle,ring,little]|thumb_distances_m[index,middle,ring,little]"
PROXIMITY_LAYOUT = "v1:stamp_ros_s|distances_m[" + ",".join(PAIR_NAMES) + "]"


def fingertip_distances(mano_points):
    points = np.asarray(mano_points, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Expected 21 finite MANO landmarks")
    tips = points[list(TIP_INDICES)]
    return np.array([np.linalg.norm(tips[b] - tips[a]) for a, b in PAIRS])


def decode_contact_packet(data, *, hybrid=False):
    values = np.asarray(data, dtype=np.float64)
    expected = 10 if hybrid else 8
    if values.shape != (expected,) or not np.all(np.isfinite(values)) or values[0] <= 0:
        raise ValueError(f"Invalid v1 contact packet: expected {expected} finite values and a timestamp")
    if hybrid:
        if np.any((values[1:6] < 0) | (values[1:6] > 1)) or np.any(values[6:] < 0):
            raise ValueError("Invalid hybrid weights/distances")
    elif np.any(values[1:] < 0):
        raise ValueError("Negative fingertip distance")
    return values.copy()


def proximity_weights(distances, thumb_start=0.055, thumb_full=0.025,
                      adjacent_start=0.030, adjacent_full=0.012):
    distances = np.asarray(distances, dtype=np.float64)
    start = np.array([thumb_start] * 4 + [adjacent_start] * 3)
    full = np.array([thumb_full] * 4 + [adjacent_full] * 3)
    if np.any(start <= full) or np.any(full < 0):
        raise ValueError("Proximity thresholds must satisfy 0 <= full < start")
    x = np.clip((start - distances) / (start - full), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
