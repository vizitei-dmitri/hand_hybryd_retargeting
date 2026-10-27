"""Visibility-aware observed-point to frontmost-triangle correspondences."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SurfaceView:
    points_m: np.ndarray
    camera_center_m: np.ndarray
    camera_id: str = "camera"

    def __post_init__(self):
        self.points_m = np.asarray(self.points_m, dtype=np.float64)
        self.camera_center_m = np.asarray(self.camera_center_m, dtype=np.float64)
        if self.points_m.ndim != 2 or self.points_m.shape[1] != 3 or not np.isfinite(self.points_m).all():
            raise ValueError("points_m must be finite [N,3]")
        if self.camera_center_m.shape != (3,) or not np.isfinite(self.camera_center_m).all():
            raise ValueError("camera_center_m must be finite [3]")


@torch.no_grad()
def raycast_front_faces(points, vertices, faces, camera_center, *, chunk_size=128, max_distance_m=0.04):
    """Intersect observation rays; select only the first positive mesh hit.

    Correspondence selection is detached (ICP). Both winding directions are
    accepted. Misses and incompatible depth are excluded, never zero-filled.
    No unobserved model-to-cloud attraction is present.
    """
    triangles = vertices[faces]
    a, b, c = triangles.unbind(1)
    e1, e2 = b - a, c - a
    answers = []
    for points_chunk in points.split(chunk_size):
        ray = points_chunk - camera_center
        length = torch.linalg.vector_norm(ray, dim=-1)
        direction = ray / length.clamp_min(1e-12)[:, None]
        h = torch.linalg.cross(direction[:, None, :].expand(-1, len(faces), -1), e2[None].expand(len(ray), -1, -1))
        det = (e1[None] * h).sum(-1)
        safe_det = torch.where(det.abs() > 1e-10, det, torch.ones_like(det))
        offset = camera_center - a
        u = (offset[None] * h).sum(-1) / safe_det
        q = torch.linalg.cross(offset, e1)
        v = (direction[:, None] * q[None]).sum(-1) / safe_det
        distance = (e2 * q).sum(-1)[None] / safe_det
        valid = (det.abs() > 1e-10) & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (distance > 1e-6)
        depth, index = torch.where(valid, distance, torch.inf).min(1)
        good = torch.isfinite(depth) & ((depth - length).abs() <= max_distance_m)
        answers.append(torch.where(good, index, -1))
    return torch.cat(answers) if answers else faces.new_empty(0)


def point_triangle_squared_distance(points, triangles):
    """Exact closest distance to paired triangles (interior or any edge)."""
    a, b, c = triangles.unbind(-2)
    ab, ac, ap = b - a, c - a, points - a
    normal = torch.linalg.cross(ab, ac)
    normal2 = normal.square().sum(-1).clamp_min(1e-20)
    dot = (ap * normal).sum(-1)
    projected = points - normal * (dot / normal2)[..., None]
    v2 = projected - a
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (v2 * ab).sum(-1), (v2 * ac).sum(-1)
    denom = (d00 * d11 - d01.square()).clamp_min(1e-20)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    distances = [torch.where(inside, dot.square() / normal2, torch.inf)]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = ((points - start) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-20)
        nearest = start + t.clamp(0, 1)[..., None] * edge
        distances.append((points - nearest).square().sum(-1))
    return torch.stack(distances, -1).min(-1).values


def robust_huber(residual, delta=1.0):
    absolute = residual.abs()
    return torch.where(absolute <= delta, 0.5 * residual.square(), delta * (absolute - 0.5 * delta))
