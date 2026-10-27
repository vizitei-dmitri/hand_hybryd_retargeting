"""Independent numerical fixtures and synthetic observations, all in metres.

The procedural hand below is NOT MANO, contains no MANO assets, and cannot be
used to validate anatomical accuracy. Its array dimensions exercise the same
blend-shape/LBS implementation. Loading it requires ``allow_test_model=True``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .fitting import Observation
from .model import ManoModel
from .surface import SurfaceView, raycast_front_faces


def _loft_faces(rings: int, sides: int, offset: int) -> np.ndarray:
    """Triangulate ring strips with a separate vertex at each closed end."""
    faces = []
    for ring in range(rings - 1):
        for side in range(sides):
            a = offset + ring * sides + side
            b = offset + ring * sides + (side + 1) % sides
            faces.extend(((a, b, b + sides), (a, b + sides, a + sides)))
    for side in range(sides):
        following = (side + 1) % sides
        faces.append((offset + rings * sides, offset + following, offset + side))
        faces.append((offset + rings * sides + 1,
                      offset + (rings - 1) * sides + side,
                      offset + (rings - 1) * sides + following))
    return np.asarray(faces, dtype=np.int64)


def make_fixture_model(path=None, seed=0) -> dict[str, np.ndarray]:
    """Return an independently authored 778-vertex, 16-joint numerical model.

    A capped ellipsoidal palm (128 vertices) overlaps five capped, tapered
    finger tubes (130 vertices each). These six closed components deliberately
    use simple geometry, not an anatomical skin reconstruction. Ten independent
    shape modes control size, palm width/thickness, individual digit lengths,
    digit girth, and thumb splay. Seeded, small pose correctives exercise all
    135 pose features. The skeleton follows the loader's joint/tip convention.

    If ``path`` is provided, save numeric/Unicode arrays to that exact NPZ path
    without pickle. Existing files are refused. The returned dictionary can be
    passed to ``ManoModel(arrays, allow_test_model=True)`` directly.
    """
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(18) / 18
    latitude = np.pi * np.arange(1, 8) / 8
    palm = np.stack([
        np.column_stack((.043 * np.sin(t) * np.cos(angles),
                         np.full(18, .032 - .038 * np.cos(t)),
                         .012 * np.sin(t) * np.sin(angles)))
        for t in latitude
    ]).reshape(-1, 3)
    palm = np.vstack((palm, [0, -.006, 0], [0, .070, 0]))
    vertices = [palm]
    faces = [_loft_faces(7, 18, 0)]
    weights = np.zeros((778, 16), dtype=np.float64)
    weights[:128, 0] = 1
    regressor = np.zeros((16, 778), dtype=np.float64)
    # The wrist is at the origin, expressed as a convex combination of poles.
    regressor[0, 126:128] = [.070 / .076, .006 / .076]
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
    shapedirs = np.zeros((778, 3, 10), dtype=np.float64)
    tips_by_name = {}
    # Skeleton order: index, middle, pinky, ring, thumb. Thumb is radial (-x).
    digits = (
        ("index", 1, (-.026, .062, 0), (-.035, 1, 0), .075, .0082),
        ("middle", 4, (-.008, .068, 0), (0, 1, 0), .084, .0086),
        ("pinky", 7, (.027, .056, 0), (.14, 1, 0), .062, .0065),
        ("ring", 10, (.010, .065, 0), (.055, 1, 0), .078, .0078),
        ("thumb", 13, (-.030, .023, -.001), (-.78, .63, 0), .062, .0095),
    )
    along = np.array([0, .15, .29, .43, .58, .72, .87, .96])
    angles = 2 * np.pi * np.arange(16) / 16
    for digit_index, (name, first_joint, base, direction, length, radius) in enumerate(digits):
        offset = 128 + digit_index * 130
        base, direction = np.asarray(base), np.asarray(direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        across = np.array([direction[1], -direction[0], 0])
        centers = base + along[:, None] * length * direction
        taper = 1 - .38 * along
        radial = (np.cos(angles)[:, None] * across
                  + .8 * np.sin(angles)[:, None] * np.array([0, 0, 1]))
        tube = (centers[:, None] + radius * taper[:, None, None] * radial).reshape(-1, 3)
        tube = np.vstack((tube, base - .003 * direction, base + length * direction))
        vertices.append(tube)
        faces.append(_loft_faces(8, 16, offset))
        for joint, ring in zip(range(first_joint, first_joint + 3), (0, 3, 5)):
            regressor[joint, offset + ring * 16:offset + (ring + 1) * 16] = 1 / 16
        for ring, fraction in enumerate(along):
            # Blend around successive pivots, with the distal cap following DIP.
            if fraction <= .43:
                blend = np.clip((fraction - .20) / .35, 0, 1)
                joint = first_joint
            else:
                blend = np.clip((fraction - .55) / .32, 0, 1)
                joint = first_joint + 1
            ids = slice(offset + ring * 16, offset + (ring + 1) * 16)
            weights[ids, joint] = 1 - blend
            weights[ids, joint + 1] = blend
        weights[offset + 128, first_joint] = 1
        weights[offset + 129, first_joint + 2] = 1
        tips_by_name[name] = offset + 129
        ids = slice(offset, offset + 130)
        axial = np.outer((tube - base) @ direction, direction)
        shapedirs[ids, :, 3 + digit_index] = .10 * axial
        shapedirs[ids, :, 8] = .12 * (tube - base - axial)
        if name == "thumb":
            shapedirs[ids, :, 9] = .10 * np.column_stack(
                (-(tube - base)[:, 1], (tube - base)[:, 0], np.zeros(130)))
    template = np.vstack(vertices)
    shapedirs[:, :, 0] = .065 * template
    shapedirs[:, 0, 1] = .10 * template[:, 0]
    shapedirs[:, 2, 2] = .15 * template[:, 2]
    # Small independent smooth displacements, localized to the affected joint.
    posedirs = np.empty((778, 3, 135), dtype=np.float64)
    for joint in range(1, 16):
        correction = rng.normal(0, 7e-5, (3, 9))
        envelope = weights[:, joint] * (1 + .2 * np.sin(template[:, 1] * 50))
        posedirs[:, :, (joint - 1) * 9:joint * 9] = envelope[:, None, None] * correction
    metadata = dict(model_type="numerical_fixture", handedness="right", units="m",
                    authoring="independent procedural ellipsoid and tapered tubes; NOT MANO",
                    fixture_version=1, seed=int(seed), tips_convention="wrist then thumb/index/middle/ring/pinky")
    arrays = dict(v_template=template, shapedirs=shapedirs, posedirs=posedirs,
                  J_regressor=regressor, weights=weights, parents=parents,
                  faces=np.vstack(faces),
                  tip_vertex_ids=np.array([tips_by_name[name] for name in
                                           ("thumb", "index", "middle", "ring", "pinky")]),
                  hands_mean=np.zeros(45), metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    return arrays


@torch.no_grad()
def _visible_points(vertices, faces, camera_center, count, rng):
    """Uniform perspective rays over the mesh's projected bounding rectangle."""
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    center = torch.as_tensor(camera_center, dtype=vertices.dtype, device=vertices.device)
    forward = vertices.mean(0) - center
    forward /= torch.linalg.vector_norm(forward)
    reference = forward.new_tensor([0., 1., 0.])
    if abs(float(forward @ reference)) > .95:
        reference = forward.new_tensor([1., 0., 0.])
    right = torch.linalg.cross(forward, reference)
    right /= torch.linalg.vector_norm(right)
    up = torch.linalg.cross(right, forward)
    offset = vertices - center
    depth = offset @ forward
    if (depth <= 0).any():
        raise ValueError("Synthetic camera must be outside and in front of the mesh")
    projected = torch.stack((offset @ right / depth, offset @ up / depth), -1)
    low, high = projected.min(0).values.cpu().numpy(), projected.max(0).values.cpu().numpy()
    samples, remaining = [], count
    for _ in range(64):
        uv = torch.as_tensor(rng.uniform(low, high, (max(64, 3 * remaining), 2)),
                             dtype=vertices.dtype, device=vertices.device)
        directions = forward + uv[:, :1] * right + uv[:, 1:] * up
        # A target defines a ray only. Infinite depth tolerance lets the core
        # raycaster select its actual first hit, even on a self-occluded hand.
        indices = raycast_front_faces(center + directions, vertices, faces, center,
                                      max_distance_m=float("inf"))
        hit = indices >= 0
        triangles = vertices[faces[indices[hit]]]
        directions = directions[hit]
        normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0])
        distance = ((triangles[:, 0] - center) * normals).sum(-1) / (directions * normals).sum(-1)
        points = (center + distance[:, None] * directions)[:remaining]
        samples.append(points.cpu().numpy())
        remaining -= len(points)
        if remaining == 0:
            return np.concatenate(samples).astype(np.float64, copy=False)
    raise RuntimeError("Could not generate enough visible mesh rays")


def generate_sequence(model: ManoModel, frames=8, points_per_camera=128, noise_m=.001,
                      joint_noise_m=.004, seed=0):
    """Return ``(observations, truth)`` for a deterministic moving hand.

    ``model`` must already have been explicitly loaded; numerical fixtures thus
    retain the loader's ``allow_test_model`` gate. Two fixed cameras observe
    actual first mesh intersections, with optional isotropic metric noise added
    only after visibility selection. Joint measurements have independent noise.
    Use zero points for inexpensive joint-only experiments. Outliers and missing
    data can be injected into the returned observations by the caller.

    Truth contains per-frame theta[F,45], global_orient[F,3], translation_m[F,3],
    joints_m[F,21,3], vertices_m[F,778,3], one shared beta[10], timestamps_s[F],
    camera_centers_m[2,3], and metadata_json identifying the source model type.
    This synthetic experiment does not measure real-world MANO accuracy.
    """
    for name, value, minimum in (("frames", frames, 1), ("points_per_camera", points_per_camera, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if any(not np.isfinite(value) or value < 0 for value in (noise_m, joint_noise_m)):
        raise ValueError("noise_m and joint_noise_m must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    time = np.arange(frames, dtype=np.float64) / 30
    phase = 2 * np.pi * time
    theta = np.zeros((frames, 15, 3))
    for finger in range(5):
        curl = .27 + .16 * np.sin(phase + .6 * finger)
        for segment, gain in enumerate((1., .8, .55)):
            theta[:, finger * 3 + segment, 0] = gain * curl
        theta[:, finger * 3, 2] = .035 * np.sin(phase + finger)
    theta = theta.reshape(frames, 45)
    beta = np.array([.8, -.55, .4, .7, -.4, .5, -.3, .2, .5, -.2])
    orient = np.column_stack((.12 + .035 * np.sin(phase),
                              -.16 + .04 * np.cos(phase), .08 + .03 * np.sin(phase)))
    translation = np.column_stack((.012 * np.sin(phase),
                                   -.025 + .006 * np.sin(phase), .55 + .008 * np.sin(.7 * phase)))
    arguments = [torch.as_tensor(value, dtype=model.dtype, device=model.device)
                 for value in (theta, beta, orient, translation)]
    with torch.no_grad():
        vertices, joints = model(*arguments)
    vertices_np, joints_np = vertices.cpu().numpy().copy(), joints.cpu().numpy().copy()
    cameras = np.array([[-.22, .10, .05], [.24, .12, .08]])
    observations = []
    for frame in range(frames):
        views = []
        for camera_index, camera in enumerate(cameras):
            points = _visible_points(vertices[frame], model.faces, camera, points_per_camera, rng)
            points = points + rng.normal(0, noise_m, points.shape)
            views.append(SurfaceView(points, camera.copy(), f"synthetic_{camera_index}"))
        measured = joints_np[frame] + rng.normal(0, joint_noise_m, (21, 3))
        observations.append(Observation(float(time[frame]), measured, np.ones(21),
                                        tuple(views), f"synthetic_{frame:04d}"))
    metadata = dict(model_type=model.model_type, handedness=model.metadata["handedness"],
                    units="m", synthetic=True, seed=int(seed))
    truth = dict(theta=theta, beta=beta, global_orient=orient, translation_m=translation,
                 joints_m=joints_np, vertices_m=vertices_np, timestamps_s=time,
                 camera_centers_m=cameras, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    return observations, truth
