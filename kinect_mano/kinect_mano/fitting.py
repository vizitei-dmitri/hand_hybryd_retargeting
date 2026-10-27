"""Joint, visible-surface and timestamp-aware temporal MANO optimization."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .model import axis_angle_to_matrix
from .surface import SurfaceView, point_triangle_squared_distance, raycast_front_faces, robust_huber


@dataclass
class Observation:
    timestamp_s: float
    joints_m: np.ndarray
    confidence: np.ndarray
    views: tuple[SurfaceView, ...] = ()
    frame_id: str = ""
    valid: bool = True

    def __post_init__(self):
        self.joints_m = np.asarray(self.joints_m, dtype=np.float64)
        self.confidence = np.asarray(self.confidence, dtype=np.float64).copy()
        if self.joints_m.shape != (21, 3) or self.confidence.shape != (21,):
            raise ValueError("Expected joints_m[21,3] and confidence[21]")
        if not np.isfinite(self.timestamp_s) or not np.isfinite(self.confidence).all() or (self.confidence < 0).any() or (self.confidence > 1).any():
            raise ValueError("timestamp must be finite; confidence in [0,1]")
        self.confidence[~np.isfinite(self.joints_m).all(1)] = 0
        self.joints_m = np.nan_to_num(self.joints_m)
        self.valid = bool(self.valid)


@dataclass
class FitConfig:
    mode: str = "hybrid"
    iterations: int = 80
    learning_rate: float = 0.015
    joint_weight: float = 1.0
    surface_weight: float = 1.0
    pose_weight: float = 0.0002
    anatomy_weight: float = 0.005
    shape_weight: float = 0.0001
    temporal_weight: float = 0.02
    acceleration_weight: float = 0.002
    joint_scale_m: float = 0.01
    surface_scale_m: float = 0.005
    correspondence_distance_m: float = 0.04
    correspondence_every: int = 3
    max_points_per_camera: int = 2048
    max_gap_s: float = 0.25
    max_joint_residual_m: float = 0.04
    min_valid_joints: int = 6
    patience: int = 25

    def __post_init__(self):
        if self.mode not in ("joint", "hybrid", "temporal"):
            raise ValueError("mode must be joint, hybrid or temporal")
        if any(not isinstance(value, Integral) or isinstance(value,bool) or value < 1 for value in (self.iterations,self.correspondence_every,self.max_points_per_camera,self.patience)) or not isinstance(self.min_valid_joints,Integral) or not 3 <= self.min_valid_joints <= 21:
            raise ValueError("Invalid iteration/point/joint limits")
        for key, value in asdict(self).items():
            if isinstance(value, Real) and (not np.isfinite(value) or value < 0):
                raise ValueError(f"Invalid configuration: {key}")
        if min(self.learning_rate, self.joint_scale_m, self.surface_scale_m, self.max_gap_s, self.max_joint_residual_m) <= 0:
            raise ValueError("Learning rate, scales and thresholds must be positive")


@dataclass
class FitState:
    timestamp_s: float
    theta: np.ndarray
    beta: np.ndarray
    global_orient: np.ndarray
    translation_m: np.ndarray
    joints_m: np.ndarray
    vertices_m: np.ndarray
    valid: bool
    status: str
    quality: dict = field(default_factory=dict)


def _tensor(model, value):
    return torch.as_tensor(np.asarray(value), device=model.device, dtype=model.dtype)


def rigid_initialize(model, observation, beta):
    with torch.no_grad():
        _, rest = model(_tensor(model, np.zeros(45)), beta, _tensor(model, np.zeros(3)), _tensor(model, np.zeros(3)))
    rest = rest[0].cpu().numpy()
    valid = observation.confidence > 0
    palm = np.array([0, 1, 5, 9, 13, 17])
    ids = palm[valid[palm]]
    if len(ids) < 3:
        ids = np.flatnonzero(valid)
    if len(ids) < 3:
        return np.zeros(3), np.zeros(3)
    x, y = rest[ids], observation.joints_m[ids]
    weights = observation.confidence[ids]
    weights /= weights.sum()
    cx, cy = (x * weights[:, None]).sum(0), (y * weights[:, None]).sum(0)
    u, _, vt = np.linalg.svd((x - cx).T @ ((y - cy) * weights[:, None]))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    return Rotation.from_matrix(rotation).as_rotvec(), cy - rotation @ cx


def anatomical_loss(joints, theta):
    """Broad geometric bend and local rotation bounds, not a learned prior.

    Avoid assigning anatomical meanings to raw MANO x/y/z axes. The geometric
    constraint is invariant to root orientation. Twist is limited conservatively
    by total local angle; fine anatomical axes remain an optional extension.
    """
    terms = []
    for start in (1, 5, 9, 13, 17):
        bones = joints[:, start + 1:start + 4] - joints[:, start:start + 3]
        unit = bones / torch.linalg.vector_norm(bones, dim=-1, keepdim=True).clamp_min(1e-9)
        cosines = (unit[:, :-1] * unit[:, 1:]).sum(-1)
        # Inter-phalangeal bend above 115 degrees is discouraged.
        terms.append(torch.relu(np.cos(np.deg2rad(115)) - cosines).square().mean())
    terms.append(torch.relu(torch.linalg.vector_norm(theta.reshape(-1, 15, 3), dim=-1) - 2.4).square().mean())
    return torch.stack(terms).mean()


class ManoFitter:
    def __init__(self, model, config=None, beta=None):
        self.model = model
        self.config = config or FitConfig()
        self.beta = np.zeros(10) if beta is None else np.asarray(beta, dtype=np.float64)
        if self.beta.shape != (10,) or not np.isfinite(self.beta).all():
            raise ValueError("beta must be finite [10]")
        self.history: list[FitState] = []

    def reset(self):
        self.history.clear()

    def _prepare(self, observation):
        views = []
        for view in observation.views:
            points = view.points_m
            if len(points) > self.config.max_points_per_camera:
                # Deterministic uniform subsampling; same policy across modes.
                points = points[np.linspace(0, len(points) - 1, self.config.max_points_per_camera).astype(int)]
            views.append((_tensor(self.model, points), _tensor(self.model, view.camera_center_m), view.camera_id))
        return _tensor(self.model, observation.joints_m), _tensor(self.model, observation.confidence), views

    def _loss(self, theta, beta, orient, translation, observation, prepared, correspondences=None, iteration=0, history=()):
        config, model = self.config, self.model
        vertices, joints = model(theta, beta, orient, translation)
        measured, confidence, views = prepared
        joint_distance = torch.linalg.vector_norm(joints[0] - measured, dim=-1)
        joint = (robust_huber(joint_distance / config.joint_scale_m) * confidence).sum() / confidence.sum().clamp_min(1)
        pose = theta.square().mean()
        anatomy = anatomical_loss(joints, theta)
        surface = vertices.sum() * 0
        updated = []
        coverages = []
        if config.mode != "joint":
            camera_terms = []
            for i, (points, center, _) in enumerate(views):
                if correspondences is None or iteration % config.correspondence_every == 0:
                    indices = raycast_front_faces(points, vertices[0], model.faces, center, max_distance_m=config.correspondence_distance_m)
                else:
                    indices = correspondences[i]
                updated.append(indices)
                valid = indices >= 0
                coverages.append(float(valid.float().mean()) if len(valid) else 0.)
                camera_sum = vertices.sum()*0
                if valid.any():
                    triangles = vertices[0, model.faces[indices[valid]]]
                    distance = point_triangle_squared_distance(points[valid], triangles).clamp_min(1e-16).sqrt()
                    camera_sum = robust_huber(distance / config.surface_scale_m).sum()
                if len(points):
                    # Fixed denominator and a truncated unmatched-ray penalty
                    # prevent disappearing correspondences from improving E.
                    # Outlier/missing rays have bounded cost and zero gradient;
                    # they never pull hidden model triangles toward the cloud.
                    unmatched_cost = robust_huber(vertices.new_tensor(config.correspondence_distance_m/config.surface_scale_m))
                    camera_terms.append((camera_sum+(~valid).sum()*unmatched_cost)/len(points))
            if camera_terms:
                surface = torch.stack(camera_terms).mean()
        temporal = vertices.sum() * 0
        acceleration = vertices.sum() * 0
        if config.mode == "temporal" and history:
            previous = history[-1]
            dt = observation.timestamp_s - previous.timestamp_s
            if 0 < dt <= config.max_gap_s:
                previous_joints = _tensor(model, previous.joints_m)
                temporal = ((joints[0] - previous_joints) / (0.5 * dt)).square().mean()
                current_rot = axis_angle_to_matrix(torch.cat([orient.reshape(1, 3), theta.reshape(15, 3)]))
                previous_rot = axis_angle_to_matrix(_tensor(model, np.concatenate([previous.global_orient[None], previous.theta.reshape(15, 3)])))
                temporal = temporal + ((current_rot - previous_rot) / (4 * dt)).square().mean()
                if len(history) >= 2:
                    earlier = history[-2]
                    prev_dt = previous.timestamp_s - earlier.timestamp_s
                    if 0 < prev_dt <= config.max_gap_s:
                        velocity = (joints[0] - previous_joints) / dt
                        previous_velocity = _tensor(model, (previous.joints_m - earlier.joints_m) / prev_dt)
                        acceleration = ((velocity - previous_velocity) / (10 * (dt + prev_dt) * .5)).square().mean()
        total = (config.joint_weight * joint + config.surface_weight * surface + config.pose_weight * pose
                 + config.anatomy_weight * anatomy + config.shape_weight * beta.square().mean()
                 + config.temporal_weight * temporal + config.acceleration_weight * acceleration)
        return total, dict(joint=joint, surface=surface, anatomy=anatomy, temporal=temporal, acceleration=acceleration), updated, coverages

    def fit(self, observation: Observation, *, initial_state=None, optimize_shape=False):
        start = perf_counter()
        if self.history and observation.timestamp_s <= self.history[-1].timestamp_s:
            raise ValueError("Frame timestamps must increase strictly; reset tracker for a new sequence")
        if self.history and observation.timestamp_s - self.history[-1].timestamp_s > self.config.max_gap_s:
            self.reset()
        previous = initial_state or (self.history[-1] if self.history else None)
        beta = _tensor(self.model, self.beta).clone().requires_grad_(optimize_shape)
        if not observation.valid or np.count_nonzero(observation.confidence) < self.config.min_valid_joints:
            # Do not emit an old frame as a fresh valid measurement.
            self.reset()
            orient, translation = rigid_initialize(self.model, observation, beta)
            with torch.no_grad():
                vertices, joints = self.model(_tensor(self.model, np.zeros(45)), beta, _tensor(self.model, orient), _tensor(self.model, translation))
            return FitState(observation.timestamp_s, np.zeros(45), self.beta.copy(), orient, translation, joints[0].cpu().numpy(), vertices[0].cpu().numpy(), False, "invalid_initializer" if not observation.valid else "insufficient_joints", {"fit_ms": (perf_counter() - start) * 1000})
        if previous is None:
            orient, translation = rigid_initialize(self.model, observation, beta)
            theta = np.zeros(45)
        else:
            theta, orient, translation = previous.theta, previous.global_orient, previous.translation_m
        theta = _tensor(self.model, theta).clone().requires_grad_()
        orient = _tensor(self.model, orient).clone().requires_grad_()
        translation = _tensor(self.model, translation).clone().requires_grad_()
        parameters = [theta, orient, translation] + ([beta] if optimize_shape else [])
        optimizer = torch.optim.Adam([{"params": [theta, orient], "lr": self.config.learning_rate}, {"params": [translation], "lr": self.config.learning_rate * .2}] + ([{"params": [beta], "lr": .02}] if optimize_shape else []))
        prepared = self._prepare(observation)
        best, best_params, stale, correspondences = float("inf"), None, 0, None
        iterations = 0
        for iteration in range(self.config.iterations):
            optimizer.zero_grad()
            loss, terms, correspondences, coverage = self._loss(theta, beta, orient, translation, observation, prepared, correspondences, iteration, self.history)
            score = float(loss.detach())
            if not np.isfinite(score):
                break
            # Snapshots compete only when correspondences have been refreshed.
            # Cached ICP objectives between refreshes are not comparable scores.
            if iteration % self.config.correspondence_every == 0 or self.config.mode == "joint":
                if score < best:
                    best, stale = score, 0
                    best_params = [p.detach().clone() for p in parameters]
                else:
                    stale += self.config.correspondence_every if self.config.mode != "joint" else 1
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                break
            torch.nn.utils.clip_grad_norm_(parameters, 50)
            optimizer.step()
            iterations += 1
            if stale >= self.config.patience:
                break
        # Include the final optimizer update with fresh visibility as well.
        with torch.no_grad():
            candidate = self._loss(theta,beta,orient,translation,observation,prepared,iteration=0,history=self.history)[0]
            if torch.isfinite(candidate) and float(candidate) < best:
                best = float(candidate)
                best_params = [p.detach().clone() for p in parameters]
        if best_params is not None:
            with torch.no_grad():
                for parameter, value in zip(parameters, best_params):
                    parameter.copy_(value)
        with torch.no_grad():
            vertices, joints = self.model(theta, beta, orient, translation)
            final, terms, _, coverage = self._loss(theta, beta, orient, translation, observation, prepared, iteration=0, history=self.history)
        residual = np.linalg.norm(joints[0].cpu().numpy() - observation.joints_m, axis=1)
        mask = observation.confidence > 0
        residual_mean = float(np.average(residual[mask], weights=observation.confidence[mask]))
        valid = np.isfinite(best) and residual_mean <= self.config.max_joint_residual_m
        cameras_used = sum(value > 0 for value in coverage)
        quality = dict(loss=float(final), joint_residual_mm=residual_mean * 1000, surface_coverage=coverage,
                       surface_cameras_used=cameras_used, surface_point_counts=[len(view.points_m) for view in observation.views],
                       iterations=iterations, fit_ms=(perf_counter() - start) * 1000,
                       terms={name: float(value) for name, value in terms.items()})
        status = "tracked" if valid else "rejected"
        if valid and self.config.mode != "joint" and cameras_used == 0:
            status = "degraded_joint_only"
        state = FitState(observation.timestamp_s, theta.detach().cpu().numpy(), beta.detach().cpu().numpy(), orient.detach().cpu().numpy(), translation.detach().cpu().numpy(), joints[0].cpu().numpy(), vertices[0].cpu().numpy(), bool(valid), status, quality)
        if valid:
            self.history = (self.history + [state])[-2:]
        else:
            self.reset()
        return state


def calibrate_shape(model, observations, *, iterations=100, mode="hybrid"):
    """Jointly optimize ONE shared beta and separate per-frame poses."""
    observations = [o for o in observations if np.count_nonzero(o.confidence) >= 6]
    if not observations:
        raise ValueError("Shape calibration requires valid observations")
    fitter = ManoFitter(model, FitConfig(mode=mode, iterations=iterations))
    beta = torch.zeros(10, dtype=model.dtype, device=model.device, requires_grad=True)
    states = []
    for obs in observations:
        orient, translation = rigid_initialize(model, obs, beta)
        states.append([_tensor(model, np.zeros(45)).requires_grad_(), _tensor(model, orient).requires_grad_(), _tensor(model, translation).requires_grad_()])
    parameters = [beta] + [value for state in states for value in state]
    optimizer = torch.optim.Adam(parameters, lr=.008)
    prepared = [fitter._prepare(obs) for obs in observations]
    best, best_parameters = float("inf"), None
    for _ in range(iterations):
        optimizer.zero_grad()
        losses = [fitter._loss(theta, beta, orient, translation, obs, inputs)[0] for (theta, orient, translation), obs, inputs in zip(states, observations, prepared)]
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite shape calibration objective")
        if float(loss.detach()) < best:
            best = float(loss.detach())
            best_parameters = [p.detach().clone() for p in parameters]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 50)
        optimizer.step()
    return best_parameters[0].cpu().numpy(), dict(loss=best, frames=len(observations), iterations=iterations)
