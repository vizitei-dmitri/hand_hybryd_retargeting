"""Goal orientation sampling with a minimum initial error (pure torch, no Isaac Sim).

Quaternions are (w, x, y, z), as in Isaac Lab.
"""

import math

import torch


def quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Shortest rotation angle (rad) between unit quaternions."""
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def axis_angle_quat(angles: torch.Tensor, axis) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=angles.dtype, device=angles.device)
    axis = axis / axis.norm()
    half = 0.5 * angles[:, None]
    return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)


def uniform_quat(count: int, device, generator=None) -> torch.Tensor:
    """Uniform random rotations (Shoemake)."""
    u1, u2, u3 = torch.rand((3, count), device=device, generator=generator)
    a, b = torch.sqrt(1.0 - u1), torch.sqrt(u1)
    return torch.stack((a * torch.sin(2 * math.pi * u2), a * torch.cos(2 * math.pi * u2),
                        b * torch.sin(2 * math.pi * u3), b * torch.cos(2 * math.pi * u3)), dim=-1)


def sample_goal_quats(initial_quats: torch.Tensor, mode: str, axis, angle_range_deg, min_error_rad: float,
                      max_rounds: int = 1000, generator=None) -> torch.Tensor:
    """Rejection-sample goals whose angle to the initial cube orientation is >= min_error_rad.

    Only rejected entries are redrawn, so the accepted distribution is the configured one
    conditioned on the minimum error.
    """
    count, device = initial_quats.shape[0], initial_quats.device
    if mode == "axis":
        lo, hi = map(math.radians, angle_range_deg)
        # With the identity start, |angle| is the goal error; the range must leave room for it.
        if max(abs(lo), abs(hi)) < min_error_rad:
            raise ValueError("target_angle_range_deg cannot satisfy min_initial_goal_error_deg")
    elif mode != "uniform":
        raise ValueError(f"Unknown target orientation mode: {mode}")
    goals = torch.empty((count, 4), device=device)
    pending = torch.arange(count, device=device)
    for _ in range(max_rounds):
        n = pending.numel()
        if mode == "uniform":
            candidates = uniform_quat(n, device, generator)
        else:
            angles = lo + (hi - lo) * torch.rand(n, device=device, generator=generator)
            candidates = axis_angle_quat(angles, axis)
        ok = quat_angle(candidates, initial_quats[pending]) >= min_error_rad
        goals[pending[ok]] = candidates[ok]
        pending = pending[~ok]
        if pending.numel() == 0:
            return goals
    raise RuntimeError(f"Goal rejection sampling did not converge for {pending.numel()} environments")


def sample_angles_haar(count: int, lo_rad: float, hi_rad: float, device, generator=None,
                       max_rounds: int = 1000) -> torch.Tensor:
    """Rotation angles on [lo, hi] with the SO(3) Haar density, which is proportional to 1 - cos.

    Sampling the angle uniformly instead would over-represent small rotations. Because this is
    the Haar density merely truncated to a band, widening the band to [0, pi] recovers exactly
    the uniform-SO(3) distribution, so a curriculum built on it converges to the real target
    rather than to a differently shaped approximation of it.
    """
    if not 0.0 <= lo_rad <= hi_rad <= math.pi + 1e-9:
        raise ValueError(f"Invalid angle band: [{lo_rad}, {hi_rad}]")
    angles = torch.empty(count, device=device)
    pending = torch.arange(count, device=device)
    for _ in range(max_rounds):
        n = pending.numel()
        candidates = lo_rad + (hi_rad - lo_rad) * torch.rand(n, device=device, generator=generator)
        # Density 1 - cos(theta) peaks at 2 (theta = pi), so that is the rejection envelope.
        accept = torch.rand(n, device=device, generator=generator) * 2.0 <= (1.0 - torch.cos(candidates))
        angles[pending[accept]] = candidates[accept]
        pending = pending[~accept]
        if pending.numel() == 0:
            return angles
    raise RuntimeError(f"Haar angle sampling did not converge for {pending.numel()} environments")


def random_axes(count: int, device, generator=None) -> torch.Tensor:
    """Uniform directions on the sphere."""
    axes = torch.randn((count, 3), device=device, generator=generator)
    return axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def sample_curriculum_goals(current_quats: torch.Tensor, min_error_rad: float, limit_rad: float,
                            band_rad: float, frontier_fraction: float, inside_fraction: float,
                            generator=None):
    """Goals at a curriculum-bounded rotation from the CURRENT cube orientation.

    POISE (2026) reports that training straight on the full goal range rarely works (6.2% vs 59.5%
    with a curriculum), and samples near the frontier, inside the exposed range, or at the limit.
    Returns (goals, at_frontier); at_frontier marks the goals whose success rate drives promotion.
    """
    count, device = current_quats.shape[0], current_quats.device
    limit_rad = max(limit_rad, min_error_rad)
    draw = torch.rand(count, device=device, generator=generator)
    at_frontier = draw < frontier_fraction
    at_limit = draw >= frontier_fraction + inside_fraction

    frontier_lo = max(min_error_rad, limit_rad - band_rad)
    lo = torch.full((count,), min_error_rad, device=device)
    hi = torch.full((count,), limit_rad, device=device)
    lo[at_frontier] = frontier_lo
    lo[at_limit] = limit_rad
    # A degenerate band (limit still at the minimum error) collapses to the single allowed angle.
    angles = torch.empty(count, device=device)
    for mask in (at_frontier, at_limit, ~(at_frontier | at_limit)):
        if not bool(mask.any()):
            continue
        band_lo = lo[mask][0].item()
        band_hi = hi[mask][0].item()
        angles[mask] = sample_angles_haar(int(mask.sum()), band_lo, band_hi, device, generator)
    deltas = axis_angle_quat_axes(angles, random_axes(count, device, generator))
    return quat_multiply(current_quats, deltas), at_frontier


STREAM_STAGES = ("A", "B", "C")


def stage_axes(stage: str, count: int, device, primary_axis, generator=None) -> torch.Tensor:
    """Rotation axes for one stage of the DIRECTIONAL curriculum, in the palm frame.

    Stage A is the single axis the previous experiment already solved, B the three palm principal
    axes, C a uniformly sampled direction. The angle is fixed by the caller: this curriculum grows
    directional complexity only, so a goal never becomes a large single-step orientation jump.
    """
    if stage == "A":
        axis = torch.as_tensor(primary_axis, dtype=torch.float32, device=device)
        return (axis / axis.norm()).expand(count, 3).clone()
    if stage == "B":
        pick = torch.randint(3, (count,), device=device, generator=generator)
        return torch.eye(3, device=device)[pick]
    if stage == "C":
        return random_axes(count, device, generator)
    raise ValueError(f"Unknown stream stage {stage!r}, expected one of {STREAM_STAGES}")


def sample_fixed_angle_goals(current_quats: torch.Tensor, angle_rad: float, stage: str, primary_axis,
                             generator=None) -> torch.Tensor:
    """Goals exactly angle_rad from the CURRENT orientation, about a stage-dependent axis.

    Measuring from the current orientation rather than from the previous ideal goal is what keeps
    every goal exactly angle_rad away: tracking error would otherwise accumulate into a target that
    is no longer the requested distance from where the object actually is.

    Stages A and B draw a random sign, which makes the stream reversible instead of a monotonic
    drift in one direction. Stage C needs no sign because the axis itself is already two-sided.
    """
    count, device = current_quats.shape[0], current_quats.device
    axes = stage_axes(stage, count, device, primary_axis, generator)
    if stage in ("A", "B"):
        sign = torch.where(torch.rand(count, device=device, generator=generator) < 0.5, -1.0, 1.0)
        axes = axes * sign[:, None]
    angles = torch.full((count,), float(angle_rad), device=device)
    return quat_multiply(current_quats, axis_angle_quat_axes(angles, axes))


def axis_angle_quat_axes(angles: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Per-entry axis-angle to quaternion (axis_angle_quat shares one axis for the whole batch)."""
    half = 0.5 * angles[:, None]
    return torch.cat((torch.cos(half), torch.sin(half) * axes), dim=-1)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (w, x, y, z); kept local so goals.py stays Isaac-Sim-free."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)
