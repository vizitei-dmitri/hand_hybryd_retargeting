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
