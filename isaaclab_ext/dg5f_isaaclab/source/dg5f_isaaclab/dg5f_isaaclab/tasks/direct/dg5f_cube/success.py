"""Reward-v2 orientation state term and held-success tracking (pure torch, no Isaac Sim)."""

import math

import torch


def orientation_state_reward(error_rad: torch.Tensor, scale: float, sigma_rad: float) -> torch.Tensor:
    """Dense reward for being near the goal: scale * exp(-(error / sigma)^2)."""
    return scale * torch.exp(-(error_rad / sigma_rad).square())


def hold_steps(hold_time_s: float, control_dt: float) -> int:
    """Consecutive control steps covering hold_time_s (rounded, at least 1)."""
    if hold_time_s <= 0 or control_dt <= 0:
        raise ValueError("hold time and control dt must be positive")
    return max(1, round(hold_time_s / control_dt))


class HeldSuccessTracker:
    """Per-env success = orientation within tolerance for `required_steps` CONSECUTIVE steps.

    Leaving the tolerance resets the counter. `update` returns the envs that complete the
    hold for the first time in this episode (the only step that earns the bonus).
    """

    def __init__(self, num_envs: int, required_steps: int, device):
        if required_steps < 1:
            raise ValueError("required_steps must be >= 1")
        self.required_steps = required_steps
        self.consecutive = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.entered = torch.zeros_like(self.succeeded)
        self.steps_in_tolerance = torch.zeros_like(self.consecutive)
        self.success_step = torch.full_like(self.consecutive, -1)

    def update(self, in_tolerance: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """in_tolerance must already exclude steps that may not count (e.g. before actions arrive)."""
        self.consecutive = torch.where(in_tolerance, self.consecutive + 1, torch.zeros_like(self.consecutive))
        self.entered |= in_tolerance
        self.steps_in_tolerance += in_tolerance.long()
        newly = (self.consecutive >= self.required_steps) & ~self.succeeded
        self.succeeded |= newly
        self.success_step[newly] = step[newly]
        return newly

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.consecutive[ids] = 0
        self.succeeded[ids] = False
        self.entered[ids] = False
        self.steps_in_tolerance[ids] = 0
        self.success_step[ids] = -1


def expected_state_rewards(errors_deg, scale: float, sigma_deg: float):
    """Table helper for the reward diagnostics."""
    errors = torch.tensor([math.radians(e) for e in errors_deg])
    return orientation_state_reward(errors, scale, math.radians(sigma_deg)).tolist()
