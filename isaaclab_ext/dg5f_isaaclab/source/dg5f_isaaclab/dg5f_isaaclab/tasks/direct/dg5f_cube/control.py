"""Control-step action FIFO and position mappings, independent of Isaac Sim."""

from collections.abc import Sequence
import math

import torch


class ActionDelayQueue:
    """Newest-first history, visible to the policy in its entirety.

    For delay D, push(a[t]) returns a[t-D]. Zero-padded on reset.
    Keep at least one history slot even with D=0 for the action-rate reward.
    """

    def __init__(self, num_envs: int, delays: Sequence[int], device):
        if not delays or any(type(d) is not int or d < 0 for d in delays):
            raise ValueError("Action delays must be nonnegative control-step integers")
        self.history_steps = max(1, max(delays))
        self.history = torch.zeros((num_envs, self.history_steps, len(delays)), device=device)
        self.delays = torch.tensor(delays, dtype=torch.long, device=device)
        self._indices = torch.arange(len(delays), device=device)
        self._slots = (self.delays - 1).clamp_min(0)

    def push(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape != self.history[:, 0].shape:
            raise ValueError("Action shape does not match the delay queue")
        delivered = torch.where(
            self.delays > 0, self.history[:, self._slots, self._indices], actions,
        )
        self.history[:, 1:] = self.history[:, :-1].clone()
        self.history[:, 0] = actions
        return delivered

    def reset(self, env_ids=None):
        if env_ids is None:
            self.history.zero_()
        else:
            self.history[env_ids] = 0


CONTROL_MODES = ("integrated_delta_position", "measured_delta_position", "absolute_position")


def position_targets(actions, measured, command, lower, upper, grasp, control_mode, delta_action_scale):
    """Decode DELIVERED (already delayed) actions into clamped joint position targets.

    integrated_delta_position: q_cmd[t] = clamp(q_cmd[t-1] + a * scale); a=0 holds q_cmd.
    measured_delta_position:   clamp(q_measured[t] + a * scale); previous version, kept for A/B.
    absolute_position:         grasp-centered -1/0/+1 -> lower/grasp/upper mapping.
    """
    if control_mode in ("integrated_delta_position", "measured_delta_position"):
        if not math.isfinite(delta_action_scale) or delta_action_scale <= 0:
            raise ValueError("delta_action_scale must be positive radians")
        base = command if control_mode == "integrated_delta_position" else measured
        targets = base + actions * delta_action_scale
    elif control_mode == "absolute_position":
        targets = torch.where(
            actions >= 0, grasp + actions * (upper - grasp), grasp + actions * (grasp - lower),
        )
    else:
        raise ValueError(f"Unknown control_mode: {control_mode}")
    return torch.clamp(targets, lower, upper)
