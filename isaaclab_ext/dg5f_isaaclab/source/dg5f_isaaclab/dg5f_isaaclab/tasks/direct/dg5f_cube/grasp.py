"""Grasp-quality shaping from fingertip contact geometry (pure torch, no Isaac Sim).

Follows "Robust In-Hand Manipulation via Priors in Reinforcement Learning and Mechanical
Design" (2026): a dense term on lambda_min of the grasp Gramian M = G G^T, where G is built
from the fingertip contact positions alone. It needs no force sensing, which matters here
because the observation already carries the five fingertip positions.

Why it is not redundant with "at least three fingertips are touching": a count cannot tell a
real grip from a bowl of fingertips holding the cube up against gravity. lambda_min measures
the worst-case direction of the wrench the contact set can resist, so a bowl scores badly and
opposed contacts score well.
"""

import torch


def skew(vectors: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) cross-product matrices."""
    x, y, z = vectors.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((
        torch.stack((zero, -z, y), dim=-1),
        torch.stack((z, zero, -x), dim=-1),
        torch.stack((-y, x, zero), dim=-1),
    ), dim=-2)


def grasp_gramian(contact_offsets: torch.Tensor, in_contact: torch.Tensor) -> torch.Tensor:
    """(N, C, 3) contact positions relative to the object centre -> (N, 6, 6) Gramian.

    Each contact contributes the unit-force wrench basis G_i = [[I], [skew(r_i)]], so
    M = sum_i G_i G_i^T over the contacts that are actually touching.
    """
    mask = in_contact.to(contact_offsets.dtype)[..., None, None]
    r = skew(contact_offsets)                                  # (N, C, 3, 3)
    identity = torch.eye(3, device=contact_offsets.device, dtype=contact_offsets.dtype)
    identity = identity.expand(r.shape)
    # G_i G_i^T = [[I, -skew(r)], [skew(r), skew(r) skew(r)^T]]; skew is antisymmetric.
    top = torch.cat((identity, -r), dim=-1)
    bottom = torch.cat((r, r @ r.transpose(-1, -2)), dim=-1)
    blocks = torch.cat((top, bottom), dim=-2)                  # (N, C, 6, 6)
    return (blocks * mask).sum(dim=1)


def gramian_min_eigenvalue(gramian: torch.Tensor) -> torch.Tensor:
    """Smallest eigenvalue of a batch of symmetric PSD matrices; 0 for a singular contact set."""
    # eigvalsh on a symmetric matrix; fewer than three contacts leaves M singular, giving ~0.
    return torch.linalg.eigvalsh(gramian)[:, 0].clamp_min(0.0)


class GraspQualityReward:
    """Bounded, EMA-normalised lambda_min shaping term.

    The paper bounds the term at r_max and rescales it by an adaptive factor kept on an
    exponential moving average so it cannot outgrow the task reward; it does not state the
    exact normaliser, so the running mean magnitude is used here, which keeps the scaled term
    at order one regardless of the object size and contact count.
    """

    def __init__(self, margin: float, clip: float, ema: float):
        if not 0.0 < ema < 1.0:
            raise ValueError("grasp_quality_ema must be in (0, 1)")
        if clip <= 0 or margin < 0:
            raise ValueError("grasp_quality_clip must be positive and the margin non-negative")
        self.margin = margin
        self.clip = clip
        self.ema = ema
        self.scale = None

    def __call__(self, contact_offsets: torch.Tensor, in_contact: torch.Tensor):
        lambda_min = gramian_min_eigenvalue(grasp_gramian(contact_offsets, in_contact))
        centred = lambda_min - self.margin
        magnitude = centred.abs().mean().detach()
        if self.scale is None:
            self.scale = magnitude.clone()
        else:
            self.scale.mul_(self.ema).add_((1.0 - self.ema) * magnitude)
        reward = (centred / self.scale.clamp_min(1e-12)).clamp(-self.clip, self.clip)
        return reward, lambda_min
