"""Opt-in observation hypotheses; no reward, physics, or batch-dependent scaling."""
import torch
from .grasp import grasp_gramian


def update_timers(timers, previous, contact, dt):
    """Contact age, release age, and recontact age, clipped at 2 seconds.

    Zero after reset means no recorded history. Update exactly once per control step.
    """
    t = timers.clone()
    released = previous & ~contact
    recontact = ~previous & contact
    t[..., 0] = torch.where(contact, t[..., 0] + dt, 0.)
    t[..., 1] = torch.where(released, dt, torch.where(t[..., 1] > 0, t[..., 1] + dt, 0.))
    t[..., 2] = torch.where(recontact, dt, torch.where(t[..., 2] > 0, t[..., 2] + dt, 0.))
    return t.clamp(0, 2.)


def mechanics_features(offsets, contacts):
    # Dimensionless moment arms: divide meters by fixed cube half-width 0.03m.
    # Use the EXISTING isotropic point-contact Gramian, not a friction-cone closure claim.
    offsets = offsets / .03
    spectrum = torch.linalg.eigvalsh(grasp_gramian(offsets, contacts)).clamp_min(0.) / 5.
    loo = []
    for finger in range(5):
        mask = contacts.clone(); mask[:, finger] = False
        loo.append(torch.linalg.eigvalsh(grasp_gramian(offsets, mask))[:, 0].clamp_min(0.) / 5.)
    return torch.cat((spectrum[:, :1], spectrum, torch.stack(loo, -1)), -1).clamp(0., 10.)


def observation_features(mode, contacts, offsets, timers):
    if mode == 'contacts': return contacts.float()
    if mode == 'timers': return torch.cat((contacts.float(), timers.flatten(1) / 2.), -1)
    if mode == 'mechanics': return mechanics_features(offsets, contacts)
    raise ValueError(mode)
