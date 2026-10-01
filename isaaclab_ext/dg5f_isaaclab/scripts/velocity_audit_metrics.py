"""Reset-aware diagnostics for physics-substep joint-position differences."""
import math

import torch


def velocity_details(q, qdot, epoch, dt, names, disabled_joint):
    valid = epoch[1:] == epoch[:-1]
    velocity = (q[1:] - q[:-1]) / dt
    # First sample, and every epoch change, begins a new physical episode.
    index = torch.arange(q.shape[0], device=q.device)[:, None].expand_as(epoch)
    starts = torch.zeros_like(epoch, dtype=torch.bool)
    starts[0] = True
    starts[1:] = ~valid
    last_start = torch.where(starts, index, 0).cummax(dim=0).values
    age = (index - last_start) * dt
    active = [j for j, name in enumerate(names) if name != disabled_joint]

    def summarize(mask):
        v = velocity[mask][:, active].abs()
        if not v.numel():
            return {"joint_samples": 0}
        return {"joint_samples": v.numel(), "max": float(v.max()),
                "p99": float(torch.quantile(v.flatten(), .99)),
                "p999": float(torch.quantile(v.flatten(), .999)),
                "fraction_above_pi": float((v > math.pi).float().mean()),
                "fraction_above_1_01_pi": float((v > 1.01 * math.pi).float().mean()),
                "fraction_above_1_1_pi": float((v > 1.1 * math.pi).float().mean())}

    magnitude = velocity.abs().masked_fill(~valid[..., None], -1)
    top = magnitude.flatten().topk(min(20, int(valid.sum()) * len(names))).indices.tolist()
    extremes = []
    for flat in top:
        sample, joint = divmod(flat, q.shape[2])
        t, env = divmod(sample, q.shape[1])
        extremes.append({"joint": names[joint], "env": env, "epoch": int(epoch[t + 1, env]),
                         "physics_sample": t + 1, "seconds_since_reset": float(age[t + 1, env]),
                         "q_before_rad": float(q[t, env, joint]),
                         "q_after_rad": float(q[t + 1, env, joint]),
                         "fd_rad_s": float(velocity[t, env, joint]),
                         "physx_rad_s": float(qdot[t + 1, env, joint])})
    return {"active_joints_only": {
                "all": summarize(valid),
                "first_0_2_seconds": summarize(valid & (age[1:] <= .2)),
                "after_0_2_seconds": summarize(valid & (age[:-1] > .2)),
                "after_1_second": summarize(valid & (age[:-1] > 1.0))},
            "largest_finite_differences": extremes}
