"""Read-only first-episode telemetry; distinguish raw actor mu from clipped commands."""
from collections import defaultdict, deque

import torch


def distribution(values):
    x = values.float().flatten()
    if not x.numel():
        return {"samples": 0}
    q = torch.quantile(x, torch.tensor([.5, .9, .95, .99, .999]))
    return {"samples": x.numel(), "mean": float(x.mean()), "mean_square": float(x.square().mean()),
            "max": float(x.max()), **{k: float(v) for k, v in zip(("p50", "p90", "p95", "p99", "p999"), q)}}


class Telemetry:
    def __init__(self, raw):
        self.raw = raw
        self.done = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.data = defaultdict(list)
        self.previous_mu = self.previous_command = self.previous_q = self.previous_velocity = None
        self.mu = None
        self.steps = []
        self.window = deque(maxlen=60)
        self.drops = []

    def physics_sample(self):
        r = self.raw
        q = r.hand.data.joint_pos[:, r.active_joint_ids].detach().clone()
        if self.previous_q is not None:
            mask = ~self.done
            fd = (q - self.previous_q) / r.cfg.sim.dt
            self.data["qdot_fd"].append(fd[mask].cpu())
            self.data["qdot_fd_after_1s"].append(fd[mask & (r.episode_length_buf * r.step_dt > 1.0)].cpu())
        self.previous_q = q

    def reward_sample(self):
        r, mask = self.raw, ~self.done
        if self.mu is None or not mask.any():
            return
        mu, command = self.mu.detach(), r.actions.detach()
        self.data["raw_mu"].append(mu[mask].cpu())
        self.data["command"].append(command[mask].cpu())
        self.data["delivered_command"].append(r.applied_actions.detach()[mask].cpu())
        if self.previous_mu is not None:
            self.data["delta_mu"].append((mu - self.previous_mu)[mask].cpu())
            self.data["delta_command"].append((command - self.previous_command)[mask].cpu())
            self.data["sign_flips"].append(((mu * self.previous_mu) < 0)[mask].cpu())
        self.previous_mu, self.previous_command = mu.clone(), command.clone()
        self.data["action_square"].append(command.square().mean(-1)[mask].cpu())
        self.data["action_rate_square"].append(r.action_delta.square().mean(-1)[mask].cpu())
        self.data["qdot_physx"].append(r.hand.data.joint_vel[:, r.active_joint_ids][mask].detach().cpu())
        for name, value in r.reward_terms.items():
            self.data["reward/" + name].append(value[mask].detach().cpu())
        velocity = r.cube.data.root_lin_vel_w.detach().clone()
        acceleration = (torch.zeros_like(velocity) if self.previous_velocity is None
                        else (velocity - self.previous_velocity) / r.step_dt)
        self.previous_velocity = velocity
        self.window.append({"tips": r.tip_in_contact.detach().cpu(),
                            "quality": r.grasp_quality_value.detach().cpu(),
                            "palm_force": r.palm_contact_force.detach().cpu(),
                            "acceleration": acceleration.norm(dim=-1).cpu(),
                            "near_limit": (command.abs() >= .95).float().cpu()})
        for env in (mask & r.reset_terminated).nonzero().flatten().tolist():
            history = [{key: value[env].tolist() for key, value in frame.items()} for frame in self.window]
            self.drops.append({"env": env, "episode_step": int(r.episode_length_buf[env]), "last_60_steps": history})
        self.steps.append({"step": len(self.steps), "active_envs": int(mask.sum()),
                           "mean_abs_mu": float(mu[mask].abs().mean()),
                           "mean_abs_command": float(command[mask].abs().mean()),
                           "near_limit": float((mu[mask].abs() >= .95).float().mean())})
        self.done |= r.reset_terminated | r.reset_time_outs

    def summary(self):
        values = {k: torch.cat(v) for k, v in self.data.items() if v and sum(t.numel() for t in v)}
        result = {"sampling": "Only first episode per environment, including its terminal step; reset jumps excluded.",
                  "mu_definition": "raw_mu is the unclipped actor mean; command is clamp(mu,-1,1); delivered_command is the delayed action actually entering the q_cmd integrator. Reward penalizes issued command squared.",
                  "step_summaries": self.steps, "drop_windows": self.drops}
        for key in ("raw_mu", "command", "delivered_command"):
            v = values[key]
            result[key] = distribution(v.abs())
            result[key]["near_limit_fraction"] = float((v.abs() >= .95).float().mean())
            result[key]["per_joint_near_limit"] = dict(zip(self.raw.cfg.active_action_joints,
                                                         (v.abs() >= .95).float().mean(0).tolist()))
        for key in ("delta_mu", "delta_command", "qdot_physx", "qdot_fd", "qdot_fd_after_1s"):
            result[key] = distribution(values.get(key, torch.empty(0)).abs())
        result["sign_flip_fraction"] = float(values.get("sign_flips", torch.zeros(1)).float().mean())
        result["mean_action_square"] = float(values["action_square"].mean())
        result["mean_action_rate_square"] = float(values["action_rate_square"].mean())
        result["action_square_sum_per_episode"] = float(values["action_square"].sum() / self.raw.num_envs)
        result["action_rate_square_sum_per_episode"] = float(values["action_rate_square"].sum() / self.raw.num_envs)
        result["reward_terms"] = {}
        for key, value in values.items():
            if key.startswith("reward/"):
                result["reward_terms"][key[7:]] = {"mean_signed": float(value.mean()),
                    "mean_abs": float(value.abs().mean()), "sum_per_episode": float(value.sum() / self.raw.num_envs),
                    "absolute_distribution": distribution(value.abs())}
        result["task_abs"] = sum(result["reward_terms"].get(name, {}).get("mean_abs", 0) for name in
                                 ("orientation_state_reward", "orientation_progress_reward", "goal_dwell_reward", "success_bonus"))
        result["action_penalty_to_task"] = result["reward_terms"]["action_penalty"]["mean_abs"] / max(result["task_abs"], 1e-12)
        result["action_penalty_scale"] = self.raw.cfg.action_penalty_scale
        result["action_rate_penalty_scale"] = self.raw.cfg.action_rate_penalty_scale
        return result
