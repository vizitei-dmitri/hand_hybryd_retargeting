"""Is max_abs_joint_velocity 5-14 rad/s real motion, or a PhysX readout artefact?

The night run logged max_abs_joint_velocity 4.9-13.9 rad/s against a readback velocity limit of
pi rad/s. This compares, per joint, the PhysX velocity state against a finite difference of the
PhysX joint POSITION state.

Sampling is at the PHYSICS rate, not the control rate: _apply_action runs once per physics substep
before sim.step, so consecutive samples are exactly sim.dt apart. A finite difference over the
whole control step (decimation 2) averages two substeps and would understate a real spike by
construction, which would make the comparison meaningless.

Samples spanning a reset are dropped (q teleports there), tracked with a per-env epoch counter.
"""

import argparse
import json
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="DG5F-Cube-Fingertip-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--run", default="logs/rsl_rl/dg5f_cube_direct/2026-09-28_21-26-31_fingertip_v3_night")
parser.add_argument("--checkpoints", default="4000,6499", help="Iterations from --run, plus the zero policy")
parser.add_argument("--steps", type=int, default=0, help="Control steps; 0 = one full episode")
parser.add_argument("--output", type=Path, default=Path("logs/velocity_audit/audit.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env import DG5FCubeEnv


class AuditEnv(DG5FCubeEnv):
    """DG5FCubeEnv that records (q, qdot, epoch) once per physics substep."""

    def _setup_scene(self):
        super()._setup_scene()
        # Must exist before __init__ finishes: _apply_action and _reset_idx both touch them.
        self.audit_epoch = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.audit = []

    def _apply_action(self):
        super()._apply_action()
        if self.audit is not None:
            self.audit.append((
                self.hand.data.joint_pos[:, self.joint_ids].clone(),
                self.hand.data.joint_vel[:, self.joint_ids].clone(),
                self.audit_epoch.clone(),
            ))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.audit_epoch[ids] += 1


def statistics(values: torch.Tensor, mask: torch.Tensor | None = None):
    """Per-joint max / p99 / rms of |values|, over the masked samples. values is (T, E, J)."""
    magnitude = values.abs()
    if mask is not None:
        flat = magnitude[mask]            # (N, J)
    else:
        flat = magnitude.reshape(-1, magnitude.shape[-1])
    return {
        "max": flat.max(dim=0).values,
        "p99": torch.quantile(flat.float(), 0.99, dim=0),
        "rms": flat.square().mean(dim=0).sqrt(),
        "samples": flat.shape[0],
    }


def probe_velocity_limits(raw):
    """Readback limits plus the PhysX articulation cap, which is what actually clamps qdot."""
    report = {"joint_vel_limits_rad_s": raw.hand.data.joint_vel_limits[0, raw.joint_ids].tolist()}
    try:
        from isaaclab.sim.utils import get_current_stage
        prim = get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")
        # Read by name: the PhysxArticulationAPI accessor is not stable across Isaac Sim versions.
        attribute = prim.GetAttribute("physxArticulation:maxJointVelocity")
        report["usd_articulation_max_joint_velocity"] = (
            attribute.Get() if attribute and attribute.HasAuthoredValue() else "not authored (PhysX default)")
    except Exception as error:  # a USD/schema difference must not fail the audit
        report["usd_articulation_max_joint_velocity"] = f"probe failed: {error}"
    try:
        view = raw.hand.root_physx_view
        report["physx_view_dof_max_velocities"] = view.get_dof_max_velocities()[0].tolist()
    except Exception as error:
        report["physx_view_dof_max_velocities"] = f"probe failed: {error}"
    return report


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(AuditEnv(env_cfg))
    raw = env.unwrapped
    names = list(env_cfg.actuated_joint_names)
    physics_dt = raw.cfg.sim.dt
    steps = args_cli.steps or int(raw.max_episode_length)

    limits = probe_velocity_limits(raw)
    print("[AUDIT] velocity limits:", json.dumps(limits, indent=2), flush=True)

    policies = ["zero"] + [f"model_{it}" for it in args_cli.checkpoints.split(",") if it]
    run_dir = Path(args_cli.run)
    runner = None
    results = {"meta": {"task": args_cli.task, "num_envs": raw.num_envs, "control_steps": steps,
                        "physics_dt": physics_dt, "control_dt": raw.step_dt, "joints": names,
                        "velocity_limits": limits}}

    for name in policies:
        if name != "zero":
            if runner is None:
                agent = load_yaml(str(run_dir / "params" / "agent.yaml"))
                runner = OnPolicyRunner(env, agent, log_dir=None, device=raw.device)
            runner.load(str(run_dir / f"{name}.pt"))
            policy = runner.get_inference_policy(device=raw.device)
        torch.manual_seed(args_cli.seed)
        with torch.inference_mode():
            env.reset()
            raw.audit.clear()
            raw.audit_epoch.zero_()
            obs = env.get_observations()
            for _ in range(steps):
                actions = (torch.zeros((raw.num_envs, env_cfg.action_space), device=raw.device)
                           if name == "zero" else policy(obs))
                obs, _, _, _ = env.step(actions)

            q = torch.stack([sample[0] for sample in raw.audit])       # (T, E, J)
            qdot = torch.stack([sample[1] for sample in raw.audit])
            epoch = torch.stack([sample[2] for sample in raw.audit])   # (T, E)
            # A difference is only physical when both samples belong to the same episode.
            valid = epoch[1:] == epoch[:-1]
            qdot_fd = (q[1:] - q[:-1]) / physics_dt
            physx = statistics(qdot[1:], valid)
            finite = statistics(qdot_fd, valid)
            # What the env itself logs: PhysX qdot, unmasked, so quote it for continuity.
            logged_max = float(qdot.abs().max())

        limit = math.pi
        entry = {
            "logged_max_abs_joint_velocity": logged_max,
            "valid_samples": finite["samples"],
            "dropped_reset_samples": int((~valid).sum()),
            "physx": {"max": float(physx["max"].max()), "p99": float(physx["p99"].max()),
                      "rms": float(physx["rms"].mean())},
            "finite_difference": {"max": float(finite["max"].max()), "p99": float(finite["p99"].max()),
                                  "rms": float(finite["rms"].mean())},
            "fraction_above_pi": {
                "physx": float((qdot[1:].abs()[valid] > limit).float().mean()),
                "finite_difference": float((qdot_fd.abs()[valid] > limit).float().mean()),
            },
            "per_joint": {
                name: {"physx_max": float(physx["max"][j]), "physx_p99": float(physx["p99"][j]),
                       "physx_rms": float(physx["rms"][j]), "fd_max": float(finite["max"][j]),
                       "fd_p99": float(finite["p99"][j]), "fd_rms": float(finite["rms"][j])}
                for j, name in enumerate(names)
            },
        }
        worst = sorted(entry["per_joint"].items(), key=lambda kv: -kv[1]["physx_max"])[:5]
        entry["worst_joints_by_physx_max"] = [
            {"joint": n, **v} for n, v in worst]
        results[name] = entry
        print(f"[AUDIT] {name:>12} physx max={entry['physx']['max']:6.2f} p99={entry['physx']['p99']:5.2f}"
              f" rms={entry['physx']['rms']:5.2f} | fd max={entry['finite_difference']['max']:6.2f}"
              f" p99={entry['finite_difference']['p99']:5.2f} rms={entry['finite_difference']['rms']:5.2f}"
              f" | >pi physx={entry['fraction_above_pi']['physx']:.5f}"
              f" fd={entry['fraction_above_pi']['finite_difference']:.5f}", flush=True)
        for row in entry["worst_joints_by_physx_max"]:
            print(f"[AUDIT]     {row['joint']:12s} physx max={row['physx_max']:6.2f} p99={row['physx_p99']:5.2f}"
                  f"  fd max={row['fd_max']:6.2f} p99={row['fd_p99']:5.2f}", flush=True)

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(results, indent=2))
    print(f"[AUDIT] wrote {args_cli.output}", flush=True)
    raw.audit = None
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
