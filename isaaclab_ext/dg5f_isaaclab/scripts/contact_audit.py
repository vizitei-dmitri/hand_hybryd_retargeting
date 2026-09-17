"""Contact-pair audit for the DG5F grasp (diagnostic only; not part of the PPO env).

Adds one filtered ContactSensor per hand link in a diagnostic subclass and prints every
link-link and link-cube pair with a nonzero contact force after a zero-action hold.
Persistent large self-contact forces reveal overlapping collision geometry.
"""

import argparse
import math
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--no_reset_noise", action="store_true")
parser.add_argument("--min_force", type=float, default=0.01, help="N")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg

from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401
from dg5f_isaaclab.assets.dg5f import URDF_PATH
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env import DG5FCubeEnv

import xml.etree.ElementTree as ET

LINKS = [link.get("name") for link in ET.parse(URDF_PATH).getroot().findall("link")
         if link.find("collision") is not None]
ENV = "/World/envs/env_.*"
TARGETS = {name: f"{ENV}/Robot/{name}" for name in LINKS} | {"cube": f"{ENV}/Cube"}


class ContactAuditEnv(DG5FCubeEnv):
    def _setup_scene(self):
        super()._setup_scene()
        self.audit = {}
        for link in LINKS:
            others = [name for name in TARGETS if name != link]
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=TARGETS[link], filter_prim_paths_expr=[TARGETS[name] for name in others]))
            self.scene.sensors[f"audit_{link}"] = sensor
            self.audit[link] = (sensor, others)


def main():
    cfg = parse_env_cfg("DG5F-Cube-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.robot_cfg.spawn.activate_contact_sensors = True
    if args_cli.no_reset_noise:
        cfg.reset_joint_position_noise_rad = 0.0
        cfg.cube_reset_position_noise_m = 0.0
    print(f"[AUDIT] collider={cfg.robot_cfg.spawn.collider_type} links={len(LINKS)}")
    env = ContactAuditEnv(cfg)
    with torch.inference_mode():
        env.reset()
        for _ in range(args_cli.steps):
            env.step(torch.zeros((env.num_envs, cfg.action_space), device=env.device))
        for env_id in range(env.num_envs):
            pairs = {}
            for link, (sensor, others) in env.audit.items():
                forces = sensor.data.force_matrix_w[env_id, 0].norm(dim=-1)
                for column, other in enumerate(others):
                    force = forces[column].item()
                    if force > args_cli.min_force:
                        key = tuple(sorted((link, other)))
                        pairs[key] = max(pairs.get(key, 0.0), force)
            self_pairs = {k: v for k, v in pairs.items() if "cube" not in k}
            cube_pairs = {k: v for k, v in pairs.items() if "cube" in k}
            q = env.hand.data.joint_pos[env_id, env.joint_ids]
            print(f"[AUDIT] env={env_id} thumb_q_deg={[round(math.degrees(v), 2) for v in q[:4].tolist()]}")
            print(f"[AUDIT] env={env_id} self_contacts(N)={ {f'{a}-{b}': round(v, 3) for (a, b), v in self_pairs.items()} }")
            print(f"[AUDIT] env={env_id} cube_contacts(N)={ {a if b == 'cube' else b: round(v, 3) for (a, b), v in cube_pairs.items()} }")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
