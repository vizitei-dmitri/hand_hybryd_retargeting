"""Check that the letter-cube visual matches the rigid-body collider and pose (no training).

Static: the visual's bounds in the body frame equal the collider box, the collider is
hidden, the visual carries no physics API and has an identity local rotation.
Dynamic: the pose PhysX publishes to Fabric for the renderer equals RigidObject.root_pose_w
(the source of the observed cube quaternion) while the cube tumbles; the renderer composes
the static visual transform under it. GUI screenshots show the tumbling colored cube.
"""

import argparse
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=45)
parser.add_argument("--screenshot_dir", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.utils.math import quat_from_euler_xyz, quat_error_magnitude, quat_mul

from isaaclab_tasks.utils import parse_env_cfg

import dg5f_isaaclab.tasks  # noqa: F401
import gymnasium as gym


def fabric_body_pose(rt_stage, path):
    prim = rt_stage.GetPrimAtPath(path)
    pos = prim.GetAttribute("_rigidBodyWorldPosition").Get()
    quat = prim.GetAttribute("_rigidBodyWorldOrientation").Get()
    return torch.tensor([pos[0], pos[1], pos[2]]), torch.tensor([quat.GetReal(), *quat.GetImaginary()])


def capture(env, path):
    import asyncio
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    capture = capture_viewport_to_file(get_active_viewport(), str(path.resolve()))
    result = asyncio.ensure_future(capture.wait_for_result())
    for _ in range(240):
        env.unwrapped.sim.render()
        if result.done() and path.exists() and path.stat().st_size > 0:
            print(f"[CUBE-CHECK] screenshot {path}")
            for _ in range(5):  # let the async PNG writer finish
                env.unwrapped.sim.render()
            return
    raise RuntimeError("Viewport capture timed out")


def main():
    cfg = parse_env_cfg("DG5F-Cube-Direct-v0", device=args_cli.device, num_envs=1)
    cfg.max_cube_distance_from_palm_m = 1e3
    cfg.min_cube_world_height_m = -1e3
    env = gym.make("DG5F-Cube-Direct-v0", cfg=cfg)
    raw = env.unwrapped
    stage = raw.sim.stage
    body = "/World/envs/env_0/Cube"
    visual = f"{body}/{cfg.object_cfg.spawn.visual_prim_name}"
    collider = f"{body}/geometry/mesh"

    # Static: visual bounds in the BODY frame must equal the collider box.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    body_prim = stage.GetPrimAtPath(body)
    rel = cache.ComputeRelativeBound(stage.GetPrimAtPath(visual), body_prim).ComputeAlignedRange()
    half = cfg.cube_size_m / 2
    lo, hi = torch.tensor(list(rel.GetMin())), torch.tensor(list(rel.GetMax()))
    print(f"[CUBE-CHECK] visual bounds in body frame min_mm={(1000 * lo).tolist()} max_mm={(1000 * hi).tolist()}"
          f" collider_half_mm={1000 * half}")
    assert torch.allclose(lo, torch.full((3,), -half), atol=2e-4) and torch.allclose(hi, torch.full((3,), half), atol=2e-4)
    collider_visible = UsdGeom.Imageable(stage.GetPrimAtPath(collider)).ComputeVisibility()
    print(f"[CUBE-CHECK] collider mesh visibility={collider_visible} (render uses only the letter-cube visual)")
    assert collider_visible == UsdGeom.Tokens.invisible
    visual_prim = stage.GetPrimAtPath(visual)
    assert visual_prim.GetParent() == body_prim
    physics_apis = (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI)
    subtree = list(Usd.PrimRange(visual_prim, Usd.TraverseInstanceProxies()))
    offending = [str(p.GetPath()) for p in subtree if any(p.HasAPI(api) for api in physics_apis)]
    print(f"[CUBE-CHECK] visual subtree prims={len(subtree)} with physics APIs={offending}")
    assert not offending
    local = Gf.Transform(UsdGeom.Xformable(visual_prim).GetLocalTransformation())
    angle = local.GetRotation().GetAngle()
    print(f"[CUBE-CHECK] visual local: rotation_deg={angle:.6f} scale={list(local.GetScale())}"
          f" translation_mm={[1000 * v for v in local.GetTranslation()]}")
    assert abs(angle) < 1e-6
    print(f"[CUBE-CHECK] cube mass={raw.cube.root_physx_view.get_masses()[0].tolist()} kg"
          f" bodies={raw.cube.num_bodies}")

    with torch.inference_mode():
        env.reset()
        for _ in range(30):
            env.step(torch.zeros((1, cfg.action_space), device=raw.device))
        if args_cli.screenshot_dir is not None:
            capture(env, args_cli.screenshot_dir / "in_hand.png")
        # Tumble the cube just above the palm with a known, non-symmetric orientation.
        rot = quat_from_euler_xyz(*(torch.tensor([v], device=raw.device) for v in
                                    (math.radians(35), math.radians(-60), math.radians(20))))
        pose = raw.cube.data.root_pose_w.clone()
        pose[:, 2] += 0.06
        pose[:, 3:7] = quat_mul(rot, pose[:, 3:7])
        raw.cube.write_root_pose_to_sim(pose)
        raw.cube.write_root_velocity_to_sim(torch.tensor([[0.0, 0.0, 0.0, 2.0, -1.0, 3.0]], device=raw.device))
        import omni.usd
        import usdrt
        rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        worst_rot, worst_pos, shots = 0.0, 0.0, 0
        for step in range(args_cli.steps):
            env.step(torch.zeros((1, cfg.action_space), device=raw.device))
            f_pos, f_quat = fabric_body_pose(rt_stage, body)
            b_pos, b_quat = raw.cube.data.root_pos_w[0].cpu(), raw.cube.data.root_quat_w[0].cpu()
            worst_rot = max(worst_rot, math.degrees(quat_error_magnitude(f_quat[None], b_quat[None]).item()))
            worst_pos = max(worst_pos, (f_pos - b_pos).norm().item())
            if args_cli.screenshot_dir is not None and step in (0, 8, args_cli.steps - 1):
                capture(env, args_cli.screenshot_dir / f"tumble_{shots}_step{step}.png")
                print(f"[CUBE-CHECK] step={step} cube quat_w={b_quat.tolist()}")
                shots += 1
        tilt = math.degrees(quat_error_magnitude(raw.cube.data.root_quat_w, raw.palm_quat_w).item())
        print(f"[CUBE-CHECK] after tumble: cube tilt vs palm={tilt:.1f} deg,"
              f" max Fabric-render-pose vs root_pose_w: rotation={worst_rot:.6f} deg position={1000 * worst_pos:.6f} mm")
        assert worst_rot < 1e-2 and worst_pos < 1e-5
        obs_quat = env.unwrapped._get_observations()["policy"][0, 43:47]
        print(f"[CUBE-CHECK] observation cube quaternion (palm frame)={obs_quat.tolist()}")
    print("[CUBE-CHECK] PASS")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
