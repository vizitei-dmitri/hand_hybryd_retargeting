"""Letter cube (NVIDIA DexCube look) as ONE rigid body with a simple box collider.

Visual: the stock Isaac Sim DexCube used by the Isaac Lab Allegro/Shadow in-hand
examples (six colored faces with the large letters X, R, E, T, M, D).
Only its `/DexCube/visuals` subtree is referenced: the stock root carries its own
rigid body, mass and collider, which must not be nested inside our body.
Physics: a hidden USD cube (collider + rigid body + mass) spawned by spawn_cuboid.
The visual is centered and uniformly scaled to the collider edge with identity
rotation, so the visible faces move exactly with the simulated body frame.
"""

from collections.abc import Callable

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
from isaaclab.sim.utils import create_prim, get_current_stage, standardize_xform_ops
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Standard Isaac Sim asset root (persistent.isaac.asset_root.cloud), not a fixed URL.
DEX_CUBE_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
DEX_CUBE_VISUAL_PRIM = "/DexCube/visuals"


def _aligned_bounds(prim: Usd.Prim) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeLocalBound(prim).ComputeAlignedRange()


@sim_utils.clone
def spawn_visual_cuboid(prim_path, cfg: "VisualCuboidCfg", translation=None, orientation=None, **kwargs):
    if len(set(cfg.size)) != 1:
        raise ValueError("The cube visual can only be fitted to a cube collider")
    prim = spawn_cuboid.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    # Physics still uses the hidden collision mesh; renderers/cameras only see the visual.
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")).MakeInvisible()
    visual = create_prim(f"{prim_path}/{cfg.visual_prim_name}", "Xform", stage=stage)
    visual.GetReferences().AddReference(cfg.visual_usd_path, cfg.visual_prim_in_asset)
    # Local bound includes only the referenced prim's own (identity) transform.
    bounds = _aligned_bounds(visual)
    extent, center = bounds.GetSize(), bounds.GetMidpoint()
    if bounds.IsEmpty() or max(extent) - min(extent) > cfg.visual_max_aspect_error * max(extent):
        raise ValueError(f"Unexpected visual bounds for {cfg.visual_usd_path}: {bounds}")
    scale = cfg.size[0] / max(extent)
    # No rotation: the visual frame is exactly the rigid-body frame.
    standardize_xform_ops(
        visual, translation=tuple(-scale * c for c in center), orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(scale,) * 3,
    )
    print(f"[CUBE] visual={cfg.visual_usd_path}{cfg.visual_prim_in_asset} native_edge_m={max(extent):.6f}"
          f" scale={scale:.6f} collider_edge_m={cfg.size[0]:.4f}")
    return prim


@configclass
class VisualCuboidCfg(sim_utils.CuboidCfg):
    """Cube collider/rigid body whose visible geometry is a referenced USD subtree."""

    func: Callable = spawn_visual_cuboid
    visual_usd_path: str = DEX_CUBE_USD_PATH
    visual_prim_in_asset: str = DEX_CUBE_VISUAL_PRIM
    visual_prim_name: str = "visual"
    visual_max_aspect_error: float = 0.01
