"""DG5F asset for Isaac Lab 2.3.2 / Isaac Sim 5.1 (all angles in radians)."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from pxr import UsdPhysics

from .sysid import DEFAULT_SYSID_PATH, load_sysid


EXTERNAL_PROJECT_ROOT = Path(__file__).resolve().parents[4]
URDF_PATH = EXTERNAL_PROJECT_ROOT.parents[1] / "models/dg5f/urdf/dg5f_right.urdf"
USD_CACHE_PATH = EXTERNAL_PROJECT_ROOT / "assets/usd_cache"

# Resolve metadata from the actual model, not from a separate SDK ordering.
_model = ET.parse(URDF_PATH).getroot()
_joints = [joint for joint in _model.findall("joint") if joint.get("type") != "fixed"]
if len(_joints) != 20 or any(joint.get("type") != "revolute" for joint in _joints):
    raise ValueError("DG5F teacher expects exactly 20 bounded revolute joints in the URDF")
DG5F_JOINT_NAMES = tuple(joint.attrib["name"] for joint in _joints)
DG5F_JOINT_LIMITS = tuple(
    (float(joint.find("limit").attrib["lower"]), float(joint.find("limit").attrib["upper"]))
    for joint in _joints
)
DG5F_JOINT_EFFORT_LIMITS = {joint.attrib["name"]: float(joint.find("limit").attrib["effort"]) for joint in _joints}
DG5F_JOINT_VELOCITY_LIMITS = {joint.attrib["name"]: float(joint.find("limit").attrib["velocity"]) for joint in _joints}
DG5F_FINGERTIP_NAMES = tuple(link.attrib["name"] for link in _model.findall("link") if link.attrib["name"].endswith("_tip"))
(_palm,) = [link.attrib["name"] for link in _model.findall("link") if link.attrib["name"].endswith("_palm")]
DG5F_PALM_NAME = _palm
DG5F_SYSID = load_sysid(DEFAULT_SYSID_PATH, DG5F_JOINT_NAMES)
# Hardware identity from the real-hand stack (not assumed from the URDF name):
# vendor/tesollo_control uses DGSDK model DG_MODEL_DG_5F_RIGHT = 0x5F22 (the 0x..02
# suffix is the M series, cf. DG_1F_M/DG_3F_M), Modbus TCP port 502, and the URDF
# comes from tesollodelto/delto_m_ros2. Physical label not inspected.
DG5F_HARDWARE_MODEL = "TESOLLO DG-5F-M (right)"
# DG-5F-M per-joint data sheet values (Knoxlabs DG-5F-M listing; stall torque and
# no-load speed also in Tesollo DG-5F press material). Metadata only unless used below.
DG5F_RATED_JOINT_TORQUE_NM = 0.4
DG5F_STALL_JOINT_TORQUE_NM = 2.0
DG5F_NO_LOAD_SPEED_RPM = 75.0
DG5F_CONTROL_FREQUENCY_HZ = 250.0
# ImplicitActuatorCfg has one effort limit and no continuous/peak or thermal model:
# use the rated (continuous) torque, never stall torque, as the simulated limit.
# TODO: replace with a current/thermal actuator model identified on the real hand.
DG5F_DEFAULT_EFFORT_CAP_NM = DG5F_RATED_JOINT_TORQUE_NM
assert len(DG5F_FINGERTIP_NAMES) == 5
for mesh in _model.iter("mesh"):
    mesh_path = (URDF_PATH.parent / mesh.attrib["filename"]).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

# Initial curl in URDF order, checked against FK; TODO validate contact preload
# with the real object's size/material, then identify the physical DG5F gains.
_grasp_deg = (10, -90, 45, 35, 0, 40, 65, 30, 0, 35, 65, 30, 0, 35, 65, 30, 0, -10, 55, 40)
DG5F_INITIAL_JOINT_POS = dict(zip(DG5F_JOINT_NAMES, map(math.radians, _grasp_deg)))
for name, (lower, upper) in zip(DG5F_JOINT_NAMES, DG5F_JOINT_LIMITS):
    assert lower <= DG5F_INITIAL_JOINT_POS[name] <= upper, name

# PhysX only filters DIRECT parent/child link contacts. With merge_fixed_joints=False the
# thumb base (child of rl_dg_palm) overlaps the fixed rl_dg_base housing and stays in
# permanent deep penetration (~0.2-1.4 MN contact force, phantom thumb joint velocity).
# These parts are rigidly mounted next to each other on the real hand.
DG5F_FILTERED_COLLISION_PAIRS = (("rl_dg_1_1", "rl_dg_base"),)


@sim_utils.clone
def spawn_dg5f(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Import with the 2.3.2 loader, then apply offsets to editable collisions.

    The 2.3.2 converter does not forward make_instanceable to the URDF importer.
    Honor it here for visual debugging, while always keeping collisions editable.
    """
    prim = sim_utils.spawn_from_urdf(
        prim_path, cfg.replace(collision_props=None), translation, orientation, **kwargs
    )
    if not cfg.make_instanceable:
        sim_utils.make_uninstanceable(prim.GetPath(), stage=prim.GetStage())
    else:
        for body in prim.GetChildren():
            collisions = body.GetChild("collisions")
            if collisions.IsValid():
                sim_utils.make_uninstanceable(collisions.GetPath(), stage=prim.GetStage())
    if cfg.collision_props is not None:
        sim_utils.modify_collision_properties(prim_path, cfg.collision_props, stage=prim.GetStage())
    stage = prim.GetStage()
    for body, other in DG5F_FILTERED_COLLISION_PAIRS:
        body_prim, other_prim = stage.GetPrimAtPath(f"{prim_path}/{body}"), stage.GetPrimAtPath(f"{prim_path}/{other}")
        if not (body_prim.IsValid() and other_prim.IsValid()):
            raise RuntimeError(f"Cannot filter collisions between missing links {body} and {other}")
        UsdPhysics.FilteredPairsAPI.Apply(body_prim).CreateFilteredPairsRel().AddTarget(other_prim.GetPath())
    return prim


DG5F_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_dg5f,
        asset_path=str(URDF_PATH),
        usd_dir=str(USD_CACHE_PATH),
        usd_file_name="dg5f_right.usd",
        fix_base=True,
        # Preserve the real palm/tip bodies so their states are directly readable.
        merge_fixed_joints=False,
        make_instanceable=False,
        collision_from_visuals=False,
        # Convex hulls of the concave palm swallow the thumb base (permanent ~1 kN contact).
        collider_type="convex_decomposition",
        self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force", target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, max_depenetration_velocity=0.2,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        # URDF palm normal +X becomes world +Z (palm up).
        rot=(math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
        joint_pos=DG5F_INITIAL_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=list(DG5F_JOINT_NAMES),
            stiffness=DG5F_SYSID.parameters["stiffness"],
            damping=DG5F_SYSID.parameters["damping"],
            armature=DG5F_SYSID.parameters["armature"],
            friction=DG5F_SYSID.parameters["friction"],
            effort_limit_sim={
                name: min(DG5F_DEFAULT_EFFORT_CAP_NM, limit)
                for name, limit in DG5F_JOINT_EFFORT_LIMITS.items()
            },
            velocity_limit_sim=DG5F_JOINT_VELOCITY_LIMITS,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
