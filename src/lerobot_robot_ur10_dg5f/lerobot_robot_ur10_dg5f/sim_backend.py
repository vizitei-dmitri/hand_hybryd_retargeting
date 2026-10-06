"""MuJoCo twin of the UR10e + DG5F cell for development without hardware."""

import mujoco
import numpy as np
from lerobot_robot_dg5f.constants import JOINT_NAMES
from scipy.spatial.transform import Rotation as R

from .backend import ARM_JOINTS, RobotState


ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
TCP_SITE = "attachment_site"  # UR flange, tool0
# The MJCF base body is yawed by pi: the MuJoCo world is ROS base_link, while the
# UR controller reports poses in its "base" frame.
UR_BASE_IN_WORLD = R.from_euler("z", np.pi)


def load_model(mjcf_path) -> mujoco.MjModel:
    """Compile with gravity compensation on the robot, as the UR controller does.

    Setting model.body_gravcomp after compilation has no effect: MuJoCo decides
    whether to compute gravcomp from model.ngravcomp, fixed at compile time.
    """
    spec = mujoco.MjSpec.from_file(str(mjcf_path))

    def enable_gravcomp(body):
        body.gravcomp = 1.0
        for child in body.bodies:
            enable_gravcomp(child)

    enable_gravcomp(spec.body("base"))
    model = spec.compile()
    # The scene asks for RK4. With the UR actuator damping (kv=500) and armature 0.1 that is
    # unstable at 2 ms, and the saturated wrists chatter. mujoco_menagerie uses implicitfast.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model


def solve_ik(model, ik_data, site_id, arm_qpos_idx, arm_dof_idx,
             target_pos_w, target_rot_w, iters=30, damping=1e-2, tol=1e-4):
    """Damped least squares: joint target that puts the site at the target pose (world frame)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)  # mj_jacSite needs both
        position = ik_data.site_xpos[site_id]
        rotation = ik_data.site_xmat[site_id].reshape(3, 3)
        error = np.concatenate([
            target_pos_w - position,
            R.from_matrix(target_rot_w @ rotation.T).as_rotvec(),
        ])
        if np.linalg.norm(error) < tol:
            break
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        jacobian = np.vstack([jacp, jacr])[:, arm_dof_idx]
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error)
        ik_data.qpos[arm_qpos_idx] += step
    return ik_data.qpos[arm_qpos_idx].copy()


class MujocoBackend:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.data = None
        self._renderer = None

    @property
    def is_connected(self) -> bool:
        return self.model is not None

    def connect(self) -> None:
        model = load_model(self.config.mjcf_path)
        self.model = model
        self.data = mujoco.MjData(model)
        self._ik_data = mujoco.MjData(model)
        self._arm_qpos = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self._arm_dof = np.array([model.joint(name).dofadr[0] for name in ARM_JOINTS])
        self._arm_act = np.array([model.actuator(name).id for name in ARM_ACTUATORS])
        self._hand_qpos = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])
        self._hand_act = np.array([model.actuator(f"{name}_ctrl").id for name in JOINT_NAMES])
        self._hand_lo, self._hand_hi = model.actuator_ctrlrange[self._hand_act].T
        self._site = model.site(TCP_SITE).id
        if self.config.sim_cameras:
            height, width = self.config.sim_image_hw
            self._renderer = mujoco.Renderer(model, height, width)
        self.go_home(np.asarray(self.config.home_joints_rad, dtype=np.float64))

    def disconnect(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.model = None
        self.data = None

    def read(self) -> RobotState:
        return RobotState(
            tcp=self._tcp_in_ur_base(),
            arm_q=self.data.qpos[self._arm_qpos].copy(),
            hand_deg=np.rad2deg(self.data.qpos[self._hand_qpos]),
        )

    def command(self, tcp_target, hand_deg) -> None:
        position_w = UR_BASE_IN_WORLD.apply(tcp_target[:3])
        rotation_w = (UR_BASE_IN_WORLD * R.from_rotvec(tcp_target[3:6])).as_matrix()
        # Start IK from the current pose so it stays on the same branch (elbow up/down).
        self._ik_data.qpos[:] = self.data.qpos
        self.data.ctrl[self._arm_act] = solve_ik(
            self.model, self._ik_data, self._site, self._arm_qpos, self._arm_dof,
            position_w, rotation_w)
        self.data.ctrl[self._hand_act] = np.clip(np.deg2rad(hand_deg), self._hand_lo, self._hand_hi)

    def wait_next_period(self, period_s: float) -> None:
        substeps = max(1, round(period_s / self.model.opt.timestep))
        mujoco.mj_step(self.model, self.data, nstep=substeps)

    def go_home(self, arm_q) -> None:
        """Simulation shortcut: reset the scene and place the arm at home instantly."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qpos] = arm_q
        self.data.ctrl[self._arm_act] = arm_q
        self.data.ctrl[self._hand_act] = np.clip(0.0, self._hand_lo, self._hand_hi)
        mujoco.mj_forward(self.model, self.data)

    def images(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            return {}
        frames = {}
        for camera in self.config.sim_cameras:
            self._renderer.update_scene(self.data, camera=camera)
            frames[camera] = self._renderer.render().copy()
        return frames

    def _tcp_in_ur_base(self) -> np.ndarray:
        to_base = UR_BASE_IN_WORLD.inv()
        position = to_base.apply(self.data.site_xpos[self._site])
        rotation = to_base * R.from_matrix(self.data.site_xmat[self._site].reshape(3, 3))
        return np.concatenate([position, rotation.as_rotvec()])
