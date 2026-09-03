"""ROS 2 position-command bridge for the local DG5F MuJoCo model."""

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from dg5f_teleop.constants import ACTUATOR_NAMES, JOINT_NAMES


SELF_COLLISION_MODES = ("full", "tip_only", "off")


def enforce_fixed_joint_state(
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    qvel_addresses: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Keep mechanically unavailable joints fixed in the digital twin."""
    data.qpos[np.asarray(qpos_addresses, dtype=np.int64)] = positions
    data.qvel[np.asarray(qvel_addresses, dtype=np.int64)] = 0.0


def configure_position_actuators(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    kp: float,
    kd: float,
) -> None:
    """Configure the affine MuJoCo actuators as position servos."""
    if not np.isfinite(kp) or kp <= 0.0:
        raise ValueError("actuator_kp must be finite and positive")
    if not np.isfinite(kd) or kd < 0.0:
        raise ValueError("actuator_kd must be finite and non-negative")

    actuator_ids = np.asarray(actuator_ids, dtype=np.int64)
    model.actuator_gainprm[actuator_ids, 0] = kp
    model.actuator_biasprm[actuator_ids, 1] = -kp
    model.actuator_biasprm[actuator_ids, 2] = -kd


def configure_hand_self_collision(
    model: mujoco.MjModel, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Select hand self-collisions while retaining environment contacts.

    The DG5F collision meshes use group 1.  Separate collision categories let
    those meshes continue to contact ordinary environment geoms (category 1)
    without every finger link blocking every other finger link.  In
    ``tip_only`` mode, fingertip collision meshes get their own category and
    can still contact one another.
    """
    mode = str(mode).lower()
    if mode not in SELF_COLLISION_MODES:
        raise ValueError(
            f"self_collision_mode must be one of {SELF_COLLISION_MODES}, got {mode!r}"
        )

    hand_geom_ids = np.flatnonzero(np.asarray(model.geom_group) == 1)
    if mode == "full":
        return hand_geom_ids, np.empty(0, dtype=np.int64)

    # Hand vs hand: disabled. Hand vs a normal environment geom: retained.
    model.geom_contype[hand_geom_ids] = 2
    model.geom_conaffinity[hand_geom_ids] = 1

    tip_geom_ids = []
    if mode == "tip_only":
        for geom_id in hand_geom_ids:
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
            )
            if name and name.endswith("_tip_collision"):
                tip_geom_ids.append(int(geom_id))

        tip_geom_ids = np.asarray(tip_geom_ids, dtype=np.int64)
        # Tip vs tip and tip vs environment: enabled. Tip vs other hand links:
        # disabled, so an unrelated phalanx cannot prevent a requested pinch.
        model.geom_contype[tip_geom_ids] = 4
        model.geom_conaffinity[tip_geom_ids] = 5
    else:
        tip_geom_ids = np.empty(0, dtype=np.int64)

    return hand_geom_ids, tip_geom_ids


class MujocoBridgeNode(Node):
    """Run MuJoCo in real time and apply named JointTrajectory targets."""

    def __init__(self) -> None:
        super().__init__("dg5f_mujoco")

        self.declare_parameter("model_path", "")
        self.declare_parameter("command_topic", "/dg5f/joint_command")
        self.declare_parameter("joint_state_topic", "/dg5f/joint_states")
        self.declare_parameter("control_hz", 200.0)
        self.declare_parameter("state_publish_hz", 50.0)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("actuator_kp", 40.0)
        self.declare_parameter("actuator_kd", 0.5)
        self.declare_parameter("self_collision_mode", "tip_only")
        self.declare_parameter("fixed_joints", ["rj_dg_5_1"])
        self.declare_parameter("fixed_joint_positions", [0.0])
        self.declare_parameter("render", True)

        model_path = Path(str(self.get_parameter("model_path").value))
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")

        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)

        self._actuator_ids = self._resolve_ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR_NAMES, "actuator"
        )
        self._joint_ids = self._resolve_ids(
            mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES, "joint"
        )
        self._qpos_addresses = self._model.jnt_qposadr[self._joint_ids]
        self._qvel_addresses = self._model.jnt_dofadr[self._joint_ids]
        self._ctrl_limits = self._model.actuator_ctrlrange[self._actuator_ids].copy()

        fixed_names = list(self.get_parameter("fixed_joints").value)
        fixed_positions = np.asarray(
            self.get_parameter("fixed_joint_positions").value,
            dtype=np.float64,
        )
        if len(fixed_names) != len(fixed_positions):
            raise ValueError(
                "fixed_joints and fixed_joint_positions must have equal length"
            )
        unknown_fixed = [name for name in fixed_names if name not in JOINT_NAMES]
        if unknown_fixed:
            raise ValueError(f"Unknown fixed DG5F joints: {unknown_fixed}")
        if not np.all(np.isfinite(fixed_positions)):
            raise ValueError("Fixed DG5F positions must be finite")
        self._fixed_indices = np.asarray(
            [JOINT_NAMES.index(name) for name in fixed_names], dtype=np.int64
        )
        self._fixed_positions = fixed_positions
        self._fixed_qpos_addresses = self._qpos_addresses[self._fixed_indices]
        self._fixed_qvel_addresses = self._qvel_addresses[self._fixed_indices]

        actuator_kp = float(self.get_parameter("actuator_kp").value)
        actuator_kd = float(self.get_parameter("actuator_kd").value)
        configure_position_actuators(
            self._model, self._actuator_ids, actuator_kp, actuator_kd
        )
        self_collision_mode = str(
            self.get_parameter("self_collision_mode").value
        ).lower()
        hand_geoms, tip_geoms = configure_hand_self_collision(
            self._model, self_collision_mode
        )

        self._target = self._data.qpos[self._qpos_addresses].copy()
        self._target[self._fixed_indices] = self._fixed_positions
        enforce_fixed_joint_state(
            self._data,
            self._fixed_qpos_addresses,
            self._fixed_qvel_addresses,
            self._fixed_positions,
        )
        self._last_command_time: Optional[float] = None
        self._timeout_reported = False

        self._command_subscription = self.create_subscription(
            JointTrajectory,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            1,
        )
        self._state_publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            5,
        )

        control_hz = float(self.get_parameter("control_hz").value)
        state_publish_hz = float(self.get_parameter("state_publish_hz").value)
        if control_hz <= 0.0 or state_publish_hz <= 0.0:
            raise ValueError("control_hz and state_publish_hz must be positive")

        self._physics_substeps = max(
            1, int(round((1.0 / control_hz) / self._model.opt.timestep))
        )
        self._state_publish_every = max(1, int(round(control_hz / state_publish_hz)))
        self._control_count = 0
        self._viewer = None

        if bool(self.get_parameter("render").value):
            try:
                from mujoco import viewer as mujoco_viewer

                self._viewer = mujoco_viewer.launch_passive(self._model, self._data)
            except Exception as error:
                self.get_logger().error(
                    f"MuJoCo viewer could not start; simulation remains active: {error}"
                )

        self._timer = self.create_timer(1.0 / control_hz, self._step)
        effective_hz = 1.0 / (self._physics_substeps * self._model.opt.timestep)
        self.get_logger().info(
            "Loaded %s: %d joints, %d actuators, effective control %.1f Hz, "
            "Kp=%.1f Kd=%.2f, self-collision=%s (%d hand/%d tip geoms)"
            % (
                model_path,
                len(self._joint_ids),
                len(self._actuator_ids),
                effective_hz,
                actuator_kp,
                actuator_kd,
                self_collision_mode,
                len(hand_geoms),
                len(tip_geoms),
            )
        )

    def _resolve_ids(self, object_type, names, label: str) -> np.ndarray:
        ids = np.asarray(
            [mujoco.mj_name2id(self._model, object_type, name) for name in names],
            dtype=np.int64,
        )
        missing = [name for name, object_id in zip(names, ids) if object_id < 0]
        if missing:
            raise RuntimeError(f"MuJoCo model is missing {label}s: {missing}")
        return ids

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_command(self, message: JointTrajectory) -> None:
        if not message.points:
            self.get_logger().warning("Ignoring JointTrajectory without points")
            return
        positions = message.points[-1].positions
        if len(message.joint_names) != len(positions):
            self.get_logger().warning(
                "Ignoring command: joint_names and positions have different lengths"
            )
            return

        command_by_name = dict(zip(message.joint_names, positions))
        missing = [name for name in JOINT_NAMES if name not in command_by_name]
        if missing:
            self.get_logger().warning(f"Ignoring command missing DG5F joints: {missing}")
            return

        target = np.asarray([command_by_name[name] for name in JOINT_NAMES])
        if not np.all(np.isfinite(target)):
            self.get_logger().warning("Ignoring command containing NaN or infinity")
            return

        self._target = np.clip(
            target,
            self._ctrl_limits[:, 0],
            self._ctrl_limits[:, 1],
        )
        self._target[self._fixed_indices] = self._fixed_positions
        self._last_command_time = self._now_seconds()
        self._timeout_reported = False

    def _step(self) -> None:
        if self._last_command_time is not None:
            timeout = float(self.get_parameter("command_timeout").value)
            if self._now_seconds() - self._last_command_time > timeout:
                if not self._timeout_reported:
                    self.get_logger().warning(
                        "DG5F command timeout: holding the last actuator targets"
                    )
                    self._timeout_reported = True

        self._data.ctrl[self._actuator_ids] = self._target
        enforce_fixed_joint_state(
            self._data,
            self._fixed_qpos_addresses,
            self._fixed_qvel_addresses,
            self._fixed_positions,
        )
        mujoco.mj_step(self._model, self._data, nstep=self._physics_substeps)
        enforce_fixed_joint_state(
            self._data,
            self._fixed_qpos_addresses,
            self._fixed_qvel_addresses,
            self._fixed_positions,
        )
        mujoco.mj_forward(self._model, self._data)
        self._control_count += 1

        if self._control_count % self._state_publish_every == 0:
            self._publish_joint_state()
        if self._viewer is not None and self._control_count % self._state_publish_every == 0:
            if self._viewer.is_running():
                self._viewer.sync()

    def _publish_joint_state(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "dg5f"
        message.name = list(JOINT_NAMES)
        message.position = self._data.qpos[self._qpos_addresses].tolist()
        message.velocity = self._data.qvel[self._qvel_addresses].tolist()
        message.effort = self._data.qfrc_actuator[self._qvel_addresses].tolist()
        self._state_publisher.publish(message)

    def destroy_node(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[MujocoBridgeNode] = None
    try:
        node = MujocoBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
