"""Quest landmarks -> DG5F retargeting.

Hybrid retargeting:
Vector keeps anatomical phalanx shape.
Near thumb-to-finger contact, DexPilot is warm-started from the Vector pose and
only a bounded part of its correction is mixed into the active fingers.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from dex_retargeting.retargeting_config import RetargetingConfig
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from dg5f_teleop.constants import JOINT_NAMES
from dg5f_teleop.hand_math import landmarks_to_mano


NON_THUMB_DISTAL_FLEXION_JOINTS = tuple(
    f"rj_dg_{finger}_{joint}" for finger in range(2, 6) for joint in (3, 4)
)
TIP_LANDMARKS = (4, 8, 12, 16, 20)


def prevent_distal_hyperextension(retargeting, minimum: float = 0.0):
    if not np.isfinite(minimum):
        raise ValueError("Distal flexion minimum must be finite")

    optimizer = retargeting.optimizer
    target_names = list(optimizer.target_joint_names)
    limits = np.asarray(retargeting.joint_limits, dtype=np.float64).copy()
    changed = []

    for name in NON_THUMB_DISTAL_FLEXION_JOINTS:
        if name not in target_names:
            continue
        index = target_names.index(name)
        if minimum > limits[index, 1]:
            raise ValueError(
                f"Distal flexion minimum {minimum} exceeds {name} upper limit "
                f"{limits[index, 1]}"
            )
        if limits[index, 0] < minimum:
            limits[index, 0] = minimum
            changed.append(name)

    optimizer.set_joint_limit(limits, epsilon=0.0)
    retargeting.joint_limits = limits
    retargeting.last_qpos = np.clip(
        np.asarray(retargeting.last_qpos), limits[:, 0], limits[:, 1]
    )
    return changed


def reference_for(retargeting, mano_points: np.ndarray) -> np.ndarray:
    indices = np.asarray(
        retargeting.optimizer.target_link_human_indices, dtype=np.int64
    )
    if retargeting.optimizer.retargeting_type == "POSITION":
        return mano_points[indices, :]
    return mano_points[indices[1, :], :] - mano_points[indices[0, :], :]


def smooth_contact_weight(distance, contact_start, contact_full, max_blend):
    if contact_start <= contact_full:
        raise ValueError("hybrid_contact_start must be > hybrid_contact_full")
    if distance >= contact_start:
        return 0.0
    if distance <= contact_full:
        return max_blend
    x = (contact_start - distance) / (contact_start - contact_full)
    return float(max_blend * x * x * (3.0 - 2.0 * x))


class RetargetNode(Node):
    def __init__(self) -> None:
        super().__init__("dg5f_retarget")

        self.declare_parameter("input_topic", "/hands/right/landmarks")
        self.declare_parameter("command_topic", "/dg5f/joint_command")
        self.declare_parameter("target_state_topic", "/dg5f/target_joint_states")
        self.declare_parameter("tracking_topic", "/dg5f/tracking_ok")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("retargeting_config", "")
        self.declare_parameter("scaling_factor", 1.0)
        self.declare_parameter("max_joint_velocity", 3.0)
        self.declare_parameter("watchdog_timeout", 0.35)
        self.declare_parameter("input_reliability", "reliable")
        self.declare_parameter("prevent_distal_hyperextension", False)
        self.declare_parameter("distal_flexion_min", 0.0)

        self.declare_parameter(
            "hybrid_dexpilot_config",
            "/workspace/src/dg5f_teleop/config/dg5f_dexpilot.yaml",
        )
        self.declare_parameter("hybrid_contact_start", 0.055)
        self.declare_parameter("hybrid_contact_full", 0.025)
        self.declare_parameter("hybrid_max_blend", 0.80)
        self.declare_parameter("hybrid_max_joint_correction", 0.45)
        self.declare_parameter("hybrid_lateral_max_correction", 0.12)
        self.declare_parameter("hybrid_correction_alpha", 0.35)
        self.declare_parameter("hybrid_release_alpha", 0.18)

        urdf_path = Path(str(self.get_parameter("urdf_path").value))
        config_path = Path(str(self.get_parameter("retargeting_config").value))
        if not urdf_path.is_file():
            raise FileNotFoundError(f"DG5F URDF does not exist: {urdf_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Retarget config does not exist: {config_path}")

        scale = float(self.get_parameter("scaling_factor").value)
        config = RetargetingConfig.load_from_file(
            config_path,
            override={"urdf_path": str(urdf_path), "scaling_factor": scale},
        )
        self._retargeting = config.build()

        if bool(self.get_parameter("prevent_distal_hyperextension").value):
            prevent_distal_hyperextension(
                self._retargeting,
                float(self.get_parameter("distal_flexion_min").value),
            )

        self._primary_names = list(self._retargeting.joint_names)
        missing = [n for n in JOINT_NAMES if n not in self._primary_names]
        if missing:
            raise RuntimeError(f"Primary retargeting is missing joints: {missing}")
        self._output_indices = np.asarray(
            [self._primary_names.index(n) for n in JOINT_NAMES], dtype=np.int64
        )

        self._hybrid = config_path.stem == "dg5f_hybrid"
        self._contact_retargeting = None
        self._contact_names = None
        self._contact_output_indices = None

        if self._hybrid:
            if self._retargeting.optimizer.retargeting_type != "VECTOR":
                raise RuntimeError("Hybrid primary config must be Vector")

            dex_path = Path(str(self.get_parameter("hybrid_dexpilot_config").value))
            dex_config = RetargetingConfig.load_from_file(
                dex_path,
                override={
                    "urdf_path": str(urdf_path),
                    "scaling_factor": scale,
                    # Vector already performs low-pass filtering.
                    "low_pass_alpha": 1.0,
                },
            )
            self._contact_retargeting = dex_config.build()
            self._contact_names = list(self._contact_retargeting.joint_names)
            self._contact_output_indices = np.asarray(
                [self._contact_names.index(n) for n in JOINT_NAMES], dtype=np.int64
            )
            self.get_logger().info(
                "Hybrid: Vector shape + bounded DexPilot contact correction"
            )

        reliability_name = str(self.get_parameter("input_reliability").value).lower()
        reliability = (
            QoSReliabilityPolicy.BEST_EFFORT
            if reliability_name == "best_effort"
            else QoSReliabilityPolicy.RELIABLE
        )
        input_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=reliability,
        )

        self._command_publisher = self.create_publisher(
            JointTrajectory, str(self.get_parameter("command_topic").value), 1
        )
        self._target_state_publisher = self.create_publisher(
            JointState, str(self.get_parameter("target_state_topic").value), 1
        )
        self._tracking_publisher = self.create_publisher(
            Bool, str(self.get_parameter("tracking_topic").value), 1
        )
        self._subscription = self.create_subscription(
            PoseArray,
            str(self.get_parameter("input_topic").value),
            self._on_landmarks,
            input_qos,
        )

        self._last_output: Optional[np.ndarray] = None
        self._last_frame_time: Optional[float] = None
        self._last_command_time: Optional[float] = None
        self._tracking_ok = False
        self._last_error_log_time = -1e9
        self._last_hybrid_log_time = -1e9
        self._hybrid_correction_state = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._watchdog_timer = self.create_timer(0.1, self._watchdog)

        mode = (
            "HYBRID(Vector+DexPilot)"
            if self._hybrid
            else self._retargeting.optimizer.retargeting_type
        )
        self.get_logger().info(
            f"Ready: {self.get_parameter('input_topic').value} -> "
            f"{self.get_parameter('command_topic').value} "
            f"({mode}, 20 DG5F joints, scale {scale:.3f})"
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_throttled(self, text: str) -> None:
        now = self._now_seconds()
        if now - self._last_error_log_time >= 1.0:
            self.get_logger().warning(text)
            self._last_error_log_time = now

    def _set_tracking(self, value: bool) -> None:
        self._tracking_publisher.publish(Bool(data=value))
        self._tracking_ok = value

    def _primary_to_contact_seed(self, primary_qpos: np.ndarray) -> np.ndarray:
        by_name = dict(zip(self._primary_names, primary_qpos))
        return np.asarray(
            [by_name[name] for name in self._contact_names], dtype=np.float64
        )

    def _hybrid_weights(self, mano_points: np.ndarray) -> np.ndarray:
        start = float(self.get_parameter("hybrid_contact_start").value)
        full = float(self.get_parameter("hybrid_contact_full").value)
        max_blend = float(self.get_parameter("hybrid_max_blend").value)

        if not 0.0 <= max_blend <= 1.0:
            raise ValueError("hybrid_max_blend must be in [0,1]")

        thumb = mano_points[TIP_LANDMARKS[0]]
        other = mano_points[np.asarray(TIP_LANDMARKS[1:], dtype=np.int64)]
        distances = np.linalg.norm(other - thumb[None, :], axis=1)
        other_weights = np.asarray(
            [
                smooth_contact_weight(float(d), start, full, max_blend)
                for d in distances
            ],
            dtype=np.float64,
        )
        return np.concatenate([[float(np.max(other_weights))], other_weights])

    def _apply_hybrid(
        self,
        mano_points: np.ndarray,
        primary_all: np.ndarray,
        primary_target: np.ndarray,
    ) -> np.ndarray:
        weights = self._hybrid_weights(mano_points)
        if np.max(weights) <= 0.0:
            release_alpha = float(
                self.get_parameter("hybrid_release_alpha").value
            )
            release_alpha = float(np.clip(release_alpha, 0.0, 1.0))
            self._hybrid_correction_state += release_alpha * (
                -self._hybrid_correction_state
            )
            return primary_target + self._hybrid_correction_state

        # Important: DexPilot starts from the current Vector pose.
        # Its previous-pose regularizer therefore prefers the natural Vector branch.
        seed = self._primary_to_contact_seed(primary_all)
        self._contact_retargeting.set_qpos(seed)

        contact_ref = reference_for(self._contact_retargeting, mano_points)
        contact_all = self._contact_retargeting.retarget(contact_ref)
        contact_target = np.asarray(
            contact_all[self._contact_output_indices], dtype=np.float64
        )

        delta = contact_target - primary_target
        max_corr = float(
            self.get_parameter("hybrid_max_joint_correction").value
        )
        if max_corr <= 0.0:
            raise ValueError("hybrid_max_joint_correction must be > 0")

        result = primary_target.copy()

        # Thumb needs full opposition freedom.
        result[0:4] += np.clip(
            delta[0:4] * weights[0], -max_corr, max_corr
        )

        # Vector still owns the anatomy, but near contact DexPilot gets
        # a small amount of the *_1 lateral/spread joint as well.
        lateral_max = float(
            self.get_parameter("hybrid_lateral_max_correction").value
        )
        if lateral_max < 0.0:
            raise ValueError("hybrid_lateral_max_correction must be >= 0")

        for finger in range(1, 5):
            base = finger * 4

            result[base] += np.clip(
                delta[base] * weights[finger],
                -lateral_max,
                lateral_max,
            )

            sl = slice(base + 1, base + 4)
            result[sl] += np.clip(
                delta[sl] * weights[finger], -max_corr, max_corr
            )

        now = self._now_seconds()
        if now - self._last_hybrid_log_time >= 1.0:
            labels = ("thumb", "index", "middle", "ring", "little")
            active = [
                f"{name}={w:.2f}"
                for name, w in zip(labels, weights)
                if w > 0.01
            ]
            self.get_logger().info("Hybrid contact: " + ", ".join(active))
            self._last_hybrid_log_time = now

        desired_correction = result - primary_target
        correction_alpha = float(
            self.get_parameter("hybrid_correction_alpha").value
        )
        correction_alpha = float(np.clip(correction_alpha, 0.0, 1.0))
        self._hybrid_correction_state += correction_alpha * (
            desired_correction - self._hybrid_correction_state
        )

        return primary_target + self._hybrid_correction_state

    def _on_landmarks(self, message: PoseArray) -> None:
        now = self._now_seconds()
        if len(message.poses) != 21:
            self._set_tracking(False)
            self._warn_throttled(
                f"Expected 21 hand landmarks, received {len(message.poses)}"
            )
            return

        points = np.asarray(
            [
                [p.position.x, p.position.y, p.position.z]
                for p in message.poses
            ],
            dtype=np.float64,
        )

        try:
            mano_points = landmarks_to_mano(points)
            primary_ref = reference_for(self._retargeting, mano_points)
            primary_all = self._retargeting.retarget(primary_ref)
            primary_target = np.asarray(
                primary_all[self._output_indices], dtype=np.float64
            )

            target = (
                self._apply_hybrid(mano_points, primary_all, primary_target)
                if self._hybrid
                else primary_target
            )
        except Exception as error:
            self._set_tracking(False)
            self._warn_throttled(f"Retargeting rejected a frame: {error}")
            return

        if not np.all(np.isfinite(target)):
            self._set_tracking(False)
            self._warn_throttled("Retargeting produced NaN or infinity")
            return

        if self._last_output is not None and self._last_command_time is not None:
            dt = np.clip(now - self._last_command_time, 1e-3, 0.1)
            max_velocity = float(self.get_parameter("max_joint_velocity").value)
            max_delta = max_velocity * dt
            target = np.clip(
                target,
                self._last_output - max_delta,
                self._last_output + max_delta,
            )

        self._last_output = target
        self._last_frame_time = now
        self._last_command_time = now
        self._set_tracking(True)
        self._publish_command(message, target)

    def _publish_command(self, source: PoseArray, target: np.ndarray) -> None:
        trajectory = JointTrajectory()
        trajectory.header = source.header
        trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = target.tolist()
        point.time_from_start = Duration(sec=0, nanosec=0)
        trajectory.points = [point]
        self._command_publisher.publish(trajectory)

        target_state = JointState()
        target_state.header = source.header
        target_state.name = list(JOINT_NAMES)
        target_state.position = target.tolist()
        self._target_state_publisher.publish(target_state)

    def _watchdog(self) -> None:
        if self._last_frame_time is None:
            return
        timeout = float(self.get_parameter("watchdog_timeout").value)
        if self._now_seconds() - self._last_frame_time > timeout:
            if self._tracking_ok:
                self.get_logger().warning(
                    "Hand tracking timeout: holding the last safe joint target"
                )
            self._set_tracking(False)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[RetargetNode] = None
    try:
        node = RetargetNode()
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
