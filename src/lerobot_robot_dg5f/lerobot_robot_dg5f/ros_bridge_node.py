"""ROS transport, watchdog and arming bridge to the DG5F LeRobot plugin."""

import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory

from .config_dg5f import Dg5fConfig
from .constants import BROKEN_PINKY_JOINT, JOINT_NAMES
from .dg5f import Dg5f


def trajectory_to_degrees(message: JointTrajectory) -> np.ndarray:
    """Read the latest complete named trajectory point in DGSDK joint order."""
    if not message.points:
        raise ValueError("JointTrajectory has no points")
    positions = message.points[-1].positions
    if len(message.joint_names) != len(positions):
        raise ValueError("joint_names and positions have different lengths")
    by_name = dict(zip(message.joint_names, positions))
    missing = [name for name in JOINT_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"JointTrajectory is missing DG5F joints: {missing}")
    radians = np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(radians)):
        raise ValueError("JointTrajectory contains NaN or infinity")
    return np.rad2deg(radians)


def command_is_fresh(
    last_command_time: Optional[float], now: float, timeout: float
) -> bool:
    """Return whether a valid target is recent enough for hardware output."""
    return (
        last_command_time is not None
        and timeout > 0.0
        and 0.0 <= now - last_command_time <= timeout
    )


class Dg5fLeRobotBridge(Node):
    """Rate-limited and explicitly armed bridge to a LeRobot ``Dg5f``."""

    def __init__(self) -> None:
        super().__init__("dg5f_lerobot_bridge")

        self.declare_parameter("command_topic", "/dg5f/joint_command")
        self.declare_parameter("tracking_topic", "/dg5f/tracking_ok")
        self.declare_parameter("joint_state_topic", "/dg5f/lerobot/joint_states")
        self.declare_parameter(
            "commanded_state_topic", "/dg5f/lerobot/commanded_joint_states"
        )
        self.declare_parameter("temperature_topic", "/dg5f/lerobot/temperatures")
        self.declare_parameter("connected_topic", "/dg5f/lerobot/connected")
        self.declare_parameter("armed_topic", "/dg5f/lerobot/armed")
        self.declare_parameter("enable_service", "/dg5f/lerobot/enable")
        self.declare_parameter("backend", "mock")
        self.declare_parameter("ip", "169.254.186.72")
        self.declare_parameter("port", 502)
        self.declare_parameter("slave_id", 1)
        self.declare_parameter("command_rate_hz", 50.0)
        self.declare_parameter("state_rate_hz", 30.0)
        self.declare_parameter("command_timeout", 0.35)
        self.declare_parameter("control_smoothing", True)
        self.declare_parameter("max_speed_deg_s", 30.0)
        self.declare_parameter("max_accel_deg_s2", 60.0)
        self.declare_parameter("response_time_s", 0.15)
        self.declare_parameter("filter_tau_s", 0.05)
        self.declare_parameter("target_deadband_deg", 0.20)
        self.declare_parameter("min_send_step_deg", 0.20)
        self.declare_parameter("max_dt_s", 0.05)
        self.declare_parameter("initial_feedback_timeout_s", 2.0)
        self.declare_parameter("telemetry_drain_limit", 16)
        # Kept only so old launch/config invocations do not fail.
        self.declare_parameter("max_relative_target_deg", 7.0)
        self.declare_parameter("auto_enable", False)
        self.declare_parameter("require_tracking", True)
        self.declare_parameter("disabled_joints", [BROKEN_PINKY_JOINT])
        self.declare_parameter("disabled_positions_deg", [0.0])

        backend = str(self.get_parameter("backend").value).lower()
        self._backend_name = backend
        auto_enable = bool(self.get_parameter("auto_enable").value)
        if backend == "tesollo" and auto_enable:
            raise ValueError("auto_enable is forbidden for the real Tesollo backend")

        disabled_names = list(self.get_parameter("disabled_joints").value)
        disabled_positions = list(
            self.get_parameter("disabled_positions_deg").value
        )
        if len(disabled_names) != len(disabled_positions):
            raise ValueError(
                "disabled_joints and disabled_positions_deg must have equal length"
            )
        disabled_map = dict(zip(disabled_names, disabled_positions))

        config = Dg5fConfig(
            id="dg5f_ros_bridge",
            backend=backend,
            ip=str(self.get_parameter("ip").value),
            port=int(self.get_parameter("port").value),
            slave_id=int(self.get_parameter("slave_id").value),
            control_smoothing=bool(
                self.get_parameter("control_smoothing").value
            ),
            max_speed_deg_s=float(
                self.get_parameter("max_speed_deg_s").value
            ),
            max_accel_deg_s2=float(
                self.get_parameter("max_accel_deg_s2").value
            ),
            response_time_s=float(
                self.get_parameter("response_time_s").value
            ),
            filter_tau_s=float(self.get_parameter("filter_tau_s").value),
            target_deadband_deg=float(
                self.get_parameter("target_deadband_deg").value
            ),
            min_send_step_deg=float(
                self.get_parameter("min_send_step_deg").value
            ),
            max_dt_s=float(self.get_parameter("max_dt_s").value),
            initial_feedback_timeout_s=float(
                self.get_parameter("initial_feedback_timeout_s").value
            ),
            telemetry_drain_limit=int(
                self.get_parameter("telemetry_drain_limit").value
            ),
            max_relative_target_deg=float(
                self.get_parameter("max_relative_target_deg").value
            ),
            disabled_joint_positions_deg=disabled_map,
        )
        self._robot = Dg5f(config)
        self._robot.connect(calibrate=False)

        self._armed = auto_enable
        self._tracking_ok = not bool(
            self.get_parameter("require_tracking").value
        )
        self._latest_command_deg: Optional[np.ndarray] = None
        self._last_command_time: Optional[float] = None
        self._last_warning_time = -1e9

        self._joint_state_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_state_topic").value), 5
        )
        self._commanded_state_pub = self.create_publisher(
            JointState, str(self.get_parameter("commanded_state_topic").value), 5
        )
        self._temperature_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("temperature_topic").value),
            5,
        )
        self._connected_pub = self.create_publisher(
            Bool, str(self.get_parameter("connected_topic").value), 1
        )
        self._armed_pub = self.create_publisher(
            Bool, str(self.get_parameter("armed_topic").value), 1
        )
        self._command_sub = self.create_subscription(
            JointTrajectory,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            1,
        )
        self._tracking_sub = self.create_subscription(
            Bool,
            str(self.get_parameter("tracking_topic").value),
            self._on_tracking,
            1,
        )
        self._enable_service = self.create_service(
            SetBool,
            str(self.get_parameter("enable_service").value),
            self._on_enable,
        )

        command_hz = float(self.get_parameter("command_rate_hz").value)
        state_hz = float(self.get_parameter("state_rate_hz").value)
        if command_hz <= 0.0 or state_hz <= 0.0:
            raise ValueError("command_rate_hz and state_rate_hz must be positive")
        self._command_timer = self.create_timer(1.0 / command_hz, self._send_latest)
        self._state_timer = self.create_timer(1.0 / state_hz, self._publish_state)

        self.get_logger().info(
            f"LeRobot DG5F connected: backend={backend}, armed={self._armed}, "
            f"disabled={disabled_map}"
        )

    def _now_seconds(self) -> float:
        return time.monotonic()

    def _warn_throttled(self, text: str) -> None:
        now = self._now_seconds()
        if now - self._last_warning_time >= 1.0:
            self.get_logger().warning(text)
            self._last_warning_time = now

    def _on_command(self, message: JointTrajectory) -> None:
        try:
            self._latest_command_deg = trajectory_to_degrees(message)
            self._last_command_time = self._now_seconds()
        except ValueError as error:
            self._warn_throttled(f"Ignoring unsafe DG5F command: {error}")

    def _on_tracking(self, message: Bool) -> None:
        self._tracking_ok = bool(message.data)
        if not self._tracking_ok:
            self._robot.hold_position()
        if not self._tracking_ok and self._backend_name == "tesollo":
            self._armed = False

    def _on_enable(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            if not self._robot.is_connected:
                response.success = False
                response.message = "DG5F backend is not connected"
                return response
            if self._latest_command_deg is None:
                response.success = False
                response.message = "No valid DG5F command has been received"
                return response
            if (
                bool(self.get_parameter("require_tracking").value)
                and not self._tracking_ok
            ):
                response.success = False
                response.message = "Hand tracking is not healthy"
                return response
            timeout = float(self.get_parameter("command_timeout").value)
            if not command_is_fresh(
                self._last_command_time, self._now_seconds(), timeout
            ):
                response.success = False
                response.message = "Latest DG5F command is stale"
                return response
        self._armed = bool(request.data)
        if not self._armed:
            self._robot.hold_position()
        response.success = True
        response.message = "DG5F output enabled" if self._armed else "DG5F output disabled"
        self.get_logger().warning(response.message)
        return response

    def _send_latest(self) -> None:
        if not self._armed or self._latest_command_deg is None:
            return
        if bool(self.get_parameter("require_tracking").value) and not self._tracking_ok:
            return
        timeout = float(self.get_parameter("command_timeout").value)
        if not command_is_fresh(
            self._last_command_time, self._now_seconds(), timeout
        ):
            self._armed = False
            self._robot.hold_position()
            self._warn_throttled("Command timeout: LeRobot output was disarmed")
            return

        action = {
            f"{joint}.pos": float(self._latest_command_deg[index])
            for index, joint in enumerate(JOINT_NAMES)
        }
        try:
            sent = self._robot.send_action(action)
        except Exception as error:
            self._armed = False
            self._robot.hold_position()
            self.get_logger().error(f"LeRobot command failed; output disarmed: {error}")
            return
        self._publish_commanded_state(sent)

    def _publish_commanded_state(self, action: dict[str, float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [
            float(np.deg2rad(action[f"{joint}.pos"])) for joint in JOINT_NAMES
        ]
        self._commanded_state_pub.publish(message)

    def _publish_state(self) -> None:
        self._connected_pub.publish(Bool(data=self._robot.is_connected))
        self._armed_pub.publish(Bool(data=self._armed))
        if not self._robot.is_connected:
            return
        try:
            observation = self._robot.get_observation()
        except Exception as error:
            self._warn_throttled(f"Could not read LeRobot DG5F state: {error}")
            return

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [
            float(np.deg2rad(observation[f"{joint}.pos"]))
            for joint in JOINT_NAMES
        ]
        message.velocity = [
            float(np.deg2rad(observation[f"{joint}.vel"]))
            for joint in JOINT_NAMES
        ]
        message.effort = [
            float(observation[f"{joint}.current"]) for joint in JOINT_NAMES
        ]
        self._joint_state_pub.publish(message)
        self._temperature_pub.publish(
            Float32MultiArray(
                data=[float(observation[f"{joint}.temp"]) for joint in JOINT_NAMES]
            )
        )

    def destroy_node(self):
        if hasattr(self, "_robot") and self._robot.is_connected:
            self._robot.disconnect()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[Dg5fLeRobotBridge] = None
    try:
        node = Dg5fLeRobotBridge()
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
