"""ROS transport, watchdog and arming bridge to the DG5F LeRobot plugin."""

import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.task import Future
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory

from .config_dg5f import Dg5fConfig
from .constants import BROKEN_PINKY_JOINT, JOINT_NAMES
from .dg5f import Dg5f
from .health import age_ms, disarm_reason


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
        self.declare_parameter(
            "diagnostics_topic", "/dg5f/lerobot/diagnostics"
        )
        self.declare_parameter("enable_service", "/dg5f/lerobot/enable")
        self.declare_parameter("recover_service", "/dg5f/lerobot/recover")
        self.declare_parameter("backend", "mock")
        self.declare_parameter("ip", "169.254.186.72")
        self.declare_parameter("port", 502)
        self.declare_parameter("slave_id", 1)
        self.declare_parameter("command_rate_hz", 50.0)
        self.declare_parameter("state_rate_hz", 30.0)
        self.declare_parameter("command_timeout", 0.35)
        self.declare_parameter("tracking_timeout", 0.35)
        self.declare_parameter("control_smoothing", True)
        self.declare_parameter("max_speed_deg_s", 30.0)
        self.declare_parameter("max_accel_deg_s2", 60.0)
        self.declare_parameter("response_time_s", 0.15)
        self.declare_parameter("filter_tau_s", 0.05)
        self.declare_parameter("target_deadband_deg", 0.20)
        self.declare_parameter("min_send_step_deg", 0.20)
        self.declare_parameter("max_direct_step_deg", 5.0)
        self.declare_parameter("startup_blend_s", 0.70)
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
            max_direct_step_deg=float(
                self.get_parameter("max_direct_step_deg").value
            ),
            startup_blend_s=float(
                self.get_parameter("startup_blend_s").value
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
        self._last_tracking_time: Optional[float] = None
        self._disarm_reason = "NONE"
        self._recovery_state = "IDLE"
        self._recovery_pending = False
        self._arm_pending = False
        self._arm_epoch = 0
        self._workers = ThreadPoolExecutor(max_workers=1)
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
        self._diagnostics_pub = self.create_publisher(
            DiagnosticArray,
            str(self.get_parameter("diagnostics_topic").value),
            5,
        )
        self._events_pub = self.create_publisher(String, "/dg5f/lerobot/events", 100)
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
        self._service_group = ReentrantCallbackGroup()
        self._enable_service = self.create_service(
            SetBool,
            str(self.get_parameter("enable_service").value),
            self._on_enable,
            callback_group=self._service_group,
        )
        self._recover_service = self.create_service(
            Trigger,
            str(self.get_parameter("recover_service").value),
            self._on_recover,
            callback_group=self._service_group,
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
        self._last_tracking_time = self._now_seconds()
        self._tracking_ok = bool(message.data)
        if not self._tracking_ok and not self._arm_pending:
            self._robot.hold_position()
        if not self._tracking_ok and self._backend_name == "tesollo":
            self._disarm("TRACKING_TIMEOUT")

    def _event(self, name, **details):
        self._events_pub.publish(String(data=json.dumps({
            "event": name, "source_monotonic_s": time.monotonic(), **details,
        })))

    def _disarm(self, reason):
        if self._armed:
            self._armed = False
            self._disarm_reason = reason
            self._robot.hold_position()
            self._event("DISARMED", reason=reason)

    def _input_failure(self):
        now = self._now_seconds()
        if bool(self.get_parameter("require_tracking").value):
            if not self._tracking_ok or not command_is_fresh(
                self._last_tracking_time, now,
                float(self.get_parameter("tracking_timeout").value),
            ):
                return "TRACKING_TIMEOUT"
        if not command_is_fresh(self._last_command_time, now,
                                float(self.get_parameter("command_timeout").value)):
            return "COMMAND_TIMEOUT"
        return None

    async def _on_enable(self, request: SetBool.Request, response: SetBool.Response):
        """Enable/disable output. ARM is a strictly passive preflight.

        In particular ARM must never call DGSDK recovery, SystemStart,
        SystemStop or MoveServoJoint. Faulted sessions are handled only by the
        explicit ``/dg5f/lerobot/recover`` service.
        """
        if not request.data:
            self._arm_epoch += 1
            self._disarm("USER_REQUEST")
            response.success, response.message = True, "DG5F output disabled"
            return response

        self._event("ARM_REQUESTED")
        if self._recovery_pending:
            response.success, response.message = False, "ARM FAILED: explicit recovery is running"
            self.get_logger().warning(response.message)
            return response
        if self._arm_pending:
            response.success, response.message = False, "ARM FAILED: preflight already running"
            self.get_logger().warning(response.message)
            return response
        if self._armed:
            response.success, response.message = True, "DG5F output already enabled"
            return response

        self._arm_pending = True
        try:
            failure = self._input_failure()
            if failure:
                raise RuntimeError(failure)
            if self._latest_command_deg is None:
                raise RuntimeError("COMMAND_TIMEOUT")

            # PASSIVE ONLY: prepare_arm reads status and validates it. It may
            # not reconnect/restart/reseed/send any hardware command.
            self._robot.prepare_arm()

            failure = self._input_failure()
            if failure:
                raise RuntimeError(failure)
            status = self._robot.get_diagnostics()
            failure = disarm_reason(status)
            if failure != "NONE":
                raise RuntimeError(failure)

            # Software-only safety transition: begin from the last accepted
            # physical setpoint and blend toward the current VR target.  This
            # does not call DGSDK or send any command while ARM is still false.
            self._robot.begin_arm_blend()
            self._disarm_reason = "NONE"
            self._armed = True
            self._event("ARMED")
            response.success, response.message = True, "DG5F output enabled (passive preflight OK)"
        except Exception as error:
            self._armed = False
            self._disarm_reason = str(error)
            response.success, response.message = False, f"ARM FAILED: {error}"
        finally:
            self._arm_pending = False
        self.get_logger().warning(response.message)
        return response

    async def _on_recover(self, request: Trigger.Request, response: Trigger.Response):
        """Explicitly recover only the Tesollo control session.

        Recovery may restart the DGSDK system, but it never sends MoveServoJoint.
        It is separated from ARM and always leaves the bridge DISARMED.
        """
        del request
        if self._backend_name != "tesollo":
            response.success, response.message = False, "RECOVERY FAILED: real Tesollo backend is not active"
            return response
        if self._recovery_pending:
            response.success, response.message = False, "RECOVERY FAILED: recovery already running"
            return response

        self._arm_epoch += 1
        self._disarm("RECOVERY_REQUESTED")
        self._armed = False
        self._arm_pending = False
        self._recovery_pending = True
        self._recovery_state = "STARTED"
        self._event("RECOVERY_STARTED")

        # Never let a pre-fault command execute after recovery. A new live
        # target must arrive before the user can ARM again.
        self._latest_command_deg = None
        self._last_command_time = None
        self._robot.hold_position()

        try:
            pending = Future(executor=self.executor)
            worker = self._workers.submit(self._robot.recover)

            def complete(job):
                try:
                    pending.set_result(job.result())
                except Exception as error:
                    pending.set_exception(error)

            worker.add_done_callback(complete)
            await pending

            status = self._robot.get_diagnostics()
            if not status.get("motion_ready", False):
                raise RuntimeError(str(status.get("motion_ready_reason", "SDK_NOT_MOTION_READY")))

            self._recovery_state = "SUCCEEDED"
            self._disarm_reason = "RECOVERY_REQUIRED_ARM"
            self._event("RECOVERY_SUCCEEDED")
            response.success = True
            response.message = (
                "DG5F recovery succeeded; output remains DISARMED. "
                "Wait for a fresh Quest command, then run arm."
            )
        except Exception as error:
            self._recovery_state = "FAILED"
            self._disarm_reason = "RECOVERY_FAILED"
            self._event("RECOVERY_FAILED", reason=str(error))
            response.success = False
            response.message = f"RECOVERY FAILED: {error}"
        finally:
            self._recovery_pending = False
            self._armed = False

        self.get_logger().warning(response.message)
        return response

    def _send_latest(self) -> None:
        if not self._armed or self._latest_command_deg is None:
            return
        failure = self._input_failure()
        if failure:
            self._disarm(failure)
            self._warn_throttled(f"{failure}: LeRobot output was disarmed")
            return
        if bool(self.get_parameter("require_tracking").value) and not self._tracking_ok:
            return

        action = {
            f"{joint}.pos": float(self._latest_command_deg[index])
            for index, joint in enumerate(JOINT_NAMES)
        }
        try:
            sent = self._robot.send_action(action)
        except Exception as error:
            try:
                reason = disarm_reason(self._robot.get_diagnostics())
            except Exception:
                reason = "INTERNAL_ERROR"
            self._disarm(reason if reason != "NONE" else "SDK_COMMAND_REJECTED")
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
        try:
            diagnostics = self._robot.get_diagnostics()
        except Exception as error:
            diagnostics = {
                "transport_connected": False,
                "control_thread_alive": False,
                "motion_ready": False,
                "system_started": False,
                "telemetry_valid": False,
                "temperature_safe": False,
                "diagnostics_error": str(error),
            }

        transport_connected = bool(
            diagnostics.get(
                "transport_connected",
                diagnostics.get("connected", self._robot.is_connected),
            )
        )
        if self._armed:
            reason = disarm_reason(diagnostics)
            if reason != "NONE":
                self._disarm(reason)

        self._connected_pub.publish(Bool(data=transport_connected))
        self._armed_pub.publish(Bool(data=self._armed))
        if not self._robot.is_connected or not transport_connected:
            self._publish_diagnostics(diagnostics)
            return
        try:
            observation = self._robot.get_observation()
        except Exception as error:
            self._warn_throttled(f"Could not read LeRobot DG5F state: {error}")
            self._publish_diagnostics(diagnostics)
            return

        try:
            diagnostics = self._robot.get_diagnostics()
        except Exception as error:
            diagnostics["diagnostics_error"] = str(error)

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
        self._publish_diagnostics(diagnostics)

    @staticmethod
    def _diagnostic_text(value: object) -> str:
        if isinstance(value, np.ndarray):
            return ",".join(f"{float(item):.9g}" for item in value)
        if isinstance(value, (list, tuple)):
            return ",".join(str(item) for item in value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _publish_diagnostics(self, values: dict[str, object]) -> None:
        """Publish one self-contained passive snapshot for the recorder."""
        status = DiagnosticStatus()
        status.name = "dg5f_lerobot_bridge"
        status.hardware_id = str(self.get_parameter("ip").value)

        transport = bool(values.get("transport_connected", False))
        thread_alive = bool(values.get("control_thread_alive", False))
        system_started = bool(values.get("system_started", False))
        telemetry_valid = bool(values.get("telemetry_valid", False))
        temperature_safe = bool(values.get("temperature_safe", False))
        motion_ready = bool(values.get("motion_ready", False))
        last_motion_result = int(values.get("last_motion_result", 0))
        if not transport or not thread_alive or last_motion_result != 0:
            status.level = DiagnosticStatus.ERROR
            status.message = "low-level transport/control fault"
        elif not system_started or not telemetry_valid or not temperature_safe:
            status.level = DiagnosticStatus.WARN
            status.message = "hardware is connected but not motion-ready"
        elif not motion_ready:
            status.level = DiagnosticStatus.WARN
            status.message = "motion is not ready"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "motion-ready"

        snapshot = dict(values)
        snapshot["armed"] = self._armed
        snapshot["tracking_ok"] = self._tracking_ok
        snapshot["disarm_reason"] = self._disarm_reason
        snapshot["recovery_state"] = self._recovery_state
        snapshot["recovery_pending"] = self._recovery_pending
        snapshot["last_command_age_ms"] = age_ms(self._last_command_time, self._now_seconds())
        snapshot["last_tracking_age_ms"] = age_ms(self._last_tracking_time, self._now_seconds())
        snapshot["backend"] = self._backend_name
        snapshot["control_smoothing"] = self.get_parameter("control_smoothing").value
        snapshot["command_profile"] = "smoothed" if snapshot["control_smoothing"] else "direct_guarded"
        snapshot["arm_blend_active"] = self._robot.command_shaper.arm_blend_active
        for name in ("max_speed_deg_s", "max_accel_deg_s2", "response_time_s",
                     "filter_tau_s", "target_deadband_deg", "min_send_step_deg",
                     "max_direct_step_deg", "startup_blend_s"):
            snapshot[name] = self.get_parameter(name).value

        measured_pos = np.asarray(
            snapshot.get("measured_pos", np.full(len(JOINT_NAMES), np.nan)),
            dtype=np.float64,
        )
        current = np.asarray(
            snapshot.get("measured_current", np.full(len(JOINT_NAMES), np.nan)),
            dtype=np.float64,
        )
        temperature = np.asarray(
            snapshot.get("measured_temp", np.full(len(JOINT_NAMES), np.nan)),
            dtype=np.float64,
        )
        command = np.asarray(
            snapshot.get("latest_command_deg", np.full(len(JOINT_NAMES), np.nan)),
            dtype=np.float64,
        )

        def add_maximum(prefix: str, data: np.ndarray, *, absolute: bool) -> None:
            if data.shape != (len(JOINT_NAMES),) or not np.any(np.isfinite(data)):
                snapshot[f"max_{prefix}"] = float("nan")
                snapshot[f"max_{prefix}_joint"] = ""
                return
            ranked = np.abs(data) if absolute else data
            index = int(np.nanargmax(ranked))
            snapshot[f"max_{prefix}"] = float(ranked[index])
            snapshot[f"max_{prefix}_joint"] = JOINT_NAMES[index]

        add_maximum("current", current, absolute=True)
        add_maximum("temperature", temperature, absolute=False)
        if bool(snapshot.get("latest_command_valid", False)) and command.shape == measured_pos.shape:
            add_maximum("tracking_error_deg", command - measured_pos, absolute=True)
        else:
            add_maximum(
                "tracking_error_deg",
                np.full(len(JOINT_NAMES), np.nan),
                absolute=True,
            )

        status.values = [
            KeyValue(key=str(key), value=self._diagnostic_text(value))
            for key, value in sorted(snapshot.items())
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self._diagnostics_pub.publish(message)

    def destroy_node(self):
        if hasattr(self, "_workers"):
            self._workers.shutdown(wait=True)
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
