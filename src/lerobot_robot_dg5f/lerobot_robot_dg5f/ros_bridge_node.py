"""ROS transport, watchdog and arming bridge to the DG5F LeRobot plugin."""

import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from vr_haptic_msgs.msg import ManoLandmarks
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from rclpy.task import Future
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory

from .config_dg5f import Dg5fConfig
from .constants import BROKEN_PINKY_JOINT, JOINT_NAMES
from .current_guard import AdaptiveCurrentGuard, ComplianceConfig
from .object_contact import ObjectContactConfig
from .contact_kinematics import ContactKinematics
from dg5f_teleop.contact_signals import decode_contact_packet, PAIR_NAMES, FINGERS
from .dg5f import Dg5f
from .health import age_ms, disarm_reason
from .tracking_diagnostics import TrackingDiagnostics
from .servo_controller import GuardedServoCommand


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
        self.declare_parameter("landmarks_topic", "/hands/right/landmarks")
        self.declare_parameter("quest_hand_topic", "/quest/hand_pose")
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
        self.declare_parameter("control_mode", "legacy")
        self.declare_parameter("servo_rate_hz", 60.0)
        self.declare_parameter("servo_max_velocity_deg_s", 120.0)
        self.declare_parameter("servo_max_acceleration_deg_s2", 720.0)
        self.declare_parameter("command_rate_hz", 50.0)
        self.declare_parameter("state_rate_hz", 30.0)
        self.declare_parameter("command_timeout", 0.35)
        self.declare_parameter("tracking_timeout", 0.35)
        # Once already ARMED, a tracking outage enters a paused grace state.
        # The hand keeps its last accepted physical setpoint and may resume
        # automatically if tracking returns within this window.
        self.declare_parameter("tracking_grace_s", 15.0)
        self.declare_parameter("tracking_resume_blend_s", 1.0)
        self.declare_parameter("arm_pose_max_age_ms", 100.0)
        # Empirical protection for the old DGSDK/control board. Normal runs in
        # our logs stayed far below these thresholds; failed runs approached
        # ~0.9 A on one joint and ~1.45 A total immediately before telemetry/
        # Ethernet loss. The guard first slows load-increasing motion, then
        # unloads at the hard threshold, and only disarms on a sustained trip.
        self.declare_parameter("current_guard_enabled", True)
        self.declare_parameter("current_guard_soft_ma", 350.0)
        self.declare_parameter("current_guard_hard_ma", 650.0)
        self.declare_parameter("current_guard_trip_ma", 850.0)
        self.declare_parameter("current_guard_total_soft_ma", 600.0)
        self.declare_parameter("current_guard_total_hard_ma", 850.0)
        self.declare_parameter("current_guard_total_trip_ma", 1050.0)
        self.declare_parameter("current_guard_trip_hold_s", 0.04)
        self.declare_parameter("current_guard_release_tau_s", 0.20)
        for name, value in vars(ComplianceConfig()).items():
            self.declare_parameter(f"compliance_{name}", value)
        for name, value in vars(ObjectContactConfig()).items():
            self.declare_parameter(f"object_contact_{name}", value)
        self.declare_parameter("compliance_contact_timeout_s", 0.15)
        self.declare_parameter("compliance_urdf_path", "/workspace/models/dg5f/urdf/dg5f_right.urdf")
        self.declare_parameter("hybrid_contact_topic", "/dg5f/hybrid_contact")
        self.declare_parameter("safety_proximity_topic", "/dg5f/safety_proximity")
        self.declare_parameter("control_smoothing", True)
        self.declare_parameter("max_speed_deg_s", 30.0)
        self.declare_parameter("max_accel_deg_s2", 60.0)
        self.declare_parameter("response_time_s", 0.15)
        self.declare_parameter("filter_tau_s", 0.05)
        self.declare_parameter("target_deadband_deg", 0.20)
        self.declare_parameter("min_send_step_deg", 0.20)
        self.declare_parameter("max_direct_step_deg", 4.0)
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
            control_mode=str(self.get_parameter("control_mode").value),
            servo_rate_hz=float(self.get_parameter("servo_rate_hz").value),
            servo_max_velocity_deg_s=float(self.get_parameter("servo_max_velocity_deg_s").value),
            servo_max_acceleration_deg_s2=float(self.get_parameter("servo_max_acceleration_deg_s2").value),
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
        self._control_mode = config.control_mode
        self._robot = Dg5f(config)
        self._robot.connect(calibrate=False)

        self._armed = auto_enable
        self._tracking_ok = not bool(
            self.get_parameter("require_tracking").value
        )
        self._tracking_diagnostics = TrackingDiagnostics()
        self._latest_command_deg: Optional[np.ndarray] = None
        self._last_command_time: Optional[float] = None
        self._last_tracking_time: Optional[float] = None
        self._disarm_reason = "NONE"
        self._sdk_block_reason = None
        self._load_threshold_active = False
        self._last_retarget_time = None
        self._last_hold_tick_time = None
        self._recovery_state = "IDLE"
        self._recovery_pending = False
        self._arm_pending = False
        self._arm_epoch = 0
        self._tracking_grace_active = False
        self._tracking_grace_started: Optional[float] = None
        self._resume_waiting_for_fresh_command = False
        compliance = ComplianceConfig(**{
            name: self.get_parameter(f"compliance_{name}").value for name in vars(ComplianceConfig())
        })
        contact_timeout = float(self.get_parameter("compliance_contact_timeout_s").value)
        if not np.isfinite(contact_timeout) or contact_timeout <= 0:
            raise ValueError("compliance_contact_timeout_s must be finite and positive")
        self._contact_kinematics = None
        self._contact_kinematics_error = ""
        if compliance.enabled:
            try:
                self._contact_kinematics = ContactKinematics(str(self.get_parameter("compliance_urdf_path").value))
            except Exception as error:
                self._contact_kinematics_error = str(error)
                self.get_logger().warning(f"Contact FK unavailable; current protection remains active: {error}")
        self._contact_samples = {}
        self._command_bounds = None
        self._compliance_active = False
        self._compliance_diagnostics = {}
        self._last_compliance_time = None
        self._current_guard = AdaptiveCurrentGuard(
            len(JOINT_NAMES),
            soft_ma=float(self.get_parameter("current_guard_soft_ma").value),
            hard_ma=float(self.get_parameter("current_guard_hard_ma").value),
            trip_ma=float(self.get_parameter("current_guard_trip_ma").value),
            total_soft_ma=float(
                self.get_parameter("current_guard_total_soft_ma").value
            ),
            total_hard_ma=float(
                self.get_parameter("current_guard_total_hard_ma").value
            ),
            total_trip_ma=float(
                self.get_parameter("current_guard_total_trip_ma").value
            ),
            trip_hold_s=float(
                self.get_parameter("current_guard_trip_hold_s").value
            ),
            release_tau_s=float(
                self.get_parameter("current_guard_release_tau_s").value
            ),
            nominal_step_deg=(config.servo_max_velocity_deg_s / config.servo_rate_hz
                              if self._control_mode == "servo" else max(
                                  1e-6, float(self.get_parameter("max_direct_step_deg").value))),
            compliance=compliance,
            object_contact=ObjectContactConfig(**{
                name: self.get_parameter(f"object_contact_{name}").value
                for name in vars(ObjectContactConfig())
            }),
        )
        self._current_guard_active = False
        self._current_guard_min_scale = 1.0
        self._current_guard_max_current_ma = 0.0
        self._current_guard_total_current_ma = 0.0
        self._current_guard_limited_joints: list[str] = []
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
        self._servo_pub = (self.create_publisher(String, "/dg5f/lerobot/servo_state", 10)
                           if self._control_mode == "servo" else None)
        self._events_pub = self.create_publisher(String, "/dg5f/lerobot/events", 100)
        self._command_sub = self.create_subscription(
            JointTrajectory,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            1,
        )
        # Passive observers only: no change to MANO conversion or retargeting.
        self.create_subscription(PoseArray, str(self.get_parameter("landmarks_topic").value),
                                 lambda msg: self._observe_tracking_input("landmarks", msg), qos_profile_sensor_data)
        self.create_subscription(ManoLandmarks, str(self.get_parameter("quest_hand_topic").value),
                                 lambda msg: self._observe_tracking_input("quest", msg), qos_profile_sensor_data)
        self._tracking_sub = self.create_subscription(
            Bool,
            str(self.get_parameter("tracking_topic").value),
            self._on_tracking,
            1,
        )
        self.create_subscription(Float64MultiArray, str(self.get_parameter("hybrid_contact_topic").value),
                                 lambda msg: self._on_contact(msg, hybrid=True), 1)
        self.create_subscription(Float64MultiArray, str(self.get_parameter("safety_proximity_topic").value),
                                 lambda msg: self._on_contact(msg, hybrid=False), 1)
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
        if self._control_mode == "servo":
            # A steady-clock timer is independent of Quest callbacks and /clock.
            self._servo_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self._command_timer = self.create_timer(
                1.0 / config.servo_rate_hz, self._send_latest, clock=self._servo_clock)
        else:
            self._command_timer = self.create_timer(1.0 / command_hz, self._send_latest)
        self._state_timer = self.create_timer(1.0 / state_hz, self._publish_state)

        self.get_logger().info(
            f"LeRobot DG5F connected: backend={backend}, control_mode={self._control_mode}, armed={self._armed}, "
            f"disabled={disabled_map}"
        )

    def _now_seconds(self) -> float:
        return time.monotonic()

    def _warn_throttled(self, text: str) -> None:
        now = self._now_seconds()
        if now - self._last_warning_time >= 1.0:
            self.get_logger().warning(text)
            self._last_warning_time = now

    def _tracking_is_fresh(self, now: float) -> bool:
        if not bool(self.get_parameter("require_tracking").value):
            return True
        return self._tracking_ok and command_is_fresh(
            self._last_tracking_time,
            now,
            float(self.get_parameter("tracking_timeout").value),
        )

    def _tracking_grace_elapsed(self, now: float) -> float:
        if self._tracking_grace_started is None:
            return 0.0
        return max(0.0, now - self._tracking_grace_started)

    def _enter_tracking_grace(self, now: float) -> None:
        if not self._armed or self._tracking_grace_active:
            return
        self._tracking_grace_active = True
        self._tracking_grace_started = now
        self._resume_waiting_for_fresh_command = False
        # Freeze only the software trajectory. Do NOT suspend the low-level
        # keepalive: during the grace window the physical hand holds the last
        # accepted setpoint and no new VR command is forwarded.
        self._robot.pause_trajectory()
        if self._control_mode == "servo":
            self._last_hold_tick_time = None
            self._event("TRACKING_LOST", reason=self._input_failure(), runtime_state="TRACKING_HOLD")
            self._warn_throttled("Tracking lost: holding last command until fresh tracking and target return")
            return
        grace_s = float(self.get_parameter("tracking_grace_s").value)
        self._event("TRACKING_GRACE_STARTED", grace_s=grace_s)
        self._warn_throttled(
            f"Tracking lost: holding last pose for up to {grace_s:.1f} s"
        )

    def _expire_tracking_grace(self) -> None:
        if self._control_mode == "servo":
            return  # Indefinite TRACKING_HOLD; legacy alone has a grace deadline.
        if not self._tracking_grace_active:
            return
        self._event("TRACKING_GRACE_EXPIRED")
        self._disarm("TRACKING_GRACE_EXPIRED")

    def _try_auto_resume(self, now: float) -> None:
        if not (
            self._armed
            and self._tracking_grace_active
            and self._resume_waiting_for_fresh_command
            and self._tracking_is_fresh(now)
        ):
            return
        grace_s = float(self.get_parameter("tracking_grace_s").value)
        elapsed = self._tracking_grace_elapsed(now)
        if elapsed > grace_s and self._control_mode != "servo":
            self._expire_tracking_grace()
            return
        if not command_is_fresh(
            self._last_command_time,
            now,
            float(self.get_parameter("command_timeout").value),
        ):
            return

        if self._control_mode == "servo":
            if self._last_command_time <= self._tracking_grace_started or self._sdk_block_reason:
                return
            self._tracking_grace_active = False
            self._tracking_grace_started = None
            self._resume_waiting_for_fresh_command = False
            self._event("TRACKING_RECOVERED", hold_elapsed_s=elapsed, runtime_state="ACTIVE")
            return  # Keep q_cmd, physical command and contact anchors intact.

        # Re-anchor exactly once to current physical feedback. This prevents a
        # stale pre-loss command from becoming the origin of the catch-up.
        self._robot.prepare_arm()
        pose = self._robot.begin_arm_blend_from_feedback(
            max_pose_age_ms=float(self.get_parameter("arm_pose_max_age_ms").value),
            duration_s=float(self.get_parameter("tracking_resume_blend_s").value),
        )
        target_offset = float(np.max(np.abs(self._latest_command_deg - pose)))
        self._tracking_grace_active = False
        self._tracking_grace_started = None
        self._resume_waiting_for_fresh_command = False
        self._disarm_reason = "NONE"
        self._event(
            "TRACKING_AUTO_RESUME_STARTED",
            grace_elapsed_s=elapsed,
            max_target_offset_deg=target_offset,
        )
        self.get_logger().warning(
            "Tracking restored: automatically blending from current physical "
            f"pose to live VR target over "
            f"{float(self.get_parameter('tracking_resume_blend_s').value):.2f} s"
        )

    def _on_contact(self, message, *, hybrid):
        key = "hybrid" if hybrid else "proximity"
        try:
            packet = decode_contact_packet(message.data, hybrid=hybrid)
            source_age = self.get_clock().now().nanoseconds * 1e-9 - packet[0]
            timeout = float(self.get_parameter("compliance_contact_timeout_s").value)
            if not -0.05 <= source_age <= timeout:
                raise ValueError("stale/future contact timestamp")
            previous = self._contact_samples.get(key)
            if previous is not None and packet[0] <= previous[0][0]:
                return  # Replays/repeated stamps must never refresh freshness.
            self._contact_samples[key] = (packet, self._now_seconds() - max(0, source_age))
        except (ValueError, TypeError) as error:
            self._contact_samples.pop(key, None)
            self._warn_throttled(f"Contact signal rejected; current guard remains active: {error}")

    def _contact_for_guard(self, effective, now):
        timeout = float(self.get_parameter("compliance_contact_timeout_s").value)
        fresh = {name: packet for name, (packet, at) in self._contact_samples.items()
                 if 0 <= now - at <= timeout}
        proximity = fresh.get("proximity")
        if proximity is None or self._contact_kinematics is None:
            return {}
        try:
            gradients = self._contact_kinematics.pair_gradients(effective)
        except Exception as error:
            self._warn_throttled(f"Contact geometry unavailable; current fallback: {error}")
            return {}
        hybrid = fresh.get("hybrid")
        return {
            "hybrid_weights": None if hybrid is None else hybrid[1:6],
            "pair_distances_m": proximity[1:], "contact_stamp_s": float(proximity[0]),
            "pair_gradients_m_deg": gradients,
        }

    def _guard_current_target(
        self, desired_deg: np.ndarray, now: float, *, operator_target_deg=None
    ) -> Optional[np.ndarray | GuardedServoCommand]:
        """Return local servo bounds, or the unchanged legacy guard decision.

        Low steady current without contact leaves the target unchanged. Contact
        and rising current can pre-emptively reduce convergence. Relief bypasses
        the adaptive limiter. Only legacy converts trip/stall into DISARM.
        """
        self._command_bounds = None
        if not bool(self.get_parameter("current_guard_enabled").value):
            self._current_guard_active = False
            self._current_guard_min_scale = 1.0
            self._current_guard_limited_joints = []
            self._compliance_diagnostics = {"compliance_enabled": False, "compliance_active": False}
            return np.asarray(desired_deg, dtype=np.float64).copy()

        status = self._robot.get_diagnostics()
        current = np.asarray(
            status.get("measured_current", status.get("raw_current", [])),
            dtype=np.float64,
        )
        measured = np.asarray(status.get("measured_pos", []), dtype=np.float64)
        if current.shape != (len(JOINT_NAMES),) or measured.shape != (
            len(JOINT_NAMES),
        ):
            # Hardware-health watchdog remains authoritative if telemetry is
            # actually missing. Do not fabricate load information here.
            return np.asarray(desired_deg, dtype=np.float64).copy()

        effective = (self._robot.command_shaper.command_pose_deg.copy()
                     if self._control_mode == "servo" else
                     self._robot.command_shaper.effective_command())
        decision = self._current_guard.update(
            current_ma=current,
            measured_deg=measured,
            effective_deg=effective,
            desired_deg=np.asarray(desired_deg, dtype=np.float64),
            now=now,
            operator_target_deg=operator_target_deg,
            trajectory_braking=(self._robot.command_shaper.telemetry.get("trajectory_braking")
                                if self._control_mode == "servo" else None),
            **self._contact_for_guard(effective, now),
        )
        self._last_compliance_time = now
        self._compliance_diagnostics = decision.diagnostics
        for transition in decision.object_contact_events:
            details = dict(transition)
            event = details.pop("event")
            if self._control_mode == "servo":
                event = event.removeprefix("OBJECT_")
            self._event(event, **details)
        if decision.active:
            lower, upper = np.full(20, -np.inf), np.full(20, np.inf)
            mask = decision.limited_mask
            lower[mask] = np.minimum(effective[mask], decision.target_deg[mask])
            upper[mask] = np.maximum(effective[mask], decision.target_deg[mask])
            self._command_bounds = (lower, upper)
        self._current_guard_min_scale = decision.min_scale
        self._current_guard_max_current_ma = decision.max_current_ma
        self._current_guard_total_current_ma = decision.total_current_ma
        self._current_guard_limited_joints = [
            JOINT_NAMES[index]
            for index, limited in enumerate(decision.limited_mask)
            if limited
        ]
        compliance_active = bool(decision.diagnostics.get("compliance_active"))
        if compliance_active != self._compliance_active:
            details = {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in decision.diagnostics.items()}
            self._event("COMPLIANCE_ACTIVE" if compliance_active else "COMPLIANCE_RELEASED", **details)
        self._compliance_active = compliance_active

        if decision.active and not self._current_guard_active:
            self._event(
                "CURRENT_GUARD_ACTIVE",
                max_current_ma=decision.max_current_ma,
                total_current_ma=decision.total_current_ma,
                min_scale=decision.min_scale,
                limited_joints=self._current_guard_limited_joints,
            )
            self._warn_throttled(
                "Current guard limiting load-increasing motion: "
                f"Imax={decision.max_current_ma:.0f} mA, "
                f"Itotal={decision.total_current_ma:.0f} mA"
            )
        elif not decision.active and self._current_guard_active:
            self._event("CURRENT_GUARD_RELEASED")
        self._current_guard_active = decision.active

        if self._control_mode == "servo":
            threshold = decision.trip or decision.stall_trip
            if threshold and not self._load_threshold_active:
                self._event("CURRENT_LOAD_THRESHOLD", current_threshold=decision.trip,
                            stall_threshold=decision.stall_trip,
                            max_current_ma=decision.max_current_ma,
                            total_current_ma=decision.total_current_ma)
            self._load_threshold_active = threshold
            return GuardedServoCommand(decision.target_deg,
                                       decision.diagnostics["physical_lower_deg"],
                                       decision.diagnostics["physical_upper_deg"])

        if decision.trip or decision.stall_trip:
            event = "CURRENT_GUARD_TRIP" if decision.trip else "STALL_GUARD_TRIP"
            reason = "OVERCURRENT_GUARD" if decision.trip else "STALL_GUARD"
            self._event(
                event,
                max_current_ma=decision.max_current_ma,
                total_current_ma=decision.total_current_ma,
                duration_s=decision.trip_duration_s if decision.trip else decision.diagnostics["stall_duration_s"],
                joint_tracking_scale=decision.diagnostics["joint_tracking_scale"].tolist(),
                contact_limited_pairs=decision.diagnostics["contact_limited_pairs"],
            )
            self._disarm(reason)
            self.get_logger().error(
                f"{reason}: sustained load fault; output disarmed "
                f"(Imax={decision.max_current_ma:.0f} mA, "
                f"Itotal={decision.total_current_ma:.0f} mA)"
            )
            return None
        return decision.target_deg

    def _on_command(self, message: JointTrajectory) -> None:
        try:
            command = trajectory_to_degrees(message)
            now = self._now_seconds()
            self._last_retarget_time = now
            if self._control_mode == "servo" and self._armed:
                if not self._tracking_is_fresh(now) or self._sdk_block_reason:
                    return  # Observe receipt, but never accept lost-tracking targets.
                if self._tracking_grace_active:
                    self._resume_waiting_for_fresh_command = True
            self._latest_command_deg = command
            self._last_command_time = now
            if self._resume_waiting_for_fresh_command:
                try:
                    self._try_auto_resume(now)
                except Exception as error:
                    self._warn_throttled(f"Auto-resume waiting for safe pose: {error}")
        except ValueError as error:
            self._warn_throttled(f"Ignoring unsafe DG5F command: {error}")

    def _observe_tracking_input(self, name, message):
        points = message.landmarks if name == "quest" else [pose.position for pose in message.poses]
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        self._tracking_diagnostics.observe(
            name, [[point.x, point.y, point.z] for point in points], stamp, self._now_seconds())

    def _on_tracking(self, message: Bool) -> None:
        now = self._now_seconds()
        self._last_tracking_time = now
        self._tracking_ok = bool(message.data)

        if not self._armed:
            return
        if not self._tracking_ok:
            self._resume_waiting_for_fresh_command = False
            self._enter_tracking_grace(now)
            return

        if self._tracking_grace_active:
            grace_s = float(self.get_parameter("tracking_grace_s").value)
            if self._tracking_grace_elapsed(now) > grace_s and self._control_mode != "servo":
                self._expire_tracking_grace()
                return
            # Grace may have started either from an explicit tracking=false or
            # because the tracking topic itself went stale. Require a command
            # generated after a fresh tracking message returns in both cases.
            if not self._resume_waiting_for_fresh_command:
                self._resume_waiting_for_fresh_command = True
                self._event("TRACKING_RESTORED_WAITING_FRESH_COMMAND")

    def _event(self, name, **details):
        self._events_pub.publish(String(data=json.dumps({
            "event": name, "source_monotonic_s": time.monotonic(), **details,
        })))

    def _disarm(self, reason):
        was_armed = self._armed
        self._armed = False
        self._tracking_grace_active = False
        self._tracking_grace_started = None
        self._resume_waiting_for_fresh_command = False
        if was_armed:
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
        """Enable/disable output after a passive hardware preflight.

        Servo ARM seeds the backend hold setpoint from measured feedback once;
        tracking only selects ACTIVE versus TRACKING_HOLD. Legacy ARM remains
        passive. Neither path recovers or restarts the SDK session.
        """
        if not request.data:
            self._arm_epoch += 1
            self._disarm("USER_REQUEST")
            response.success, response.message = True, "DG5F output disabled"
            return response

        self._event("ARM_REQUESTED")
        if self._control_mode == "servo" and self._sdk_block_reason:
            response.success, response.message = False, f"ARM FAILED: explicit recovery required ({self._sdk_block_reason})"
            return response
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
            if self._control_mode != "servo":
                failure = self._input_failure()
                if failure:
                    raise RuntimeError(failure)
                if self._latest_command_deg is None:
                    raise RuntimeError("COMMAND_TIMEOUT")

            # PASSIVE ONLY: prepare_arm reads status and validates it. It may
            # not reconnect/restart or send any hardware command.
            self._robot.prepare_arm()

            if self._control_mode != "servo":
                failure = self._input_failure()
                if failure:
                    raise RuntimeError(failure)
            status = self._robot.get_diagnostics()
            failure = disarm_reason(status)
            if failure != "NONE":
                raise RuntimeError(failure)

            # Re-anchor the shaper ONCE from the freshest physical pose. This
            # prevents a stale pre-arm command_pose from producing a large
            # first setpoint. Servo also initializes the physical hold setpoint;
            # subsequent live targets go through the normal servo timer.
            pose = self._robot.begin_arm_blend_from_feedback(
                max_pose_age_ms=float(self.get_parameter("arm_pose_max_age_ms").value),
                duration_s=float(self.get_parameter("startup_blend_s").value),
            )
            tracking_hold = self._control_mode == "servo" and (
                self._input_failure() is not None or self._latest_command_deg is None)
            if self._control_mode == "servo":
                if tracking_hold:
                    # Discard pre-arm stale intent. A new valid target received
                    # with fresh tracking is required to leave HOLD.
                    self._latest_command_deg = None
                    self._last_command_time = None
                self._robot.backend.send_positions(
                    self._robot.command_shaper.effective_command())
            target_offset = (None if self._latest_command_deg is None else
                             float(np.max(np.abs(self._latest_command_deg - pose))))
            self._tracking_grace_active = False
            self._tracking_grace_started = None
            self._resume_waiting_for_fresh_command = False
            self._current_guard.reset(self._now_seconds())
            self._current_guard_active = False
            self._current_guard_min_scale = 1.0
            self._current_guard_limited_joints = []
            self._disarm_reason = "NONE"
            self._armed = True
            self._event("ARMED", max_target_offset_deg=target_offset)
            if tracking_hold:
                self._enter_tracking_grace(self._now_seconds())
            response.success = True
            response.message = (f"DG5F output enabled ({self._runtime_state()})"
                                if self._control_mode == "servo" else
                                "DG5F output enabled (passive preflight OK)")
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

        # Never let a pre-fault command execute after recovery. Servo may ARM
        # into HOLD; legacy still requires a new live target before ARM.
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
            self._sdk_block_reason = None
            self._current_guard.reset(self._now_seconds())
            self._current_guard_active = False
            self._disarm_reason = "RECOVERY_REQUIRED_ARM"
            self._event("RECOVERY_SUCCEEDED")
            response.success = True
            response.message = (
                "DG5F recovery succeeded; output remains DISARMED. "
                + ("Run arm; fresh tracking will resume motion automatically."
                   if self._control_mode == "servo" else
                   "Wait for a fresh Quest command, then run arm.")
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
        if not self._armed:
            return
        now = self._now_seconds()

        if self._control_mode == "servo":
            if self._sdk_block_reason:
                self._publish_servo_hold(now)
                return
            failure = self._input_failure()
            if failure or self._latest_command_deg is None:
                self._enter_tracking_grace(now)
            if self._tracking_grace_active:
                self._publish_servo_hold(now)
                return
            self._send_servo_tick(now)
            return

        if self._latest_command_deg is None:
            return

        if bool(self.get_parameter("require_tracking").value):
            if not self._tracking_is_fresh(now):
                self._enter_tracking_grace(now)
            if self._tracking_grace_active:
                if self._tracking_grace_elapsed(now) > float(
                    self.get_parameter("tracking_grace_s").value
                ):
                    self._expire_tracking_grace()
                # During grace, or while waiting for the first post-restore
                # target, send absolutely no new motion command. Low-level
                # keepalive continues holding the last accepted pose.
                return

        failure = self._input_failure()
        if failure:
            self._disarm(failure)
            self._warn_throttled(f"{failure}: LeRobot output was disarmed")
            return

        guarded_target = self._guard_current_target(self._latest_command_deg, now)
        if guarded_target is None:
            return

        action = {
            f"{joint}.pos": float(guarded_target[index])
            for index, joint in enumerate(JOINT_NAMES)
        }
        try:
            sent = self._robot.send_action(action, command_bounds=self._command_bounds)
        except Exception as error:
            try:
                reason = disarm_reason(self._robot.get_diagnostics())
            except Exception:
                reason = "INTERNAL_ERROR"
            self._disarm(reason if reason != "NONE" else "SDK_COMMAND_REJECTED")
            self.get_logger().error(f"LeRobot command failed; output disarmed: {error}")
            return
        self._publish_commanded_state(sent)

    def _send_servo_tick(self, now: float) -> None:
        try:
            # Snapshot is read for logging only; it never seeds an ordinary tick.
            measured = self._robot.get_diagnostics()["measured_pos"]
            target = self._latest_command_deg.copy()
            controller = self._robot.command_shaper
            operator_target = controller._apply_disabled(np.clip(
                target, controller.lower_limits_deg, controller.upper_limits_deg))
            sent = self._robot.servo_tick(
                target, now=now,
                guard=lambda proposed: self._guard_current_target(
                    proposed, now, operator_target_deg=operator_target),
            )
            if sent is None:
                return
            controller = self._robot.command_shaper
            warning_joints = controller.observe(measured)
            if warning_joints:
                self.get_logger().warning(
                    "Servo tracking error >10 deg for >=300 ms: "
                    + ", ".join(JOINT_NAMES[index] for index in warning_joints))
            telemetry = controller.telemetry
            telemetry["command_submitted"] = True
            for key in ("post_compliance_cmd", "post_contact_cmd", "post_guard_cmd"):
                telemetry[key] = np.asarray(self._compliance_diagnostics.get(
                    key, telemetry["q_proposed"])).tolist()
            telemetry["command_direction_reason"] = self._compliance_diagnostics.get(
                "command_direction_reason", ["NONE"] * len(JOINT_NAMES))
            # This is an asynchronous SDK snapshot, not an acknowledgement of
            # this very tick. Keep submitted and observed commands distinct.
            status = self._robot.get_diagnostics()
            low_level = status.get("latest_command_deg")
            telemetry["low_level_cmd"] = None if low_level is None else np.asarray(low_level).tolist()
            telemetry["low_level_valid"] = bool(status.get("latest_command_valid", False))
            telemetry["low_level_sample_monotonic_s"] = self._now_seconds()
            telemetry["measured"] = telemetry["q_measured"]
            self._enrich_servo_telemetry(telemetry, status, now)
            self._log_servo_direction(telemetry)
            self._servo_pub.publish(String(data=json.dumps(telemetry)))
            self._publish_commanded_state(sent)
        except Exception as error:
            try:
                reason = disarm_reason(self._robot.get_diagnostics())
            except Exception:
                reason = "INTERNAL_ERROR"
            self._block_servo_backend(reason if reason != "NONE" else "COMMAND_PIPELINE_ERROR", error=str(error))

    def _runtime_state(self):
        if self._sdk_block_reason:
            return "SDK_ERROR"
        if not self._armed:
            return "DISARMED"
        return "TRACKING_HOLD" if self._tracking_grace_active else "ACTIVE"

    def _block_servo_backend(self, reason, *, error=None, diagnostics=None):
        """Latch backend inhibition without changing initial ARM authorization.

        No reconnect/reset/rearm here. Only successful explicit recovery clears
        this latch. Backend readiness faults may include wrapper health gates;
        they are not necessarily device-reported faults.
        """
        if self._sdk_block_reason:
            return
        self._sdk_block_reason = reason
        if diagnostics is None:
            try:
                diagnostics = self._robot.get_diagnostics()
            except Exception as read_error:
                diagnostics = {"diagnostics_error": str(read_error)}
        details = {key: diagnostics.get(key) for key in (
            "last_motion_result", "motion_ready_reason", "transport_connected",
            "control_thread_alive", "telemetry_valid", "temperature_safe",
            "system_started", "data_processing_status", "diagnostics_error",
            "diagnosis_process", "diagnosis_step", "diagnosis_joint_id",
            "diagnosis_period", "diagnosis_joint", "diagnosis_temperature")}
        self._event("SDK_ERROR", reason=reason, error=error, **details)
        self.get_logger().error(f"SDK_ERROR: backend inhibited; reason={reason}, error={error}, status={details}")
        try:
            self._robot.hold_position()
        except Exception as suspend_error:
            self.get_logger().error(f"SDK_ERROR: suspend failed: {suspend_error}")

    def _enrich_servo_telemetry(self, telemetry, status, now):
        for key, value in self._compliance_diagnostics.items():
            if key not in telemetry:
                telemetry[key] = value.tolist() if isinstance(value, np.ndarray) else value
        telemetry.update(self._tracking_diagnostics.snapshot(
            now=now, retarget_time=self._last_retarget_time,
            tracking_ok=self._tracking_ok, tracking_time=self._last_tracking_time,
            timeout=float(self.get_parameter("tracking_timeout").value)))
        telemetry["runtime_state"] = self._runtime_state()
        telemetry["compliance_age_ms"] = age_ms(self._last_compliance_time, now)
        telemetry["sdk_block_reason"] = self._sdk_block_reason
        current = np.asarray(status.get("measured_current", np.full(20, np.nan)))
        telemetry["current_ma"] = current.tolist()
        telemetry["total_current_ma"] = float(np.sum(np.abs(current)))
        measured = np.asarray(telemetry["q_measured"])
        physical = np.asarray(telemetry.get("low_level_submitted", self._robot.command_shaper.effective_command()))
        telemetry["submitted_command_lead_deg"] = (physical - measured).tolist()
        actual = telemetry.get("low_level_cmd")
        telemetry["actual_command_lead_deg"] = (
            (np.asarray(actual) - measured).tolist()
            if actual is not None and telemetry.get("low_level_valid") else None)
        telemetry.setdefault("physical_limit_reason", ["HOLD_LAST_COMMAND"] * 20)

    def _publish_servo_hold(self, now):
        """Continue the diagnostic clock while SDK keepalive holds the pose."""
        try:
            status = self._robot.get_diagnostics()
            controller = self._robot.command_shaper
            command = controller.command_pose_deg.tolist()
            measured = np.asarray(status.get("measured_pos", np.full(20, np.nan)))
            dt = 1 / controller.rate_hz if self._last_hold_tick_time is None else now - self._last_hold_tick_time
            self._last_hold_tick_time = now
            telemetry = dict(source_monotonic_s=now, command_submitted=False,
                q_target=(self._latest_command_deg.tolist() if self._latest_command_deg is not None
                          else [float("nan")] * 20), previous_q_cmd=command,
                q_cmd=command, q_measured=measured.tolist(), raw_servo_step=[0.0] * 20,
                post_contact_cmd=command, post_guard_cmd=command,
                desired_velocity_deg_s=[0.0] * 20,
                commanded_velocity_deg_s=controller.velocity_deg_s.tolist(),
                acceleration_deg_s2=[0.0] * 20, acceleration_limited=[False] * 20,
                max_acceleration_deg_s2=controller.max_acceleration_deg_s2,
                tracking_error=np.abs(np.asarray(command) - measured).tolist(),
                dt=dt, actual_rate_hz=1 / dt if dt > 0 else None, max_step_deg=0.0,
                low_level_submitted=controller.effective_command().tolist(),
                low_level_cmd=np.asarray(status["latest_command_deg"]).tolist()
                    if status.get("latest_command_deg") is not None else None,
                low_level_valid=bool(status.get("latest_command_valid", False)))
            self._enrich_servo_telemetry(telemetry, status, now)
            self._servo_pub.publish(String(data=json.dumps(telemetry)))
        except Exception as error:
            self._block_servo_backend("DIAGNOSTICS_ERROR", error=str(error))

    def _log_servo_direction(self, telemetry):
        previous = np.asarray(telemetry["previous_q_cmd"])
        desired_delta = np.asarray(telemetry["target"]) - previous
        post_contact = np.asarray(telemetry["post_contact_cmd"])
        post_violation = desired_delta * (post_contact - previous) < -1e-8
        moved_away = np.asarray(telemetry["distance_after"]) > np.asarray(telemetry["distance_before"]) + 1e-8
        braking = np.asarray(telemetry.get("trajectory_braking", [False] * 20))
        planned_step = np.abs(np.asarray(telemetry["raw_servo_step"]))
        post_violation &= ~(braking & (np.abs(post_contact - previous) <= planned_step + 1e-8))
        moved_away &= ~(braking & (np.abs(np.asarray(telemetry["q_cmd"]) - previous) <= planned_step + 1e-8))
        telemetry["post_contact_direction_violation"] = post_violation.tolist()
        telemetry["command_moved_away_from_target"] = moved_away.tolist()
        for index in np.flatnonzero(post_violation | moved_away | np.asarray(telemetry["direction_violation"])):
            reason = telemetry["command_direction_reason"][index]
            details = {key: telemetry[key][index] for key in (
                "target", "previous_q_cmd", "raw_servo_step", "new_q_cmd",
                "post_compliance_cmd", "post_contact_cmd", "post_guard_cmd", "measured")}
            details["low_level_cmd"] = (None if telemetry["low_level_cmd"] is None
                                        else telemetry["low_level_cmd"][index])
            event = "COMMAND_MOVED_AWAY_FROM_TARGET" if moved_away[index] else "COMMAND_DIRECTION_VIOLATION_BLOCKED"
            self.get_logger().error(f"{event} joint={JOINT_NAMES[index]} reason={reason} {details}")
            self._event(event, joint=JOINT_NAMES[index], reason=reason, **details)

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
                if self._control_mode == "servo":
                    self._block_servo_backend(reason, diagnostics=diagnostics)
                else:
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
        snapshot["runtime_state"] = self._runtime_state()
        snapshot["sdk_block_reason"] = self._sdk_block_reason or "NONE"
        snapshot.update(self._tracking_diagnostics.snapshot(
            now=self._now_seconds(), retarget_time=self._last_retarget_time,
            tracking_ok=self._tracking_ok, tracking_time=self._last_tracking_time,
            timeout=float(self.get_parameter("tracking_timeout").value)))
        snapshot["disarm_reason"] = self._disarm_reason
        snapshot["recovery_state"] = self._recovery_state
        snapshot["recovery_pending"] = self._recovery_pending
        snapshot["last_command_age_ms"] = age_ms(self._last_command_time, self._now_seconds())
        now = self._now_seconds()
        snapshot["last_tracking_age_ms"] = age_ms(self._last_tracking_time, now)
        snapshot["tracking_grace_active"] = self._tracking_grace_active
        snapshot["tracking_grace_s"] = self.get_parameter("tracking_grace_s").value
        snapshot["tracking_grace_elapsed_s"] = (
            self._tracking_grace_elapsed(now) if self._tracking_grace_active else 0.0
        )
        snapshot["tracking_resume_pending"] = self._resume_waiting_for_fresh_command
        snapshot["tracking_resume_blend_s"] = self.get_parameter(
            "tracking_resume_blend_s"
        ).value
        snapshot["arm_pose_max_age_ms"] = self.get_parameter("arm_pose_max_age_ms").value
        snapshot["current_guard_enabled"] = self.get_parameter(
            "current_guard_enabled"
        ).value
        snapshot["current_guard_active"] = self._current_guard_active
        snapshot["current_guard_min_scale"] = self._current_guard_min_scale
        snapshot["current_guard_max_current_ma"] = self._current_guard_max_current_ma
        snapshot["current_guard_total_current_ma"] = self._current_guard_total_current_ma
        snapshot["current_guard_limited_joints"] = self._current_guard_limited_joints
        snapshot.update(self._compliance_diagnostics)
        measured_for_lead = values.get("measured_pos")
        physical_for_lead = values.get("latest_command_deg")
        snapshot["actual_command_lead_deg"] = (
            np.asarray(physical_for_lead) - np.asarray(measured_for_lead)
            if (measured_for_lead is not None and physical_for_lead is not None
                and values.get("latest_command_valid", False))
            else np.full(20, np.nan))
        snapshot["compliance_age_ms"] = age_ms(self._last_compliance_time, now)
        snapshot["contact_kinematics_error"] = self._contact_kinematics_error
        snapshot["contact_finger_order"] = list(FINGERS)
        snapshot["contact_pair_order"] = list(PAIR_NAMES)
        timeout = float(self.get_parameter("compliance_contact_timeout_s").value)
        for key in ("hybrid", "proximity"):
            sample = self._contact_samples.get(key)
            snapshot[f"{key}_signal_age_ms"] = age_ms(None if sample is None else sample[1], now)
            valid = sample is not None and 0 <= now - sample[1] <= timeout
            snapshot[f"{key}_signal_fresh"] = valid
            if key == "hybrid":
                snapshot["hybrid_contact_weights"] = sample[0][1:6] if valid else np.full(5, np.nan)
            else:
                snapshot["pair_distances_m"] = sample[0][1:] if valid else np.full(len(PAIR_NAMES), np.nan)
        for name in vars(ComplianceConfig()):
            snapshot[f"compliance_{name}"] = self.get_parameter(f"compliance_{name}").value
        for name in vars(ObjectContactConfig()):
            snapshot[f"object_contact_{name}"] = self.get_parameter(f"object_contact_{name}").value
        for name in (
            "current_guard_soft_ma",
            "current_guard_hard_ma",
            "current_guard_trip_ma",
            "current_guard_total_soft_ma",
            "current_guard_total_hard_ma",
            "current_guard_total_trip_ma",
            "current_guard_trip_hold_s",
            "current_guard_release_tau_s",
        ):
            snapshot[name] = self.get_parameter(name).value
        snapshot["backend"] = self._backend_name
        snapshot["control_mode"] = self._control_mode
        snapshot["servo_rate_hz"] = self.get_parameter("servo_rate_hz").value
        snapshot["servo_max_velocity_deg_s"] = self.get_parameter("servo_max_velocity_deg_s").value
        snapshot["servo_max_acceleration_deg_s2"] = self.get_parameter("servo_max_acceleration_deg_s2").value
        snapshot["control_smoothing"] = self.get_parameter("control_smoothing").value
        snapshot["command_profile"] = "smoothed" if snapshot["control_smoothing"] else "direct_guarded"
        if self._control_mode == "servo":
            snapshot["command_profile"] = "servo"
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
