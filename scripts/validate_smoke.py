#!/usr/bin/env python3
"""Validate all integration topics concurrently during the mock smoke test."""

import argparse
import sys
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from dg5f_teleop.contact_signals import decode_contact_packet
from trajectory_msgs.msg import JointTrajectory
from vr_haptic_msgs.msg import ManoLandmarks


FAULT_JOINT = "rj_dg_5_1"


class SmokeValidator(Node):
    def __init__(self) -> None:
        super().__init__("dg5f_lerobot_smoke_validator")
        self.received = set()
        self.errors = []
        self.create_subscription(
            ManoLandmarks, "/quest/hand_pose", self._quest, 10
        )
        self.create_subscription(
            PoseArray, "/hands/right/landmarks", self._landmarks, 10
        )
        self.create_subscription(
            JointTrajectory, "/dg5f/joint_command", self._command, 10
        )
        self.create_subscription(
            JointState, "/dg5f/joint_states", self._mujoco_state, 10
        )
        self.create_subscription(
            JointState,
            "/dg5f/lerobot/commanded_joint_states",
            self._lerobot_command,
            10,
        )
        self.create_subscription(
            JointState,
            "/dg5f/lerobot/joint_states",
            self._lerobot_state,
            10,
        )
        self.create_subscription(Bool, "/dg5f/tracking_ok", self._tracking, 10)
        self.create_subscription(Bool, "/dg5f/lerobot/armed", self._armed, 10)
        for topic, hybrid in [("hybrid_contact", True), ("safety_proximity", False)]:
            self.create_subscription(Float64MultiArray, "/dg5f/" + topic,
                                     lambda msg, h=hybrid: self._contact(msg, h), 10)
        self.create_subscription(
            DiagnosticArray,
            "/dg5f/lerobot/diagnostics",
            self._diagnostics,
            10,
        )

    def _contact(self, message, hybrid):
        try:
            decode_contact_packet(message.data, hybrid=hybrid)
            self.received.add("hybrid_contact" if hybrid else "safety_proximity")
        except ValueError as error:
            self.errors.append(str(error))

    def _quest(self, message: ManoLandmarks) -> None:
        if len(message.landmarks) == 21:
            self.received.add("quest")

    def _landmarks(self, message: PoseArray) -> None:
        if len(message.poses) == 21:
            self.received.add("landmarks")

    def _check_fixed_joint(self, names, positions, label: str) -> None:
        if FAULT_JOINT not in names:
            return
        value = float(positions[list(names).index(FAULT_JOINT)])
        if abs(value) > 1e-6:
            self.errors.append(f"{label}: {FAULT_JOINT} moved to {value}")
            return
        self.received.add(label)

    def _command(self, message: JointTrajectory) -> None:
        if message.points:
            self._check_fixed_joint(
                message.joint_names, message.points[-1].positions, "command"
            )

    def _mujoco_state(self, message: JointState) -> None:
        self._check_fixed_joint(message.name, message.position, "mujoco")

    def _lerobot_command(self, message: JointState) -> None:
        self._check_fixed_joint(
            message.name, message.position, "lerobot_command"
        )

    def _lerobot_state(self, message: JointState) -> None:
        self._check_fixed_joint(message.name, message.position, "lerobot_state")

    def _tracking(self, message: Bool) -> None:
        if message.data:
            self.received.add("tracking")

    def _armed(self, message: Bool) -> None:
        if message.data:
            self.received.add("armed")

    def _diagnostics(self, message: DiagnosticArray) -> None:
        for status in message.status:
            if status.name != "dg5f_lerobot_bridge":
                continue
            values = {item.key: item.value for item in status.values}
            required = {
                "transport_connected",
                "control_thread_alive",
                "motion_ready",
                "telemetry_valid",
                "latest_command_deg",
                "max_current",
                "max_temperature",
                "max_tracking_error_deg",
                "joint_tracking_scale",
                "total_current_slope_ma_s",
                "contact_signal_valid",
            }
            if required <= values.keys():
                if values.get("current_unit") != "mA" or values.get("raw_velocity_unit") != "rpm":
                    self.errors.append("Unexpected telemetry units in diagnostics")
                    return
                if values.get("command_profile") != "direct_guarded":
                    self.errors.append("Default launch did not enable guarded direct command profile")
                    return
                if values.get("contact_signal_valid", "").lower() != "true":
                    return
                self.received.add("diagnostics")


class DisarmValidator(Node):
    """Verify the existing 15-second tracking grace, then its reasoned disarm."""

    def __init__(self) -> None:
        super().__init__("dg5f_lerobot_disarm_validator")
        self.disarmed = False
        self.in_grace = False
        self.create_subscription(DiagnosticArray, "/dg5f/lerobot/diagnostics", self._diagnostics, 10)

    def _diagnostics(self, message):
        for status in message.status:
            if status.name != "dg5f_lerobot_bridge":
                continue
            values = {item.key: item.value for item in status.values}
            self.in_grace = (values.get("armed", "").lower() == "true" and
                             values.get("tracking_grace_active", "").lower() == "true")
            self.disarmed = (values.get("armed", "").lower() == "false" and
                             values.get("disarm_reason") == "TRACKING_GRACE_EXPIRED")


def validate_disarmed(expect_grace=False) -> int:
    rclpy.init()
    node = DisarmValidator()
    deadline = time.monotonic() + (3.0 if expect_grace else 18.0)
    ready = lambda: node.in_grace if expect_grace else node.disarmed
    try:
        while time.monotonic() < deadline and not ready():
            rclpy.spin_once(node, timeout_sec=0.1)
        if not ready():
            print("Tracking watchdog did not enter expected grace/disarmed state", file=sys.stderr)
            return 1
        print("Validated tracking grace hold." if expect_grace else "Validated TRACKING_GRACE_EXPIRED disarm.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-disarmed", action="store_true")
    parser.add_argument("--expect-tracking-grace", action="store_true")
    args = parser.parse_args()
    if args.expect_disarmed or args.expect_tracking_grace:
        return validate_disarmed(args.expect_tracking_grace)

    required = {
        "quest",
        "landmarks",
        "command",
        "mujoco",
        "lerobot_command",
        "lerobot_state",
        "tracking",
        "armed",
        "diagnostics",
        "hybrid_contact",
        "safety_proximity",
    }
    rclpy.init()
    node = SmokeValidator()
    deadline = time.monotonic() + 12.0
    try:
        while time.monotonic() < deadline and not required <= node.received:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.errors:
                print(node.errors[0], file=sys.stderr)
                return 1
        missing = sorted(required - node.received)
        if missing:
            print(f"Missing valid smoke topics: {missing}", file=sys.stderr)
            return 1
        print("Validated topics and fixed rj_dg_5_1 across ROS, MuJoCo and LeRobot.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
