#!/usr/bin/env python3
"""Validate all integration topics concurrently during the mock smoke test."""

import sys
import time

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
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


def main() -> int:
    required = {
        "quest",
        "landmarks",
        "command",
        "mujoco",
        "lerobot_command",
        "lerobot_state",
        "tracking",
        "armed",
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
