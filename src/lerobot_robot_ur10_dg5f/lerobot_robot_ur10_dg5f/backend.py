"""Backend contract shared by the MuJoCo twin and the real cell."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass
class RobotState:
    tcp: np.ndarray            # (6,) UR base frame, metres and rotation vector
    arm_q: np.ndarray          # (6,) radians
    hand_deg: np.ndarray       # (20,) degrees, JOINT_NAMES order
    protective_stop: bool = False


class Backend(Protocol):
    @property
    def is_connected(self) -> bool: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read(self) -> RobotState: ...
    def command(self, tcp_target: np.ndarray, hand_deg: np.ndarray) -> None: ...
    def wait_next_period(self, period_s: float) -> None: ...
    def go_home(self, arm_q: np.ndarray) -> None: ...
    def images(self) -> dict[str, np.ndarray]: ...


def make_backend(config) -> Backend:
    # Lazy imports: MuJoCo lives only in the Docker image, ur_rtde only next to the robot.
    if config.backend == "mujoco":
        from .sim_backend import MujocoBackend

        return MujocoBackend(config)
    from .real_backend import RealBackend

    return RealBackend(config)
