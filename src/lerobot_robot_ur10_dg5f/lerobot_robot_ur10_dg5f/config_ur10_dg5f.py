"""LeRobot configuration for the UR10e + DG5F cell."""

from dataclasses import dataclass, field
from math import isfinite

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot_robot_dg5f.constants import BROKEN_PINKY_JOINT


@RobotConfig.register_subclass("ur10_dg5f")
@dataclass
class Ur10Dg5fConfig(RobotConfig):
    """Either the MuJoCo twin or the real UR10e + DG5F cell.

    Poses are in the UR base frame, as reported by ``getActualTCPPose()``.
    The workspace box below is sized for the colleague's MuJoCo table; measure
    the real one on the teach pendant before using hardware.
    """

    backend: str = "mujoco"
    mjcf_path: str = "models/ur10e_dg5f/scene.xml"
    sim_cameras: tuple[str, ...] = ("cam_front", "cam_side")
    sim_image_hw: tuple[int, int] = (240, 320)

    ur_ip: str = "192.168.1.10"
    ur_servo_hz: float = 500.0
    ur_lookahead_s: float = 0.1
    ur_gain: float = 300.0

    hand_backend: str = "tesollo"
    hand_ip: str = "169.254.186.72"
    hand_servo_hz: float = 60.0

    max_step_m: float = 0.010
    max_step_rad: float = 0.05
    workspace_lo: tuple[float, float, float] = (-0.40, 0.35, 0.44)
    workspace_hi: tuple[float, float, float] = (0.40, 0.90, 0.80)
    # initial_pose of the colleague's env raised by 10 cm: the original one is inside the table.
    home_joints_rad: tuple[float, ...] = (1.4201, -1.8540, 2.2389, -1.9449, -1.5715, -0.1499)
    hand_disabled_joints_deg: dict[str, float] = field(
        default_factory=lambda: {BROKEN_PINKY_JOINT: 0.0}
    )
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend not in {"mujoco", "real"}:
            raise ValueError("backend must be 'mujoco' or 'real'")
        if self.hand_backend not in {"tesollo", "mock"}:
            raise ValueError("hand_backend must be 'tesollo' or 'mock'")
        if len(self.home_joints_rad) != 6:
            raise ValueError("home_joints_rad must have 6 values")
        if len(self.workspace_lo) != 3 or len(self.workspace_hi) != 3:
            raise ValueError("workspace bounds must have 3 values each")
        if any(lo >= hi for lo, hi in zip(self.workspace_lo, self.workspace_hi)):
            raise ValueError("workspace_lo must be below workspace_hi on every axis")
        if not 0.03 <= self.ur_lookahead_s <= 0.2:
            raise ValueError("ur_lookahead_s must be in [0.03, 0.2] (servoL range)")
        if not 100.0 <= self.ur_gain <= 2000.0:
            raise ValueError("ur_gain must be in [100, 2000] (servoL range)")
        positive = {
            "ur_servo_hz": self.ur_servo_hz,
            "hand_servo_hz": self.hand_servo_hz,
            "max_step_m": self.max_step_m,
            "max_step_rad": self.max_step_rad,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
