"""LeRobot configuration for the Tesollo DG5F hand."""

from dataclasses import dataclass, field
from math import isfinite

from lerobot.robots import RobotConfig

from .constants import BROKEN_PINKY_JOINT


@RobotConfig.register_subclass("dg5f")
@dataclass
class Dg5fConfig(RobotConfig):
    """Configuration for either a real Tesollo hand or the mock test backend."""

    ip: str = "169.254.186.72"
    port: int = 502
    slave_id: int = 1
    backend: str = "mock"
    control_smoothing: bool = True
    max_speed_deg_s: float = 30.0
    max_accel_deg_s2: float = 60.0
    response_time_s: float = 0.15
    filter_tau_s: float = 0.05
    target_deadband_deg: float = 0.20
    min_send_step_deg: float = 0.20
    max_direct_step_deg: float = 4.0
    startup_blend_s: float = 0.70
    max_dt_s: float = 0.05
    initial_feedback_timeout_s: float = 2.0
    telemetry_drain_limit: int = 16
    # Deprecated compatibility setting. PositionCommandShaper no longer uses it.
    max_relative_target_deg: float | None = 7.0
    disabled_joint_positions_deg: dict[str, float] = field(
        default_factory=lambda: {BROKEN_PINKY_JOINT: 0.0}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend not in {"mock", "tesollo"}:
            raise ValueError("backend must be 'mock' or 'tesollo'")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.slave_id < 0:
            raise ValueError("slave_id must be non-negative")
        positive = {
            "max_speed_deg_s": self.max_speed_deg_s,
            "max_accel_deg_s2": self.max_accel_deg_s2,
            "response_time_s": self.response_time_s,
            "max_dt_s": self.max_dt_s,
            "initial_feedback_timeout_s": self.initial_feedback_timeout_s,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        non_negative = {
            "filter_tau_s": self.filter_tau_s,
            "target_deadband_deg": self.target_deadband_deg,
            "min_send_step_deg": self.min_send_step_deg,
            "max_direct_step_deg": self.max_direct_step_deg,
            "startup_blend_s": self.startup_blend_s,
        }
        for name, value in non_negative.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.telemetry_drain_limit <= 0:
            raise ValueError("telemetry_drain_limit must be positive")
        if (
            self.max_relative_target_deg is not None
            and (
                not isfinite(self.max_relative_target_deg)
                or self.max_relative_target_deg <= 0.0
            )
        ):
            raise ValueError("max_relative_target_deg must be positive or None")
