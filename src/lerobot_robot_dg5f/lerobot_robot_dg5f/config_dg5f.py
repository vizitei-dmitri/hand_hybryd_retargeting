"""LeRobot configuration for the Tesollo DG5F hand."""

from dataclasses import dataclass, field

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
        if (
            self.max_relative_target_deg is not None
            and self.max_relative_target_deg <= 0.0
        ):
            raise ValueError("max_relative_target_deg must be positive or None")
