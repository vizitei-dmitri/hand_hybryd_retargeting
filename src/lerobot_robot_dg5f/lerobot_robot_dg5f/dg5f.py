"""LeRobot ``Robot`` implementation for the Tesollo DG5F hand."""

from functools import cached_property

import numpy as np
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot

from .backends import MockDg5fBackend, TesolloDg5fBackend
from .config_dg5f import Dg5fConfig
from .constants import (
    JOINT_NAMES,
    LOWER_LIMITS_DEG,
    TELEMETRY_FIELDS,
    UPPER_LIMITS_DEG,
)


def apply_disabled_joints(
    positions_deg: np.ndarray,
    disabled_positions_deg: dict[str, float],
) -> np.ndarray:
    """Return a copy with mechanically unavailable joints held fixed."""
    result = np.asarray(positions_deg, dtype=np.float64).copy()
    if result.shape != (len(JOINT_NAMES),):
        raise ValueError(f"Expected {len(JOINT_NAMES)} DG5F positions")
    for name, fixed_position in disabled_positions_deg.items():
        if name not in JOINT_NAMES:
            raise ValueError(f"Unknown disabled DG5F joint: {name}")
        if not np.isfinite(fixed_position):
            raise ValueError(f"Disabled position for {name} must be finite")
        result[JOINT_NAMES.index(name)] = float(fixed_position)
    return result


class Dg5f(Robot):
    """Twenty-DoF DG5F hand exposed through the standard LeRobot API.

    LeRobot ``*.pos`` action and observation values are degrees because that is
    the native unit of Tesollo DGSDK. The ROS bridge performs radian conversion.
    """

    config_class = Dg5fConfig
    name = "dg5f"

    def __init__(self, config: Dg5fConfig):
        super().__init__(config)
        self.config = config
        if config.backend == "tesollo":
            self.backend = TesolloDg5fBackend(
                config.ip, config.port, config.slave_id
            )
        else:
            self.backend = MockDg5fBackend()
        self._telemetry = {
            field: np.zeros(len(JOINT_NAMES), dtype=np.float64)
            for field in TELEMETRY_FIELDS
        }
        self._last_sent_deg = np.zeros(len(JOINT_NAMES), dtype=np.float64)

        # Validate the fault map immediately, before any hardware connection.
        apply_disabled_joints(
            self._last_sent_deg, self.config.disabled_joint_positions_deg
        )

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return {
            f"{joint}.{field}": float
            for field in TELEMETRY_FIELDS
            for joint in JOINT_NAMES
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        return self.backend.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        self.backend.connect()
        telemetry = self.backend.read_telemetry()
        if telemetry.get("pos") is not None:
            self._telemetry["pos"] = telemetry["pos"]
            self._last_sent_deg = telemetry["pos"].copy()
        self._last_sent_deg = apply_disabled_joints(
            self._last_sent_deg, self.config.disabled_joint_positions_deg
        )
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Tesollo reports calibrated joint angles; no LeRobot calibration needed."""

    def configure(self) -> None:
        """The bundled DGSDK configures gains and developer mode on ``start``."""

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise RuntimeError("DG5F is not connected")
        fresh = self.backend.read_telemetry()
        for field in TELEMETRY_FIELDS:
            values = fresh.get(field)
            if values is not None:
                self._telemetry[field] = values
        return {
            f"{joint}.{field}": float(self._telemetry[field][index])
            for field in TELEMETRY_FIELDS
            for index, joint in enumerate(JOINT_NAMES)
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise RuntimeError("DG5F is not connected")

        missing = [
            f"{joint}.pos"
            for joint in JOINT_NAMES
            if f"{joint}.pos" not in action
        ]
        if missing:
            raise ValueError(f"Action is missing DG5F position keys: {missing}")

        requested = np.asarray(
            [float(action[f"{joint}.pos"]) for joint in JOINT_NAMES],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(requested)):
            raise ValueError("DG5F action contains NaN or infinity")

        safe = np.clip(requested, LOWER_LIMITS_DEG, UPPER_LIMITS_DEG)
        max_step = self.config.max_relative_target_deg
        if max_step is not None:
            safe = np.clip(
                safe,
                self._last_sent_deg - max_step,
                self._last_sent_deg + max_step,
            )
        safe = apply_disabled_joints(
            safe, self.config.disabled_joint_positions_deg
        )

        self.backend.send_positions(safe)
        self._last_sent_deg = safe
        return {
            f"{joint}.pos": float(safe[index])
            for index, joint in enumerate(JOINT_NAMES)
        }

    def disconnect(self) -> None:
        if self.is_connected:
            self.backend.disconnect()
