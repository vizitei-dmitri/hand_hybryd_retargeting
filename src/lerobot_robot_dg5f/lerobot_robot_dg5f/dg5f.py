"""LeRobot ``Robot`` implementation for the Tesollo DG5F hand."""

from functools import cached_property
from typing import Callable

import numpy as np
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot

from .backends import MockDg5fBackend, TesolloDg5fBackend
from .command_shaper import PositionCommandShaper
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

    def __init__(
        self,
        config: Dg5fConfig,
        *,
        clock: Callable[[], float] | None = None,
    ):
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

        # Validate the fault map immediately, before any hardware connection.
        initial = apply_disabled_joints(
            np.zeros(len(JOINT_NAMES)), self.config.disabled_joint_positions_deg
        )
        disabled_by_index = {
            JOINT_NAMES.index(name): float(position)
            for name, position in self.config.disabled_joint_positions_deg.items()
        }
        shaper_kwargs = {}
        if clock is not None:
            shaper_kwargs["clock"] = clock
        self.command_shaper = PositionCommandShaper(
            LOWER_LIMITS_DEG,
            UPPER_LIMITS_DEG,
            disabled_positions_deg=disabled_by_index,
            smoothing=config.control_smoothing,
            max_speed_deg_s=config.max_speed_deg_s,
            max_accel_deg_s2=config.max_accel_deg_s2,
            response_time_s=config.response_time_s,
            filter_tau_s=config.filter_tau_s,
            target_deadband_deg=config.target_deadband_deg,
            min_send_step_deg=config.min_send_step_deg,
            max_direct_step_deg=config.max_direct_step_deg,
            startup_blend_s=config.startup_blend_s,
            max_dt_s=config.max_dt_s,
            **shaper_kwargs,
        )
        self._telemetry["pos"] = initial

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
        del calibrate
        if self.is_connected:
            return
        self.backend.connect()
        try:
            initial = self.backend.read_initial_position(
                self.config.initial_feedback_timeout_s,
                self.config.telemetry_drain_limit,
            )
            initial = apply_disabled_joints(
                initial, self.config.disabled_joint_positions_deg
            )
            self._telemetry["pos"] = initial.copy()
            self.command_shaper.reset(initial)
        except Exception:
            self.backend.disconnect()
            raise
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
        fresh = self.backend.read_telemetry(self.config.telemetry_drain_limit)
        for field in TELEMETRY_FIELDS:
            values = fresh.get(field)
            if values is not None:
                self._telemetry[field] = values
        return {
            f"{joint}.{field}": float(self._telemetry[field][index])
            for field in TELEMETRY_FIELDS
            for index, joint in enumerate(JOINT_NAMES)
        }

    def get_diagnostics(self) -> dict[str, object]:
        """Return passive low-level health and the most recent telemetry."""
        if not self.is_connected:
            status: dict[str, object] = {
                "connected": False,
                "transport_connected": False,
                "control_running": False,
                "control_thread_alive": False,
                "motion_ready": False,
                "system_started": False,
                "telemetry_valid": False,
                "temperature_safe": False,
            }
        else:
            status = dict(self.backend.read_control_status())

        for field in TELEMETRY_FIELDS:
            status.setdefault(f"measured_{field}", self._telemetry[field].copy())
        if "raw_velocity" in status:
            status["measured_vel"] = np.asarray(status["raw_velocity"]) * 6.0
        status.update(current_unit="mA", raw_velocity_unit="rpm", velocity_unit="degree/s",
                      position_unit="degree", temperature_unit="C")
        return status

    def prepare_arm(self) -> bool:
        """Passive arm preflight.

        This method is deliberately side-effect free for the physical hand: it
        must never call DGSDK recovery, ``SystemStart`` or send a position.  A
        faulted session has to be recovered explicitly via :meth:`recover`.
        """
        status = self.get_diagnostics()
        if not status.get("transport_connected", False):
            raise RuntimeError("DGSDK transport not connected")
        if not status.get("control_thread_alive", status.get("control_running", False)):
            raise RuntimeError("DGSDK control thread is not alive")
        if not status.get("telemetry_valid", False):
            raise RuntimeError("DGSDK telemetry is stale; run explicit recover first")
        if not status.get("temperature_safe", False):
            raise RuntimeError("DGSDK temperature is unsafe")
        if not status.get("system_started", False):
            raise RuntimeError("DGSDK system is not started; run explicit recover first")
        if status.get("recovery_required", False):
            raise RuntimeError("DGSDK recovery is required; run explicit recover first")
        if not status.get("motion_ready", False):
            raise RuntimeError(str(status.get("motion_ready_reason", "SDK_NOT_MOTION_READY")))
        return False

    def recover(self) -> np.ndarray:
        """Explicitly recover a faulted DGSDK session while remaining disarmed.

        Recovery reseeds the software command state after the low-level session
        has been restored. Manual ARM and tracking auto-resume may also perform
        a one-shot software reseed from already-fresh feedback, but never on
        every control tick.
        """
        pose = self.backend.recover(self.config.initial_feedback_timeout_s)
        pose = apply_disabled_joints(pose, self.config.disabled_joint_positions_deg)
        self.command_shaper.reset(pose)
        self._telemetry["pos"] = pose.copy()
        status = self.get_diagnostics()
        if not status.get("motion_ready", False):
            raise RuntimeError(str(status.get("motion_ready_reason", "SDK_NOT_MOTION_READY")))
        return pose.copy()

    def begin_arm_blend_from_feedback(
        self, *, max_pose_age_ms: float, duration_s: float | None = None
    ) -> np.ndarray:
        """Reseed once from fresh physical feedback, then start a safe blend.

        This is intentionally used only at a discrete transition (manual ARM
        or automatic resume after tracking loss), never on every control tick.
        No hardware command is sent by this method.
        """
        if not self.is_connected:
            raise RuntimeError("DG5F is not connected")
        if not np.isfinite(max_pose_age_ms) or max_pose_age_ms <= 0.0:
            raise ValueError("max_pose_age_ms must be finite and positive")

        status = self.backend.read_control_status()
        if not status.get("telemetry_valid", False):
            raise RuntimeError("PHYSICAL_POSE_STALE")
        age = float(status.get("last_position_sample_age_ms", float("inf")))
        if not np.isfinite(age) or age > max_pose_age_ms:
            raise RuntimeError("PHYSICAL_POSE_STALE")

        fresh = self.backend.read_telemetry(self.config.telemetry_drain_limit)
        pose = fresh.get("pos")
        if pose is None:
            raise RuntimeError("PHYSICAL_POSE_STALE")
        pose = apply_disabled_joints(
            np.asarray(pose, dtype=np.float64),
            self.config.disabled_joint_positions_deg,
        )
        self._telemetry["pos"] = pose.copy()
        self.command_shaper.reset(pose)
        self.command_shaper.begin_arm_blend(duration_s=duration_s)
        return pose.copy()

    def pause_trajectory(self) -> None:
        """Freeze software trajectory at the last accepted setpoint only.

        Unlike :meth:`hold_position`, this deliberately keeps the low-level
        servo keepalive alive, so a short tracking outage holds the last pose
        without accepting any new VR targets.
        """
        if self.command_shaper.is_initialized:
            self.command_shaper.hold()

    def send_action(self, action: RobotAction, *, command_bounds=None) -> RobotAction:
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

        step = self.command_shaper.step(requested, command_bounds=command_bounds)
        if step.should_send:
            self.backend.send_positions(step.output_deg)
            self.command_shaper.accept_output(step.output_deg)
        effective = self.command_shaper.effective_command()
        return {
            f"{joint}.pos": float(effective[index])
            for index, joint in enumerate(JOINT_NAMES)
        }

    def hold_position(self) -> None:
        """Cancel motion and stop low-level keepalive without a new setpoint."""
        if self.command_shaper.is_initialized:
            self.command_shaper.hold()
        if self.is_connected:
            self.backend.suspend_motion()

    def disconnect(self) -> None:
        if self.is_connected:
            self.backend.disconnect()
