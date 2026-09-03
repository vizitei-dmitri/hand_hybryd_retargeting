"""Stateful position-command shaping for DG5F streaming teleoperation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class CommandStep:
    """Result of one command-shaper update, expressed in degrees."""

    command_deg: np.ndarray
    output_deg: np.ndarray
    should_send: bool
    velocity_deg_s: np.ndarray


class PositionCommandShaper:
    """Convert target positions into bounded position setpoints.

    Measured feedback is deliberately absent from this class. After ``reset``
    the authoritative trajectory state is ``command_pose_deg`` and
    ``velocity_deg_s``; asynchronous hardware feedback can therefore never
    reseed or rewind the command trajectory.
    """

    def __init__(
        self,
        lower_limits_deg: np.ndarray,
        upper_limits_deg: np.ndarray,
        *,
        disabled_positions_deg: Mapping[int, float] | None = None,
        smoothing: bool = True,
        max_speed_deg_s: float = 30.0,
        max_accel_deg_s2: float = 60.0,
        response_time_s: float = 0.15,
        filter_tau_s: float = 0.05,
        target_deadband_deg: float = 0.20,
        min_send_step_deg: float = 0.20,
        max_dt_s: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.lower_limits_deg = np.asarray(lower_limits_deg, dtype=np.float64)
        self.upper_limits_deg = np.asarray(upper_limits_deg, dtype=np.float64)
        if self.lower_limits_deg.ndim != 1:
            raise ValueError("Joint limits must be one-dimensional")
        if self.upper_limits_deg.shape != self.lower_limits_deg.shape:
            raise ValueError("Lower and upper joint limits must have equal shape")
        if not np.all(np.isfinite(self.lower_limits_deg)) or not np.all(
            np.isfinite(self.upper_limits_deg)
        ):
            raise ValueError("Joint limits must be finite")
        if np.any(self.lower_limits_deg > self.upper_limits_deg):
            raise ValueError("Lower joint limits exceed upper limits")

        self.joint_count = int(self.lower_limits_deg.size)
        self.disabled_positions_deg = dict(disabled_positions_deg or {})
        for index, position in self.disabled_positions_deg.items():
            if not 0 <= index < self.joint_count:
                raise ValueError(f"Disabled joint index is out of range: {index}")
            if not np.isfinite(position):
                raise ValueError(f"Disabled joint {index} position must be finite")
            if not self.lower_limits_deg[index] <= position <= self.upper_limits_deg[index]:
                raise ValueError(f"Disabled joint {index} position is outside its limits")

        positive = {
            "max_speed_deg_s": max_speed_deg_s,
            "max_accel_deg_s2": max_accel_deg_s2,
            "response_time_s": response_time_s,
            "max_dt_s": max_dt_s,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        non_negative = {
            "filter_tau_s": filter_tau_s,
            "target_deadband_deg": target_deadband_deg,
            "min_send_step_deg": min_send_step_deg,
        }
        for name, value in non_negative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        self.smoothing = bool(smoothing)
        self.max_speed_deg_s = float(max_speed_deg_s)
        self.max_accel_deg_s2 = float(max_accel_deg_s2)
        self.response_time_s = float(response_time_s)
        self.filter_tau_s = float(filter_tau_s)
        self.target_deadband_deg = float(target_deadband_deg)
        self.min_send_step_deg = float(min_send_step_deg)
        self.max_dt_s = float(max_dt_s)
        self._clock = clock

        self.command_pose_deg: np.ndarray | None = None
        self.velocity_deg_s: np.ndarray | None = None
        self.last_sent_pose_deg: np.ndarray | None = None
        self.filtered_target_deg: np.ndarray | None = None
        self.last_update_time: float | None = None

    @property
    def is_initialized(self) -> bool:
        return self.command_pose_deg is not None

    def _validate_pose(self, pose_deg: np.ndarray, label: str) -> np.ndarray:
        result = np.asarray(pose_deg, dtype=np.float64)
        if result.shape != (self.joint_count,):
            raise ValueError(f"{label} must contain {self.joint_count} positions")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{label} contains NaN or infinity")
        return result.copy()

    def _apply_disabled(self, pose_deg: np.ndarray) -> np.ndarray:
        result = np.asarray(pose_deg, dtype=np.float64).copy()
        for index, position in self.disabled_positions_deg.items():
            result[index] = position
        return result

    def reset(self, initial_pose_deg: np.ndarray, now: float | None = None) -> None:
        """Initialize the internal trajectory once from measured position."""
        initial = self._validate_pose(initial_pose_deg, "Initial pose")
        initial = np.clip(initial, self.lower_limits_deg, self.upper_limits_deg)
        initial = self._apply_disabled(initial)
        timestamp = self._clock() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("Reset timestamp must be finite")

        self.command_pose_deg = initial.copy()
        self.velocity_deg_s = np.zeros(self.joint_count, dtype=np.float64)
        self.last_sent_pose_deg = initial.copy()
        self.filtered_target_deg = initial.copy()
        self.last_update_time = timestamp

    def step(self, target_deg: np.ndarray, now: float | None = None) -> CommandStep:
        """Advance the internal trajectory and propose the next SDK setpoint."""
        if not self.is_initialized:
            raise RuntimeError("PositionCommandShaper must be reset before use")
        assert self.command_pose_deg is not None
        assert self.velocity_deg_s is not None
        assert self.last_sent_pose_deg is not None
        assert self.filtered_target_deg is not None
        assert self.last_update_time is not None

        target = self._validate_pose(target_deg, "Target")
        target = np.clip(target, self.lower_limits_deg, self.upper_limits_deg)
        target = self._apply_disabled(target)

        timestamp = self._clock() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("Step timestamp must be finite")
        elapsed = timestamp - self.last_update_time
        dt = float(np.clip(elapsed, 1e-3, self.max_dt_s))
        self.last_update_time = timestamp

        if self.smoothing:
            alpha = 1.0 if self.filter_tau_s == 0.0 else dt / (self.filter_tau_s + dt)
            self.filtered_target_deg += alpha * (
                target - self.filtered_target_deg
            )
            self.filtered_target_deg = self._apply_disabled(self.filtered_target_deg)

            error = self.filtered_target_deg - self.command_pose_deg
            error[np.abs(error) < self.target_deadband_deg] = 0.0
            desired_velocity = np.clip(
                error / self.response_time_s,
                -self.max_speed_deg_s,
                self.max_speed_deg_s,
            )
            max_delta_velocity = self.max_accel_deg_s2 * dt
            delta_velocity = np.clip(
                desired_velocity - self.velocity_deg_s,
                -max_delta_velocity,
                max_delta_velocity,
            )
            self.velocity_deg_s += delta_velocity
            new_command = self.command_pose_deg + self.velocity_deg_s * dt

            before = self.filtered_target_deg - self.command_pose_deg
            after = self.filtered_target_deg - new_command
            crossed = before * after <= 0.0
            new_command[crossed] = self.filtered_target_deg[crossed]
            self.velocity_deg_s[crossed] = 0.0
            self.command_pose_deg = np.clip(
                new_command, self.lower_limits_deg, self.upper_limits_deg
            )
        else:
            self.filtered_target_deg = target.copy()
            self.command_pose_deg = target.copy()
            self.velocity_deg_s.fill(0.0)

        self.command_pose_deg = self._apply_disabled(self.command_pose_deg)
        self.velocity_deg_s[list(self.disabled_positions_deg)] = 0.0

        delta = self.command_pose_deg - self.last_sent_pose_deg
        ready = np.abs(delta) >= self.min_send_step_deg
        should_send = bool(np.any(ready))
        output = self.last_sent_pose_deg.copy()
        if should_send:
            output[ready] = self.command_pose_deg[ready]
            output = self._apply_disabled(output)

        return CommandStep(
            command_deg=self.command_pose_deg.copy(),
            output_deg=output,
            should_send=should_send,
            velocity_deg_s=self.velocity_deg_s.copy(),
        )

    def accept_output(self, output_deg: np.ndarray) -> None:
        """Commit a proposed output only after the backend accepted it."""
        if self.last_sent_pose_deg is None:
            raise RuntimeError("PositionCommandShaper must be reset before use")
        output = self._validate_pose(output_deg, "Accepted output")
        output = np.clip(output, self.lower_limits_deg, self.upper_limits_deg)
        self.last_sent_pose_deg = self._apply_disabled(output)

    def hold(self, now: float | None = None) -> None:
        """Stop the internal trajectory at the last backend-accepted setpoint."""
        if self.last_sent_pose_deg is None or self.velocity_deg_s is None:
            raise RuntimeError("PositionCommandShaper must be reset before use")
        timestamp = self._clock() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("Hold timestamp must be finite")
        held = self._apply_disabled(self.last_sent_pose_deg)
        self.command_pose_deg = held.copy()
        self.filtered_target_deg = held.copy()
        self.velocity_deg_s.fill(0.0)
        self.last_update_time = timestamp

    def effective_command(self) -> np.ndarray:
        """Return the last position setpoint accepted by the backend."""
        if self.last_sent_pose_deg is None:
            raise RuntimeError("PositionCommandShaper must be reset before use")
        return self.last_sent_pose_deg.copy()
