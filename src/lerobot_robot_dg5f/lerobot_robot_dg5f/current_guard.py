"""Adaptive current-aware target limiting for DG5F position teleoperation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurrentGuardDecision:
    """One current-guard update."""

    target_deg: np.ndarray
    active: bool
    trip: bool
    trip_duration_s: float
    max_current_ma: float
    total_current_ma: float
    min_scale: float
    limited_mask: np.ndarray


def _linear_scale(value: float | np.ndarray, soft: float, hard: float):
    """Map load to 1 below ``soft`` and 0 at/above ``hard``."""
    values = np.asarray(value, dtype=np.float64)
    result = (hard - values) / (hard - soft)
    result = np.clip(result, 0.0, 1.0)
    if np.ndim(value) == 0:
        return float(result)
    return result


class AdaptiveCurrentGuard:
    """Slow only load-increasing motion as motor current approaches a limit.

    Below the soft thresholds the target is returned unchanged. Between soft
    and hard thresholds, load-increasing target motion is progressively rate
    limited. At/above a hard threshold the affected setpoint is walked back
    toward the measured physical pose, while a command that already relieves
    the load is allowed through normally. A sustained trip threshold requests
    a high-level disarm as a final backstop.
    """

    def __init__(
        self,
        joint_count: int,
        *,
        soft_ma: float,
        hard_ma: float,
        trip_ma: float,
        total_soft_ma: float,
        total_hard_ma: float,
        total_trip_ma: float,
        trip_hold_s: float,
        release_tau_s: float,
        nominal_step_deg: float,
    ) -> None:
        if joint_count <= 0:
            raise ValueError("joint_count must be positive")
        self.joint_count = int(joint_count)
        self.soft_ma = self._validate_thresholds("joint", soft_ma, hard_ma, trip_ma)
        self.hard_ma = float(hard_ma)
        self.trip_ma = float(trip_ma)
        self.total_soft_ma = self._validate_thresholds(
            "total", total_soft_ma, total_hard_ma, total_trip_ma
        )
        self.total_hard_ma = float(total_hard_ma)
        self.total_trip_ma = float(total_trip_ma)
        if not math.isfinite(trip_hold_s) or trip_hold_s < 0.0:
            raise ValueError("trip_hold_s must be finite and non-negative")
        if not math.isfinite(release_tau_s) or release_tau_s < 0.0:
            raise ValueError("release_tau_s must be finite and non-negative")
        if not math.isfinite(nominal_step_deg) or nominal_step_deg <= 0.0:
            raise ValueError("nominal_step_deg must be finite and positive")
        self.trip_hold_s = float(trip_hold_s)
        self.release_tau_s = float(release_tau_s)
        self.nominal_step_deg = float(nominal_step_deg)
        self.reset()

    @staticmethod
    def _validate_thresholds(name: str, soft: float, hard: float, trip: float) -> float:
        values = (float(soft), float(hard), float(trip))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(f"{name} current thresholds must be finite and positive")
        if not values[0] < values[1] < values[2]:
            raise ValueError(f"{name} current thresholds must satisfy soft < hard < trip")
        return values[0]

    def reset(self, now: float | None = None) -> None:
        self._joint_scale = np.ones(self.joint_count, dtype=np.float64)
        self._global_scale = 1.0
        self._last_time = None if now is None else float(now)
        self._trip_started: float | None = None

    def _release_smoothed_scale(
        self, previous: np.ndarray | float, raw: np.ndarray | float, dt: float
    ):
        """Attack immediately; release slowly to avoid guard chatter."""
        prev = np.asarray(previous, dtype=np.float64)
        raw_values = np.asarray(raw, dtype=np.float64)
        if self.release_tau_s <= 0.0:
            updated = raw_values
        else:
            alpha = 1.0 - math.exp(-max(0.0, dt) / self.release_tau_s)
            recovering = prev + alpha * (raw_values - prev)
            updated = np.where(raw_values < prev, raw_values, recovering)
        if np.ndim(previous) == 0:
            return float(updated)
        return updated

    def update(
        self,
        *,
        current_ma: np.ndarray,
        measured_deg: np.ndarray,
        effective_deg: np.ndarray,
        desired_deg: np.ndarray,
        now: float,
    ) -> CurrentGuardDecision:
        currents = np.asarray(current_ma, dtype=np.float64)
        measured = np.asarray(measured_deg, dtype=np.float64)
        effective = np.asarray(effective_deg, dtype=np.float64)
        desired = np.asarray(desired_deg, dtype=np.float64)
        for label, values in (
            ("current", currents),
            ("measured", measured),
            ("effective", effective),
            ("desired", desired),
        ):
            if values.shape != (self.joint_count,):
                raise ValueError(f"{label} must contain {self.joint_count} values")
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(effective)) or not np.all(
            np.isfinite(desired)
        ):
            raise ValueError("current guard poses must be finite")
        if not math.isfinite(now):
            raise ValueError("current guard timestamp must be finite")

        # A missing current channel must not manufacture a false over-current.
        current_abs = np.where(np.isfinite(currents), np.abs(currents), 0.0)
        max_current = float(np.max(current_abs))
        total_current = float(np.sum(current_abs))

        raw_joint_scale = _linear_scale(current_abs, self.soft_ma, self.hard_ma)
        raw_global_scale = _linear_scale(
            total_current, self.total_soft_ma, self.total_hard_ma
        )
        dt = 0.0 if self._last_time is None else max(0.0, float(now) - self._last_time)
        self._last_time = float(now)
        self._joint_scale = self._release_smoothed_scale(
            self._joint_scale, raw_joint_scale, dt
        )
        self._global_scale = self._release_smoothed_scale(
            self._global_scale, raw_global_scale, dt
        )
        combined_scale = np.minimum(self._joint_scale, self._global_scale)

        # Decide whether the requested target would increase the position error
        # that the actuator is already fighting. Relief motion is never slowed.
        current_error = effective - measured
        desired_error = desired - measured
        worsening = np.abs(desired_error) > np.abs(current_error) + 1e-6

        target = desired.copy()
        limited = worsening & (combined_scale < 0.999999)
        if np.any(limited):
            delta = desired - effective
            allowed_step = self.nominal_step_deg * combined_scale
            target[limited] = effective[limited] + np.clip(
                delta[limited], -allowed_step[limited], allowed_step[limited]
            )

        # Once a hard threshold is reached, actively unload instead of merely
        # refusing further advance. Moving toward measured pose reduces the
        # commanded position error without asking the physical joint to jump.
        hard_loaded = (current_abs >= self.hard_ma) | (
            (total_current >= self.total_hard_ma) & (np.abs(current_error) > 1e-3)
        )
        hard_retreat = hard_loaded & ~(
            np.abs(desired_error) + 1e-6 < np.abs(current_error)
        )
        if np.any(hard_retreat):
            target[hard_retreat] = measured[hard_retreat]
            limited |= hard_retreat

        over_trip = bool(
            max_current >= self.trip_ma or total_current >= self.total_trip_ma
        )
        if over_trip:
            if self._trip_started is None:
                self._trip_started = float(now)
        else:
            self._trip_started = None
        trip_duration = (
            0.0
            if self._trip_started is None
            else max(0.0, float(now) - self._trip_started)
        )
        trip = over_trip and trip_duration >= self.trip_hold_s

        return CurrentGuardDecision(
            target_deg=target,
            active=bool(np.any(limited)),
            trip=bool(trip),
            trip_duration_s=float(trip_duration),
            max_current_ma=max_current,
            total_current_ma=total_current,
            min_scale=float(np.min(combined_scale)),
            limited_mask=limited.copy(),
        )
