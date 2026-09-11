"""Adaptive current-aware target limiting for DG5F position teleoperation."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from dg5f_teleop.contact_signals import FINGERS, PAIRS, PAIR_NAMES, proximity_weights
from .object_contact import ObjectContactConfig, PerFingerObjectContact


@dataclass(frozen=True)
class ComplianceConfig:
    """Additional software envelope. Existing emergency thresholds are separate."""

    enabled: bool = True
    attack_tau_s: float = 0.04
    release_tau_s: float = 0.10
    joint_slope_soft_ma_s: float = 800.0
    joint_slope_hard_ma_s: float = 3500.0
    total_slope_soft_ma_s: float = 1200.0
    total_slope_hard_ma_s: float = 5000.0
    slope_filter_tau_s: float = 0.08
    slope_max_reduction: float = 0.60
    contact_preemptive_reduction: float = 0.06
    contact_current_start_ma: float = 150.0
    lead_min_deg: float = 1.0
    lead_max_deg: float = 8.0
    thumb_contact_start_m: float = 0.055
    thumb_contact_full_m: float = 0.025
    adjacent_contact_start_m: float = 0.025
    adjacent_contact_full_m: float = 0.010
    stall_enabled: bool = True
    stall_current_ma: float = 350.0
    stall_error_deg: float = 8.0
    stall_velocity_deg_s: float = 0.5
    stall_hold_s: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool):
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"compliance {name} must be finite and positive")
        for prefix in ("joint", "total"):
            if getattr(self, f"{prefix}_slope_soft_ma_s") >= getattr(self, f"{prefix}_slope_hard_ma_s"):
                raise ValueError("Slope thresholds must satisfy soft < hard")
        if self.lead_min_deg > self.lead_max_deg:
            raise ValueError("lead_min_deg must not exceed lead_max_deg")
        if self.slope_max_reduction > 1 or self.contact_preemptive_reduction > 1:
            raise ValueError("Scale reductions must not exceed 1")
        proximity_weights(np.ones(7), self.thumb_contact_start_m, self.thumb_contact_full_m,
                          self.adjacent_contact_start_m, self.adjacent_contact_full_m)


class RobustCurrentTrend:
    """Median-of-three followed by a five-sample fit and EMA of the slope."""

    def __init__(self, size, tau):
        self.raw = deque(maxlen=3)
        self.history = deque(maxlen=5)
        self.slope = np.zeros(size)
        self.tau = tau

    def update(self, values, now, dt):
        self.raw.append(np.asarray(values).copy())
        self.history.append((now, np.median(np.stack(self.raw), axis=0)))
        if len(self.history) >= 3:
            times = np.array([item[0] for item in self.history])
            times -= times.mean()
            denominator = float(times @ times)
            if denominator > 1e-8:
                fitted = times @ np.stack([item[1] for item in self.history]) / denominator
                alpha = 1 - math.exp(-dt / self.tau)
                self.slope += alpha * (fitted - self.slope)
        return self.slope.copy()


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
    stall_trip: bool = False
    diagnostics: dict = field(default_factory=dict)
    object_contact_events: list[dict] = field(default_factory=list)


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
    limited. At/above a hard threshold, further load-increasing motion is
    frozen at the last effective setpoint; the guard does not invent an active
    retreat trajectory of its own. A command that relieves the load is still
    allowed through normally. A sustained trip threshold requests a high-level
    disarm as an additional software backstop. Vendor/DGSDK/firmware protection
    remains untouched and authoritative underneath this guard.
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
        compliance: ComplianceConfig | None = None,
        object_contact: ObjectContactConfig | None = None,
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
        self.compliance = compliance
        if object_contact is not None and joint_count != 20:
            raise ValueError("Object contact requires the DG5F 20-joint layout")
        self._object_contact = None if object_contact is None else PerFingerObjectContact(object_contact)
        if compliance is not None and compliance.contact_current_start_ma >= self.soft_ma:
            raise ValueError("contact_current_start_ma must be below joint current soft threshold")
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
        if self._object_contact is not None:
            self._object_contact.reset()
        self._joint_scale = np.ones(self.joint_count, dtype=np.float64)
        self._global_scale = 1.0
        self._last_time = None if now is None else float(now)
        self._trip_started: float | None = None
        self._adaptive_scale = np.ones(self.joint_count)
        self._trend = RobustCurrentTrend(self.joint_count, self.compliance.slope_filter_tau_s if self.compliance else 0.05)
        self._last_measured = None
        self._velocity = np.full(self.joint_count, np.inf)
        self._last_contact_stamp = None
        self._last_distances = None
        self._distance_rates = np.zeros(len(PAIRS))
        self._stall_started = np.full(self.joint_count, np.nan)

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
        hybrid_weights: np.ndarray | None = None,
        pair_distances_m: np.ndarray | None = None,
        pair_gradients_m_deg: np.ndarray | None = None,
        contact_stamp_s: float | None = None,
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
        if dt > 0.2:
            # Do not infer sustained faults or slopes across missing telemetry/control ticks.
            self._trend = RobustCurrentTrend(self.joint_count, self.compliance.slope_filter_tau_s if self.compliance else 0.05)
            self._last_measured = None
            self._trip_started = None
            self._stall_started.fill(np.nan)
        trend_dt = min(dt, 0.05)
        slope = self._trend.update(current_abs, float(now), trend_dt)
        total_slope = float(np.sum(slope))
        if self._last_measured is not None and dt > 0:
            speed = np.abs(measured - self._last_measured) / dt
            alpha = 1 - math.exp(-trend_dt / 0.05)
            self._velocity = np.where(np.isfinite(self._velocity),
                                      self._velocity + alpha * (speed - self._velocity), speed)
        else:
            self._velocity = np.full(self.joint_count, np.inf)
        self._last_measured = measured.copy()
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
        delta = desired - effective
        # A large opening request may cross measured pose. Its INITIAL direction
        # is still relief; comparing final absolute errors would wrongly block it.
        relief = delta * current_error < -1e-6
        worsening = (np.abs(desired_error) > np.abs(current_error) + 1e-6) & ~relief
        config = self.compliance
        enabled = config is not None and config.enabled
        contact_valid = False
        contact_pair_weights = np.zeros(len(PAIRS))
        contact_scale = np.ones(self.joint_count)
        joint_slope_scale = np.ones(self.joint_count)
        global_slope_scale = 1.0
        contact_limited_pairs = []
        contact_closing = np.zeros(self.joint_count, dtype=bool)
        load_evidence = np.zeros(self.joint_count)
        lead_budget = np.full(self.joint_count, np.inf)

        if enabled:
            joint_slope_risk = 1 - _linear_scale(slope, config.joint_slope_soft_ma_s, config.joint_slope_hard_ma_s)
            global_slope_risk = 1 - _linear_scale(total_slope, config.total_slope_soft_ma_s, config.total_slope_hard_ma_s)
            joint_slope_scale = 1 - config.slope_max_reduction * joint_slope_risk
            global_slope_scale = 1 - config.slope_max_reduction * global_slope_risk
            load_evidence = np.maximum(1 - raw_joint_scale, 1 - raw_global_scale)
            load_evidence = np.maximum(load_evidence, np.maximum(joint_slope_risk, global_slope_risk))
            distances = np.asarray(pair_distances_m, dtype=float)
            gradients = np.asarray(pair_gradients_m_deg, dtype=float)
            contact_valid = (
                self.joint_count == 20 and distances.shape == (len(PAIRS),)
                and gradients.shape == (len(PAIRS), 20)
                and np.all(np.isfinite(distances)) and np.all(distances >= 0)
                and np.all(np.isfinite(gradients))
                and contact_stamp_s is not None and math.isfinite(contact_stamp_s)
            )
            if contact_valid:
                contact_pair_weights = proximity_weights(
                    distances, config.thumb_contact_start_m, config.thumb_contact_full_m,
                    config.adjacent_contact_start_m, config.adjacent_contact_full_m,
                )
                weights = np.asarray(hybrid_weights, dtype=float)
                if weights.shape == (5,) and np.all(np.isfinite(weights)) and np.all((weights >= 0) & (weights <= 1)):
                    contact_pair_weights[:4] = np.maximum(contact_pair_weights[:4], np.minimum(weights[0], weights[1:]))
                if self._last_contact_stamp is not None and contact_stamp_s > self._last_contact_stamp:
                    contact_dt = contact_stamp_s - self._last_contact_stamp
                    if contact_dt <= 0.2:
                        raw_rates = (distances - self._last_distances) / contact_dt
                        alpha = 1 - math.exp(-contact_dt / 0.05)
                        self._distance_rates += alpha * (raw_rates - self._distance_rates)
                    else:
                        self._distance_rates.fill(0)
                if self._last_contact_stamp is None or contact_stamp_s > self._last_contact_stamp:
                    self._last_contact_stamp = contact_stamp_s
                    self._last_distances = distances.copy()
                contribution = gradients * np.clip(delta, -self.nominal_step_deg, self.nominal_step_deg)
                closing = (contribution < -1e-7) & (contact_pair_weights[:, None] > 0)
                opening = (contribution > 1e-7) & (contact_pair_weights[:, None] > 0)
                # Separation only exempts joints actually opening that pair, and
                # never overrides another pair that this same joint is closing.
                geometry_relief = np.any(opening & (self._distance_rates[:, None] > 0.002), axis=0) & ~np.any(closing, axis=0)
                relief |= geometry_relief
                worsening &= ~relief
                pair_load = np.maximum(
                    np.clip((current_abs - config.contact_current_start_ma) /
                            (self.soft_ma - config.contact_current_start_ma), 0, 1), joint_slope_risk,
                )
                for pair, (a, b) in enumerate(PAIRS):
                    involved = np.zeros(20, dtype=bool)
                    involved[a * 4:a * 4 + 4] = True
                    involved[b * 4:b * 4 + 4] = True
                    affected = closing[pair] & involved & ~relief
                    weight = contact_pair_weights[pair]
                    risk = float(np.max(pair_load[involved]))
                    scale = (1 - config.contact_preemptive_reduction * weight) * (1 - weight * risk)
                    if np.any(affected) and scale < 0.999999:
                        contact_limited_pairs.append(PAIR_NAMES[pair])
                        contact_scale[affected] = np.minimum(contact_scale[affected], scale)
                        load_evidence[affected] = np.maximum(load_evidence[affected], weight * risk)
                        contact_closing |= affected
            else:
                self._last_contact_stamp = self._last_distances = None
                self._distance_rates.fill(0)
            raw_adaptive = np.minimum.reduce([
                joint_slope_scale, np.full(self.joint_count, global_slope_scale), contact_scale,
            ])
            # Smooth only contact + dI/dt. Keep current protection out of this
            # state so its immediate attack and existing release stay separate.
            tau = np.where(raw_adaptive < self._adaptive_scale,
                           config.attack_tau_s, config.release_tau_s)
            alpha = 1.0 - np.exp(-min(dt, 0.05) / tau)
            self._adaptive_scale += alpha * (raw_adaptive - self._adaptive_scale)
            combined_scale = np.minimum(combined_scale, self._adaptive_scale)
            lead_budget = config.lead_min_deg + (config.lead_max_deg - config.lead_min_deg) * combined_scale

        target = desired.copy()
        limited = (worsening | contact_closing) & ~relief & (combined_scale < 0.999999)
        if np.any(limited):
            allowed_step = self.nominal_step_deg * combined_scale
            if enabled:
                # Scale the requested convergence, not just a fixed maximum step.
                allowed_step = np.minimum(allowed_step, np.abs(delta) * combined_scale)
                # Cap further error accumulation; never command a retreat toward
                # feedback. If already over budget, simply don't advance further.
                loaded = limited & (load_evidence > 0.01) & (delta * current_error >= 0)
                allowed_step[loaded] = np.minimum(allowed_step[loaded],
                    np.maximum(0, lead_budget[loaded] - np.abs(current_error[loaded])))
            target[limited] = effective[limited] + np.clip(
                delta[limited], -allowed_step[limited], allowed_step[limited]
            )

        object_decision = None
        if self._object_contact is not None:
            object_decision = self._object_contact.update(
                desired_deg=desired, effective_deg=effective, measured_deg=measured,
                current_ma=current_abs, slope_ma_s=slope, soft_target_deg=target, now=float(now),
            )
            target = object_decision.target_deg
            limited |= object_decision.limited_mask
            # Confirmed local preload relief passes the same emergency rule as
            # operator relief. All load-increasing motion still freezes below.
            object_mask = object_decision.limited_mask
            relief[object_mask] = ((target[object_mask] - effective[object_mask])
                                   * current_error[object_mask] < -1e-6)

        # At a hard threshold, stop adding load rather than generating an
        # automatic retreat. Freezing at the last effective setpoint is less
        # aggressive and avoids the guard itself commanding a sudden motion.
        # If the operator is already asking to reduce the position error, that
        # relief command remains allowed through.
        hard_loaded = (current_abs >= self.hard_ma) | (
            (total_current >= self.total_hard_ma) & (np.abs(current_error) > 1e-3)
        )
        hard_freeze = hard_loaded & ~relief
        if np.any(hard_freeze):
            target[hard_freeze] = effective[hard_freeze]
            limited |= hard_freeze

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

        stall_trip = False
        stall_duration = np.zeros(self.joint_count)
        if enabled and config.stall_enabled:
            stalled = ((current_abs >= config.stall_current_ma)
                       & (np.abs(current_error) >= config.stall_error_deg)
                       & (self._velocity <= config.stall_velocity_deg_s) & ~relief)
            self._stall_started = np.where(stalled,
                np.where(np.isnan(self._stall_started), now, self._stall_started), np.nan)
            stall_duration = np.where(stalled, now - self._stall_started, 0.0)
            stall_trip = bool(np.any(stall_duration >= config.stall_hold_s))
        adaptive_tracking_scale = np.where(limited, combined_scale, 1.0)
        adaptive_tracking_scale[hard_freeze] = 0.0
        # Report actual command gain, including a depleted lead budget.
        nonzero_delta = limited & (np.abs(delta) > 1e-8)
        adaptive_tracking_scale[nonzero_delta] = np.minimum(adaptive_tracking_scale[nonzero_delta],
            np.abs(target[nonzero_delta] - effective[nonzero_delta]) /
            np.minimum(np.abs(delta[nonzero_delta]), self.nominal_step_deg))
        diagnostics = {
            "compliance_enabled": enabled, "compliance_active": bool(enabled and np.any(limited)),
            "contact_signal_valid": bool(contact_valid),
            "global_current_scale": self._global_scale,
            "global_slope_scale": float(global_slope_scale),
            "total_current_ma": total_current, "total_current_slope_ma_s": total_slope,
            "joint_current_scale": self._joint_scale.copy(),
            "joint_current_slope_ma_s": slope, "joint_slope_scale": joint_slope_scale,
            "joint_contact_scale": contact_scale,
            "joint_tracking_scale": adaptive_tracking_scale,
            "joint_lead_budget_deg": lead_budget,
            "hybrid_contact_weights": np.asarray(hybrid_weights).copy() if contact_valid and np.asarray(hybrid_weights).shape == (5,) else np.full(5, np.nan),
            "pair_distances_m": np.asarray(pair_distances_m).copy() if contact_valid else np.full(len(PAIRS), np.nan),
            "pair_distance_rates_m_s": self._distance_rates.copy(),
            "pair_contact_weights": contact_pair_weights,
            "contact_limited_pairs": contact_limited_pairs,
            "limited_fingers": [name for finger, name in enumerate(FINGERS)
                                if np.any(limited[finger * 4:finger * 4 + 4])],
            "stall_duration_s": float(np.max(stall_duration)),
        }

        if object_decision is not None:
            diagnostics.update(object_decision.diagnostics)

        return CurrentGuardDecision(
            target_deg=target,
            active=bool(np.any(limited)),
            trip=bool(trip),
            trip_duration_s=float(trip_duration),
            max_current_ma=max_current,
            total_current_ma=total_current,
            min_scale=float(np.min(adaptive_tracking_scale if enabled else combined_scale)),
            limited_mask=limited.copy(),
            stall_trip=stall_trip,
            diagnostics=diagnostics,
            object_contact_events=[] if object_decision is None else object_decision.transitions,
        )
