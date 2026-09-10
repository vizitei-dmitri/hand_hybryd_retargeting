"""Adaptive current-aware target limiting for DG5F position teleoperation."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from dg5f_teleop.contact_signals import FINGERS, PAIRS, PAIR_NAMES, proximity_weights


@dataclass(frozen=True)
class ComplianceConfig:
    """Additional software envelope. Existing emergency thresholds are separate."""

    enabled: bool = True
    joint_slope_soft_ma_s: float = 800.0
    joint_slope_hard_ma_s: float = 3500.0
    total_slope_soft_ma_s: float = 1200.0
    total_slope_hard_ma_s: float = 5000.0
    slope_filter_tau_s: float = 0.05
    slope_max_reduction: float = 0.8
    contact_preemptive_reduction: float = 0.12
    contact_current_start_ma: float = 100.0
    lead_min_deg: float = 1.0
    lead_max_deg: float = 8.0
    thumb_contact_start_m: float = 0.055
    thumb_contact_full_m: float = 0.025
    adjacent_contact_start_m: float = 0.030
    adjacent_contact_full_m: float = 0.012
    thumb_index_contact_distance_mm: float = 25.0
    thumb_index_slowdown_start_mm: float = 35.0
    soft_attack_tau_s: float = 0.03
    soft_release_tau_s: float = 0.06
    other_contact_max_reduction: float = 0.5
    yield_enabled: bool = True
    yield_current_ma: float = 240.0
    yield_error_deg: float = 4.0
    yield_velocity_deg_s: float = 1.0
    yield_hold_s: float = 0.25
    yield_rate_deg_s: float = 5.0
    stall_enabled: bool = True
    stall_current_ma: float = 350.0
    stall_error_deg: float = 8.0
    stall_velocity_deg_s: float = 0.5
    stall_hold_s: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool):
                continue
            allow_zero = name in {"slope_max_reduction", "contact_preemptive_reduction", "other_contact_max_reduction"}
            if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f"compliance {name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
        for prefix in ("joint", "total"):
            if getattr(self, f"{prefix}_slope_soft_ma_s") >= getattr(self, f"{prefix}_slope_hard_ma_s"):
                raise ValueError("Slope thresholds must satisfy soft < hard")
        if self.lead_min_deg > self.lead_max_deg:
            raise ValueError("lead_min_deg must not exceed lead_max_deg")
        if self.thumb_index_contact_distance_mm >= self.thumb_index_slowdown_start_mm:
            raise ValueError("Thumb-index contact distance must be below slowdown start")
        if self.slope_max_reduction > 1 or self.contact_preemptive_reduction > 1:
            raise ValueError("Scale reductions must not exceed 1")
        if self.other_contact_max_reduction >= 1:
            raise ValueError("Other-pair soft contact reduction must stay below 1")
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

    With the optional compliance layer disabled, below the soft thresholds the
    target is returned unchanged. The layer also anticipates contact and rising
    current, and bounds additional reference error under load. Between soft
    and hard thresholds, load-increasing target motion is progressively rate
    limited. At/above a hard threshold, further load-increasing motion is
    frozen at the last effective setpoint. Separately, confirmed moderate
    stationary load may yield reference lead at a bounded rate. A command that relieves the load is still
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
        self._joint_scale = np.ones(self.joint_count, dtype=np.float64)
        self._global_scale = 1.0
        self._last_time = None if now is None else float(now)
        self._trip_started: float | None = None
        self._adaptive_scale = np.ones(self.joint_count)
        self._soft_contact_scale = np.ones(self.joint_count)
        self._thumb_scale = np.ones(self.joint_count)
        self._trend = RobustCurrentTrend(self.joint_count, self.compliance.slope_filter_tau_s if self.compliance else 0.05)
        self._last_measured = None
        self._velocity = np.full(self.joint_count, np.inf)
        self._last_contact_stamp = None
        self._last_distances = None
        self._distance_rates = np.zeros(len(PAIRS))
        self._stall_started = np.full(self.joint_count, np.nan)
        self._yield_started = np.full(self.joint_count, np.nan)
        self._yield_latched = np.zeros(self.joint_count, dtype=bool)

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

    def _soft_smoothed_scale(self, previous, raw, dt):
        """Short asymmetric soft response; never used for hard freeze/trip."""
        tau = np.where(raw < previous, self.compliance.soft_attack_tau_s,
                       self.compliance.soft_release_tau_s)
        result = previous + (1 - np.exp(-min(max(dt, 0), 0.05) / tau)) * (raw - previous)
        # Stop an insignificant asymptotic tail from reporting ACTIVE forever.
        return np.where((raw == 1) & (result > 0.995), 1.0, result)

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
        robot_pair_distances_m: np.ndarray | None = None,
        robot_distance_fn=None,
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
            self._yield_started.fill(np.nan)
            self._yield_latched.fill(False)
        trend_dt = min(dt, 0.05)
        slope = self._trend.update(current_abs, float(now), trend_dt)
        total_slope = float(np.sum(slope))
        if self._last_measured is not None and dt > 0:
            speed = np.abs(measured - self._last_measured) / dt
            alpha = 1 - math.exp(-trend_dt / 0.05)
            previous_speed = np.where(np.isfinite(self._velocity), self._velocity, speed)
            self._velocity = previous_speed + alpha * (speed - previous_speed)
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
        actuator_relief = delta * current_error < -1e-6
        worsening = (np.abs(desired_error) > np.abs(current_error) + 1e-6) & ~actuator_relief
        config = self.compliance
        enabled = config is not None and config.enabled
        contact_valid = False
        contact_pair_weights = np.zeros(len(PAIRS))
        contact_scale = np.ones(self.joint_count)
        thumb_scale = np.ones(self.joint_count)
        contact_relief = np.zeros(self.joint_count, dtype=bool)
        joint_slope_scale = np.ones(self.joint_count)
        global_slope_scale = 1.0
        contact_limited_pairs = []
        contact_closing = np.zeros(self.joint_count, dtype=bool)
        load_evidence = np.zeros(self.joint_count)
        lead_budget = np.full(self.joint_count, np.inf)
        yield_pending = np.zeros(self.joint_count, dtype=bool)
        human_weights = np.asarray(hybrid_weights, dtype=float)
        human_valid = (human_weights.shape == (5,) and np.all(np.isfinite(human_weights))
                       and np.all((human_weights >= 0) & (human_weights <= 1)))
        human_distances = np.asarray(pair_distances_m, dtype=float)
        human_distances_valid = (human_distances.shape == (len(PAIRS),)
                                 and np.all(np.isfinite(human_distances)) and np.all(human_distances >= 0))
        robot_distances = np.asarray(robot_pair_distances_m, dtype=float)
        thumb_closing = np.zeros(self.joint_count, dtype=bool)
        thumb_motion_closing = np.zeros(self.joint_count, dtype=bool)
        thumb_contact_scale = 1.0

        if enabled:
            joint_slope_risk = 1 - _linear_scale(slope, config.joint_slope_soft_ma_s, config.joint_slope_hard_ma_s)
            global_slope_risk = 1 - _linear_scale(total_slope, config.total_slope_soft_ma_s, config.total_slope_hard_ma_s)
            # A single rising motor has its own limiter. Spread anticipatory
            # total-slope risk only for distributed finger loading, or when
            # the absolute total is already entering the shared envelope.
            rising_fingers = sum(float(np.sum(np.maximum(slope[i:i + 4], 0))) >
                                 config.total_slope_soft_ma_s / 5
                                 for i in range(0, self.joint_count, 4))
            if rising_fingers < 2 and total_current < self.total_soft_ma:
                global_slope_risk = 0.0
            joint_slope_scale = 1 - config.slope_max_reduction * joint_slope_risk
            global_slope_scale = 1 - config.slope_max_reduction * global_slope_risk
            load_evidence = np.maximum(1 - raw_joint_scale, 1 - raw_global_scale)
            load_evidence = np.maximum(load_evidence, np.maximum(joint_slope_risk, global_slope_risk))
            gradients = np.asarray(pair_gradients_m_deg, dtype=float)
            contact_valid = (
                self.joint_count == 20 and robot_distances.shape == (3, len(PAIRS))
                and gradients.shape == (len(PAIRS), 20)
                and np.all(np.isfinite(robot_distances)) and np.all(robot_distances >= 0)
                and np.all(np.isfinite(gradients))
            )
            if contact_valid:
                # Human proximity is intent, NEVER evidence of physical contact.
                # Measured pose detects actual proximity; effective pose also
                # prevents reference penetration before the mechanism catches up.
                distances = np.minimum(robot_distances[0], robot_distances[1])
                contact_pair_weights = proximity_weights(
                    distances, config.thumb_contact_start_m, config.thumb_contact_full_m,
                    config.adjacent_contact_start_m, config.adjacent_contact_full_m,
                )
                intent = (proximity_weights(human_distances, config.thumb_contact_start_m,
                          config.thumb_contact_full_m, config.adjacent_contact_start_m,
                          config.adjacent_contact_full_m) if human_distances_valid else np.zeros(len(PAIRS)))
                if human_valid:
                    intent[:4] = np.maximum(intent[:4], np.minimum(human_weights[0], human_weights[1:]))
                contact_pair_weights[1:4] *= intent[1:4]
                # Adjacent human fingertips are naturally close. For these
                # pairs robot proximity is diagnostic; only LOAD reduces gain.
                # Other pairs retain mild generic proximity/current behavior;
                # no new hard contact thresholds are invented for them.
                contact_mm = config.thumb_index_contact_distance_mm
                start_mm = config.thumb_index_slowdown_start_mm
                x = float(np.clip((start_mm - distances[0] * 1000) / (start_mm - contact_mm), 0, 1))
                contact_pair_weights[0] = x * x * (3 - 2 * x)
            # Human distance rates remain diagnostics only, not a relief gate.
            if human_distances_valid and contact_stamp_s is not None and math.isfinite(contact_stamp_s):
                if self._last_contact_stamp is not None and contact_stamp_s > self._last_contact_stamp:
                    contact_dt = contact_stamp_s - self._last_contact_stamp
                    if contact_dt <= 0.2:
                        raw_rates = (human_distances - self._last_distances) / contact_dt
                        alpha = 1 - math.exp(-contact_dt / 0.05)
                        self._distance_rates += alpha * (raw_rates - self._distance_rates)
                    else:
                        self._distance_rates.fill(0)
                if self._last_contact_stamp is None or contact_stamp_s > self._last_contact_stamp:
                    self._last_contact_stamp = contact_stamp_s
                    self._last_distances = human_distances.copy()
            else:
                self._last_contact_stamp = self._last_distances = None
                self._distance_rates.fill(0)
            if contact_valid:
                contribution = gradients * np.clip(delta, -self.nominal_step_deg, self.nominal_step_deg)
                thumb_motion_closing[:8] = contribution[0, :8] < -1e-7
                closing = (contribution < -1e-7) & (contact_pair_weights[:, None] > 0)
                opening = (contribution > 1e-7) & (contact_pair_weights[:, None] > 0)
                # Separation only exempts joints actually opening that pair, and
                # never overrides another pair that this same joint is closing.
                # Test movement relative to the CURRENT effective pose, not
                # human distances or a previous deeper desired target. A target
                # changing from 7 to 10 mm still closes an effective 25 mm gap.
                contact_relief = np.any(opening & (robot_distances[2, :, None] >
                                                   robot_distances[1, :, None]), axis=0) & ~np.any(closing, axis=0)
                # NEVER merge this with actuator_relief. Increasing a fingertip
                # gap says nothing about whether the servo's error is relieved.
                pair_load = np.maximum(
                    np.clip((current_abs - config.contact_current_start_ma) /
                            (self.soft_ma - config.contact_current_start_ma), 0, 1), joint_slope_risk,
                )
                for pair, (a, b) in enumerate(PAIRS):
                    involved = np.zeros(20, dtype=bool)
                    involved[a * 4:a * 4 + 4] = True
                    involved[b * 4:b * 4 + 4] = True
                    affected = closing[pair] & involved & ~contact_relief
                    weight = contact_pair_weights[pair]
                    risk = float(np.max(pair_load[involved]))
                    scale = 1 - config.other_contact_max_reduction * weight * risk
                    if pair == 0:
                        # Geometry is decisive near the measured contact region.
                        # Current strengthens this BEFORE emergency thresholds.
                        scale = (1 - weight) * (1 - weight * risk)
                        thumb_closing = affected
                        if np.any(affected):
                            thumb_contact_scale = scale
                        thumb_scale[affected] = scale
                    else:
                        affected &= ~actuator_relief
                    if np.any(affected) and scale < 0.999999:
                        contact_limited_pairs.append(PAIR_NAMES[pair])
                        if pair == 0:
                            load_evidence[affected] = np.maximum(load_evidence[affected], weight * risk)
                        else:
                            contact_scale[affected] = np.minimum(contact_scale[affected], scale)
                        contact_closing |= affected

            # Sustained moderate current + stationary feedback + reference lead.
            # Latch below the entry error too, so yielding can reduce 6 -> 2 deg
            # rather than dropping out as soon as the error crosses 4 deg.
            yield_base = (config.yield_enabled & (current_abs >= config.yield_current_ma)
                          & (self._velocity <= config.yield_velocity_deg_s) & ~actuator_relief)
            yield_entry = yield_base & (np.abs(current_error) >= config.yield_error_deg)
            self._yield_started = np.where(yield_entry,
                np.where(np.isnan(self._yield_started), now, self._yield_started), np.nan)
            self._yield_latched = yield_base & (self._yield_latched |
                (now - self._yield_started >= config.yield_hold_s))
            yield_pending = yield_entry | self._yield_latched
            sustained_risk = np.where(self._yield_latched, np.clip(current_abs / self.soft_ma, 0, 1), 0)
            load_evidence = np.maximum(load_evidence, sustained_risk)
            raw_adaptive = np.minimum.reduce([
                joint_slope_scale, np.full(self.joint_count, global_slope_scale), 1 - sustained_risk,
            ])
            self._adaptive_scale = self._soft_smoothed_scale(self._adaptive_scale, raw_adaptive, dt)
            self._soft_contact_scale = self._soft_smoothed_scale(self._soft_contact_scale, contact_scale, dt)
            # Preserve thumb-index's immediate attack/old release and its exact
            # bounded robot-space reference below. No non-TI geometric stop.
            self._thumb_scale = self._release_smoothed_scale(self._thumb_scale, thumb_scale, min(dt, 0.05))
            current_path_scale = np.minimum(combined_scale, self._adaptive_scale)
            # Soft non-TI contact must not exhaust a 5-degree actuator lead at
            # 180-220 mA. Lead limits come from current/sustained load or TI only.
            lead_scale = np.minimum(current_path_scale, self._thumb_scale)
            lead_budget = config.lead_min_deg + (config.lead_max_deg - config.lead_min_deg) * lead_scale
            current_applied = np.where(worsening & ~actuator_relief, current_path_scale, 1.0)
            geometry_scale = np.minimum(self._soft_contact_scale, self._thumb_scale)
            geometry_applied = np.where(contact_closing & ~contact_relief, geometry_scale, 1.0)
            combined_scale = np.minimum(current_applied, geometry_applied)
            contact_scale = geometry_scale

        target = desired.copy()
        limited = ((combined_scale < 0.999999) if enabled else
                   (worsening & ~actuator_relief & (combined_scale < 0.999999)))
        # While verifying sustained load, don't accumulate unlimited reference
        # lead during the dwell time. No automatic retreat before confirmation.
        limited |= yield_pending & worsening & (np.abs(delta) > np.maximum(0, lead_budget - np.abs(current_error)))
        if np.any(limited):
            allowed_step = self.nominal_step_deg * combined_scale
            if enabled:
                # Scale the requested convergence, not just a fixed maximum step.
                allowed_step = np.minimum(allowed_step, np.abs(delta) * combined_scale)
                # Cap further error accumulation; never command a retreat toward
                # feedback. If already over budget, simply don't advance further.
                loaded = limited & ((load_evidence > 0.01) | yield_pending) & (delta * current_error >= 0)
                allowed_step[loaded] = np.minimum(allowed_step[loaded],
                    np.maximum(0, lead_budget[loaded] - np.abs(current_error[loaded])))
            target[limited] = effective[limited] + np.clip(
                delta[limited], -allowed_step[limited], allowed_step[limited]
            )

        # At a hard threshold, stop adding load rather than generating an
        # automatic retreat. Freezing at the last effective setpoint is less
        # aggressive and avoids the guard itself commanding a sudden motion.
        # If the operator is already asking to reduce the position error, that
        # relief command remains allowed through.
        hard_loaded = (current_abs >= self.hard_ma) | (
            (total_current >= self.total_hard_ma) & (np.abs(current_error) > 1e-3)
        )
        hard_freeze = hard_loaded & ~actuator_relief
        if np.any(hard_freeze):
            target[hard_freeze] = effective[hard_freeze]
            limited |= hard_freeze

        yielding = np.zeros(self.joint_count, dtype=bool)
        if enabled and config.yield_enabled:
            yielding = (self._yield_latched & ~actuator_relief & ~hard_loaded
                        & (np.abs(current_error) > lead_budget + 1e-6))
            yield_step = np.minimum(np.maximum(0, np.abs(current_error) - lead_budget),
                                    min(config.yield_rate_deg_s * min(dt, 0.05), self.nominal_step_deg))
            target[yielding] = effective[yielding] - np.sign(current_error[yielding]) * yield_step[yielding]
            limited |= yielding
            if contact_valid:
                # A yield step may geometrically CLOSE TI even though the
                # operator's requested direction did not. Keep the same bound.
                thumb_motion_closing[:8] |= yielding[:8] & (gradients[0, :8] * (target[:8] - effective[:8]) < 0)

        # Bound total thumb-index CLOSING displacement against the robot-space
        # reference boundary. Do not spend apparent opening motion as a budget:
        # the shaper may realize different joint steps at different rates.
        if contact_valid and np.any(thumb_motion_closing):
            boundary = config.thumb_index_contact_distance_mm * 0.001
            gradient = gradients[0]
            mask = thumb_motion_closing & (gradient * (target - effective) < 0)
            advance = np.clip(target - effective, -self.nominal_step_deg, self.nominal_step_deg)
            original_step = advance.copy()
            projected = float(np.sum(np.minimum(gradient[mask] * advance[mask], 0)))
            budget = max(0.0, float(robot_distances[1, 0]) - boundary)
            ratio = min(1.0, budget / -projected) if projected < 0 else 1.0
            advance[mask] *= ratio
            # Optional exact FK confirmation reuses ContactKinematics. Only
            # bounded backtracking near contact, no optimizer or new model.
            if robot_distance_fn is not None and np.any(mask):
                trial = effective.copy()
                floor = min(boundary, float(robot_distances[1, 0]))
                for _ in range(5):
                    trial[mask] = effective[mask] + advance[mask]
                    try:
                        checked = np.asarray(robot_distance_fn(trial), dtype=float)
                    except (ValueError, ArithmeticError):
                        checked = np.full(len(PAIRS), np.nan)
                    if checked.shape == (len(PAIRS),) and np.all(np.isfinite(checked)) and checked[0] >= floor - 1e-9:
                        break
                    advance[mask] *= 0.5
                else:
                    advance[mask] = 0
            if np.any(np.abs(advance[mask]) < np.abs(original_step[mask]) - 1e-9):
                target[mask] = effective[mask] + advance[mask]
                limited |= mask
                thumb_closing |= mask
                if PAIR_NAMES[0] not in contact_limited_pairs:
                    contact_limited_pairs.append(PAIR_NAMES[0])

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
                       & (self._velocity <= config.stall_velocity_deg_s) & ~actuator_relief)
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
        # Yield is retreat, not progress towards a loading desired command.
        adaptive_tracking_scale[yielding] = 0.0
        yield_delta = np.where(yielding, target - effective, 0.0)
        yielding &= np.abs(yield_delta) > 1e-9
        pair_tracking_scale = np.ones(len(PAIRS))
        if contact_valid:
            for pair in range(len(PAIRS)):
                participating = np.abs(gradients[pair] * delta) > 1e-7
                if np.any(participating):
                    pair_tracking_scale[pair] = float(np.min(adaptive_tracking_scale[participating]))
        contact_limited_pairs = [name for name in contact_limited_pairs
                                 if pair_tracking_scale[PAIR_NAMES.index(name)] < 0.999999]
        robot_contact_active = bool(contact_valid and contact_limited_pairs and np.any(limited))
        if np.any(thumb_closing):
            thumb_contact_scale = min(thumb_contact_scale, float(np.min(adaptive_tracking_scale[thumb_closing])))
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
            "yield_active": bool(np.any(yielding)),
            "yield_joints": [index for index in range(self.joint_count) if yielding[index]],
            "yield_delta_deg": yield_delta,
            "robot_measured_pair_distances_mm": robot_distances[0].copy() * 1000 if contact_valid else np.full(len(PAIRS), np.nan),
            "robot_effective_pair_distances_mm": robot_distances[1].copy() * 1000 if contact_valid else np.full(len(PAIRS), np.nan),
            "robot_desired_pair_distances_mm": robot_distances[2].copy() * 1000 if contact_valid else np.full(len(PAIRS), np.nan),
            "pair_tracking_scale": pair_tracking_scale,
            "hybrid_contact_weights": human_weights.copy() if human_valid else np.full(5, np.nan),
            "pair_distances_m": human_distances.copy() if human_distances_valid else np.full(len(PAIRS), np.nan),
            "pair_distance_rates_m_s": self._distance_rates.copy(),
            "pair_contact_weights": contact_pair_weights,
            "contact_limited_pairs": contact_limited_pairs,
            "limited_fingers": [name for finger, name in enumerate(FINGERS)
                                if np.any(limited[finger * 4:finger * 4 + 4])],
            "stall_duration_s": float(np.max(stall_duration)),
            "robot_contact_limit_active": robot_contact_active,
            "thumb_index_human_contact_weight": float(min(human_weights[0], human_weights[1])) if human_valid else math.nan,
            "thumb_index_measured_tip_distance_mm": float(robot_distances[0, 0] * 1000) if contact_valid else math.nan,
            "thumb_index_effective_tip_distance_mm": float(robot_distances[1, 0] * 1000) if contact_valid else math.nan,
            "thumb_index_desired_tip_distance_mm": float(robot_distances[2, 0] * 1000) if contact_valid else math.nan,
            "thumb_index_contact_tracking_scale": thumb_contact_scale,
        }

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
        )
