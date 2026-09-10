"""Read-only debug fields for the unmodified 2026-09-08 current guard.

Nothing in this module supplies a target, scale or fault to motion control.
The experimental controller lives separately in experimental_current_guard.py.
"""

import numpy as np

from dg5f_teleop.contact_signals import FINGERS, PAIR_NAMES
from .experimental_current_guard import RobustCurrentTrend


def guard_snapshot(guard, decision, effective, desired):
    """Copy guard state and describe its decision, without changing either."""
    count = guard.joint_count
    scale = np.ones(count)
    delta = desired - effective
    moving = decision.limited_mask & (np.abs(delta) > 1e-8)
    scale[moving] = np.minimum(1.0,
        np.abs(decision.target_deg[moving] - effective[moving]) /
        np.minimum(np.abs(delta[moving]), guard.nominal_step_deg))
    scale[decision.limited_mask & ~moving] = 0.0
    return {
        "compliance_enabled": False, "compliance_active": False,
        "global_current_scale": float(guard._global_scale),
        "joint_current_scale": guard._joint_scale.copy(),
        "total_current_ma": decision.total_current_ma,
        "joint_tracking_scale": scale,
        "joint_slope_scale": np.ones(count), "global_slope_scale": 1.0,
        "joint_contact_scale": np.ones(count),
        "joint_lead_budget_deg": np.full(count, np.inf),
        "yield_active": False, "yield_joints": [],
        "yield_delta_deg": np.zeros(count), "stall_duration_s": 0.0,
        "robot_contact_limit_active": False, "contact_limited_pairs": [],
        "pair_tracking_scale": np.ones(len(PAIR_NAMES)),
        "pair_contact_weights": np.zeros(len(PAIR_NAMES)),
        "pair_distance_rates_m_s": np.zeros(len(PAIR_NAMES)),
        "contact_signal_valid": False,
        "robot_measured_pair_distances_mm": np.full(len(PAIR_NAMES), np.nan),
        "robot_effective_pair_distances_mm": np.full(len(PAIR_NAMES), np.nan),
        "robot_desired_pair_distances_mm": np.full(len(PAIR_NAMES), np.nan),
        "thumb_index_contact_tracking_scale": 1.0,
        "thumb_index_measured_tip_distance_mm": np.nan,
        "thumb_index_effective_tip_distance_mm": np.nan,
        "thumb_index_desired_tip_distance_mm": np.nan,
        "limited_fingers": [name for finger, name in enumerate(FINGERS)
                            if np.any(decision.limited_mask[finger * 4:finger * 4 + 4])],
    }


class CurrentTrendDiagnostics:
    """Observe dI/dt at diagnostic publication rate, outside the command timer."""

    def __init__(self, count):
        self.count = count
        self.last_time = None
        self.trend = RobustCurrentTrend(count, 0.05)

    def observe(self, currents, now):
        values = np.asarray(currents, dtype=float)
        if values.shape != (self.count,) or not np.all(np.isfinite(values)):
            self.last_time = None
            self.trend = RobustCurrentTrend(self.count, 0.05)
            return {"joint_current_slope_ma_s": np.full(self.count, np.nan),
                    "total_current_slope_ma_s": np.nan}
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        if dt > 0.2:
            self.trend = RobustCurrentTrend(self.count, 0.05)
        self.last_time = now
        slope = self.trend.update(np.abs(values), now, min(dt, 0.05))
        return {"joint_current_slope_ma_s": slope,
                "total_current_slope_ma_s": float(np.sum(slope))}
