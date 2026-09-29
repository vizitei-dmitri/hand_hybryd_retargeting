"""Minimal persistent position servo, in degrees, with no feedback correction.

The bridge owns the periodic timer and latest-target mailbox. This controller
proposes one step; the existing current guard runs before backend submission.
Accepted guarded q_cmd remains persistent; a separate physical output obeys
the current guard's existing lead envelope without reseeding q_cmd.
"""

import time
from dataclasses import dataclass

import numpy as np

from .command_shaper import PositionCommandShaper


@dataclass(frozen=True)
class GuardedServoCommand:
    trajectory_deg: np.ndarray
    lower_deg: np.ndarray
    upper_deg: np.ndarray


class ServoController(PositionCommandShaper):
    """Reuse pose validation/lifecycle, but replace the entire legacy step law.

No legacy filter, deadband, blend or send threshold is used.
Explicit ARM/recovery transitions retain the existing measured-pose reset.
Ordinary targets and feedback never reset the command.
"""

    def __init__(self, lower_limits_deg, upper_limits_deg, *,
                 disabled_positions_deg=None, rate_hz=60.0,
                 max_velocity_deg_s=120.0, max_acceleration_deg_s2=720.0,
                 clock=time.monotonic):
        for name, value in (("rate_hz", rate_hz),
                            ("max_velocity_deg_s", max_velocity_deg_s),
                            ("max_acceleration_deg_s2", max_acceleration_deg_s2)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        super().__init__(lower_limits_deg, upper_limits_deg,
                         disabled_positions_deg=disabled_positions_deg, clock=clock)
        self.rate_hz = float(rate_hz)
        self.max_velocity_deg_s = float(max_velocity_deg_s)
        self.max_acceleration_deg_s2 = float(max_acceleration_deg_s2)
        self._tick_time = None
        self._tracking_since = np.full(self.joint_count, np.nan)
        self._tracking_warned = np.zeros(self.joint_count, dtype=bool)
        self.telemetry = {}

    def reset(self, initial_pose_deg, now=None):
        super().reset(initial_pose_deg, now)
        self._tick_time = None
        self._tracking_since.fill(np.nan)
        self._tracking_warned.fill(False)
        self.telemetry = {}

    def begin_arm_blend(self, now=None, duration_s=None):
        # ARM already initialized the pose and zero velocity; servo ramps up.
        self._cancel_blend()

    def hold(self, now=None):
        # Tracking HOLD freezes both states, without replacing persistent q_cmd
        # with either measured feedback or the separately bounded SDK command.
        self._cancel_blend()
        self._tick_time = None
        self.velocity_deg_s.fill(0.0)
        self._tracking_since.fill(np.nan)
        self._tracking_warned.fill(False)

    def propose(self, target_deg, now=None):
        if not self.is_initialized:
            raise RuntimeError("ServoController must be reset before use")
        target = self._validate_pose(target_deg, "Target")
        timestamp = self._clock() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("Servo timestamp must be finite")
        first_tick = self._tick_time is None
        dt = 1.0 / self.rate_hz if first_tick else timestamp - self._tick_time
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Servo timestamps must increase")
        self._tick_time = timestamp
        # Use measured dt for normal 60 Hz jitter. A scheduler outage is not
        # permission to integrate a large catch-up step; log both dt values.
        limiter_dt = dt if dt <= 0.05 else 1.0 / self.rate_hz
        self._max_step = self.max_velocity_deg_s * limiter_dt
        limited_target = self._apply_disabled(np.clip(
            target, self.lower_limits_deg, self.upper_limits_deg))
        error = limited_target - self.command_pose_deg
        acceleration = self.max_acceleration_deg_s2
        # Discrete conservative stopping speed: reserve this tick's travel as
        # well as v^2/(2*a). This brakes before the final target clamp is needed.
        dv_max = acceleration * limiter_dt
        v_stop = np.sqrt(dv_max ** 2 + 2 * acceleration * np.abs(error)) - dv_max
        desired_velocity = np.sign(error) * np.minimum(self.max_velocity_deg_s, v_stop)
        self._previous_velocity = self.velocity_deg_s.copy()
        velocity = self.velocity_deg_s + np.clip(
            desired_velocity - self.velocity_deg_s,
            -dv_max, dv_max)
        # A reversal goes through zero, never directly from +v to -v.
        reversing = velocity * self.velocity_deg_s < 0
        velocity[reversing] = 0.0
        step = velocity * limiter_dt
        crossed = ((step * error > 0) & (np.abs(step) >= np.abs(error))) | (np.abs(error) < 1e-9)
        step[crossed] = error[crossed]
        proposed = self._apply_disabled(np.clip(
            self.command_pose_deg + step, self.lower_limits_deg, self.upper_limits_deg))
        terminal = crossed | (np.abs(proposed - self.command_pose_deg - step) > 1e-8)
        velocity[terminal] = 0.0
        velocity[list(self.disabled_positions_deg)] = 0.0
        self._proposed_velocity = velocity
        braking = error * (proposed - self.command_pose_deg) < -1e-8
        self.telemetry = {
            "source_monotonic_s": timestamp,
            "q_target": target.tolist(),
            "target": target.tolist(),
            "limited_target": limited_target.tolist(),
            "previous_q_cmd": self.command_pose_deg.tolist(),
            "raw_servo_step": (proposed - self.command_pose_deg).tolist(),
            "q_proposed": proposed.tolist(),
            "dt": dt,
            "limiter_dt": limiter_dt,
            "actual_rate_hz": 1.0 / dt,
            "first_tick": first_tick,
            "desired_velocity_deg_s": desired_velocity.tolist(),
            "trajectory_velocity_deg_s": velocity.tolist(),
            "commanded_velocity_deg_s": velocity.tolist(),
            "acceleration_deg_s2": ((velocity - self._previous_velocity) / limiter_dt).tolist(),
            "acceleration_limited": (np.abs(desired_velocity - self._previous_velocity) > acceleration * limiter_dt + 1e-8).tolist(),
            "max_acceleration_deg_s2": acceleration,
            "trajectory_braking": braking.tolist(),
            "target_clamped": terminal.tolist(),
        }
        return proposed

    def constrain_guarded_output(self, guarded_deg):
        """Bound even safety relief steps, preserving the guard's direction.

        This bounds persistent q_cmd. The physical lead envelope is applied
        separately after this step, so a shrinking budget cannot be undone.
        """
        guarded = self._validate_pose(guarded_deg, "Guarded command")
        guarded = np.clip(guarded, self.lower_limits_deg, self.upper_limits_deg)
        return self._apply_disabled(self.command_pose_deg + np.clip(
            guarded - self.command_pose_deg, -self._max_step, self._max_step))

    def physical_output(self, command, envelope=None):
        output = self.last_sent_pose_deg + np.clip(
            command - self.last_sent_pose_deg, -self._max_step, self._max_step)
        reasons = np.where(np.abs(output - command) > 1e-8, "PHYSICAL_RATE_LIMIT", "NONE").astype(object)
        if envelope is not None:
            # The existing lead envelope has priority over speed during load
            # relief / sudden feedback changes. Never undo it by rate-clipping.
            bounded = np.clip(output, envelope.lower_deg, envelope.upper_deg)
            lead_limited = ((command < envelope.lower_deg) | (command > envelope.upper_deg)
                            | (np.abs(bounded - output) > 1e-8))
            reasons[lead_limited] = "LEAD_BUDGET_YIELD_OR_HOLD"
            output = bounded
        self.telemetry["physical_limit_reason"] = reasons.tolist()
        return self._apply_disabled(np.clip(output, self.lower_limits_deg, self.upper_limits_deg))

    def accept_output(self, output_deg, *, physical_deg=None):
        before = self.command_pose_deg.copy()
        previous_physical = self.last_sent_pose_deg.copy()
        super().accept_output(output_deg if physical_deg is None else physical_deg)
        self.command_pose_deg = self._apply_disabled(np.clip(
            output_deg, self.lower_limits_deg, self.upper_limits_deg))
        dt = self.telemetry["limiter_dt"]
        proposal = np.asarray(self.telemetry["q_proposed"])
        trajectory_limited = np.abs(self.command_pose_deg - proposal) > 1e-8
        physical_limited = np.abs(self.last_sent_pose_deg - self.command_pose_deg) > 1e-8
        # Safety can stop/yield immediately. Carry the accepted trajectory
        # velocity forward, never a hidden pre-contact full-speed velocity.
        self.velocity_deg_s = self._proposed_velocity.copy()
        self.velocity_deg_s[trajectory_limited] = (self.command_pose_deg - before)[trajectory_limited] / dt
        self.velocity_deg_s[physical_limited] = 0.0
        self.velocity_deg_s = np.clip(self.velocity_deg_s, -self.max_velocity_deg_s, self.max_velocity_deg_s)
        braking = np.asarray(self.telemetry["trajectory_braking"])
        planned_braking = braking & (np.abs(self.command_pose_deg - before) <= np.abs(proposal - before) + 1e-8)
        self.telemetry.update(
            commanded_velocity_deg_s=self.velocity_deg_s.tolist(),
            acceleration_deg_s2=((self.velocity_deg_s - self._previous_velocity) / dt).tolist(),
            safety_velocity_override=(trajectory_limited | physical_limited).tolist(),
            previous_low_level_submitted=previous_physical.tolist(),
            low_level_submitted=self.last_sent_pose_deg.tolist(),
            physical_step_deg=(self.last_sent_pose_deg - previous_physical).tolist(),
            q_cmd=self.command_pose_deg.tolist(),
            new_q_cmd=self.command_pose_deg.tolist(),
            direction_violation=(((np.asarray(self.telemetry["q_target"]) - before)
                                  * (self.command_pose_deg - before) < -1e-8)
                                 & ~planned_braking).tolist(),
            distance_before=np.abs(np.asarray(self.telemetry["q_target"]) - before).tolist(),
            distance_after=np.abs(np.asarray(self.telemetry["q_target"]) - self.command_pose_deg).tolist(),
            max_step_deg=float(np.max(np.abs(self.command_pose_deg - before))),
        )

    def observe(self, measured_deg):
        """Log raw feedback and warn once per sustained per-joint episode."""
        measured = self._validate_pose(measured_deg, "Measured pose")
        error = np.abs(self.command_pose_deg - measured)
        high = error > 10.0
        self._tracking_since[~high] = np.nan
        self._tracking_warned[~high] = False
        newly_high = high & np.isnan(self._tracking_since)
        self._tracking_since[newly_high] = self._tick_time
        warning = high & ~self._tracking_warned & (
            self._tick_time - self._tracking_since >= 0.300 - 1e-9)
        self._tracking_warned[warning] = True
        self.telemetry.update(q_measured=measured.tolist(), tracking_error=error.tolist())
        return np.flatnonzero(warning).tolist()

    def step(self, *args, **kwargs):
        raise RuntimeError("Servo output requires propose -> current guard -> accept_output")
