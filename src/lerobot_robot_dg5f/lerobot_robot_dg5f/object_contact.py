"""Physical finger contact inferred from servo load and windowed position progress.

No MANO proximity, SDK velocity, hardware access, or force-control model.
Targets from this layer must still pass the existing emergency guard/shaper.
"""

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .constants import FINGER_FLEXION_JOINTS, JOINT_NAMES


@dataclass(frozen=True)
class ObjectContactConfig:
    enabled: bool = True
    current_ma: float = 250.0
    error_deg: float = 3.0
    window_s: float = 0.12
    hold_s: float = 0.08
    progress_ratio: float = 0.25
    min_motion_deg: float = 0.10
    intent_deadband_deg: float = 0.30
    evidence_grace_s: float = 0.04
    fast_current_ma: float = 350.0
    fast_error_deg: float = 4.0
    fast_hold_s: float = 0.01
    fast_slope_ma_s: float = 800.0
    closing_gain: float = 0.05
    preload_deg: float = 1.5
    min_preload_deg: float = 0.5
    preload_high_current_ma: float = 400.0
    yield_rate_deg_s: float = 15.0
    release_open_deg: float = 3.0
    release_current_ma: float = 120.0
    release_error_deg: float = 1.5
    release_hold_s: float = 0.15
    resume_rate_deg_s: float = 15.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "enabled":
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"object_contact_{name} must be finite and positive")
        if not 0 < self.progress_ratio < 1 or self.closing_gain > 1:
            raise ValueError("Object contact progress ratio/gain must be at most 1")
        if self.fast_current_ma < self.current_ma or self.fast_error_deg < self.error_deg:
            raise ValueError("Fast contact thresholds must not be below ordinary thresholds")
        if self.fast_hold_s > self.hold_s or self.min_preload_deg > self.preload_deg:
            raise ValueError("Invalid fast contact hold or preload range")
        if self.preload_high_current_ma <= self.current_ma:
            raise ValueError("Preload high current must exceed contact current")
        if self.release_current_ma >= self.current_ma or self.release_error_deg >= self.error_deg:
            raise ValueError("Object contact release needs current/error hysteresis")
        if self.release_open_deg <= self.intent_deadband_deg:
            raise ValueError("Release opening must exceed the intent deadband")


@dataclass
class FingerContact:
    state: str = "FREE"
    reason: str = "NONE"
    pending_since: float | None = None
    last_evidence_time: float | None = None
    quiet_since: float | None = None
    free_progress_seen: bool = False
    measured: np.ndarray | None = None
    effective: np.ndarray | None = None
    desired: np.ndarray | None = None
    peak_desired: np.ndarray | None = None
    opening_desired: np.ndarray | None = None
    opening_effective: np.ndarray | None = None
    closing_offset: np.ndarray | None = None
    resume_offset: np.ndarray | None = None
    released_at: float = -math.inf


@dataclass
class ObjectContactDecision:
    target_deg: np.ndarray
    limited_mask: np.ndarray
    diagnostics: dict
    transitions: list[dict] = field(default_factory=list)


class PerFingerObjectContact:
    def __init__(self, config: ObjectContactConfig):
        self.config = config
        self.chains = {
            name: np.array([JOINT_NAMES.index(joint) for joint in joints])
            for name, joints in FINGER_FLEXION_JOINTS.items()
        }
        self.reset()

    def reset(self):
        self.fingers = {name: FingerContact() for name in self.chains}
        self.history = deque()
        self.positions = deque(maxlen=3)
        self.last_time = None
        self.last_desired = None

    @staticmethod
    def _transition(finger, name, state, reason, events):
        if finger.state == state:
            return
        finger.state, finger.reason = state, reason
        event = {"CONTACT_PENDING": "OBJECT_CONTACT_PENDING",
                 "CONTACT_HOLD": "OBJECT_CONTACT_LATCHED",
                 "FREE": "OBJECT_CONTACT_RELEASED"}[state]
        events.append(dict(event=event, finger=name, state=state, reason=reason))

    def update(self, *, desired_deg, effective_deg, measured_deg, current_ma,
               slope_ma_s, soft_target_deg, now):
        config = self.config
        desired, effective, measured = map(np.asarray, (desired_deg, effective_deg, measured_deg))
        current, slope = np.asarray(current_ma), np.asarray(slope_ma_s)
        target = np.asarray(soft_target_deg).copy()
        limited = np.zeros(len(JOINT_NAMES), dtype=bool)
        events = []
        elapsed = 0.0 if self.last_time is None else now - self.last_time
        if elapsed < 0 or elapsed > max(0.2, 2 * config.window_s):
            # Do not confirm contact/release across a tracking or telemetry gap.
            # Preserve confirmed anchors until fresh evidence or explicit reset.
            self.history.clear()
            self.positions.clear()
            for name, finger in self.fingers.items():
                finger.pending_since = finger.quiet_since = None
                finger.last_evidence_time = None
                finger.free_progress_seen = False
                if finger.state == "CONTACT_PENDING":
                    self._transition(finger, name, "FREE", "SAMPLE_GAP", events)
            self.last_desired = None
            self.last_time = None
        dt = min(max(elapsed, 0.0), 0.05)
        human_step = np.zeros(20) if self.last_desired is None else desired - self.last_desired
        if self.last_time is None or now > self.last_time:
            self.positions.append(measured.copy())
            filtered = np.median(np.stack(self.positions), axis=0)
            self.history.append((now, desired.copy(), effective.copy(), filtered))
        self.last_time, self.last_desired = now, desired.copy()
        while len(self.history) > 2 and self.history[1][0] <= now - config.window_s:
            self.history.popleft()
        span = now - self.history[0][0]
        ready = len(self.history) >= 4 and span >= config.window_s - 1e-9
        motion = np.zeros((3, 20))
        if ready:
            times = np.array([sample[0] for sample in self.history])
            times -= times.mean()
            poses = np.array([sample[1:] for sample in self.history])
            motion = np.tensordot(times, poses, axes=(0, 0)) / (times @ times) * span
        requested = np.maximum(np.maximum(motion[0], motion[1]), 0)
        # A pre-existing guard may have frozen effective. An outstanding close
        # still requests progress; do not let a huge human error inflate the ratio.
        stalled_request = np.minimum(np.maximum(desired - measured, 0), config.error_deg)
        requested = np.where(requested >= config.min_motion_deg, requested, stalled_request)
        ratios = np.divide(np.maximum(motion[2], 0), requested,
                           out=np.ones(20), where=requested >= config.min_motion_deg)
        errors = np.maximum(effective - measured, 0)
        # Compare two overlapping median-of-three human poses. At 50 Hz their
        # centres span 40 ms; an isolated retarget step cannot cancel pending.
        recent_human = np.array([sample[1] for sample in list(self.history)[-5:]])
        intent_motion = np.zeros(20)
        if len(recent_human) >= 4:
            intent_motion = np.median(recent_human[-3:], axis=0) - np.median(recent_human[:3], axis=0)
        diag = {key: [] for key in (
            "object_contact_state", "object_contact_active", "object_contact_reason",
            "finger_current_ma", "finger_max_joint_current_ma", "finger_tracking_error_deg",
            "finger_progress_ratio", "post_contact_gain", "contact_preload_deg")}
        anchors = {f"contact_anchor_{key}": np.full(20, np.nan)
                   for key in ("measured", "effective", "desired")}

        for number, (name, joints) in enumerate(self.chains.items()):
            finger = self.fingers[name]
            joint_current = current[joints]
            peak_current = float(np.max(joint_current))
            opening_now = intent_motion[joints] < -config.intent_deadband_deg
            # An already outward target is also unambiguous relief, even after
            # the operator stops moving and windowed intent returns to zero.
            clear_opening = bool(np.any(opening_now)
                                 or np.any(desired[joints] < measured[joints] - config.intent_deadband_deg))
            closing = ((desired[joints] > measured[joints] + config.intent_deadband_deg)
                       & (desired[joints] >= effective[joints] - config.intent_deadband_deg)
                       & (motion[0, joints] >= -config.intent_deadband_deg))
            eligible = ready and not clear_opening and now - finger.released_at >= config.release_hold_s
            poor_progress = ratios[joints] < config.progress_ratio
            evidence = (eligible & closing
                        & (joint_current >= config.current_ma)
                        & (errors[joints] >= config.error_deg)
                        & poor_progress)
            fast_load = ((joint_current >= config.fast_current_ma)
                         & (errors[joints] >= config.fast_error_deg))
            stalled = np.abs(motion[2, joints]) < config.min_motion_deg
            # High physical load does not depend on ordinary desired/effective
            # alignment or one flickering requested-motion flag. Progress and
            # opening intent remain mandatory; slope alone can never latch.
            fast_load_evidence = eligible & fast_load & (poor_progress | stalled)
            fast_slope_evidence = (eligible & poor_progress
                                   & (joint_current >= config.current_ma)
                                   & (errors[joints] >= config.error_deg)
                                   & (slope[joints] >= config.fast_slope_ma_s))
            fast = bool(np.any(fast_load_evidence | fast_slope_evidence))
            physical_evidence = evidence | fast_load_evidence | fast_slope_evidence
            reason = "HIGH_LOAD_LOW_PROGRESS" if np.any(fast_load_evidence) else (
                "RISING_CURRENT_LOW_PROGRESS" if fast else "LOW_PROGRESS")

            if config.enabled and finger.state != "CONTACT_HOLD":
                expired = (finger.last_evidence_time is not None
                           and now - finger.last_evidence_time > config.evidence_grace_s + 1e-9)
                if clear_opening or expired:
                    finger.pending_since = finger.last_evidence_time = None
                    self._transition(finger, name, "FREE",
                                     "OPERATOR_OPENING" if clear_opening else "EVIDENCE_CLEARED", events)
                if np.any(physical_evidence):
                    finger.last_evidence_time = now
                    if finger.pending_since is None:
                        finger.pending_since = now
                    self._transition(finger, name, "CONTACT_PENDING", reason, events)
                    hold = config.fast_hold_s if fast else config.hold_s
                    if now - finger.pending_since >= hold - 1e-9:
                        self._transition(finger, name, "CONTACT_HOLD", reason, events)
                        finger.measured = measured[joints].copy()
                        finger.effective = effective[joints].copy()
                        finger.desired = desired[joints].copy()
                        finger.peak_desired = desired[joints].copy()
                        finger.opening_desired = finger.opening_effective = None
                        finger.closing_offset = np.zeros(len(joints))
                        finger.resume_offset = None
                # During grace, retain accumulated hold but require fresh
                # physical evidence before confirming CONTACT_HOLD.

            preload = config.preload_deg - (config.preload_deg - config.min_preload_deg) * np.clip(
                (peak_current - config.current_ma) / (config.preload_high_current_ma - config.current_ma), 0, 1)
            gain = 1.0
            if config.enabled and finger.state == "CONTACT_HOLD":
                base = np.minimum(finger.effective, finger.measured + preload)
                if (finger.opening_desired is not None and np.any(human_step[joints] > 1e-6)
                        and not np.any(human_step[joints] < -1e-6)):
                    # A close following a partial opening gets the small gain
                    # immediately, without recapturing the original contact anchors.
                    previous_human = desired[joints] - human_step[joints]
                    finger.closing_offset = effective[joints] - (
                        base + config.closing_gain * (previous_human - finger.desired))
                    finger.opening_desired = finger.opening_effective = None
                    finger.peak_desired = previous_human.copy()
                finger.peak_desired = np.maximum(finger.peak_desired, desired[joints])
                if (finger.opening_desired is None
                        and np.max(finger.peak_desired - desired[joints]) >= config.intent_deadband_deg):
                    finger.opening_desired = finger.peak_desired.copy()
                    finger.opening_effective = effective[joints].copy()
                mapped = base + config.closing_gain * (desired[joints] - finger.desired) + finger.closing_offset
                # The anchor prevents feedback drift from ratcheting closure;
                # live measured pose caps residual servo preload if the object moves.
                mapped = np.minimum(mapped, measured[joints] + preload)
                # Automatic yield only removes excessive positive servo preload;
                # never pull past measured or invent a retreat to an old anchor.
                mapped = np.maximum(mapped, np.minimum(effective[joints], measured[joints] + preload))
                bounded = np.maximum(mapped, effective[joints] - config.yield_rate_deg_s * dt)
                gain = config.closing_gain
                if finger.opening_desired is not None:
                    opening_delta = np.minimum(desired[joints] - finger.opening_desired, 0)
                    opening_joints = opening_delta < 0
                    bounded[opening_joints] = np.minimum(bounded[opening_joints],
                        finger.opening_effective[opening_joints] + opening_delta[opening_joints])
                    gain = 1.0
                target[joints] = np.minimum(target[joints], bounded)
                limited[joints] = True
                quiet = (peak_current < config.release_current_ma
                         and np.max(np.abs(effective[joints] - measured[joints])) < config.release_error_deg)
                if quiet:
                    if finger.quiet_since is None:
                        finger.quiet_since = now
                    finger.free_progress_seen |= bool(ready and np.any(motion[2, joints] >= config.min_motion_deg))
                else:
                    finger.quiet_since = None
                    finger.free_progress_seen = False
                opened = (finger.opening_desired is not None
                          and np.max(finger.opening_desired - desired[joints]) >= config.release_open_deg)
                unloaded = (finger.quiet_since is not None and finger.free_progress_seen
                            and now - finger.quiet_since >= config.release_hold_s)
                if opened or unloaded:
                    self._transition(finger, name, "FREE", "OPERATOR_OPENING" if opened else "LOAD_RELEASED", events)
                    # Keep this tick's bounded reference; remove its offset slowly
                    # on subsequent FREE ticks instead of jumping to human desired.
                    finger.resume_offset = np.maximum(desired[joints] - target[joints], 0)
                    finger.released_at = now
                    finger.pending_since = finger.quiet_since = None
                    finger.last_evidence_time = None
                    finger.free_progress_seen = False
            elif config.enabled and finger.resume_offset is not None:
                if (not np.any(human_step[joints] < -1e-6)
                        and (finger.reason == "LOAD_RELEASED" or np.any(human_step[joints] > 1e-6))):
                    finger.resume_offset = np.maximum(0, finger.resume_offset - config.resume_rate_deg_s * dt)
                target[joints] = np.minimum(target[joints], desired[joints] - finger.resume_offset)
                target[joints] = np.minimum(target[joints], effective[joints] + config.resume_rate_deg_s * dt)
                limited[joints] = True
                if np.max(finger.resume_offset) <= 1e-6 and np.all(desired[joints] <= target[joints] + 1e-6):
                    finger.resume_offset = None

            # Physical current diagnostics include lateral/base motors; the latch
            # and preload decisions above use only this finger's flexion chain.
            all_current = current[number * 4:number * 4 + 4]
            diag["object_contact_state"].append(finger.state)
            diag["object_contact_active"].append(finger.state == "CONTACT_HOLD")
            diag["object_contact_reason"].append(finger.reason)
            diag["finger_current_ma"].append(float(np.sum(all_current)))
            diag["finger_max_joint_current_ma"].append(float(np.max(all_current)))
            diag["finger_tracking_error_deg"].append(float(np.max(errors[joints])))
            loaded = physical_evidence if np.any(physical_evidence) else np.ones(len(joints), dtype=bool)
            diag["finger_progress_ratio"].append(float(np.min(ratios[joints][loaded])) if ready else math.nan)
            diag["post_contact_gain"].append(gain)
            diag["contact_preload_deg"].append(float(preload) if finger.state == "CONTACT_HOLD" else 0.0)
            for key in ("measured", "effective", "desired"):
                value = getattr(finger, key)
                if value is not None:
                    anchors[f"contact_anchor_{key}"][joints] = value

        diag.update(anchors)
        diag["object_contact_limited_joints"] = [JOINT_NAMES[j] for j in np.flatnonzero(limited)]
        for event in events:
            number = list(self.chains).index(event["finger"])
            joints = self.chains[event["finger"]]
            for key in ("finger_current_ma", "finger_max_joint_current_ma",
                        "finger_tracking_error_deg", "finger_progress_ratio", "contact_preload_deg"):
                event[key] = diag[key][number]
            if event["state"] == "CONTACT_HOLD":
                event["flexion_joints"] = [JOINT_NAMES[j] for j in joints]
                for key, values in anchors.items():
                    event[key] = values[joints].tolist()
        return ObjectContactDecision(target, limited, diag, events)
