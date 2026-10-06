"""Residual RL over a frozen base policy with operator takeover and the RLIF reward."""

from dataclasses import dataclass
from typing import Callable, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.teleoperators.utils import TeleopEvents

from .pose_math import apply_offset, pose_difference, rotate_offset
from .synergies import HandSynergies


@dataclass(frozen=True)
class ResidualScale:
    """beta of the plan, one per unit: metres, radians, synergy units (degrees along a component)."""

    lin_m: float = 0.02
    ang_rad: float = 0.10
    syn: float = 10.0

    def wrist(self) -> np.ndarray:
        return np.array([self.lin_m] * 3 + [self.ang_rad] * 3)


class OperatorSource(Protocol):
    def get_action(self) -> dict:
        """{"wrist_pose": (6,) in the Quest frame, "hand_deg": (20,) retargeted DG5F joints}."""

    def get_teleop_events(self) -> dict:
        """Keys from lerobot.teleoperators.utils.TeleopEvents."""


def apply_residual(base, residual, synergies: HandSynergies, scale: ResidualScale) -> np.ndarray:
    """a = a_base (+) scale * clip(residual): wrist offset in the base frame, hand via synergies."""
    residual = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
    tcp = apply_offset(base[:6], residual[:6] * scale.wrist())
    hand = base[6:] + synergies.to_joint_offset(residual[6:] * scale.syn)
    return np.concatenate([tcp, hand])


def residual_label(executed, base, synergies: HandSynergies, scale: ResidualScale):
    """Normalised residual that would turn base into executed; clipped, plus a saturation flag."""
    raw = np.concatenate([
        pose_difference(executed[:6], base[:6]),
        synergies.project_offset(np.asarray(executed[6:]) - np.asarray(base[6:])),
    ])
    scales = np.concatenate([scale.wrist(), np.full(synergies.k, scale.syn)])
    normalised = np.divide(raw, scales, out=np.zeros_like(raw), where=scales > 0)
    saturated = bool(np.any(np.abs(normalised) > 1.0) or np.any((scales == 0) & (np.abs(raw) > 1e-9)))
    return np.clip(normalised, -1.0, 1.0), saturated


def follow_operator(operator_pose, operator_anchor, robot_anchor, quest_to_base) -> np.ndarray:
    """Relative mode: move the robot from its anchor by the operator's motion since takeover."""
    delta_quest = pose_difference(operator_pose, operator_anchor)
    return apply_offset(robot_anchor, rotate_offset(delta_quest, quest_to_base))


class ResidualInterventionEnv(gym.Wrapper):
    """Action: residual in [-1, 1]^(6+k). Reward: 1[success] - 1[takeover started this step]."""

    def __init__(
        self,
        env: gym.Env,
        base_policy: Callable[[dict], np.ndarray],
        synergies: HandSynergies,
        operator: OperatorSource,
        scale: ResidualScale = ResidualScale(),
        quest_to_base=np.eye(3),
        blend_s: float = 0.4,
    ):
        super().__init__(env)
        self.base_policy = base_policy
        self.synergies = synergies
        self.operator = operator
        self.scale = scale
        self.quest_to_base = np.asarray(quest_to_base, dtype=np.float64)
        self.blend_steps = max(1, round(blend_s / env.unwrapped.period_s))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6 + synergies.k,), dtype=np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._observation = observation
        self._intervening = False
        self._last_executed = self.env.unwrapped.robot.last_command.copy()
        self._operator_anchor = None
        self._robot_anchor = None
        self._blend_from = None
        self._blend_step = 0
        return observation, info

    def step(self, residual):
        events = self.operator.get_teleop_events()
        intervening = bool(events.get(TeleopEvents.IS_INTERVENTION, False))
        started = intervening and not self._intervening
        released = self._intervening and not intervening
        # The base policy runs during takeover too: labels need a_base.
        base = np.asarray(self.base_policy(self._observation), dtype=np.float64)

        if started or released:
            self._blend_from = self._last_executed[6:].copy()
            self._blend_step = 0

        if intervening:
            operator = self.operator.get_action()
            operator_pose = np.asarray(operator["wrist_pose"], dtype=np.float64)
            if started:
                self._operator_anchor = operator_pose.copy()
                self._robot_anchor = self._last_executed[:6].copy()
            tcp = follow_operator(operator_pose, self._operator_anchor, self._robot_anchor,
                                  self.quest_to_base)
            hand_goal = np.asarray(operator["hand_deg"], dtype=np.float64)
            applied = None
        else:
            applied = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
            proposal = apply_residual(base, applied, self.synergies, self.scale)
            tcp, hand_goal = proposal[:6], proposal[6:]

        hand = self._blend(hand_goal)
        observation, reward, terminated, truncated, info = self.env.step(np.concatenate([tcp, hand]))
        executed = np.asarray(info["sent_action"], dtype=np.float64)

        if intervening:
            label, saturated = residual_label(executed, base, self.synergies, self.scale)
        else:
            label, saturated = applied, False

        success = bool(events.get(TeleopEvents.SUCCESS, False))
        reward = float(reward) + float(success) - float(started)
        terminated = bool(terminated or success or events.get(TeleopEvents.TERMINATE_EPISODE, False))
        info.update(
            base_action=base,
            executed_action=executed,
            residual_label=label,
            label_saturated=saturated,
            is_intervention=intervening,
            intervention_started=started,
            success=success,
        )
        self._observation = observation
        self._intervening = intervening
        self._last_executed = executed
        return observation, reward, terminated, truncated, info

    def _blend(self, goal):
        """Linear finger transition over blend_s after takeover and after release."""
        if self._blend_from is None:
            return goal
        self._blend_step += 1
        alpha = min(1.0, self._blend_step / self.blend_steps)
        blended = (1.0 - alpha) * self._blend_from + alpha * goal
        if alpha >= 1.0:
            self._blend_from = None
        return blended
