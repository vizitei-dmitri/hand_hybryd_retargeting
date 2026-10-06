"""Gymnasium environment around the UR10e + DG5F LeRobot robot."""

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .pose_math import rotvec_to_6d
from .ur10_dg5f import ACTION_KEYS, ARM_KEYS, HAND_KEYS, TCP_KEYS, Ur10Dg5f, vector_to_action


# TCP position (3) + 6D orientation (6) + arm joints (6) + hand joints (20).
STATE_DIM = 3 + 6 + len(ARM_KEYS) + len(HAND_KEYS)


class Ur10Dg5fEnv(gym.Env):
    """Absolute 26-value actions: TCP pose (6) and DG5F joints in degrees (20).

    Limits live in the robot, not in the action space, so replay and any policy
    go through the same safety path. Reward is zero here; the residual wrapper adds it.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot: Ur10Dg5f,
        fps: float = 10.0,
        episode_s: float = 30.0,
        wait_for_scene_reset: Callable[[], None] | None = None,
    ):
        super().__init__()
        self.robot = robot
        self.period_s = 1.0 / fps
        self.max_steps = round(episode_s * fps)
        self.wait_for_scene_reset = wait_for_scene_reset
        if not robot.is_connected:
            robot.connect()
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(len(ACTION_KEYS),), dtype=np.float32)
        observation_spaces = {
            OBS_STATE: spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
        }
        for name, shape in robot.observation_features.items():
            if isinstance(shape, tuple):
                observation_spaces[f"{OBS_IMAGES}.{name}"] = spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        self.observation_space = spaces.Dict(observation_spaces)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.go_home()
        if self.wait_for_scene_reset is not None:
            self.wait_for_scene_reset()  # real cell: operator puts the object back, presses a key
        self._step_count = 0
        return self._observation(), {}

    def step(self, action):
        sent = self.robot.send_action(vector_to_action(np.asarray(action, dtype=np.float64)))
        self.robot.wait_next_period(self.period_s)
        observation = self._observation()
        self._step_count += 1
        protective_stop = self.robot.protective_stop()
        info = {
            "sent_action": np.asarray([sent[key] for key in ACTION_KEYS], dtype=np.float64),
            "protective_stop": protective_stop,
        }
        return observation, 0.0, protective_stop, self._step_count >= self.max_steps, info

    def close(self):
        self.robot.disconnect()

    def _observation(self) -> dict[str, np.ndarray]:
        raw = self.robot.get_observation()
        tcp = np.asarray([raw[key] for key in TCP_KEYS])
        state = np.concatenate([
            tcp[:3],
            rotvec_to_6d(tcp[3:]),
            [raw[key] for key in ARM_KEYS],
            [raw[key] for key in HAND_KEYS],
        ]).astype(np.float32)
        observation = {OBS_STATE: state}
        for key, value in raw.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                observation[f"{OBS_IMAGES}.{key}"] = value
        return observation
