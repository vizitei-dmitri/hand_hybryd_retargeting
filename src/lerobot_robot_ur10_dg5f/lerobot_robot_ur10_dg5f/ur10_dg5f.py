"""LeRobot ``Robot`` for a UR10e arm carrying a DG5F hand."""

from functools import cached_property

import numpy as np
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot_robot_dg5f.constants import JOINT_NAMES

from .backend import ARM_JOINTS, make_backend
from .config_ur10_dg5f import Ur10Dg5fConfig
from .safety import clip_hand, clip_workspace, limit_step


TCP_KEYS = ("tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz")
ARM_KEYS = tuple(f"{joint}.pos" for joint in ARM_JOINTS)
HAND_KEYS = tuple(f"{joint}.pos" for joint in JOINT_NAMES)
ACTION_KEYS = TCP_KEYS + HAND_KEYS


def action_to_vector(action: RobotAction) -> np.ndarray:
    missing = [key for key in ACTION_KEYS if key not in action]
    if missing:
        raise ValueError(f"Action is missing keys: {missing}")
    return np.asarray([float(action[key]) for key in ACTION_KEYS], dtype=np.float64)


def vector_to_action(vector) -> RobotAction:
    return {key: float(value) for key, value in zip(ACTION_KEYS, vector, strict=True)}


class Ur10Dg5f(Robot):
    """UR10e TCP pose (UR base frame, metres + rotation vector) and 20 DG5F joints (degrees)."""

    config_class = Ur10Dg5fConfig
    name = "ur10_dg5f"

    def __init__(self, config: Ur10Dg5fConfig):
        super().__init__(config)
        self.config = config
        self.backend = make_backend(config)
        self._last_tcp = None
        self._last_hand = None

    @cached_property
    def _camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        if self.config.backend == "mujoco":
            height, width = self.config.sim_image_hw
            return {name: (height, width, 3) for name in self.config.sim_cameras}
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict:
        features = {key: float for key in TCP_KEYS + ARM_KEYS + HAND_KEYS}
        features.update(self._camera_shapes)
        return features

    @cached_property
    def action_features(self) -> dict:
        return {key: float for key in ACTION_KEYS}

    @property
    def is_connected(self) -> bool:
        return self.backend.is_connected

    @property
    def last_command(self) -> np.ndarray:
        """Last target actually sent: 6 TCP values followed by 20 hand degrees."""
        return np.concatenate([self._last_tcp, self._last_hand])

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            return
        self.backend.connect()
        self._seed_from_measurement()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """UR and Tesollo report calibrated values; no LeRobot calibration needed."""

    def configure(self) -> None:
        """Nothing to configure beyond the backend connection."""

    def get_observation(self) -> RobotObservation:
        self._require_connected()
        state = self.backend.read()
        observation = dict(zip(TCP_KEYS, map(float, state.tcp)))
        observation.update(zip(ARM_KEYS, map(float, state.arm_q)))
        observation.update(zip(HAND_KEYS, map(float, state.hand_deg)))
        observation.update(self.backend.images())
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        """Limit and send a target; return what was actually sent (record this, not the request)."""
        self._require_connected()
        requested = action_to_vector(action)
        if not np.all(np.isfinite(requested)):
            raise ValueError("UR10e + DG5F action contains NaN or infinity")
        # Box first, then step: the segment between two points of a box stays inside it,
        # so both limits hold as long as the previous target was inside the box.
        tcp = clip_workspace(requested[:6], self.config.workspace_lo, self.config.workspace_hi)
        tcp = limit_step(self._last_tcp, tcp, self.config.max_step_m, self.config.max_step_rad)
        hand = clip_hand(requested[6:], self.config.hand_disabled_joints_deg)
        self.backend.command(tcp, hand)
        self._last_tcp, self._last_hand = tcp, hand
        return vector_to_action(self.last_command)

    def go_home(self) -> None:
        self._require_connected()
        self.backend.go_home(np.asarray(self.config.home_joints_rad, dtype=np.float64))
        self._seed_from_measurement()

    def wait_next_period(self, period_s: float) -> None:
        self.backend.wait_next_period(period_s)

    def protective_stop(self) -> bool:
        return self.backend.read().protective_stop

    def disconnect(self) -> None:
        if self.is_connected:
            self.backend.disconnect()

    def _seed_from_measurement(self) -> None:
        """Seed the command trajectory from feedback once, at connect and after go_home only."""
        state = self.backend.read()
        self._last_tcp = state.tcp.copy()
        self._last_hand = clip_hand(state.hand_deg, self.config.hand_disabled_joints_deg)

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("UR10e + DG5F is not connected")
