import numpy as np
import pytest

from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.backends import TesolloDg5fBackend
from lerobot_robot_dg5f.constants import JOINT_NAMES


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class EndlessFeedbackApi:
    def __init__(self):
        self.calls = 0

    def get_current_position(self):
        self.calls += 1
        return True, np.full(20, self.calls, dtype=np.float32)


class StaleFeedbackBackend:
    def __init__(self):
        self.connected = False
        self.samples = iter((60.2, 43.0, 60.1))
        self.sent = []

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def read_initial_position(self, timeout_s, drain_limit):
        del timeout_s, drain_limit
        return np.full(20, 41.7)

    def read_telemetry(self, drain_limit):
        del drain_limit
        value = next(self.samples, 60.1)
        zeros = np.zeros(20)
        return {
            "pos": np.full(20, value),
            "vel": zeros,
            "current": zeros,
            "temp": zeros,
        }

    def send_positions(self, positions_deg):
        self.sent.append(np.asarray(positions_deg).copy())


class MissingInitialFeedbackBackend(StaleFeedbackBackend):
    def read_initial_position(self, timeout_s, drain_limit):
        del timeout_s, drain_limit
        raise TimeoutError("no initial position")


class RejectingCommandBackend(StaleFeedbackBackend):
    def read_initial_position(self, timeout_s, drain_limit):
        del timeout_s, drain_limit
        return np.zeros(20)

    def send_positions(self, positions_deg):
        del positions_deg
        raise RuntimeError("queue rejected")


def test_latest_feedback_read_is_bounded():
    api = EndlessFeedbackApi()
    backend = TesolloDg5fBackend("127.0.0.1", 502, 1)
    backend._api = api
    backend._connected = True

    latest = backend._read_latest("get_current_position", drain_limit=4)

    assert api.calls == 4
    assert np.array_equal(latest, np.full(20, 4.0))


def test_stale_feedback_does_not_reseed_command_trajectory():
    clock = FakeClock()
    robot = Dg5f(
        Dg5fConfig(
            id="stale-feedback",
            backend="mock",
            filter_tau_s=0.0,
            target_deadband_deg=0.0,
            min_send_step_deg=0.0,
        ),
        clock=clock,
    )
    backend = StaleFeedbackBackend()
    robot.backend = backend
    robot.connect()

    target = {f"{joint}.pos": 61.7 for joint in JOINT_NAMES}
    clock.now = 0.05
    robot.send_action(target)
    command_before_feedback = robot.command_shaper.command_pose_deg.copy()

    observation = robot.get_observation()
    assert observation[f"{JOINT_NAMES[0]}.pos"] == 60.2
    assert np.array_equal(
        robot.command_shaper.command_pose_deg, command_before_feedback
    )

    clock.now = 0.10
    robot.send_action(target)
    assert np.all(
        robot.command_shaper.command_pose_deg[:16] >= command_before_feedback[:16]
    )
    assert np.max(robot.command_shaper.command_pose_deg[:16]) < 60.2
    robot.disconnect()


def test_connect_fails_closed_without_initial_position():
    robot = Dg5f(Dg5fConfig(id="missing-initial", backend="mock"))
    backend = MissingInitialFeedbackBackend()
    robot.backend = backend

    try:
        robot.connect()
    except TimeoutError:
        pass
    else:
        raise AssertionError("connect() accepted missing initial feedback")

    assert not backend.is_connected
    assert not robot.command_shaper.is_initialized


def test_rejected_backend_command_is_not_reported_as_effective():
    robot = Dg5f(
        Dg5fConfig(
            id="reject-command",
            backend="mock",
            control_smoothing=False,
            min_send_step_deg=0.0,
        )
    )
    robot.backend = RejectingCommandBackend()
    robot.connect()
    before = robot.command_shaper.effective_command()

    with pytest.raises(RuntimeError, match="queue rejected"):
        robot.send_action({f"{joint}.pos": 5.0 for joint in JOINT_NAMES})

    assert np.array_equal(robot.command_shaper.effective_command(), before)
    robot.disconnect()
