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


class SuspendApi:
    def __init__(self):
        self.calls = 0

    def suspend_motion(self):
        self.calls += 1


class CommandQueueApi:
    def __init__(self, results, temperature=None, control_status=None):
        self.results = iter(results)
        self.commands = []
        self.temperature = temperature
        self.control_status = control_status

    def set_target_position(self, command):
        self.commands.append(np.asarray(command).copy())
        return next(self.results)

    def get_current_temp(self):
        if self.temperature is None:
            return False, np.zeros(20, dtype=np.float32)
        return True, np.asarray(self.temperature, dtype=np.float32)

    def get_control_status(self):
        if self.control_status is None:
            raise RuntimeError("status unavailable")
        return self.control_status


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


def test_tesollo_backend_suspend_motion_forwards_to_sdk():
    api = SuspendApi()
    backend = TesolloDg5fBackend("127.0.0.1", 502, 1)
    backend._api = api
    backend._connected = True

    backend.suspend_motion()

    assert api.calls == 1


def test_latest_feedback_read_is_bounded():
    api = EndlessFeedbackApi()
    backend = TesolloDg5fBackend("127.0.0.1", 502, 1)
    backend._api = api
    backend._connected = True

    latest = backend._read_latest("get_current_position", drain_limit=4)

    assert api.calls == 4
    assert np.array_equal(latest, np.full(20, 4.0))


def test_transient_command_queue_rejection_retries_latest_target(monkeypatch):
    api = CommandQueueApi((False, True))
    backend = TesolloDg5fBackend("127.0.0.1", 502, 1)
    backend._api = api
    backend._connected = True
    sleeps = []
    monkeypatch.setattr("lerobot_robot_dg5f.backends.time.sleep", sleeps.append)

    target = np.arange(20, dtype=np.float64)
    backend.send_positions(target)

    assert sleeps == [0.002]
    assert len(api.commands) == 2
    assert np.array_equal(api.commands[0], target.astype(np.float32))
    assert np.array_equal(api.commands[1], target.astype(np.float32))


def test_persistent_command_queue_rejection_reports_stalled_sdk(monkeypatch):
    api = CommandQueueApi((False, False))
    backend = TesolloDg5fBackend("127.0.0.1", 502, 1)
    backend._api = api
    backend._connected = True
    monkeypatch.setattr("lerobot_robot_dg5f.backends.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="rejected the position target twice"):
        backend.send_positions(np.zeros(20))

    assert len(api.commands) == 2


def test_persistent_queue_rejection_reports_hottest_joint(monkeypatch):
    temperature = np.full(20, 31.0)
    temperature[7] = 68.5
    api = CommandQueueApi((False, False), temperature=temperature)
    backend = TesolloDg5fBackend("127.0.0.1", 502, 1)
    backend._api = api
    backend._connected = True
    monkeypatch.setattr("lerobot_robot_dg5f.backends.time.sleep", lambda _: None)

    with pytest.raises(
        RuntimeError, match=r"hottest rj_dg_2_4=68.5 C; DGSDK motion limit=65.0 C"
    ):
        backend.send_positions(np.zeros(20))


def test_rejected_command_reports_low_level_control_status(monkeypatch):
    api = CommandQueueApi(
        (False, False),
        control_status={
            "connected": False,
            "control_running": False,
            "system_started": True,
            "temperature_safe": False,
            "servo_keepalive_enabled": True,
            "communication_rate_hz": 200,
            "last_motion_result": 7,
        },
    )
    backend = TesolloDg5fBackend("127.0.0.1", 502, 1)
    backend._api = api
    backend._connected = True
    monkeypatch.setattr("lerobot_robot_dg5f.backends.time.sleep", lambda _: None)

    with pytest.raises(
        RuntimeError,
        match=(
            r"connected=False, control_running=False, system_started=True, "
            r"temperature_safe=False, servo_keepalive_enabled=True, "
            r"communication_rate_hz=200, last_motion_result=7"
        ),
    ):
        backend.send_positions(np.zeros(20))


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
