"""The real backend against fake ur_rtde modules and the DG5F mock: threading, not physics."""

import sys
import threading
import time
import types

import numpy as np
import pytest
from lerobot_robot_dg5f.constants import JOINT_NAMES

from lerobot_robot_ur10_dg5f import Ur10Dg5fConfig
from lerobot_robot_ur10_dg5f.real_backend import RealBackend


START_TCP = [-0.10, 0.50, 0.45, 2.2, -2.2, 0.0]


class FakeReceive:
    def __init__(self, ip):
        self.tcp = list(START_TCP)
        self.q = [0.0] * 6
        self.protective = False

    def getActualTCPPose(self):
        return list(self.tcp)

    def getActualQ(self):
        return list(self.q)

    def isProtectiveStopped(self):
        return self.protective

    def isEmergencyStopped(self):
        return False

    def disconnect(self):
        pass


class FakeControl:
    def __init__(self, ip):
        self.targets = []
        self.servo_active = False
        self.events = []
        self.lock = threading.Lock()

    def initPeriod(self):
        return time.monotonic()

    def waitPeriod(self, t_start):
        time.sleep(max(0.0, t_start + 0.002 - time.monotonic()))

    def servoL(self, pose, speed, acceleration, dt, lookahead, gain):
        assert 0.03 <= lookahead <= 0.2 and 100 <= gain <= 2000
        with self.lock:
            self.servo_active = True
            self.targets.append(list(pose))
        return True

    def servoStop(self, a=10.0):
        self.servo_active = False
        self.events.append("servoStop")
        return True

    def moveJ(self, q, speed, acceleration, asynchronous=False):
        assert not self.servo_active, "moveJ while servoing"
        self.events.append(("moveJ", list(q)))
        return True

    def stopScript(self):
        self.events.append("stopScript")

    def disconnect(self):
        self.events.append("disconnect")


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=FakeReceive))
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=FakeControl))
    config = Ur10Dg5fConfig(id="test", calibration_dir=tmp_path, backend="real", hand_backend="mock")
    backend = RealBackend(config)
    backend.connect()
    yield backend
    backend.disconnect()


def wait_for(condition, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


def test_servo_thread_streams_the_latest_target(backend):
    target = np.array(START_TCP) + np.r_[0.01, 0, 0, 0, 0, 0]
    backend.command(target, np.zeros(20))
    control = backend.arm.control
    assert wait_for(lambda: control.targets and np.allclose(control.targets[-1], target))


def test_hand_follows_with_servo_velocity_limit(backend):
    target = np.zeros(20)
    target[JOINT_NAMES.index("rj_dg_2_2")] = 30.0
    start = time.monotonic()
    backend.command(np.array(START_TCP), target)
    index = JOINT_NAMES.index("rj_dg_2_2")
    assert wait_for(lambda: abs(backend.read().hand_deg[index] - 30.0) < 0.5, timeout=2.0)
    elapsed = time.monotonic() - start
    assert elapsed >= 30.0 / 120.0 * 0.9  # not faster than servo_max_velocity_deg_s


def test_go_home_stops_servo_before_movej(backend):
    backend.go_home(np.zeros(6))
    events = backend.arm.control.events
    assert events.index("servoStop") < next(i for i, e in enumerate(events) if e[0] == "moveJ")
    assert backend.arm._thread is not None and backend.arm._thread.is_alive()


def test_protective_stop_is_reported(backend):
    backend.arm.receive.protective = True
    assert backend.read().protective_stop


def test_disconnect_stops_script(backend):
    control = backend.arm.control
    backend.disconnect()
    assert "servoStop" in control.events and "stopScript" in control.events
    assert not backend.hand.is_connected
