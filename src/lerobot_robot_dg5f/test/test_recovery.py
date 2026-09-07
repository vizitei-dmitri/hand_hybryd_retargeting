import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.backends import MockDg5fBackend, TesolloDg5fBackend
from lerobot_robot_dg5f.health import disarm_reason
from lerobot_robot_dg5f.ros_bridge_node import Dg5fLeRobotBridge
from lerobot_robot_dg5f.constants import JOINT_NAMES


class RecoveryBackend(MockDg5fBackend):
    def __init__(self):
        super().__init__()
        self.needed = True

    def read_control_status(self):
        result = super().read_control_status()
        result.update(recovery_required=self.needed, motion_ready=not self.needed)
        return result

    def recover(self, timeout_s):
        time.sleep(0.1)
        self.needed = False
        self._positions[:] = 15
        self._positions[1] = -80
        self._positions[16] = 0
        return self._positions.copy()


def test_arm_is_passive_and_recovery_is_explicit():
    robot = Dg5f(Dg5fConfig(backend="mock"))
    robot.backend = RecoveryBackend()
    robot.connect()
    before = robot.command_shaper.command_pose_deg.copy()
    robot.get_observation()
    np.testing.assert_array_equal(robot.command_shaper.command_pose_deg, before)

    # ARM preflight must never invoke backend.recover or reseed command state.
    with pytest.raises(RuntimeError, match="recovery is required"):
        robot.prepare_arm()
    assert robot.backend.needed
    np.testing.assert_array_equal(robot.command_shaper.command_pose_deg, before)

    pose = robot.recover()
    np.testing.assert_array_equal(pose, robot.backend._positions)
    np.testing.assert_array_equal(robot.command_shaper.command_pose_deg, robot.backend._positions)
    assert robot.command_shaper.command_pose_deg[16] == 0
    assert not robot.prepare_arm()  # healthy passive preflight only
    robot.disconnect()


def test_sdk_units_rpm_to_degrees_per_second():
    backend = TesolloDg5fBackend("127.0.0.1", 502, 1)
    backend._connected = True
    backend._api = SimpleNamespace(get_control_status=lambda: {
        "measured_pos": np.zeros(20), "raw_velocity": np.full(20, -12),
        "measured_current": np.full(20, 311), "measured_temp": np.full(20, 42.5),
    })
    values = backend.read_telemetry(16)
    assert values["current"][0] == 311.0
    assert values["vel"][0] == -72.0
    assert np.deg2rad(values["vel"][0]) == pytest.approx(-12 * 2 * np.pi / 60)


def test_direct_profile_uses_fast_step_guard_and_keeps_fault_map():
    params = yaml.safe_load((Path(__file__).parents[1] / "config/bridge.params.yaml").read_text())["dg5f_lerobot_bridge"]["ros__parameters"]
    robot = Dg5f(Dg5fConfig(
        backend="mock", control_smoothing=params["control_smoothing"],
        min_send_step_deg=params["min_send_step_deg"],
        max_direct_step_deg=params["max_direct_step_deg"],
        startup_blend_s=0.0,
    ))
    robot.connect()
    action = {f"{joint}.pos": 60.0 for joint in JOINT_NAMES}
    sent = robot.send_action(action)
    assert sent["rj_dg_2_2.pos"] == params["max_direct_step_deg"]
    assert sent["rj_dg_5_1.pos"] == 0.0
    assert sent["rj_dg_1_2.pos"] == 0.0  # joint limit still active
    robot.disconnect()


def test_telemetry_loss_is_not_temperature_fault():
    backend = MockDg5fBackend()
    backend.connect()
    status = backend.read_control_status()
    status.update(telemetry_valid=False, temperature_safe=True)
    assert disarm_reason(status) == "TELEMETRY_STALE"
    status.update(transport_connected=False)
    assert disarm_reason(status) == "SDK_DISCONNECTED"


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "92")
    rclpy.init()
    node = Dg5fLeRobotBridge()  # default mock; never constructs the SDK
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    yield node, executor
    executor.remove_node(node)
    node.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


@pytest.mark.parametrize("failure", ["TRACKING_TIMEOUT", "COMMAND_TIMEOUT"])
def test_watchdog_disarm_reason_is_sticky(bridge, failure):
    node, _ = bridge
    node._armed = True
    node._tracking_ok = True
    now = time.monotonic()
    node._last_tracking_time = now if failure == "COMMAND_TIMEOUT" else now - 1
    node._last_command_time = now if failure == "TRACKING_TIMEOUT" else now - 1
    node._latest_command_deg = np.zeros(20)
    node._send_latest()
    assert not node._armed
    assert node._disarm_reason == failure
    node._send_latest()
    assert node._disarm_reason == failure


def test_explicit_recovery_service_keeps_ros_callbacks_alive_and_stays_disarmed(bridge):
    node, executor = bridge
    backend = RecoveryBackend()
    backend.connect()
    node._backend_name = "tesollo"
    node._robot.backend = backend
    ticks = []

    def feed():
        ticks.append(time.monotonic())
        node._last_command_time = time.monotonic()
        node._latest_command_deg = np.zeros(20)
        node._on_tracking(Bool(data=True))
    feed()
    node.create_timer(0.01, feed)
    client = node.create_client(Trigger, "/dg5f/lerobot/recover")
    assert client.wait_for_service(timeout_sec=2)
    future = client.call_async(Trigger.Request())
    executor.spin_until_future_complete(future, timeout_sec=4)
    assert future.done()
    assert future.result().success, future.result().message
    assert len(ticks) > 3
    assert node._recovery_state == "SUCCEEDED"
    assert not node._armed

    # A subsequent ARM is now only a passive preflight.
    arm = node.create_client(SetBool, "/dg5f/lerobot/enable")
    assert arm.wait_for_service(timeout_sec=2)
    future = arm.call_async(SetBool.Request(data=True))
    executor.spin_until_future_complete(future, timeout_sec=2)
    assert future.result().success, future.result().message
    assert node._armed
    node._disarm("USER_REQUEST")


def test_arm_fails_without_transport(bridge):
    node, executor = bridge
    node._robot.backend.disconnect()
    node._tracking_ok = True
    node._last_tracking_time = node._last_command_time = time.monotonic()
    node._latest_command_deg = np.zeros(20)
    task = executor.create_task(node._on_enable, SetBool.Request(data=True), SetBool.Response())
    executor.spin_until_future_complete(task, timeout_sec=3)
    assert not task.result().success
    assert "not connected" in task.result().message
    assert not node._armed
