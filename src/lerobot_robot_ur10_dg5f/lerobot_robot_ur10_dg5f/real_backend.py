"""Real cell: UR10e through ur_rtde, DG5F through the existing servo controller."""

import threading
import time

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.constants import JOINT_NAMES

from .backend import RobotState


class RtdeArm:
    """Streams the latest TCP target with servoL at the controller rate (500 Hz on e-Series)."""

    def __init__(self, ip: str, servo_hz: float, lookahead_s: float = 0.1, gain: float = 300.0):
        self.ip = ip
        self.dt = 1.0 / servo_hz
        self.lookahead_s = lookahead_s
        self.gain = gain
        self.control = None
        self.receive = None
        self.fault = None
        self._target = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    @property
    def is_connected(self) -> bool:
        return self.control is not None

    def connect(self) -> None:
        import rtde_control
        import rtde_receive

        self.receive = rtde_receive.RTDEReceiveInterface(self.ip)
        self.control = rtde_control.RTDEControlInterface(self.ip)
        self._target = self.tcp()  # seed once from measurement
        self._start_servo()

    def tcp(self) -> np.ndarray:
        return np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)

    def q(self) -> np.ndarray:
        return np.asarray(self.receive.getActualQ(), dtype=np.float64)

    def protective_stop(self) -> bool:
        return bool(self.receive.isProtectiveStopped() or self.receive.isEmergencyStopped())

    def set_target(self, tcp) -> None:
        with self._lock:
            self._target = np.asarray(tcp, dtype=np.float64).copy()

    def move_home(self, q, speed: float = 0.3, acceleration: float = 0.3) -> None:
        """Blocking moveJ; servoing is stopped first because both use the same script."""
        self._stop_servo()
        self.control.moveJ([float(value) for value in q], speed, acceleration)
        self.set_target(self.tcp())
        self._start_servo()

    def disconnect(self) -> None:
        if self.control is None:
            return
        self._stop_servo()
        self.control.stopScript()
        self.control.disconnect()
        self.receive.disconnect()
        self.control = None
        self.receive = None

    def _start_servo(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="rtde-servo", daemon=True)
        self._thread.start()

    def _stop_servo(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.control.servoStop()

    def _loop(self) -> None:
        while self._running:
            t_start = self.control.initPeriod()
            with self._lock:
                target = self._target.tolist()
            try:
                # speed and acceleration are ignored by servoL; time, lookahead and gain matter.
                self.control.servoL(target, 0.0, 0.0, self.dt, self.lookahead_s, self.gain)
            except Exception as error:  # surface to the env instead of dying silently
                self.fault = error
                self._running = False
                return
            self.control.waitPeriod(t_start)


def _pass_through_guard(proposed):
    # Same as the ROS bridge with current_guard_enabled=false: the hand has its own protection.
    return np.asarray(proposed, dtype=np.float64).copy()


class HandServo:
    """Owns the DG5F SDK in one thread: servo_tick and telemetry reads never run concurrently."""

    def __init__(self, config: Dg5fConfig, arm_pose_max_age_ms: float = 100.0):
        if config.control_mode != "servo":
            raise ValueError("HandServo requires control_mode='servo'")
        self.config = config
        self.arm_pose_max_age_ms = arm_pose_max_age_ms
        self.robot = None
        self.fault = None
        self._target = None
        self._positions = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    @property
    def is_connected(self) -> bool:
        return self.robot is not None and self.robot.is_connected

    def connect(self) -> None:
        self.robot = Dg5f(self.config)
        self.robot.connect()
        self.robot.prepare_arm()  # passive preflight, sends nothing
        pose = self.robot.begin_arm_blend_from_feedback(max_pose_age_ms=self.arm_pose_max_age_ms)
        self._target = pose.copy()
        self._positions = pose.copy()
        self.fault = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="dg5f-servo", daemon=True)
        self._thread.start()

    def set_target(self, hand_deg) -> None:
        with self._lock:
            self._target = np.asarray(hand_deg, dtype=np.float64).copy()

    def positions(self) -> np.ndarray:
        with self._lock:
            return self._positions.copy()

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.robot is not None and self.robot.is_connected:
            self.robot.hold_position()
            self.robot.disconnect()

    def _loop(self) -> None:
        period = 1.0 / self.config.servo_rate_hz
        next_tick = time.monotonic()
        while self._running:
            with self._lock:
                target = self._target.copy()
            try:
                self.robot.servo_tick(target, guard=_pass_through_guard)
                observation = self.robot.get_observation()
            except Exception as error:  # recovery is explicit (Dg5f.recover), never automatic
                self.fault = error
                self._running = False
                return
            with self._lock:
                self._positions = np.asarray([observation[f"{joint}.pos"] for joint in JOINT_NAMES])
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))


class RealBackend:
    def __init__(self, config):
        self.config = config
        self.arm = RtdeArm(config.ur_ip, config.ur_servo_hz, config.ur_lookahead_s, config.ur_gain)
        self.hand = HandServo(Dg5fConfig(
            id=f"{config.id or 'ur10_dg5f'}_hand",
            backend=config.hand_backend,
            ip=config.hand_ip,
            control_mode="servo",
            servo_rate_hz=config.hand_servo_hz,
            disabled_joint_positions_deg=dict(config.hand_disabled_joints_deg),
            calibration_dir=config.calibration_dir,
        ))
        self.cameras = make_cameras_from_configs(config.cameras)
        self._next_tick = None

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and self.hand.is_connected

    def connect(self) -> None:
        self.arm.connect()
        try:
            self.hand.connect()
            for camera in self.cameras.values():
                camera.connect()
        except Exception:
            self.disconnect()
            raise
        self._next_tick = time.monotonic()

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self.hand.disconnect()
        self.arm.disconnect()

    def read(self) -> RobotState:
        return RobotState(
            tcp=self.arm.tcp(),
            arm_q=self.arm.q(),
            hand_deg=self.hand.positions(),
            protective_stop=(self.arm.protective_stop() or self.arm.fault is not None
                             or self.hand.fault is not None),
        )

    def command(self, tcp_target, hand_deg) -> None:
        self.arm.set_target(tcp_target)
        self.hand.set_target(hand_deg)

    def wait_next_period(self, period_s: float) -> None:
        self._next_tick += period_s
        delay = self._next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_tick = time.monotonic()  # overrun: do not try to catch up

    def go_home(self, arm_q) -> None:
        self.arm.move_home(arm_q)
        self._next_tick = time.monotonic()

    def images(self) -> dict[str, np.ndarray]:
        return {name: camera.async_read() for name, camera in self.cameras.items()}
