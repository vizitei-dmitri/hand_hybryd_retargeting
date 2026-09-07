"""Small hardware adapters used by the DG5F LeRobot implementation."""

import time
from typing import Optional, Protocol

import numpy as np

from .constants import (
    JOINT_NAMES,
    TELEMETRY_FIELDS,
    TESOLLO_TEMPERATURE_LIMIT_C,
)


class Dg5fBackend(Protocol):
    @property
    def is_connected(self) -> bool:
        pass

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def send_positions(self, positions_deg: np.ndarray) -> None:
        pass

    def suspend_motion(self) -> None:
        pass

    def read_initial_position(
        self, timeout_s: float, drain_limit: int
    ) -> np.ndarray:
        pass

    def read_telemetry(
        self, drain_limit: int
    ) -> dict[str, Optional[np.ndarray]]:
        pass

    def read_control_status(self) -> dict[str, object]:
        pass


class MockDg5fBackend:
    """In-memory backend for tests and no-hardware pipeline validation."""

    def __init__(self) -> None:
        self._connected = False
        self._positions = np.zeros(len(JOINT_NAMES), dtype=np.float64)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def send_positions(self, positions_deg: np.ndarray) -> None:
        if not self._connected:
            raise RuntimeError("Mock DG5F backend is not connected")
        command = np.asarray(positions_deg, dtype=np.float64)
        if command.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(command)):
            raise ValueError("Expected 20 finite DG5F positions")
        self._positions = command.copy()

    def suspend_motion(self) -> None:
        # Mock backend has no persistent low-level keepalive.
        return

    def read_initial_position(
        self, timeout_s: float, drain_limit: int
    ) -> np.ndarray:
        del timeout_s, drain_limit
        if not self._connected:
            raise RuntimeError("Mock DG5F backend is not connected")
        return self._positions.copy()

    def read_telemetry(
        self, drain_limit: int
    ) -> dict[str, Optional[np.ndarray]]:
        del drain_limit
        if not self._connected:
            raise RuntimeError("Mock DG5F backend is not connected")
        zeros = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        return {
            "pos": self._positions.copy(),
            "vel": zeros.copy(),
            "current": zeros.copy(),
            "temp": zeros.copy(),
        }

    def read_control_status(self) -> dict[str, object]:
        connected = bool(self._connected)
        return {
            "connected": connected,
            "transport_connected": connected,
            "control_running": connected,
            "control_thread_alive": connected,
            "motion_ready": connected,
            "motion_ready_reason": "READY" if connected else "DISCONNECTED",
            "recovery_required": False,
            "last_telemetry_age_ms": 0.0,
            "last_sdk_packet_age_ms": 0.0,
            "last_position_sample_age_ms": 0.0,
            "last_communication_callback_age_ms": 0.0,
            "raw_current": np.zeros(20),
            "raw_velocity": np.zeros(20),
            "system_started": connected,
            "telemetry_valid": connected,
            "temperature_safe": connected,
            "servo_keepalive_enabled": True,
            "communication_rate_hz": 1000 if connected else 0,
            "data_processing_status": 0,
            "last_motion_result": 0,
            "disconnect_count": 0,
            "reconnect_count": 0,
            "diagnosis_process": 0,
            "diagnosis_step": 0,
            "diagnosis_joint_id": 0,
            "diagnosis_period": 0,
            "diagnosis_joint": 0,
            "diagnosis_temperature": 0,
            "latest_command_valid": connected,
            "latest_command_deg": self._positions.copy(),
        }


class TesolloDg5fBackend:
    """Thin adapter around the bundled ``dg5f_python`` DGSDK binding."""

    def __init__(
        self,
        ip: str,
        port: int,
        slave_id: int,
        *,
        servo_keepalive: bool = True,
    ) -> None:
        self.ip = ip
        self.port = port
        self.slave_id = slave_id
        self.servo_keepalive = bool(servo_keepalive)
        self._api = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return
        try:
            import dg5f_python
        except ImportError as error:
            raise RuntimeError(
                "dg5f_python is not installed; rebuild the project Docker image"
            ) from error

        self._api = dg5f_python.DGApi.instance(
            self.ip, int(self.port), int(self.slave_id)
        )
        try:
            self._api.start(self.servo_keepalive)
        except Exception:
            self._api.stop()
            raise
        self._connected = True

    def disconnect(self) -> None:
        if self._api is not None and self._connected:
            try:
                self._api.stop()
            finally:
                self._connected = False

    def suspend_motion(self) -> None:
        """Clear pending low-level motion/keepalive without sending a new target."""
        if not self._connected or self._api is None:
            return
        suspend = getattr(self._api, "suspend_motion", None)
        if suspend is not None:
            suspend()

    def send_positions(self, positions_deg: np.ndarray) -> None:
        if not self._connected or self._api is None:
            raise RuntimeError("Tesollo DG5F backend is not connected")
        command = np.asarray(positions_deg, dtype=np.float32)
        if command.shape != (len(JOINT_NAMES),):
            raise ValueError(f"Expected {len(JOINT_NAMES)} DG5F positions")
        if not np.all(np.isfinite(command)):
            raise ValueError("DG5F positions must be finite")
        if bool(self._api.set_target_position(command)):
            return
        # A single rejection can be a producer/consumer boundary race. Give
        # the SDK loop one short opportunity to consume and retry the latest
        # setpoint; never enqueue a burst of stale intermediate commands.
        time.sleep(0.002)
        if not bool(self._api.set_target_position(command)):
            diagnostic_parts = []
            try:
                status = self.read_control_status()
                if status:
                    diagnostic_parts.extend(
                        [
                            f"connected={status.get('connected')}",
                            f"control_running={status.get('control_running')}",
                            f"system_started={status.get('system_started')}",
                            f"temperature_safe={status.get('temperature_safe')}",
                            "servo_keepalive_enabled="
                            f"{status.get('servo_keepalive_enabled')}",
                            "communication_rate_hz="
                            f"{status.get('communication_rate_hz')}",
                            f"last_motion_result={status.get('last_motion_result')}",
                            "transport_connected="
                            f"{status.get('transport_connected', status.get('connected'))}",
                            "control_thread_alive="
                            f"{status.get('control_thread_alive', status.get('control_running'))}",
                            f"motion_ready={status.get('motion_ready')}",
                            f"telemetry_valid={status.get('telemetry_valid')}",
                        ]
                    )
            except Exception:
                pass
            try:
                temperature = self._read_latest("get_current_temp", 16)
                if temperature is not None:
                    hottest_index = int(np.argmax(temperature))
                    diagnostic_parts.append(
                        f"hottest {JOINT_NAMES[hottest_index]}="
                        f"{float(temperature[hottest_index]):.1f} C; "
                        f"DGSDK motion limit={TESOLLO_TEMPERATURE_LIMIT_C:.1f} C"
                    )
            except Exception:
                # Queue failure remains authoritative even if diagnostic
                # telemetry is unavailable or the connection is disappearing.
                pass
            context = ", ".join(diagnostic_parts)
            if not context:
                context = "check connection and temperatures"
            raise RuntimeError(
                "Tesollo rejected the position target twice "
                f"({context})"
            )

    def read_control_status(self) -> dict[str, object]:
        """Return low-level loop health when supported by the SDK binding."""
        if not self._connected or self._api is None:
            raise RuntimeError("Tesollo DG5F backend is not connected")
        get_status = getattr(self._api, "get_control_status", None)
        if get_status is None:
            return {}
        return dict(get_status())

    def recover(self, timeout_s: float) -> np.ndarray:
        if self._api is None or not self._connected:
            raise RuntimeError("DGSDK session is not initialized")
        return np.asarray(self._api.recover(timeout_s), dtype=np.float64)

    def _read_latest(
        self, method_name: str, drain_limit: int
    ) -> Optional[np.ndarray]:
        """Read at most ``drain_limit`` queued samples and return the newest."""
        if drain_limit <= 0:
            raise ValueError("drain_limit must be positive")
        latest = None
        for _ in range(drain_limit):
            ok, values = getattr(self._api, method_name)()
            if not ok:
                break
            result = np.asarray(values, dtype=np.float64)
            if result.shape == (len(JOINT_NAMES),) and np.all(np.isfinite(result)):
                latest = result.copy()
        return latest

    def read_initial_position(
        self, timeout_s: float, drain_limit: int
    ) -> np.ndarray:
        """Wait for one valid position and return the freshest bounded sample."""
        if not self._connected or self._api is None:
            raise RuntimeError("Tesollo DG5F backend is not connected")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("Initial feedback timeout must be positive")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            position = self._read_latest("get_current_position", drain_limit)
            if position is not None:
                return position
            time.sleep(0.01)
        raise TimeoutError(
            f"No valid DG5F position feedback within {timeout_s:.2f} s"
        )

    def read_telemetry(
        self, drain_limit: int
    ) -> dict[str, Optional[np.ndarray]]:
        if not self._connected or self._api is None:
            raise RuntimeError("Tesollo DG5F backend is not connected")
        status = self.read_control_status()
        if "measured_pos" in status:
            # Snapshot is copied under the SDK callback mutex, no queue lag.
            return {
                "pos": np.asarray(status["measured_pos"], dtype=np.float64),
                "vel": np.asarray(status["raw_velocity"], dtype=np.float64) * 6.0,
                "current": np.asarray(status["measured_current"], dtype=np.float64),
                "temp": np.asarray(status["measured_temp"], dtype=np.float64),
            }
        method_by_field = {
            "pos": "get_current_position",
            "vel": "get_current_velocity",
            "current": "get_current_current",
            "temp": "get_current_temp",
        }
        telemetry = {
            field: self._read_latest(method_by_field[field], drain_limit)
            for field in TELEMETRY_FIELDS
        }
        if telemetry["vel"] is not None:
            telemetry["vel"] *= 6.0  # rpm -> deg/s (ROS subsequently uses rad/s)
        return telemetry
