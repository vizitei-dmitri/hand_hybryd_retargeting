"""Small hardware adapters used by the DG5F LeRobot implementation."""

import time
from typing import Optional, Protocol

import numpy as np

from .constants import JOINT_NAMES, TELEMETRY_FIELDS


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

    def read_initial_position(
        self, timeout_s: float, drain_limit: int
    ) -> np.ndarray:
        pass

    def read_telemetry(
        self, drain_limit: int
    ) -> dict[str, Optional[np.ndarray]]:
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


class TesolloDg5fBackend:
    """Thin adapter around the bundled ``dg5f_python`` DGSDK binding."""

    def __init__(self, ip: str, port: int, slave_id: int) -> None:
        self.ip = ip
        self.port = port
        self.slave_id = slave_id
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
        self._api.start()
        self._connected = True

    def disconnect(self) -> None:
        if self._api is not None and self._connected:
            try:
                self._api.stop()
            finally:
                self._connected = False

    def send_positions(self, positions_deg: np.ndarray) -> None:
        if not self._connected or self._api is None:
            raise RuntimeError("Tesollo DG5F backend is not connected")
        command = np.asarray(positions_deg, dtype=np.float32)
        if command.shape != (len(JOINT_NAMES),):
            raise ValueError(f"Expected {len(JOINT_NAMES)} DG5F positions")
        if not np.all(np.isfinite(command)):
            raise ValueError("DG5F positions must be finite")
        if not bool(self._api.set_target_position(command)):
            raise RuntimeError("Tesollo command queue rejected the target")

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
        method_by_field = {
            "pos": "get_current_position",
            "vel": "get_current_velocity",
            "current": "get_current_current",
            "temp": "get_current_temp",
        }
        return {
            field: self._read_latest(method_by_field[field], drain_limit)
            for field in TELEMETRY_FIELDS
        }
