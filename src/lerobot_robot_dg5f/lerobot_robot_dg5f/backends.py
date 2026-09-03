"""Hardware backends used by the DG5F LeRobot implementation."""

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

    def read_telemetry(self) -> dict[str, Optional[np.ndarray]]:
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
        self._positions = np.asarray(positions_deg, dtype=np.float64).copy()

    def read_telemetry(self) -> dict[str, Optional[np.ndarray]]:
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
            self._api.stop()
        self._connected = False

    def send_positions(self, positions_deg: np.ndarray) -> None:
        if not self._connected or self._api is None:
            raise RuntimeError("Tesollo DG5F backend is not connected")
        command = np.asarray(positions_deg, dtype=np.float32)
        if command.shape != (len(JOINT_NAMES),):
            raise ValueError(f"Expected {len(JOINT_NAMES)} DG5F positions")
        if not bool(self._api.set_target_position(command)):
            raise RuntimeError("Tesollo command queue rejected the target")

    def _read(self, method_name: str) -> Optional[np.ndarray]:
        ok, values = getattr(self._api, method_name)()
        if not ok:
            return None
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(result)):
            return None
        return result

    def read_telemetry(self) -> dict[str, Optional[np.ndarray]]:
        if not self._connected or self._api is None:
            raise RuntimeError("Tesollo DG5F backend is not connected")
        method_by_field = {
            "pos": "get_current_position",
            "vel": "get_current_velocity",
            "current": "get_current_current",
            "temp": "get_current_temp",
        }
        return {
            field: self._read(method_by_field[field])
            for field in TELEMETRY_FIELDS
        }
