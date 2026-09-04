"""Read-only DG5F SDK connectivity, position and temperature check."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .backends import TesolloDg5fBackend
from .constants import JOINT_NAMES, TESOLLO_TEMPERATURE_LIMIT_C


def _read_temperatures(
    backend: TesolloDg5fBackend, timeout_s: float
) -> np.ndarray | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        temperature = backend.read_telemetry(drain_limit=16)["temp"]
        if temperature is not None:
            return temperature
        time.sleep(0.01)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Connect to DG5F and read positions without sending commands"
    )
    parser.add_argument("--ip", default="169.254.186.72")
    parser.add_argument("--port", type=int, default=502)
    parser.add_argument("--slave-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    backend = TesolloDg5fBackend(args.ip, args.port, args.slave_id)
    backend.connect()
    try:
        position = backend.read_initial_position(args.timeout, drain_limit=16)
        print("DG5F position feedback (degrees):")
        for name, value in zip(JOINT_NAMES, position):
            print(f"  {name}: {float(value):8.3f}")

        temperatures = _read_temperatures(backend, args.timeout)
        if temperatures is None:
            print("ERROR: no DG5F temperature feedback was received")
            return 2

        print("DG5F temperature feedback (C):")
        for name, value in zip(JOINT_NAMES, temperatures):
            print(f"  {name}: {float(value):8.3f}")
        hottest_index = int(np.argmax(temperatures))
        hottest = float(temperatures[hottest_index])
        print(
            f"Maximum temperature: {hottest:.3f} C "
            f"({JOINT_NAMES[hottest_index]})"
        )
        if hottest >= TESOLLO_TEMPERATURE_LIMIT_C:
            print(
                "UNSAFE: DGSDK blocks motion at "
                f"{TESOLLO_TEMPERATURE_LIMIT_C:.1f} C; "
                "let the hand cool and inspect the reported joint"
            )
            return 3
        print(
            "Temperature safety check: OK "
            f"(< {TESOLLO_TEMPERATURE_LIMIT_C:.1f} C)"
        )
        return 0
    finally:
        backend.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
