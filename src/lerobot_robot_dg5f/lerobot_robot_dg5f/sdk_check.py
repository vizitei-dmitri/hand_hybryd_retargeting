"""Read-only DG5F SDK connectivity and initial-position check."""

from __future__ import annotations

import argparse

from .backends import TesolloDg5fBackend
from .constants import JOINT_NAMES


def main() -> None:
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
    finally:
        backend.disconnect()


if __name__ == "__main__":
    main()
