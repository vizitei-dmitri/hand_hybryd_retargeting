#!/usr/bin/env python3
"""Measure the 60 Hz step response to one fixed position command."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import dg5f_python
import numpy as np


JOINT_COUNT = 20

# Test settings.
JOINT = 6
MIN_POSITION_DEG = 0.0
TARGET_POSITION_DEG = 60.0
COMMAND_RATE_HZ = 60.0
ZERO_TIME_S = 2.0
TEST_DURATION_S = 5.0

GRIPPER_IP = "169.254.186.72"
GRIPPER_PORT = 502
GRIPPER_SLAVE_ID = 1
OUTPUT = Path("fixed_step_rate_test")


def latest_position(hand, previous: np.ndarray) -> tuple[bool, np.ndarray]:
    """Drain feedback and return only the newest position."""
    received = False
    latest = previous
    while True:
        ok, position = hand.get_current_position()
        if not ok:
            return received, latest
        received = True
        latest = np.asarray(position, dtype=np.float32).copy()


def wait_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)


def save_results(rows: list[dict[str, float | int]]) -> None:
    csv_path = Path(f"{OUTPUT}_target_{TARGET_POSITION_DEG:g}.csv")
    png_path = Path(f"{OUTPUT}_target_{TARGET_POSITION_DEG:g}.png")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(f"CSV saved to {csv_path}; plotting requires matplotlib") from error

    t = np.asarray([row["time_s"] for row in rows])
    command = np.asarray([row["command_deg"] for row in rows])
    measured = np.asarray([row["measured_deg"] for row in rows])
    measured_delta = np.asarray([row["measured_delta_deg"] for row in rows])
    velocity = np.asarray([row["measured_velocity_deg_s"] for row in rows])
    period_ms = 1000.0 * np.asarray([row["sample_period_s"] for row in rows])

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, command, label="command")
    axes[0].plot(t, measured, label="measured")
    axes[0].set_ylabel("Position, deg")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, measured_delta, label="measured delta/tick")
    axes[1].plot(t, velocity / COMMAND_RATE_HZ, linestyle="--", label="velocity / 60 Hz")
    axes[1].set_ylabel("Delta, deg")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, period_ms)
    axes[2].axhline(1000.0 / COMMAND_RATE_HZ, color="tab:red", linestyle="--")
    axes[2].set_xlabel("Time, s")
    axes[2].set_ylabel("Loop period, ms")
    axes[2].grid(True)

    figure.tight_layout()
    figure.savefig(png_path, dpi=160)
    # plt.close(figure)

    plt.show()

    print(f"CSV: {csv_path}")
    print(f"Graph: {png_path}")
    print(f"Maximum measured delta: {np.max(np.abs(measured_delta)):.4f} deg/tick")
    print(f"Maximum measured speed: {np.max(np.abs(velocity)):.2f} deg/s")


def main() -> None:
    if not 0 <= JOINT < JOINT_COUNT:
        raise ValueError("JOINT must be between 0 and 19")
    if COMMAND_RATE_HZ <= 0.0 or TEST_DURATION_S <= 0.0:
        raise ValueError("Invalid test settings")

    zero = np.zeros(JOINT_COUNT, dtype=np.float32)
    hand = dg5f_python.DGApi.instance(GRIPPER_IP, GRIPPER_PORT, GRIPPER_SLAVE_ID)
    period = 1.0 / COMMAND_RATE_HZ
    rows: list[dict[str, float | int]] = []
    measured = zero.copy()

    print("Connecting to the gripper...")
    hand.start()
    try:
        zero_end = time.perf_counter() + ZERO_TIME_S
        next_tick = time.perf_counter()
        while time.perf_counter() < zero_end:
            hand.set_target_position(zero)
            next_tick += period
            wait_until(next_tick)

        start = time.perf_counter()
        next_tick = start
        previous_time: float | None = None
        previous_measured = float(measured[JOINT])
        previous_feedback_time: float | None = None

        tick_count = math.ceil(TEST_DURATION_S * COMMAND_RATE_HZ) + 1
        for tick in range(tick_count):
            now = time.perf_counter()
            elapsed = now - start
            if elapsed > TEST_DURATION_S:
                break
            sample_period = 0.0 if previous_time is None else elapsed - previous_time

            command = TARGET_POSITION_DEG
            target = zero.copy()
            target[JOINT] = command
            accepted = hand.set_target_position(target)
            feedback_ok, measured = latest_position(hand, measured)
            measured_value = float(measured[JOINT])
            measured_delta = measured_value - previous_measured if feedback_ok else 0.0
            velocity = (
                0.0
                if previous_feedback_time is None or not feedback_ok
                else (measured_value - previous_measured) / (elapsed - previous_feedback_time)
            )
            rows.append(
                {
                    "tick": tick,
                    "time_s": elapsed,
                    "command_deg": command,
                    "measured_deg": measured_value,
                    "measured_delta_deg": measured_delta,
                    "error_deg": measured_value - command,
                    "sample_period_s": sample_period,
                    "measured_velocity_deg_s": velocity,
                    "command_accepted": int(accepted),
                    "feedback_received": int(feedback_ok),
                }
            )
            previous_time = elapsed
            if feedback_ok:
                previous_measured = measured_value
                previous_feedback_time = elapsed

            next_tick += period
            # Skip missed deadlines instead of sending catch-up bursts.
            now = time.perf_counter()
            if now > next_tick:
                next_tick += math.floor((now - next_tick) / period) * period
            wait_until(next_tick)
    except KeyboardInterrupt:
        print("Test interrupted; saving collected samples.")
    finally:
        hand.set_target_position(zero)
        time.sleep(1.0)
        hand.stop()

    if rows:
        save_results(rows)


if __name__ == "__main__":
    main()
