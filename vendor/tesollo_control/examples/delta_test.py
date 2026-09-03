#!/usr/bin/env python3
"""Fixed-rate linear joint test with CSV logging and plots."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import dg5f_python
import numpy as np


JOINT_COUNT = 20

# Test settings. Change these values before running the script.
JOINT = 6
AMPLITUDE_DEG = 60.0
COMMAND_RATE_HZ = 60.0
DELTAS_DEG = [5.0]
HOLD_TIME_S = 1.0

ZERO_TIME_S = 2.0
TEST_DURATION_S = 10.0

# Gripper connection and result file prefix.
GRIPPER_IP = "169.254.186.72"
GRIPPER_PORT = 502
GRIPPER_SLAVE_ID = 1
OUTPUT = Path("delta_test")


def latest_position(hand, previous: np.ndarray) -> tuple[bool, np.ndarray]:
    """Drain the non-blocking queue and return its newest sample."""
    received = False
    latest = previous
    while True:
        ok, position = hand.get_current_position()
        if not ok:
            return received, latest
        received = True
        latest = np.asarray(position, dtype=np.float32).copy()


def wait_for_next_tick(next_tick: float, period: float) -> float:
    """Wait for a control deadline and return the adjusted deadline."""
    now = time.perf_counter()
    if now < next_tick:
        time.sleep(next_tick - now)
        return next_tick

    # Do not send a burst of commands when one or more periods were missed.
    missed_ticks = math.floor((now - next_tick) / period)
    return next_tick + missed_ticks * period


def linear_command(elapsed: float, delta: float) -> float:
    """Repeat: ramp up, hold, ramp down, hold."""
    ramp_time = AMPLITUDE_DEG / (delta * COMMAND_RATE_HZ)
    phase = elapsed % (2.0 * (ramp_time + HOLD_TIME_S))

    if phase < ramp_time:
        return delta * COMMAND_RATE_HZ * phase
    if phase < ramp_time + HOLD_TIME_S:
        return AMPLITUDE_DEG
    if phase < 2.0 * ramp_time + HOLD_TIME_S:
        return AMPLITUDE_DEG - delta * COMMAND_RATE_HZ * (
            phase - ramp_time - HOLD_TIME_S
        )
    return 0.0


def save_results(rows: list[dict[str, float | int]], prefix: Path, joint: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Append the extension: a fractional delta (for example, 0.25) must not be
    # interpreted by pathlib as an existing file suffix and replaced.
    csv_path = Path(f"{prefix}.csv")
    png_path = Path(f"{prefix}.png")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            f"CSV saved to {csv_path}, but plotting requires: "
            "python3 -m pip install '.[examples]'"
        ) from error

    t = np.array([row["time_s"] for row in rows], dtype=float)
    command = np.array([row["command_deg"] for row in rows], dtype=float)
    measured = np.array([row["measured_deg"] for row in rows], dtype=float)
    error = measured - command
    command_delta = np.array([row["command_delta_deg"] for row in rows], dtype=float)
    sample_period = np.array([row["sample_period_s"] for row in rows], dtype=float)

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, command, label="command")
    axes[0].plot(t, measured, label="measured", alpha=0.85)
    axes[0].set_ylabel("Position, deg")
    axes[0].set_title(f"DG-5F joint {joint}: commanded and measured position")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, error, color="tab:red")
    axes[1].set_ylabel("Error, deg")
    axes[1].grid(True)

    axes[2].plot(t, command_delta, label="command delta, deg")
    axes[2].plot(t, sample_period * 1000.0, label="loop period, ms", alpha=0.8)
    axes[2].set_xlabel("Time, s")
    axes[2].set_ylabel("Delta / period")
    axes[2].legend()
    axes[2].grid(True)

    figure.tight_layout()
    figure.savefig(png_path, dpi=160)
    plt.close(figure)

    print(f"CSV:   {csv_path}")
    print(f"Graph: {png_path}")
    print(f"Maximum commanded delta/sample: {np.max(np.abs(command_delta)):.4f} deg")
    print(f"Maximum absolute tracking error: {np.max(np.abs(error)):.4f} deg")
    print(f"RMS tracking error: {math.sqrt(np.mean(error**2)):.4f} deg")


def play_linear(hand, delta, name):

    zero = np.zeros(JOINT_COUNT, dtype=np.float32)
    rows: list[dict[str, float | int]] = []
    measured = zero.copy()
    previous_command = 0.0
    previous_sample_time: float | None = None

    print(f"Commanding zero position for {ZERO_TIME_S:.1f} s...")
    control_period = 1.0 / COMMAND_RATE_HZ
    test_start = time.perf_counter()
    next_tick = test_start
    motion_stage = None

    # ================================================================================================

    while time.perf_counter() - test_start < 2*ZERO_TIME_S + TEST_DURATION_S: 

        if time.perf_counter() - test_start < ZERO_TIME_S:
            sample_time = time.perf_counter()
            actual_elapsed = sample_time - test_start
            elapsed = min(actual_elapsed, TEST_DURATION_S)

            command = 0
            accepted = hand.set_target_position(zero)

            feedback_ok, measured = latest_position(hand, measured)
            sample_period = (
                0.0 if previous_sample_time is None else elapsed - previous_sample_time
            )

            # rows.append(
            #     {
            #         "time_s": elapsed,
            #         "command_deg": command,
            #         "measured_deg": float(measured[JOINT]),
            #         "error_deg": float(measured[JOINT]) - command,
            #         "command_delta_deg": command - previous_command,
            #         "sample_period_s": sample_period,
            #         "command_accepted": int(accepted),
            #         "feedback_received": int(feedback_ok),
            #     }
            # )

            previous_command = command
            previous_sample_time = elapsed

            next_tick = wait_for_next_tick(next_tick + control_period, control_period)

        # ================================================================================================

        elif time.perf_counter() - test_start < ZERO_TIME_S + TEST_DURATION_S:

            if motion_stage is None:
                motion_stage = time.perf_counter() - test_start

            sample_time = time.perf_counter()
            actual_elapsed = sample_time - test_start
            elapsed = min(actual_elapsed, ZERO_TIME_S + TEST_DURATION_S)

            target = zero.copy()
            command = linear_command(elapsed - motion_stage, delta)

            target[JOINT] = command
            accepted = hand.set_target_position(target)
            feedback_ok, measured = latest_position(hand, measured)
            sample_period = (
                0.0 if previous_sample_time is None else elapsed - previous_sample_time
            )
            rows.append(
                {
                    "time_s": elapsed,
                    "command_deg": command,
                    "measured_deg": float(measured[JOINT]),
                    "error_deg": float(measured[JOINT]) - command,
                    "command_delta_deg": command - previous_command,
                    "sample_period_s": sample_period,
                    "command_accepted": int(accepted),
                    "feedback_received": int(feedback_ok),
                }
            )
            previous_command = command
            previous_sample_time = elapsed

            next_tick = wait_for_next_tick(next_tick + control_period, control_period)

        # ================================================================================================

        else:
            hand.set_target_position(zero)
            next_tick = wait_for_next_tick(next_tick + control_period, control_period)

        # ================================================================================================
    
    if rows:
        save_results(rows, name, JOINT)

        
def main() -> None:
    if not 0 <= JOINT < JOINT_COUNT:
        raise ValueError("JOINT must be between 0 and 19")
    if not DELTAS_DEG or min(AMPLITUDE_DEG, *DELTAS_DEG, COMMAND_RATE_HZ, TEST_DURATION_S) <= 0:
        raise ValueError("Amplitude, command delta, command rate and duration must be positive")
    if min(ZERO_TIME_S, HOLD_TIME_S) < 0:
        raise ValueError("ZERO_TIME_S and HOLD_TIME_S must be non-negative")

    hand = dg5f_python.DGApi.instance(GRIPPER_IP, GRIPPER_PORT, GRIPPER_SLAVE_ID)

    print("Connecting to the gripper...")
    hand.start()

    try:
        for delta in DELTAS_DEG:
            output = OUTPUT.parent / f"{OUTPUT.name}_delta_{str(delta)}"
            print(f"Testing delta={delta:g} deg -> {output}.csv/.png")
            play_linear(hand, delta, output)
    except KeyboardInterrupt:
        print("Test interrupted; saving collected samples.")
    finally:
        hand.stop()


if __name__ == "__main__":
    main()
