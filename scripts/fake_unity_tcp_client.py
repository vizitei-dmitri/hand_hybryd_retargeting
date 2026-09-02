#!/usr/bin/env python3
"""Minimal ROS-TCP-Connector-compatible Mano publisher for V2 smoke tests."""

import argparse
import json
import math
import socket
import struct
import time

from geometry_msgs.msg import Point
from rclpy.serialization import serialize_message
from vr_haptic_msgs.msg import ManoLandmarks


def packet(destination: str, payload: bytes) -> bytes:
    """Frame a destination and payload as expected by ROS-TCP-Endpoint."""
    destination_bytes = destination.encode("utf-8")
    return (
        struct.pack("<I", len(destination_bytes))
        + destination_bytes
        + struct.pack("<I", len(payload))
        + payload
    )


def finger(base_x: float, base_y: float, lengths, curl: float):
    points = []
    x, y, z = base_x, base_y, 0.0
    points.append((x, y, z))
    angle = 0.0
    for segment_index, length in enumerate(lengths):
        angle += curl * (0.35 + 0.15 * segment_index)
        y += length * math.cos(angle)
        z -= length * math.sin(angle)
        points.append((x, y, z))
    return points


def hand_points(phase: float):
    curl = 0.55 + 0.45 * math.sin(phase)
    points = [(0.0, 0.0, 0.0)]
    points.extend(
        [
            (0.025, 0.025, 0.0),
            (0.045, 0.042, -0.005 * curl),
            (0.060, 0.057, -0.012 * curl),
            (0.073, 0.068, -0.022 * curl),
        ]
    )
    for base_x, base_y, lengths in [
        (0.030, 0.065, [0.038, 0.026, 0.020]),
        (0.010, 0.072, [0.043, 0.030, 0.022]),
        (-0.012, 0.067, [0.040, 0.028, 0.021]),
        (-0.032, 0.057, [0.032, 0.023, 0.018]),
    ]:
        points.extend(finger(base_x, base_y, lengths, curl))
    return points


def connect_with_retry(host: str, port: int) -> socket.socket:
    last_error = None
    for _ in range(30):
        try:
            return socket.create_connection((host, port), timeout=1.0)
        except OSError as error:
            last_error = error
            time.sleep(0.1)
    raise ConnectionError(f"could not connect to {host}:{port}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args()

    connection = connect_with_retry(args.host, args.port)
    with connection:
        registration = {
            "topic": "/quest/hand_pose",
            "message_name": "vr_haptic_msgs/ManoLandmarks",
            "queue_size": 1,
            "latch": False,
        }
        connection.sendall(
            packet("__publish", json.dumps(registration).encode("utf-8"))
        )
        time.sleep(0.25)

        for frame_index in range(args.frames):
            message = ManoLandmarks()
            message.header.frame_id = "quest_origin"
            for xyz in hand_points(frame_index * 0.04):
                message.landmarks.append(Point(x=xyz[0], y=xyz[1], z=xyz[2]))
            connection.sendall(
                packet("/quest/hand_pose", serialize_message(message))
            )
            time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    main()
