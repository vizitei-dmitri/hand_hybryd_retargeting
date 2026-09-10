"""Passive ROS 2 subscriptions -> local official LeRobotDataset episodes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory
import yaml

from .dataset_recording import EpisodeRecorder, SampleBuffer, decode_diagnostics, ordered_degrees


TOPICS = {
    "diagnostics": "/dg5f/lerobot/diagnostics",
    "effective": "/dg5f/lerobot/commanded_joint_states",
    "raw": "/dg5f/joint_command",
    "tracking": "/dg5f/tracking_ok",
}


def camera_config(path):
    if not path:
        return {}
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or not isinstance(config.get("cameras"), dict):
        raise ValueError("Camera YAML must contain a cameras mapping (empty {} is allowed)")
    result = {}
    for name, camera in config["cameras"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid camera name: {name}")
        if not isinstance(camera, dict) or not isinstance(camera.get("topic"), str) or not camera["topic"]:
            raise ValueError(f"Missing camera topic for {name}")
        result[name] = camera["topic"]
    return result


def image_to_rgb(message):
    """Decode 8-bit ROS images with row padding, without cv_bridge/NumPy ABI issues."""
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if message.encoding not in channels:
        raise ValueError(f"Unsupported image encoding {message.encoding}; use rgb8/bgr8/rgba8/bgra8/mono8")
    count = channels[message.encoding]
    if message.height <= 0 or message.width <= 0 or message.step < message.width * count:
        raise ValueError("Invalid Image dimensions/step")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != message.height * message.step:
        raise ValueError("Image data length does not match height * step")
    pixels = raw.reshape(message.height, message.step)[:, :message.width * count]
    pixels = pixels.reshape(message.height, message.width, count)
    if count == 1:
        return np.repeat(pixels, 3, axis=2)
    pixels = pixels[:, :, :3]
    if message.encoding in ("bgr8", "bgra8"):
        pixels = pixels[:, :, ::-1]
    return pixels.copy()


class LeRobotDatasetRecorder(Node):
    def __init__(self, options):
        super().__init__("dg5f_dataset_recorder")
        # Two different dataset IDs must not accidentally advertise the same services.
        identity = f"{os.getuid()}:{os.environ.get('ROS_DOMAIN_ID', '0')}:{options.service_prefix}"
        lock_name = hashlib.sha256(identity.encode()).hexdigest()[:16]
        self._service_lock = (Path(tempfile.gettempdir()) / f"dg5f_dataset_{lock_name}.lock").open("a+")
        try:
            fcntl.flock(self._service_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._service_lock.close()
            self.destroy_node()
            raise RuntimeError("A dataset recorder already owns these services in this ROS domain") from None
        try:
            cameras = camera_config(options.cameras)
            self.samples = SampleBuffer(options.required_data_max_age_ms, cameras)
            self.samples.camera_topics = cameras
            self.recorder = EpisodeRecorder(
                options.repo_id, options.root, options.task, self.samples,
                options.fps, options.resume, options.project_dir,
            )
        except Exception:
            self._service_lock.close()
            self.destroy_node()
            raise
        self._last_log = -float("inf")
        self._last_tick = -float("inf")
        self._period = 1.0 / options.fps
        # Best effort never backpressures reliable control publishers. Keep only newest packet.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(DiagnosticArray, TOPICS["diagnostics"], self._diagnostics, qos)
        self.create_subscription(JointState, TOPICS["effective"], self._effective, qos)
        self.create_subscription(JointTrajectory, TOPICS["raw"], self._raw, qos)
        self.create_subscription(Bool, TOPICS["tracking"], lambda msg: self.samples.put("tracking", bool(msg.data)), qos)
        for name, topic in cameras.items():
            self.create_subscription(Image, topic, lambda msg, name=name: self._image(name, msg), qos)
        for name, operation in (
            ("start_episode", self.recorder.start), ("finish_episode", self.recorder.finish),
            ("discard_episode", self.recorder.discard), ("status", lambda: None),
        ):
            self.create_service(Trigger, f"{options.service_prefix.rstrip('/')}/{name}",
                                lambda req, res, op=operation: self._service(op, res))
        self.create_timer(self._period, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"PASSIVE recorder IDLE | {self.recorder.path} | {options.fps} Hz | cameras={list(cameras)}. "
            "No SDK/control connection. Use dataset-start after ARM. "
            "Ctrl+C saves a nonempty active episode as interrupted."
        )

    def _source_age(self, message):
        stamp = message.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns == 0:
            return 0.0  # Unstamped camera/producer: receive-time freshness only.
        age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        if age < -0.1:
            raise ValueError("Message timestamp is in the future; check ROS clocks")
        return max(0.0, age)

    def _receive(self, key, message, decoder):
        try:
            self.samples.put(key, decoder(message), self._source_age(message))
        except (ValueError, TypeError, KeyError) as error:
            self.samples.invalidate(key)
            self._warn(f"Ignoring {key}: {error}")

    def _diagnostics(self, message):
        self._receive("diagnostics", message, decode_diagnostics)

    def _effective(self, message):
        self._receive("effective", message, lambda msg: ordered_degrees(msg.name, msg.position))

    def _raw(self, message):
        def decode(msg):
            if not msg.points:
                raise ValueError("Empty JointTrajectory")
            return ordered_degrees(msg.joint_names, msg.points[-1].positions)
        self._receive("raw", message, decode)

    def _image(self, key, message):
        self._receive(f"image:{key}", message, image_to_rgb)

    def _warn(self, message):
        now = time.monotonic()
        if now - self._last_log >= 2:
            self.get_logger().warning(message)
            self._last_log = now

    def _tick(self):
        now = time.monotonic()
        # After disk work/executor delays, don't create a burst of catch-up duplicates.
        if now - self._last_tick < self._period * 0.9:
            return
        self._last_tick = now
        previous = self.recorder.state
        self.recorder.tick()
        if self.recorder.state != previous:
            self.get_logger().error(f"Dataset {self.recorder.state}: {self.recorder.reason}; finish/discard explicitly")
        elif self.recorder.state == "RECORDING" and self.recorder.reason:
            self._warn(self.recorder.reason)

    def _service(self, operation, response):
        try:
            operation()
            response.success = True
            response.message = json.dumps(self.recorder.status(), ensure_ascii=False)
        except Exception as error:
            response.success, response.message = False, str(error)
        return response

    def finalize(self):
        try:
            self.recorder.close()
        finally:
            self._service_lock.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default="/workspace/lerobot_datasets", help="Base directory; repo-id is appended")
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--required-data-max-age-ms", type=float, default=100.0)
    parser.add_argument("--cameras", help="YAML camera topic mapping; omitted = no cameras")
    parser.add_argument("--resume", action="store_true", help="Explicitly append to a closed existing dataset")
    parser.add_argument("--project-dir", default="/workspace")
    parser.add_argument("--service-prefix", default="/dg5f_dataset")
    options, ros_args = parser.parse_known_args(argv)
    if not re.fullmatch(r"(/[A-Za-z_][A-Za-z0-9_]*)+", options.service_prefix):
        parser.error("--service-prefix must be an absolute ROS namespace, e.g. /dg5f_dataset")
    # Unknown application flags shouldn't silently become unused ROS arguments.
    if ros_args and ros_args[0] != "--ros-args":
        parser.error(f"Unrecognized arguments: {' '.join(ros_args)}")
    return options, ros_args


def main(argv=None):
    options, ros_args = parse_args(argv)
    node = None
    rclpy.init(args=ros_args)
    # Defer termination during parquet writes; stop at the next spin boundary.
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    old_signals = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        node = LeRobotDatasetRecorder(options)
        while rclpy.ok() and not stopping:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.finalize()
                print("Dataset finalized:", json.dumps(node.recorder.status()), flush=True)
                node.destroy_node()
        finally:
            for sig, handler in old_signals.items():
                signal.signal(sig, handler)
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
