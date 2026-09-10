#!/usr/bin/env python3
"""End-to-end fake ROS publishers + recorder subprocess + Trigger services.

No backend/SDK/Quest. A separate ROS domain prevents fake data entering teleop.
Run inside the sourced project container. Creates a real, local training dataset.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot_robot_dg5f.constants import JOINT_NAMES
from lerobot_robot_dg5f.dataset_check import check_dataset
from lerobot_robot_dg5f.lerobot_dataset_recorder import TOPICS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="mock_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--root", default="/workspace/lerobot_datasets")
    parser.add_argument("--interrupt-active", action="store_true")
    parser.add_argument("--with-camera", action="store_true")
    parser.add_argument("--ros-domain-id", type=int, default=93)
    args = parser.parse_args(argv)
    if not 1 <= args.ros_domain_id <= 232:
        parser.error("Use an isolated nonzero ROS domain in 1..232")
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
    rclpy.init()
    node = rclpy.create_node("dg5f_dataset_synthetic_source")
    publishers = {key: node.create_publisher(kind, TOPICS[key], 1) for key, kind in (
        ("diagnostics", DiagnosticArray), ("effective", JointState),
        ("raw", JointTrajectory), ("tracking", Bool),
    )}
    camera_pub = node.create_publisher(Image, "/dataset_mock/camera", 1) if args.with_camera else None
    step = 0
    def publish():
        nonlocal step
        step += 1
        measured = (np.arange(20) * 0.5 + np.sin(step / 40)).tolist()
        effective = (np.asarray(measured) + 2).tolist()
        effective[16] = 0.0
        raw = (np.asarray(measured) + 10).tolist()
        raw[16] = 0.0
        diag = {
            "measured_pos": measured, "measured_vel": [6.0] * 20,
            "measured_current": [20.0] * 20, "measured_temp": [32.0] * 20,
            "transport_connected": True, "motion_ready": True, "temperature_safe": True,
            "telemetry_valid": True, "armed": True, "tracking_ok": True,
            "current_guard_active": False, "current_guard_min_scale": 1.0,
            "disarm_reason": "NONE", "backend": "synthetic_ros_no_hardware",
            "last_position_sample_age_ms": 0.0, "last_telemetry_age_ms": 0.0,
            "last_command_age_ms": 0.0, "last_tracking_age_ms": 0.0,
            "startup_blend_s": 0.7, "max_direct_step_deg": 5.0,
            "tracking_grace_s": 15.0, "tracking_resume_blend_s": 1.0,
        }
        status = DiagnosticStatus(name="dg5f_lerobot_bridge", level=DiagnosticStatus.OK)
        status.values = [KeyValue(key=key, value=(",".join(map(str, value)) if isinstance(value, list)
                            else str(value).lower() if isinstance(value, bool) else str(value)))
                         for key, value in diag.items()]
        message = DiagnosticArray(status=[status])
        message.header.stamp = node.get_clock().now().to_msg()
        publishers["diagnostics"].publish(message)
        publishers["tracking"].publish(Bool(data=True))
        command = JointState(name=list(JOINT_NAMES), position=np.deg2rad(effective).tolist())
        command.header.stamp = message.header.stamp
        publishers["effective"].publish(command)
        trajectory = JointTrajectory(joint_names=list(JOINT_NAMES), points=[
            JointTrajectoryPoint(positions=np.deg2rad(raw).tolist())])
        trajectory.header.stamp = message.header.stamp
        publishers["raw"].publish(trajectory)
        if camera_pub:
            camera = Image(height=16, width=24, encoding="rgb8", step=24 * 3,
                           data=np.full((16, 24, 3), 128, dtype=np.uint8).tobytes())
            camera.header.stamp = message.header.stamp
            camera_pub.publish(camera)
    node.create_timer(1 / 60, publish)
    clients = {key: node.create_client(Trigger, f"/dg5f_dataset/{key}") for key in
               ("start_episode", "finish_episode", "discard_episode", "status")}
    def call(key):
        future = clients[key].call_async(Trigger.Request())
        deadline = time.monotonic() + 20
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not future.done():
            raise RuntimeError(f"Timeout calling {key}")
        response = future.result()
        if not response.success:
            raise RuntimeError(response.message)
        return json.loads(response.message)
    def start():
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            status = call("status")
            if status["preflight"] == "OK":
                return call("start_episode")
        raise RuntimeError(f"Preflight did not become ready: {status}")
    def collect(count=5):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            status = call("status")
            if status["episode_frames"] >= count:
                return
        raise RuntimeError(f"No recording progress: {status}")

    process = None
    with tempfile.TemporaryDirectory(prefix="dg5f_dataset_mock_") as temporary, tempfile.TemporaryFile(mode="w+") as log:
        command = [sys.executable, "-m", "lerobot_robot_dg5f.lerobot_dataset_recorder",
                   "--repo-id", args.repo_id, "--root", args.root, "--task", "synthetic dataset smoke test"]
        if args.with_camera:
            config = Path(temporary) / "cameras.yaml"
            config.write_text("cameras:\n  front:\n    topic: /dataset_mock/camera\n")
            command += ["--cameras", str(config)]
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 20
            while not clients["status"].wait_for_service(timeout_sec=0.1):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Recorder did not start")
            start()
            collect()
            call("finish_episode")
            start()
            collect(2)
            call("discard_episode")
            if args.interrupt_active:
                start()
                collect(2)
            process.send_signal(signal.SIGINT)
            process.wait(timeout=20)
            if process.returncode != 0:
                raise RuntimeError(f"Recorder exited with {process.returncode}")
            report = check_dataset(Path(args.root) / args.repo_id)
            assert report["episodes"] == (2 if args.interrupt_active else 1)
            assert report["interrupted_episodes"] == ([1] if args.interrupt_active else [])
            print(json.dumps({key: report[key] for key in
                              ("path", "episodes", "frames", "fps", "cameras", "interrupted_episodes")}, indent=2))
            print("PASS: ROS subscriptions, Trigger services, discard, SIGINT, official reopen; no hardware")
        except Exception:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
