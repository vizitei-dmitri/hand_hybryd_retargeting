"""Passive ROS 2 recorder for DG5F fault reproduction experiments."""

from __future__ import annotations

import argparse
import json
import time
import platform
import shutil
import signal
import socket
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import rclpy
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, String
from trajectory_msgs.msg import JointTrajectory

from .constants import JOINT_NAMES, TESOLLO_TEMPERATURE_LIMIT_C
from .debug_recording import DebugRunWriter, NETWORK_FIELDS, CONTACT_ARRAY_FIELDS
from .current_guard import ComplianceConfig


TOPICS = {
    "target": "/dg5f/joint_command",
    "command": "/dg5f/lerobot/commanded_joint_states",
    "measured": "/dg5f/lerobot/joint_states",
    "temperature": "/dg5f/lerobot/temperatures",
    "connected": "/dg5f/lerobot/connected",
    "armed": "/dg5f/lerobot/armed",
    "tracking": "/dg5f/tracking_ok",
    "diagnostics": "/dg5f/lerobot/diagnostics",
    "marker": "/dg5f/debug_marker",
    "events": "/dg5f/lerobot/events",
}


def _run_read_only(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return f"$ {' '.join(command)}\n{result.stdout}{result.stderr}".rstrip() + "\n"
    except (OSError, subprocess.SubprocessError) as error:
        return f"$ {' '.join(command)}\nUNAVAILABLE: {error}\n"


def collect_manifest(args: argparse.Namespace) -> dict[str, object]:
    project = Path(args.project_dir)
    config_path = project / "src/lerobot_robot_dg5f/config/bridge.params.yaml"
    configuration: dict[str, object] = {}
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        parameters = config["dg5f_lerobot_bridge"]["ros__parameters"]
        wanted = (
            "control_smoothing",
            "max_speed_deg_s",
            "max_accel_deg_s2",
            "response_time_s",
            "filter_tau_s",
            "target_deadband_deg",
            "min_send_step_deg",
            "disabled_joints",
            "disabled_positions_deg",
        )
        configuration = {key: parameters.get(key) for key in wanted}
        configuration.update({key: value for key, value in parameters.items()
                              if key.startswith(("compliance_", "current_guard_"))})
    except (OSError, TypeError, KeyError, yaml.YAMLError):
        configuration = {"config_read_error": str(config_path)}
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "git_commit": _run_read_only(
            ["git", "-C", str(project), "rev-parse", "HEAD"]
        ).splitlines()[-1],
        "git_status_short": _run_read_only(
            ["git", "-C", str(project), "status", "--short"]
        ),
        "backend": args.backend,
        "dg5f_ip": args.hand_ip,
        "dg5f_port": args.hand_port,
        "record_rate_hz": args.rate,
        "network_interface": args.interface,
        "temperature_limit_c": TESOLLO_TEMPERATURE_LIMIT_C,
        "configuration": configuration,
        "ros_topics": dict(TOPICS),
        "tcpdump_requested": bool(args.tcpdump),
        "ping_requested": bool(args.ping),
    }


def collect_system_text(interface: str) -> str:
    sections = [
        f"hostname: {socket.gethostname()}\n",
        f"platform: {platform.platform()}\n",
        _run_read_only(["uname", "-a"]),
        _run_read_only(["ip", "-details", "link", "show", "dev", interface]),
        _run_read_only(["ip", "addr", "show", "dev", interface]),
        _run_read_only(["ros2", "node", "list"]),
    ]
    return "\n".join(sections)


class NetworkCounters:
    def __init__(self, interface: str) -> None:
        self.path = Path("/sys/class/net") / interface / "statistics"

    def read(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for key in NETWORK_FIELDS:
            try:
                path = self.path.parent / key if key in {"carrier", "operstate", "carrier_changes"} else self.path / key
                raw = path.read_text().strip()
                values[key] = raw if key == "operstate" else int(raw)
            except (OSError, ValueError):
                continue
        return values


def _ordered_joint_values(
    names: Sequence[str], values: Sequence[float]
) -> list[float] | None:
    if len(names) != len(values):
        return None
    by_name = dict(zip(names, values))
    if any(name not in by_name for name in JOINT_NAMES):
        return None
    result = [float(by_name[name]) for name in JOINT_NAMES]
    return result if np.all(np.isfinite(result)) else None


def _parse_scalar(value: str) -> object:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_diagnostics(message: DiagnosticArray) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for status in message.status:
        if status.name != "dg5f_lerobot_bridge":
            continue
        level = status.level
        decoded["diagnostic_level"] = (
            level[0] if isinstance(level, (bytes, bytearray)) else int(level)
        )
        decoded["diagnostic_message"] = status.message
        for item in status.values:
            if item.key in {
                "latest_command_deg",
                "measured_pos",
                "measured_vel",
                "measured_current",
                "measured_temp",
                "raw_velocity",
                "raw_current",
                "joint_current_scale", "joint_current_slope_ma_s", "joint_slope_scale",
                "joint_contact_scale", "joint_tracking_scale", "joint_lead_budget_deg",
                "yield_delta_deg",
            } or item.key in CONTACT_ARRAY_FIELDS:
                try:
                    array = [float(value) for value in item.value.split(",")]
                except ValueError:
                    continue
                expected = len(CONTACT_ARRAY_FIELDS.get(item.key, JOINT_NAMES))
                if len(array) == expected:
                    decoded[item.key] = array
            elif item.key in {"limited_fingers", "contact_limited_pairs"}:
                decoded[item.key] = [value for value in item.value.split(",") if value]
            else:
                decoded[item.key] = _parse_scalar(item.value)
    return decoded


class PassiveProcesses:
    """Optional read-only ping/tcpdump helpers; failures are non-fatal."""

    def __init__(self, run_dir: Path, args: argparse.Namespace) -> None:
        self._processes: list[tuple[subprocess.Popen, object]] = []
        if args.ping:
            self._start(
                ["ping", "-D", "-i", "0.1", args.hand_ip], run_dir / "ping.log", "ping"
            )
        if args.tcpdump:
            expression = ["host", args.hand_ip, "and", "tcp", "port", str(args.hand_port)]
            self._start(
                [
                    "tcpdump",
                    "-i",
                    args.interface,
                    "-U",
                    "-w",
                    str(run_dir / "dg5f_tcp.pcap"),
                    *expression,
                ],
                run_dir / "tcpdump.log",
                "TCP capture",
            )

    def _start(self, command: list[str], log_path: Path, label: str) -> None:
        if shutil.which(command[0]) is None:
            log_path.write_text(f"{label} unavailable: {command[0]} not found\n")
            return
        log = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, text=True
            )
        except OSError as error:
            log.write(f"{label} unavailable: {error}\n")
            log.close()
            return
        self._processes.append((process, log))

    def stop(self) -> None:
        for process, log in self._processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=2.0)
            log.close()


class Dg5fDebugRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dg5f_debug_recorder")
        self.writer = DebugRunWriter(args.output_root, collect_manifest(args),
                                     run_stamp=args.run_stamp, defer_archive=args.defer_archive,
                                     arm_events_from_bridge=True)
        self._stop_file = Path(args.stop_file) if args.stop_file else None
        self._last_diagnostics_at = None
        self.writer.write_static_file(
            "README.txt",
            "Passive DG5F recording. No hardware commands are sent.\n"
            "Joint position/error columns use radians; velocity uses rad/s.\n"
            "NaN/empty/false means that a stream was unavailable at that sample.\n",
        )
        self.writer.write_static_file(
            "ros_topics.txt",
            "Expected recorder topics:\n"
            + "\n".join(TOPICS.values())
            + "\n\nROS graph at recorder start:\n"
            + _run_read_only(["ros2", "topic", "list"]),
        )
        self.writer.write_static_file(
            "system.txt", collect_system_text(args.interface)
        )
        # Host orchestration owns network processes and closes them before archive.
        if args.external_network:
            args.ping = args.tcpdump = False
        self._passive_processes = PassiveProcesses(self.writer.run_dir, args)
        self._network_reader = NetworkCounters(args.interface)
        self._network: dict[str, int] = {}
        self._last_network_sample = -1e9
        self._last_status_print = -1e9
        self._rate = float(args.rate)
        self._state: dict[str, object] = {
            "target": None,
            "command": None,
            "low_level_command": None,
            "measured": None,
            "velocity": None,
            "current": None,
            "temperature": None,
            "tracking_ok": None,
            "armed": None,
            "transport_connected": None,
        }
        self._diagnostics: dict[str, object] = {}

        self.create_subscription(
            JointTrajectory, TOPICS["target"], self._on_target, 10
        )
        self.create_subscription(JointState, TOPICS["command"], self._on_command, 10)
        self.create_subscription(JointState, TOPICS["measured"], self._on_measured, 10)
        self.create_subscription(
            Float32MultiArray, TOPICS["temperature"], self._on_temperature, 10
        )
        self.create_subscription(
            Bool, TOPICS["connected"], lambda msg: self._set_bool("transport_connected", msg), 10
        )
        self.create_subscription(
            Bool, TOPICS["armed"], lambda msg: self._set_bool("armed", msg), 10
        )
        self.create_subscription(
            Bool, TOPICS["tracking"], lambda msg: self._set_bool("tracking_ok", msg), 10
        )
        self.create_subscription(
            DiagnosticArray, TOPICS["diagnostics"], self._on_diagnostics, 10
        )
        self.create_subscription(String, TOPICS["marker"], self._on_marker, 10)
        self.create_subscription(String, TOPICS["events"], self._on_event, 100)
        self.create_timer(1.0 / self._rate, self._sample)

    def _set_bool(self, key: str, message: Bool) -> None:
        self._state[key] = bool(message.data)

    def _on_target(self, message: JointTrajectory) -> None:
        if not message.points:
            return
        values = _ordered_joint_values(message.joint_names, message.points[-1].positions)
        if values is not None:
            self._state["target"] = values

    def _on_command(self, message: JointState) -> None:
        values = _ordered_joint_values(message.name, message.position)
        if values is not None:
            self._state["command"] = values

    def _on_measured(self, message: JointState) -> None:
        position = _ordered_joint_values(message.name, message.position)
        if position is not None:
            self._state["measured"] = position
        velocity = _ordered_joint_values(message.name, message.velocity)
        if velocity is not None:
            self._state["velocity"] = velocity
        effort = _ordered_joint_values(message.name, message.effort)
        if effort is not None:
            self._state["current"] = effort

    def _on_temperature(self, message: Float32MultiArray) -> None:
        if len(message.data) == len(JOINT_NAMES):
            self._state["temperature"] = [float(value) for value in message.data]

    def _on_diagnostics(self, message: DiagnosticArray) -> None:
        self._last_diagnostics_at = time.monotonic()
        self._diagnostics.update(_parse_diagnostics(message))
        for name in vars(ComplianceConfig()):
            key = f"compliance_{name}"
            if key in self._diagnostics:
                self.writer.manifest.setdefault("runtime_configuration", {})[key] = self._diagnostics[key]
        for key in (
            "control_smoothing", "command_profile", "max_speed_deg_s",
            "max_accel_deg_s2", "response_time_s", "filter_tau_s",
            "target_deadband_deg", "min_send_step_deg", "max_direct_step_deg",
            "startup_blend_s", "current_guard_enabled", "current_guard_soft_ma",
            "current_guard_hard_ma", "current_guard_trip_ma",
            "current_guard_total_soft_ma", "current_guard_total_hard_ma",
            "current_guard_total_trip_ma", "current_guard_trip_hold_s",
            "current_guard_release_tau_s",
            "compliance_contact_timeout_s", "compliance_urdf_path",
            "hybrid_contact_topic", "safety_proximity_topic",
        ):
            if key in self._diagnostics:
                self.writer.manifest.setdefault("runtime_configuration", {})[key] = self._diagnostics[key]
        if "backend" in self._diagnostics:
            self.writer.manifest["backend"] = self._diagnostics["backend"]
        low_level_deg = self._diagnostics.get("latest_command_deg")
        command_valid = bool(self._diagnostics.get("latest_command_valid", False))
        if (
            command_valid
            and isinstance(low_level_deg, list)
            and len(low_level_deg) == len(JOINT_NAMES)
        ):
            self._state["low_level_command"] = np.deg2rad(low_level_deg).tolist()
        elif not command_valid:
            self._state["low_level_command"] = None
        for raw, normal, scale in (("measured_pos", "measured", np.pi / 180),
                                   ("measured_vel", "velocity", np.pi / 180),
                                   ("measured_current", "current", 1),
                                   ("measured_temp", "temperature", 1)):
            values = self._diagnostics.get(raw)
            if isinstance(values, list) and len(values) == 20:
                self._state[normal] = (np.asarray(values) * scale).tolist()

    def _on_event(self, message):
        try:
            event = json.loads(message.data)
        except (ValueError, TypeError):
            return
        # Arm/recovery transitions can occur between timeline samples.
        if event.get("event") in {"ARM_REQUESTED", "ARMED", "DISARMED", "RECOVERY_STARTED", "RECOVERY_SUCCEEDED", "RECOVERY_FAILED",
                                 "COMPLIANCE_ACTIVE", "COMPLIANCE_RELEASED", "CURRENT_GUARD_TRIP", "STALL_GUARD_TRIP",
                                 "ROBOT_CONTACT_LIMIT_ACTIVE", "ROBOT_CONTACT_LIMIT_RELEASED",
                                 "CURRENT_GUARD_ACTIVE", "CURRENT_GUARD_RELEASED"}:
            name = event.pop("event")
            source_time = event.get("source_monotonic_s")
            relative = None if source_time is None else source_time - self.writer.manifest["start_monotonic_s"]
            self.writer.add_event(name, t=relative, **event)

    def _on_marker(self, message: String) -> None:
        self.writer.add_marker(message.data)

    def _sample(self) -> None:
        if self._stop_file is not None and self._stop_file.exists():
            raise KeyboardInterrupt
        elapsed = self.writer.elapsed()
        if elapsed - self._last_network_sample >= 1.0:
            self._network = self._network_reader.read()
            self.writer.record_network(self._network)
            self._last_network_sample = elapsed
        snapshot = dict(self._state)
        snapshot.update(self._diagnostics)
        # Prefer one coherent diagnostics snapshot so reason and state agree.
        for key in ("tracking_ok", "armed", "transport_connected"):
            if key not in self._diagnostics and self._state[key] is not None:
                snapshot[key] = self._state[key]
        snapshot["diagnostics_age_ms"] = (
            (time.monotonic() - self._last_diagnostics_at) * 1000
            if self._last_diagnostics_at is not None else float("nan")
        )
        snapshot["network"] = self._network
        row = self.writer.record(snapshot)
        if elapsed - self._last_status_print >= 1.0:
            print(
                f"REC {elapsed:7.1f}s | armed={row['armed']} | "
                f"tracking={row['tracking_ok']} | "
                f"DG={'CONNECTED' if row['transport_connected'] else 'DISCONNECTED'} | "
                f"ready={row['motion_ready']} | "
                f"rate={row['communication_rate_hz']}Hz | "
                f"Imax={row['max_current']} | errmax={row['max_tracking_error']}",
                flush=True,
            )
            self._last_status_print = elapsed

    def finalize(self) -> Path:
        self._passive_processes.stop()
        return self.writer.finalize()


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/workspace/debug_runs")
    parser.add_argument("--project-dir", default="/workspace")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--interface", default="enp49s0")
    parser.add_argument("--hand-ip", default="169.254.186.72")
    parser.add_argument("--hand-port", type=int, default=502)
    parser.add_argument("--backend", default="tesollo")
    parser.add_argument("--ping", action="store_true")
    parser.add_argument("--tcpdump", action="store_true")
    parser.add_argument("--run-stamp")
    parser.add_argument("--stop-file")
    parser.add_argument("--defer-archive", action="store_true")
    parser.add_argument("--external-network", action="store_true")
    args, ros_args = parser.parse_known_args(argv)
    if not 1.0 <= args.rate <= 200.0:
        parser.error("--rate must be between 1 and 200 Hz")
    return args, ros_args


def main(args: Sequence[str] | None = None) -> None:
    options, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    node: Dg5fDebugRecorder | None = None
    try:
        node = Dg5fDebugRecorder(options)
        print(
            "\n============================================\n"
            "DG5F DEBUG RECORDER\n"
            "============================================\n"
            f"Output: {node.writer.run_dir}\n\n"
            "Listening only.\n"
            "No commands will be sent to DG-5F.\n\n"
            "Press Ctrl+C when experiment is finished.\n"
            "============================================",
            flush=True,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            archive = node.finalize()
            print(f"Recording finalized: {node.writer.run_dir}", flush=True)
            if not options.defer_archive:
                print(f"Archive: {archive}", flush=True)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
