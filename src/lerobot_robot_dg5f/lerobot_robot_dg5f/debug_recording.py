"""Pure file writer for passive DG5F experiments.

This module deliberately has no hardware or ROS dependencies.  It accepts
already-decoded snapshots, writes an analysis-friendly timeline and detects
only state transitions.  It never decides whether a collision happened.
"""

from __future__ import annotations

import csv
import json
import math
import tarfile
import time
from collections import deque
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .constants import JOINT_NAMES


ARRAY_FIELDS = (
    "target",
    "command",
    "low_level_command",
    "measured",
    "velocity",
    "current",
    "temperature",
    "error",
    "raw_current",
    "raw_velocity",
)

BOOL_FIELDS = (
    "tracking_ok",
    "armed",
    "transport_connected",
    "control_thread_alive",
    "motion_ready",
    "system_started",
    "telemetry_valid",
    "temperature_safe",
)

STATUS_FIELDS = (
    "communication_rate_hz",
    "data_processing_status",
    "last_motion_result",
    "disconnect_count",
    "reconnect_count",
    "diagnosis_process",
    "diagnosis_step",
    "diagnosis_joint_id",
    "diagnosis_period",
    "diagnosis_joint",
    "diagnosis_temperature",
    "disarm_reason",
    "motion_ready_reason",
    "recovery_state",
    "last_command_age_ms",
    "last_tracking_age_ms",
    "last_telemetry_age_ms",
    "last_sdk_packet_age_ms",
    "last_communication_callback_age_ms",
    "last_position_sample_age_ms",
    "diagnostics_age_ms",
)

NETWORK_FIELDS = (
    "rx_packets",
    "tx_packets",
    "rx_errors",
    "tx_errors",
    "rx_dropped",
    "tx_dropped",
    "carrier",
    "operstate",
    "carrier_changes",
)


def _array_column(prefix: str, index: int) -> str:
    if prefix in {"target", "command", "low_level_command", "measured"}:
        return f"{prefix}_q{index}"
    return f"{prefix}_{index}"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _finite_values(values: Sequence[float] | None) -> list[float]:
    if values is None or len(values) != len(JOINT_NAMES):
        return [math.nan] * len(JOINT_NAMES)
    result: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = math.nan
        result.append(parsed if math.isfinite(parsed) else math.nan)
    return result


class DebugRunWriter:
    """Own one recording directory and finalize it into a tar archive."""

    def __init__(
        self,
        output_root: str | Path,
        manifest: Mapping[str, object],
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        run_stamp: str | None = None,
        defer_archive: bool = False,
        arm_events_from_bridge: bool = False,
    ) -> None:
        self._clock = clock
        self._wall_time = wall_time
        self._defer_archive = defer_archive
        self._arm_events_from_bridge = arm_events_from_bridge
        self._start_monotonic = clock()
        stamp = run_stamp or time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime(wall_time())
        )
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / stamp
        suffix = 1
        while run_dir.exists():
            run_dir = root / f"{stamp}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=False)
        self.run_dir = run_dir
        self.archive_path = root / f"dg5f_debug_{run_dir.name}.tar.gz"
        self._finalized = False

        payload = dict(manifest)
        payload.setdefault(
            "experiment_start_wall_time",
            time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(wall_time())),
        )
        payload["timestamp_source"] = "time.monotonic"
        payload["schema_version"] = 2
        payload["start_monotonic_s"] = self._start_monotonic
        payload["start_wall_time_unix_s"] = wall_time()
        payload["timeline_position_unit"] = "radian"
        payload["timeline_velocity_unit"] = "radian/second"
        payload["timeline_current_unit"] = "mA"
        payload["raw_current_unit"] = "mA"
        payload["raw_velocity_unit"] = "rpm"
        payload["sdk_packet_age_source"] = "ReceivedGripperData callback; raw TCP packet timestamp unavailable"
        payload["timeline_temperature_unit"] = "degree Celsius"
        payload["joint_names"] = list(JOINT_NAMES)
        self.manifest = payload
        (run_dir / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        self._timeline_file = (run_dir / "timeline.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._events_file = (run_dir / "events.jsonl").open(
            "w", encoding="utf-8"
        )
        self._network_file = (run_dir / "network.log").open(
            "w", encoding="utf-8"
        )
        columns = ["time_s", "wall_time_unix_s"]
        for prefix in ARRAY_FIELDS:
            columns.extend(
                _array_column(prefix, index) for index in range(len(JOINT_NAMES))
            )
        columns.extend(BOOL_FIELDS)
        columns.extend(STATUS_FIELDS)
        columns.extend(
            (
                "max_current",
                "max_current_joint",
                "max_temperature",
                "max_temperature_joint",
                "max_tracking_error",
                "max_tracking_error_joint",
            )
        )
        columns.extend(NETWORK_FIELDS)
        self._timeline = csv.DictWriter(self._timeline_file, fieldnames=columns)
        self._timeline.writeheader()

        self.sample_count = 0
        self.event_counts: dict[str, int] = {}
        self.reason_counts: dict[str, dict[str, int]] = {}
        self._fault_snapshots = []
        self.first_disconnect_time: float | None = None
        self._previous: dict[str, bool | None] = {
            key: None
            for key in (
                "armed",
                "tracking_ok",
                "transport_connected",
                "motion_ready",
                "telemetry_valid",
            )
        }
        self._recent_rates: deque[tuple[float, float]] = deque()
        self._rate_before_disconnect: list[dict[str, float]] = []
        self._maxima = {
            "current": (math.nan, ""),
            "temperature": (math.nan, ""),
            "tracking_error": (math.nan, ""),
        }

    def elapsed(self) -> float:
        return max(0.0, self._clock() - self._start_monotonic)

    def write_static_file(self, name: str, content: str) -> None:
        if Path(name).name != name:
            raise ValueError("Static debug file name must be a basename")
        (self.run_dir / name).write_text(content, encoding="utf-8")

    def add_event(self, event: str, *, label: str | None = None, t: float | None = None, **details) -> None:
        event_time = self.elapsed() if t is None else float(t)
        payload: dict[str, object] = {"t": round(event_time, 6), "event": event}
        payload.update(details)
        if label is not None:
            payload["label"] = label
        self._events_file.write(json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n")
        self._events_file.flush()
        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        if "reason" in details:
            counts = self.reason_counts.setdefault(event, {})
            reason = str(details["reason"])
            counts[reason] = counts.get(reason, 0) + 1
        print(f"[{event_time:8.3f}] {event.replace('_', ' ')}", flush=True)

    def add_marker(self, label: str) -> None:
        self.add_event("USER_MARKER", label=label.strip() or "manual")

    def record_network(self, counters: Mapping[str, int | float]) -> None:
        payload = {"t": round(self.elapsed(), 6), "wall_time_unix_s": self._wall_time()}
        payload.update({key: counters.get(key) for key in NETWORK_FIELDS})
        self._network_file.write(json.dumps(payload) + "\n")
        self._network_file.flush()

    def _edge_events(self, row: Mapping[str, object], t: float) -> None:
        armed = bool(row["armed"])
        old = self._previous["armed"]
        if armed and old is not True and not self._arm_events_from_bridge:
            self.add_event("ARMED", t=t)
        elif old is True and not armed and not self._arm_events_from_bridge:
            self.add_event("DISARMED", t=t, reason=row.get("disarm_reason") or "UNKNOWN")
        self._previous["armed"] = armed

        tracking = bool(row["tracking_ok"])
        old = self._previous["tracking_ok"]
        if old is True and not tracking:
            self.add_event("TRACKING_LOST", t=t)
        elif old is False and tracking:
            self.add_event("TRACKING_RESTORED", t=t)
        self._previous["tracking_ok"] = tracking

        connected = bool(row["transport_connected"])
        old = self._previous["transport_connected"]
        if old is True and not connected:
            fault = dict(row)
            for age, label in ((0.1, "100ms"), (0.5, "500ms"), (1.0, "1s")):
                before = [rate for at, rate in self._recent_rates if at <= t - age]
                fault[f"communication_rate_{label}_ago"] = before[-1] if before else None
            self.add_event("DGSDK_DISCONNECTED", t=t, snapshot=fault)
            self._fault_snapshots.append(fault)
            if self.first_disconnect_time is None:
                self.first_disconnect_time = t
            rates = [value for _, value in self._recent_rates if math.isfinite(value)]
            if rates:
                self._rate_before_disconnect.append(
                    {
                        "min": min(rates),
                        "mean": sum(rates) / len(rates),
                        "max": max(rates),
                    }
                )
        elif old is False and connected:
            self.add_event("DGSDK_RECONNECTED", t=t, snapshot=dict(row))
        self._previous["transport_connected"] = connected

        ready = bool(row["motion_ready"])
        old = self._previous["motion_ready"]
        if old is True and not ready:
            self.add_event("MOTION_READY_FALSE", t=t, reason=row.get("motion_ready_reason") or "UNKNOWN")
        elif old is False and ready:
            self.add_event("MOTION_READY_TRUE", t=t)
        self._previous["motion_ready"] = ready
        valid = bool(row["telemetry_valid"])
        old = self._previous["telemetry_valid"]
        if old is True and not valid:
            self.add_event("TELEMETRY_STALE", t=t)
        elif old is False and valid:
            self.add_event("TELEMETRY_RESTORED", t=t)
        self._previous["telemetry_valid"] = valid

    def _sample_maximum(
        self, values: Sequence[float], *, absolute: bool
    ) -> tuple[float, str]:
        candidates = [
            (abs(value) if absolute else value, index)
            for index, value in enumerate(values)
            if math.isfinite(value)
        ]
        if not candidates:
            return math.nan, ""
        value, index = max(candidates)
        return value, JOINT_NAMES[index]

    def _update_global_maximum(self, key: str, value: float, joint: str) -> None:
        old_value, _ = self._maxima[key]
        if math.isfinite(value) and (not math.isfinite(old_value) or value > old_value):
            self._maxima[key] = (value, joint)

    def record(self, snapshot: Mapping[str, object]) -> dict[str, object]:
        if self._finalized:
            raise RuntimeError("Cannot record into a finalized debug run")
        t = self.elapsed()
        arrays = {
            key: _finite_values(snapshot.get(key))  # type: ignore[arg-type]
            for key in ARRAY_FIELDS
            if key != "error"
        }
        arrays["error"] = [
            command - measured
            if math.isfinite(command) and math.isfinite(measured)
            else math.nan
            for command, measured in zip(arrays["command"], arrays["measured"])
        ]

        row: dict[str, object] = {"time_s": f"{t:.9f}", "wall_time_unix_s": self._wall_time()}
        for prefix, values in arrays.items():
            row.update(
                {
                    _array_column(prefix, index): value
                    for index, value in enumerate(values)
                }
            )
        for key in BOOL_FIELDS:
            row[key] = int(bool(snapshot.get(key, False)))
        for key in STATUS_FIELDS:
            row[key] = snapshot.get(key, "")

        max_current, current_joint = self._sample_maximum(
            arrays["current"], absolute=True
        )
        max_temperature, temperature_joint = self._sample_maximum(
            arrays["temperature"], absolute=False
        )
        max_error, error_joint = self._sample_maximum(arrays["error"], absolute=True)
        row.update(
            {
                "max_current": max_current,
                "max_current_joint": current_joint,
                "max_temperature": max_temperature,
                "max_temperature_joint": temperature_joint,
                "max_tracking_error": max_error,
                "max_tracking_error_joint": error_joint,
            }
        )
        self._update_global_maximum("current", max_current, current_joint)
        self._update_global_maximum("temperature", max_temperature, temperature_joint)
        self._update_global_maximum("tracking_error", max_error, error_joint)

        network = snapshot.get("network")
        network_values = network if isinstance(network, Mapping) else {}
        for key in NETWORK_FIELDS:
            row[key] = network_values.get(key, "")

        try:
            rate = float(row["communication_rate_hz"])
        except (TypeError, ValueError):
            rate = math.nan
        # Detect a disconnect before adding its (usually zero) rate so the
        # summary describes the ten seconds leading up to the fault.
        self._edge_events(row, t)
        self._recent_rates.append((t, rate))
        while self._recent_rates and t - self._recent_rates[0][0] > 10.0:
            self._recent_rates.popleft()

        self._timeline.writerow(row)
        self.sample_count += 1
        if self.sample_count % 30 == 0:
            self._timeline_file.flush()
        return row

    def finalize(self) -> Path:
        if self._finalized:
            return self.archive_path
        duration = self.elapsed()
        self._timeline_file.flush()
        self._events_file.flush()
        self._network_file.flush()
        self._timeline_file.close()
        self._events_file.close()
        self._network_file.close()

        lines = [
            f"duration_s: {duration:.3f}",
            f"samples: {self.sample_count}",
            f"arm_count: {self.event_counts.get('ARMED', 0)}",
            f"tracking_lost_count: {self.event_counts.get('TRACKING_LOST', 0)}",
            f"disarm_count_by_reason: {json.dumps(self.reason_counts.get('DISARMED', {}))}",
            f"motion_ready_false_by_reason: {json.dumps(self.reason_counts.get('MOTION_READY_FALSE', {}))}",
            f"network_capture_requested: {self.manifest.get('tcpdump_requested', False)}",
            f"ping_capture_requested: {self.manifest.get('ping_requested', False)}",
            "network_capture_status: see network_status.json (requested does not imply available)",
            f"pcap_path: {'dg5f_tcp.pcap' if (self.run_dir / 'dg5f_tcp.pcap').exists() else 'unavailable'}",
            f"disconnect_events: {self.event_counts.get('DGSDK_DISCONNECTED', 0)}",
            f"reconnect_events: {self.event_counts.get('DGSDK_RECONNECTED', 0)}",
            "first_disconnect_time_s: "
            + (
                f"{self.first_disconnect_time:.6f}"
                if self.first_disconnect_time is not None
                else "none"
            ),
        ]
        for key, title in (
            ("current", "max_motor_current"),
            ("temperature", "max_temperature"),
            ("tracking_error", "max_tracking_error_rad"),
        ):
            value, joint = self._maxima[key]
            lines.extend((f"{title}:", f"  joint: {joint or 'none'}", f"  value: {value}"))
        lines.append("communication_rate_before_disconnect:")
        if self._rate_before_disconnect:
            for index, values in enumerate(self._rate_before_disconnect, start=1):
                lines.append(
                    f"  event_{index}: min={values['min']:.3f}, "
                    f"mean={values['mean']:.3f}, max={values['max']:.3f} Hz"
                )
        else:
            lines.append("  none")
        for index, fault in enumerate(self._fault_snapshots, 1):
            lines.append(f"disconnect_{index}_telemetry_age_ms: {fault.get('last_telemetry_age_ms')}")
            lines.append(f"disconnect_{index}_communication_history: " + json.dumps({
                key: value for key, value in fault.items() if key.startswith('communication_rate')
            }))
        files = [path.name for path in sorted(self.run_dir.iterdir())]
        files.append("summary.txt")
        lines.extend(("files:", *[f"  {name}" for name in files]))
        (self.run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")
        if not self._defer_archive:
            with tarfile.open(self.archive_path, "w:gz") as archive:
                archive.add(self.run_dir, arcname=self.run_dir.name)
        self._finalized = True
        return self.archive_path
