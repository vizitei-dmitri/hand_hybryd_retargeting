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
)

NETWORK_FIELDS = (
    "rx_packets",
    "tx_packets",
    "rx_errors",
    "tx_errors",
    "rx_dropped",
    "tx_dropped",
)


def _array_column(prefix: str, index: int) -> str:
    if prefix in {"target", "command", "low_level_command", "measured"}:
        return f"{prefix}_q{index}"
    return f"{prefix}_{index}"


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
    ) -> None:
        self._clock = clock
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
        payload["timeline_position_unit"] = "radian"
        payload["timeline_velocity_unit"] = "radian/second"
        payload["timeline_current_unit"] = "DGSDK native current unit"
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
        columns = ["time_s"]
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
        self.first_disconnect_time: float | None = None
        self._previous: dict[str, bool | None] = {
            key: None
            for key in (
                "armed",
                "tracking_ok",
                "transport_connected",
                "motion_ready",
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

    def add_event(self, event: str, *, label: str | None = None, t: float | None = None) -> None:
        event_time = self.elapsed() if t is None else float(t)
        payload: dict[str, object] = {"t": round(event_time, 6), "event": event}
        if label is not None:
            payload["label"] = label
        self._events_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._events_file.flush()
        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        print(f"[{event_time:8.3f}] {event.replace('_', ' ')}", flush=True)

    def add_marker(self, label: str) -> None:
        self.add_event("USER_MARKER", label=label.strip() or "manual")

    def record_network(self, counters: Mapping[str, int | float]) -> None:
        payload = {"t": round(self.elapsed(), 6)}
        payload.update({key: counters.get(key) for key in NETWORK_FIELDS})
        self._network_file.write(json.dumps(payload) + "\n")
        self._network_file.flush()

    def _edge_events(self, row: Mapping[str, object], t: float) -> None:
        armed = bool(row["armed"])
        old = self._previous["armed"]
        if armed and old is not True:
            self.add_event("ARMED", t=t)
        elif old is True and not armed:
            self.add_event("DISARMED", t=t)
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
            self.add_event("DGSDK_DISCONNECTED", t=t)
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
            self.add_event("DGSDK_RECONNECTED", t=t)
        self._previous["transport_connected"] = connected

        ready = bool(row["motion_ready"])
        old = self._previous["motion_ready"]
        if old is True and not ready:
            self.add_event("MOTION_READY_FALSE", t=t)
        elif old is False and ready:
            self.add_event("MOTION_READY_TRUE", t=t)
        self._previous["motion_ready"] = ready

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

        row: dict[str, object] = {"time_s": f"{t:.9f}"}
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
        files = [path.name for path in sorted(self.run_dir.iterdir())]
        files.append("summary.txt")
        lines.extend(("files:", *[f"  {name}" for name in files]))
        (self.run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        with tarfile.open(self.archive_path, "w:gz") as archive:
            archive.add(self.run_dir, arcname=self.run_dir.name)
        self._finalized = True
        return self.archive_path
