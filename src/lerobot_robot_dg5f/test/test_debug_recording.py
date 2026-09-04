import csv
import json
import tarfile
from pathlib import Path

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from lerobot_robot_dg5f.debug_recorder import _parse_diagnostics
from lerobot_robot_dg5f.debug_recording import DebugRunWriter


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def snapshot(**overrides):
    values = {
        "target": [0.1] * 20,
        "command": [0.2] * 20,
        "low_level_command": [0.2] * 20,
        "measured": [0.15] * 20,
        "velocity": [0.01] * 20,
        "current": [1.0] * 20,
        "temperature": [32.0] * 20,
        "tracking_ok": True,
        "armed": True,
        "transport_connected": True,
        "control_thread_alive": True,
        "motion_ready": True,
        "system_started": True,
        "telemetry_valid": True,
        "temperature_safe": True,
        "communication_rate_hz": 850,
        "data_processing_status": 0,
        "last_motion_result": 0,
        "disconnect_count": 0,
        "reconnect_count": 0,
    }
    values.update(overrides)
    return values


def test_timeline_events_summary_and_archive(tmp_path):
    clock = FakeClock()
    writer = DebugRunWriter(
        tmp_path,
        {"backend": "fake"},
        clock=clock,
        wall_time=lambda: 0.0,
        run_stamp="fake-run",
    )
    writer.write_static_file("ros_topics.txt", "/fake\n")
    writer.write_static_file("system.txt", "fake system\n")
    writer.write_static_file("README.txt", "fake readme\n")

    writer.record(snapshot())
    clock.now += 0.1
    high = snapshot()
    high["current"] = [1.0] * 7 + [12.0] + [1.0] * 12
    high["command"] = [0.2] * 4 + [0.8] + [0.2] * 15
    writer.record(high)
    clock.now += 0.1
    writer.record(snapshot(transport_connected=False, motion_ready=False))
    clock.now += 0.1
    writer.record(snapshot(transport_connected=False, motion_ready=False))
    writer.add_marker("collision")
    clock.now += 0.1
    writer.record(snapshot(transport_connected=True, motion_ready=True))

    archive_path = writer.finalize()
    assert archive_path.exists()
    assert archive_path.name == "dg5f_debug_fake-run.tar.gz"

    with (writer.run_dir / "timeline.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert float(rows[1]["max_current"]) == 12.0
    assert rows[1]["max_current_joint"] == "rj_dg_2_4"
    assert float(rows[1]["max_tracking_error"]) == 0.65

    events = [
        json.loads(line)
        for line in (writer.run_dir / "events.jsonl").read_text().splitlines()
    ]
    names = [event["event"] for event in events]
    assert names.count("DGSDK_DISCONNECTED") == 1
    assert names.count("DGSDK_RECONNECTED") == 1
    assert names.count("MOTION_READY_FALSE") == 1
    assert names.count("USER_MARKER") == 1

    summary = (writer.run_dir / "summary.txt").read_text()
    assert "disconnect_events: 1" in summary
    assert "reconnect_events: 1" in summary
    assert "rj_dg_2_4" in summary
    with tarfile.open(archive_path) as archive:
        members = {Path(name).name for name in archive.getnames()}
    assert {"timeline.csv", "events.jsonl", "summary.txt", "manifest.json"} <= members


def test_missing_streams_do_not_break_recording(tmp_path):
    clock = FakeClock()
    writer = DebugRunWriter(
        tmp_path,
        {},
        clock=clock,
        wall_time=lambda: 0.0,
        run_stamp="missing",
    )
    row = writer.record({})
    writer.finalize()

    assert row["armed"] == 0
    assert row["transport_connected"] == 0
    assert str(row["target_q0"]).lower() == "nan"


def test_recorder_source_has_no_dg5f_sdk_client():
    package = Path(__file__).parents[1] / "lerobot_robot_dg5f"
    source = "\n".join(
        (package / name).read_text()
        for name in ("debug_recording.py", "debug_recorder.py")
    )
    assert "import dg5f_python" not in source
    assert "DGApi" not in source
    assert "set_target_position" not in source
    assert "send_action" not in source
    assert "MoveServoJoint" not in source


def test_fake_ros_diagnostics_are_decoded_without_hardware():
    message = DiagnosticArray()
    status = DiagnosticStatus()
    status.name = "dg5f_lerobot_bridge"
    status.level = DiagnosticStatus.ERROR
    status.message = "fake disconnect"
    status.values = [
        KeyValue(key="transport_connected", value="false"),
        KeyValue(key="disconnect_count", value="1"),
        KeyValue(
            key="latest_command_deg",
            value=",".join(str(index) for index in range(20)),
        ),
    ]
    message.status = [status]

    decoded = _parse_diagnostics(message)
    assert decoded["transport_connected"] is False
    assert decoded["disconnect_count"] == 1
    assert decoded["latest_command_deg"] == list(range(20))
