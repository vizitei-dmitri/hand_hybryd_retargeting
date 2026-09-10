import csv
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import String

from lerobot_robot_dg5f.debug_recorder import _parse_diagnostics, Dg5fDebugRecorder
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


def test_fault_snapshot_preserves_rate_history_ages_and_reasons(tmp_path):
    clock = FakeClock()
    writer = DebugRunWriter(tmp_path, {}, clock=clock, run_stamp="fault")
    for i in range(31):
        clock.now = 100 + i / 30
        writer.record(snapshot(communication_rate_hz=430, disarm_reason="NONE"))
    clock.now = 101.1
    writer.record(snapshot(communication_rate_hz=100))
    clock.now = 101.3
    writer.record(snapshot(communication_rate_hz=0))
    clock.now = 101.4
    fault = snapshot(transport_connected=False, armed=False, motion_ready=False,
                     telemetry_valid=False, temperature_safe=True,
                     disarm_reason="SDK_DISCONNECTED", motion_ready_reason="DISCONNECTED",
                     last_telemetry_age_ms=640, last_sdk_packet_age_ms=640,
                     raw_current=[311] * 20, raw_velocity=[12] * 20,
                     communication_rate_hz=0)
    writer.record(fault)
    writer.record(fault)
    writer.finalize()
    events = [json.loads(line) for line in (writer.run_dir / "events.jsonl").read_text().splitlines()]
    disconnect = next(e for e in events if e["event"] == "DGSDK_DISCONNECTED")
    assert disconnect["snapshot"]["communication_rate_500ms_ago"] == 430
    assert disconnect["snapshot"]["last_telemetry_age_ms"] == 640
    assert disconnect["snapshot"]["raw_current_0"] == 311
    assert next(e for e in events if e["event"] == "DISARMED")["reason"] == "SDK_DISCONNECTED"
    assert sum(e["event"] == "TELEMETRY_STALE" for e in events) == 1
    assert "SDK_DISCONNECTED" in (writer.run_dir / "summary.txt").read_text()


def test_brief_arm_disarm_between_samples_recorded_once_with_reason(tmp_path):
    clock = FakeClock()
    writer = DebugRunWriter(tmp_path, {}, clock=clock, arm_events_from_bridge=True)
    node = SimpleNamespace(writer=writer)
    for name, reason in (("ARMED", "NONE"), ("DISARMED", "USER_REQUEST")):
        Dg5fDebugRecorder._on_event(node, String(data=json.dumps({
            "event": name, "reason": reason, "source_monotonic_s": 100.01,
        })))
    writer.record(snapshot(armed=False))
    writer.finalize()
    assert writer.event_counts["ARMED"] == 1
    assert writer.event_counts["DISARMED"] == 1
    assert writer.reason_counts["DISARMED"] == {"USER_REQUEST": 1}


def test_compliance_diagnostics_survive_csv_events_and_summary(tmp_path):
    values = dict(
        compliance_active="true", contact_signal_valid="true",
        joint_tracking_scale=",".join(["0.19"] * 20),
        joint_current_slope_ma_s=",".join(["1400"] * 20),
        hybrid_contact_weights="0.82,0.82,0,0,0",
        pair_distances_m="0.024,0.1,0.1,0.1,0.1,0.011,0.1",
        pair_contact_weights="0.82,0,0,0,0,1,0",
        total_current_slope_ma_s="1800", global_current_scale="0.9",
        contact_limited_pairs="thumb-index,middle-ring", limited_fingers="thumb,index,middle,ring",
    )
    message = DiagnosticArray(status=[DiagnosticStatus(
        name="dg5f_lerobot_bridge", values=[KeyValue(key=k, value=v) for k, v in values.items()],
    )])
    decoded = _parse_diagnostics(message)
    assert decoded["contact_limited_pairs"] == ["thumb-index", "middle-ring"]
    writer = DebugRunWriter(tmp_path, {}, arm_events_from_bridge=True, run_stamp="compliance")
    writer.record(snapshot(**decoded))
    node = SimpleNamespace(writer=writer)
    for name in ["COMPLIANCE_ACTIVE", "COMPLIANCE_RELEASED", "CURRENT_GUARD_TRIP", "STALL_GUARD_TRIP"]:
        Dg5fDebugRecorder._on_event(node, String(data=json.dumps({"event": name, **decoded})))
    writer.finalize()
    with (writer.run_dir / "timeline.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["joint_tracking_scale_6"]) == 0.19
    assert float(row["joint_current_slope_ma_s_6"]) == 1400
    assert float(row["pair_distances_m_middle-ring"]) == 0.011
    assert float(row["hybrid_contact_weights_index"]) == 0.82
    assert writer.event_counts["COMPLIANCE_ACTIVE"] == 1
    events = [json.loads(line) for line in (writer.run_dir / "events.jsonl").read_text().splitlines()]
    active = next(e for e in events if e["event"] == "COMPLIANCE_ACTIVE")
    assert active["joint_tracking_scale"][6] == 0.19
    summary = (writer.run_dir / "summary.txt").read_text()
    for line in ["minimum_joint_tracking_scale: 0.19", "compliance_active_events: 1",
                 "compliance_released_events: 1", "current_guard_trip_events: 1", "stall_guard_trip_events: 1"]:
        assert line in summary
    assert '"middle-ring": 1' in summary
