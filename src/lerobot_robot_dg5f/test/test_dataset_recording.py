"""Real local LeRobotDataset round trips. No hardware or network is needed."""

import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_dg5f.dataset_check import check_dataset
from lerobot_robot_dg5f.dataset_recording import (
    BOOL_KEYS, VECTOR_KEYS, EpisodeRecorder, SampleBuffer, dataset_class,
    decode_diagnostics, feature_schema, ordered_degrees,
)
from lerobot_robot_dg5f.constants import JOINT_NAMES
from lerobot_robot_dg5f.lerobot_dataset_recorder import camera_config, image_to_rgb


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    from lerobot_robot_dg5f.backends import TesolloDg5fBackend
    def forbidden(*args, **kwargs):
        raise AssertionError("Recorder tests must never access DGSDK")
    monkeypatch.setattr(TesolloDg5fBackend, "connect", forbidden)
    original = socket.socket.connect
    def guarded(sock, address):
        if isinstance(address, tuple) and address[1] == 502:
            forbidden()
        return original(sock, address)
    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


class Clock:
    def __init__(self):
        self.now = 10.0
    def __call__(self):
        return self.now


def feed(samples, tracking=True, **changes):
    diag = {
        "measured_pos": np.arange(20), "measured_vel": np.full(20, 6),
        "measured_current": np.full(20, 20), "measured_temp": np.full(20, 32),
        "transport_connected": True, "motion_ready": True, "temperature_safe": True,
        "telemetry_valid": True, "armed": True, "tracking_ok": tracking,
        "current_guard_active": False, "current_guard_min_scale": 1.0,
        "disarm_reason": "NONE", "diagnostic_level": 0,
        "last_position_sample_age_ms": 0.0, "last_telemetry_age_ms": 0.0,
        "last_command_age_ms": 0.0, "last_tracking_age_ms": 0.0,
        "backend": "synthetic", "startup_blend_s": 0.7, "max_direct_step_deg": 5.0,
        "tracking_grace_s": 15.0, "tracking_resume_blend_s": 1.0,
    }
    diag.update(changes)
    samples.put("tracking", tracking)
    samples.put("diagnostics", diag)
    samples.put("raw", np.full(20, 30))
    effective = np.full(20, 10.0)
    effective[16] = 0
    samples.put("effective", effective)
    for camera in samples.cameras:
        samples.put(f"image:{camera}", np.full((8, 10, 3), 127, dtype=np.uint8))


@pytest.fixture
def rig(tmp_path):
    clock = Clock()
    samples = SampleBuffer(clock=clock)
    feed(samples)
    writer = EpisodeRecorder("test_hand", tmp_path, "grasp the cube", samples)
    yield writer, samples, clock
    writer.close()


def test_one_episode_official_reopen(rig):
    writer, samples, clock = rig
    writer.start()
    for _ in range(3):
        feed(samples)
        assert writer.tick()
        clock.now += 1 / 30
    writer.finish()
    writer.close()
    report = check_dataset(writer.path)
    assert (report["episodes"], report["frames"], report["fps"]) == (1, 3, 30)
    assert report["tasks"] == ["grasp the cube"]
    assert report["cameras"] == []
    assert report["first_frame"]["observation.state"][1] == 1
    assert report["first_frame"]["action"][1] == 10
    assert report["first_frame"]["expert.raw_action"][1] == 30
    assert report["first_frame"]["action"][16] == 0
    assert report["first_frame"]["expert.raw_action"][16] == 30


def test_two_episodes_and_discard_do_not_overwrite(rig):
    writer, samples, clock = rig
    writer.start()
    writer.tick()
    writer.finish()
    sealed = {path: path.read_bytes() for path in writer.path.glob("data/**/*.parquet")}
    writer.start()
    writer.tick()
    writer.discard()
    assert writer.episodes == 1 and writer.frames == 0
    writer.start()
    writer.tick()
    writer.tick()
    writer.finish()
    writer.close()
    assert all(path.read_bytes() == data for path, data in sealed.items())
    report = check_dataset(writer.path)
    assert (report["episodes"], report["frames"]) == (2, 3)


def test_discard_first_then_record(rig):
    writer, _, _ = rig
    writer.start()
    writer.tick()
    writer.discard()
    writer.start()
    writer.tick()
    writer.finish()
    assert check_dataset(writer.path)["frames"] == 1


def test_schema():
    schema = feature_schema()
    for key in VECTOR_KEYS:
        assert schema[key] == {"dtype": "float32", "shape": (20,), "names": list(JOINT_NAMES)}
    for key in BOOL_KEYS:
        assert schema[f"diagnostic.{key}"]["shape"] == (1,)
    assert "task" not in schema  # Official task/task_index mechanism, not a custom feature.


@pytest.mark.parametrize("key", ["diagnostics", "effective", "raw", "tracking"])
def test_stale_receive_skips(rig, key):
    writer, samples, clock = rig
    writer.start()
    clock.now += 0.2
    feed(samples)
    value, at = samples.samples[key]
    samples.samples[key] = value, at - 0.2
    assert not writer.tick()
    assert writer.frames == 0 and writer.state == "RECORDING"
    writer.discard()


@pytest.mark.parametrize("key", ["last_position_sample_age_ms", "last_telemetry_age_ms",
                                  "last_command_age_ms", "last_tracking_age_ms"])
def test_stale_hardware_even_with_fresh_ros(rig, key):
    writer, samples, _ = rig
    writer.start()
    feed(samples, **{key: 500})
    assert not writer.tick()
    assert "Stale source" in writer.reason
    writer.discard()


def test_tracking_loss_requires_post_restore_commands(rig):
    writer, samples, clock = rig
    writer.start()
    assert writer.tick()
    clock.now += 0.01
    feed(samples, tracking=False)
    assert not writer.tick()
    clock.now += 0.01
    samples.put("tracking", True)
    samples.samples["diagnostics"][0]["tracking_ok"] = True
    assert not writer.tick()  # Raw and effective still pre-restore.
    feed(samples)
    assert writer.tick()
    assert writer.frames == 2 and writer.segment == 1
    writer.finish()
    assert check_dataset(writer.path)["segments_per_episode"] == {0: 2}


@pytest.mark.parametrize("change", [{"motion_ready": False}, {"transport_connected": False},
                                     {"armed": False}, {"diagnostic_level": 2}])
def test_hardware_fault_pauses_preserves_good_frames(rig, change):
    writer, samples, _ = rig
    writer.start()
    assert writer.tick()
    feed(samples, **change)
    assert not writer.tick()
    assert writer.state == "PAUSED" and writer.frames == 1
    feed(samples)
    assert not writer.tick()  # No automatic rearm/resume by recorder.
    writer.finish()
    assert check_dataset(writer.path)["frames"] == 1


def test_nonzero_disabled_joint_refused_not_silently_relabelled(rig):
    writer, samples, _ = rig
    writer.start()
    samples.samples["effective"][0][16] = 5
    assert not writer.tick()
    assert writer.state == "PAUSED" and "rj_dg_5_1" in writer.reason
    writer.discard()


def test_finalize_active_episode_preserves_saved_episodes(rig):
    writer, _, _ = rig
    writer.start()
    writer.tick()
    writer.finish()
    writer.start()
    writer.tick()
    writer.close()  # Same path as Ctrl+C, save partial with explicit metadata.
    writer.close()
    report = check_dataset(writer.path)
    assert report["episodes"] == 2 and report["interrupted_episodes"] == [1]


def test_explicit_resume(rig):
    writer, samples, _ = rig
    writer.start()
    writer.tick()
    writer.finish()
    writer.close()
    with pytest.raises(FileExistsError):
        EpisodeRecorder("test_hand", writer.path.parent, "new task", samples)
    resumed = EpisodeRecorder("test_hand", writer.path.parent, "new task", samples, resume=True)
    try:
        resumed.start()
        resumed.tick()
        resumed.finish()
    finally:
        resumed.close()
    assert set(check_dataset(writer.path)["tasks"]) == {"grasp the cube", "new task"}


def test_missing_preflight_does_not_create_dataset(rig):
    writer, samples, _ = rig
    samples.invalidate("raw")
    with pytest.raises(RuntimeError, match="Waiting for raw"):
        writer.start()
    assert not writer.path.exists()


def test_camera_roundtrip_and_missing_camera(tmp_path):
    samples = SampleBuffer(cameras=["front", "wrist"], clock=Clock())
    feed(samples)
    writer = EpisodeRecorder("camera_test", tmp_path, "observe", samples)
    try:
        samples.invalidate("image:front")
        with pytest.raises(RuntimeError, match="front"):
            writer.start()
        feed(samples)
        writer.start()
        feed(samples)
        assert writer.tick()
        writer.finish()
        feed(samples)
        writer.start()
        writer.tick()
        writer.discard()
    finally:
        writer.close()
    report = check_dataset(writer.path)
    assert report["cameras"] == ["observation.images.front", "observation.images.wrist"]
    assert report["features"]["observation.images.front"]["shape"] == (8, 10, 3)


def test_image_decode_stride_and_color():
    msg = SimpleNamespace(encoding="bgr8", height=1, width=2, step=8,
                          data=bytes([1, 2, 3, 4, 5, 6, 0, 0]))
    assert image_to_rgb(msg).tolist() == [[[3, 2, 1], [6, 5, 4]]]
    msg.encoding = "16UC1"
    with pytest.raises(ValueError, match="encoding"):
        image_to_rgb(msg)


def test_named_conversion_and_diagnostic_decoder():
    names = list(reversed(JOINT_NAMES))
    values = np.deg2rad(list(reversed(range(20))))
    np.testing.assert_allclose(ordered_degrees(names, values), np.arange(20), atol=1e-5)
    with pytest.raises(ValueError):
        ordered_degrees([JOINT_NAMES[0]] * 20, values)
    msg = SimpleNamespace(status=[SimpleNamespace(name="dg5f_lerobot_bridge", level=b"\x00", values=[
        SimpleNamespace(key="measured_current", value=",".join(["123"] * 20)),
        SimpleNamespace(key="armed", value="true"),
    ])])
    decoded = decode_diagnostics(msg)
    assert decoded["armed"] is True and decoded["measured_current"][0] == 123


def test_camera_example_config():
    path = Path(__file__).parents[1] / "config/dataset_cameras.example.yaml"
    assert set(camera_config(path)) == {"front", "wrist"}
    assert camera_config(None) == {}


def test_corrupt_dataset_reports_error(tmp_path):
    with pytest.raises(ValueError, match="meta/info.json"):
        check_dataset(tmp_path)


def test_single_writer_lock(rig):
    writer, samples, _ = rig
    with pytest.raises(RuntimeError, match="Another recorder"):
        EpisodeRecorder("test_hand", writer.path.parent, "task", samples)


def test_recorder_sources_have_no_hardware_entrypoints():
    import ast
    package = Path(__file__).parents[1] / "lerobot_robot_dg5f"
    forbidden = {"connect", "send_action", "send_positions", "set_target_position",
                 "SystemStart", "MoveServoJoint", "ConnectToGripper", "DGApi"}
    for name in ("dataset_recording.py", "lerobot_dataset_recorder.py", "dataset_check.py"):
        tree = ast.parse((package / name).read_text())
        for item in ast.walk(tree):
            if isinstance(item, ast.Call):
                function = item.func
                target = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
                assert target not in forbidden


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.5])
def test_validator_detects_corrupt_effective_action(rig, value):
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer, _, _ = rig
    writer.start()
    writer.tick()
    writer.finish()
    writer.close()
    # Deliberately corrupt only this temporary test dataset, never user data.
    path = next(writer.path.glob("data/**/*.parquet"))
    table = pq.read_table(path)
    index = table.schema.get_field_index("action")
    values = table.column("action").to_pylist()
    values[0][16] = value
    table = table.set_column(index, "action", pa.array(values, type=table.schema.field(index).type))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="NaN/Inf|Nonzero rj_dg_5_1"):
        check_dataset(writer.path)


def test_discard_empty_after_reopen(rig):
    writer, _, _ = rig
    writer.start()
    writer.tick()
    writer.finish()
    writer.start()
    writer.discard()
    writer.close()
    assert check_dataset(writer.path)["episodes"] == 1


def test_slow_timer_gap_is_explicit(rig):
    writer, samples, clock = rig
    writer.start()
    writer.tick()
    clock.now += 1
    feed(samples)
    writer.tick()
    assert writer.segment == 1
    writer.finish()


def test_camera_names_cannot_replace_control_samples():
    samples = SampleBuffer(cameras=["raw"], clock=Clock())
    feed(samples)
    frame, reason, _ = samples.snapshot()
    assert not reason
    assert frame["expert.raw_action"].shape == (20,)
    assert frame["observation.images.raw"].shape == (8, 10, 3)


def test_close_finalizes_after_save_failure(rig, monkeypatch):
    writer, _, _ = rig
    writer.start()
    writer.tick()
    dataset = writer.dataset
    calls = []
    def fail_save(*args, **kwargs):
        raise OSError("simulated disk full before write")
    real_finalize = dataset.finalize
    def finalize():
        calls.append(True)
        real_finalize()
    monkeypatch.setattr(dataset, "save_episode", fail_save)
    monkeypatch.setattr(dataset, "finalize", finalize)
    with pytest.raises(RuntimeError, match="disk full"):
        writer.close()
    assert calls and writer.state == "ERROR"
    assert writer.frames == 1 and dataset.episode_buffer["size"] == 1


def test_read_only_nodes_can_import_without_sdk_construction(monkeypatch):
    import importlib
    from lerobot_robot_dg5f.dg5f import Dg5f
    def forbidden(*args, **kwargs):
        raise AssertionError("Cannot construct Robot from passive recorder")
    monkeypatch.setattr(Dg5f, "__init__", forbidden)
    importlib.reload(importlib.import_module("lerobot_robot_dg5f.lerobot_dataset_recorder"))
    importlib.reload(importlib.import_module("lerobot_robot_dg5f.dataset_check"))
