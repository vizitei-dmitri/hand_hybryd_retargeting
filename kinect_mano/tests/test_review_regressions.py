"""Regressions for the three core review defects; numerical fixtures, NOT MANO.

These CPU tests use no licensed assets, detectors, devices, ROS, or network.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from kinect_mano import cli, io, visualization
from kinect_mano.data import CameraCalibration, CameraFrame, JointEstimate
from kinect_mano.fitting import FitConfig, FitState, ManoFitter, Observation
from kinect_mano.model import ManoModel
from kinect_mano.pipeline import RGBDProcessor
from kinect_mano.surface import SurfaceView
from kinect_mano.synthetic import generate_sequence, make_fixture_model


@pytest.fixture(scope="module")
def numerical_model():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield ManoModel(make_fixture_model(), allow_test_model=True, device="cpu")
    finally:
        torch.set_num_threads(previous_threads)


def _flat_state(model, timestamp_s=0.):
    theta, beta, orient = np.zeros(45), np.zeros(10), np.zeros(3)
    translation = np.array([0., 0., .5])
    with torch.no_grad():
        vertices, joints = model(*(torch.as_tensor(value, dtype=model.dtype)
                                   for value in (theta, beta, orient, translation)))
    return FitState(timestamp_s, theta, beta, orient, translation,
                    joints[0].numpy(), vertices[0].numpy(), True, "tracked")


def test_surface_dropout_cannot_win_and_best_state_uses_fresh_scores(numerical_model, monkeypatch):
    model = numerical_model
    observations, truth = generate_sequence(
        model, frames=1, points_per_camera=32, noise_m=0, joint_noise_m=0, seed=0,
    )
    view = observations[0].views[0]
    rays = view.points_m - view.camera_center_m
    directions = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    observation = Observation(
        0., observations[0].joints_m, np.ones(21),
        (SurfaceView(view.points_m + .025 * directions, view.camera_center_m, "A"),),
    )
    fitter = ManoFitter(model, FitConfig(mode="hybrid", iterations=4, correspondence_every=3),
                        beta=truth["beta"])
    prepared = fitter._prepare(observation)
    parameters = [torch.as_tensor(truth[key] if key == "beta" else truth[key][0],
                                  dtype=model.dtype)
                  for key in ("theta", "beta", "global_orient", "translation_m")]
    forward = directions.mean(0)
    forward /= np.linalg.norm(forward)
    escaped = list(parameters)
    escaped[3] = parameters[3] - torch.as_tensor(.020 * forward, dtype=model.dtype)
    with torch.no_grad():
        supported_loss, _, _, supported_coverage = fitter._loss(*parameters, observation, prepared)
        escaped_loss, escaped_terms, _, escaped_coverage = fitter._loss(*escaped, observation, prepared)
    assert supported_coverage == [1.]
    assert escaped_coverage == [0.]
    assert float(escaped_terms["surface"]) > 0, "Unmatched rays must retain bounded cost"
    assert float(escaped_loss) > float(supported_loss), "Dropping every ray must not improve the score"

    # Observe real Adam proposals with independent fresh visibility. Only refresh
    # proposals and the final update compete; cached ICP scores must not win.
    original_loss, original_step = fitter._loss, torch.optim.Adam.step
    active_parameters = []
    fresh_scores = [float(supported_loss)]

    def observe_loss(*args, **kwargs):
        active_parameters[:] = args[:4]
        return original_loss(*args, **kwargs)

    def observe_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            score = original_loss(*active_parameters, observation, prepared, iteration=0)[0]
        fresh_scores.append(float(score))
        return result

    monkeypatch.setattr(fitter, "_loss", observe_loss)
    monkeypatch.setattr(torch.optim.Adam, "step", observe_step)
    initial = FitState(-1 / 30, truth["theta"][0], truth["beta"], truth["global_orient"][0],
                       truth["translation_m"][0], truth["joints_m"][0], truth["vertices_m"][0],
                       True, "tracked")
    state = fitter.fit(observation, initial_state=initial)
    assert state.valid
    assert len(fresh_scores) == state.quality["iterations"] + 1
    assert len(fresh_scores) == 5, "Exercise cached steps and the final update"
    returned = [torch.as_tensor(getattr(state, key), dtype=model.dtype)
                for key in ("theta", "beta", "global_orient", "translation_m")]
    with torch.no_grad():
        fresh_returned = float(original_loss(*returned, observation, prepared, iteration=0)[0])
    eligible_scores = [score for update, score in enumerate(fresh_scores)
                       if update % fitter.config.correspondence_every == 0
                       or update == len(fresh_scores) - 1]
    assert fresh_returned == pytest.approx(min(eligible_scores), abs=1e-5)
    assert state.quality["loss"] == pytest.approx(fresh_returned, abs=1e-5)


def test_invalid_initializer_stays_invalid_through_npz_and_fitting(numerical_model, tmp_path):
    model = numerical_model
    truth = _flat_state(model)
    invalid = JointEstimate(truth.joints_m, np.ones(21), np.ones(21, dtype=bool), False)

    class Precomputed:
        def estimate_joints(self, *args, **kwargs):
            return invalid

    calibrations = {
        name: CameraCalibration(name, np.eye(3), [], np.eye(4), 1, 1)
        for name in ("A", "B")
    }
    frames = [CameraFrame(name, 0., None, np.full((1, 1), .5), np.ones((1, 1), dtype=bool))
              for name in calibrations]
    observation = RGBDProcessor(calibrations, precomputed=Precomputed()).process_pair(*frames)
    assert not observation.valid
    assert np.count_nonzero(observation.confidence) == 21, "This is not an insufficient-joints case"
    path = tmp_path / "invalid_observation.npz"
    io.save_observations(path, [observation])
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["valid"], [False])
    restored, _, _ = io.load_observations(path)
    assert not restored[0].valid
    np.testing.assert_array_equal(restored[0].confidence, np.ones(21))

    fitter = ManoFitter(model, FitConfig(mode="joint", iterations=2))
    previous = fitter.fit(Observation(-1 / 30, truth.joints_m, np.ones(21)))
    assert previous.valid and fitter.history
    state = fitter.fit(restored[0])
    assert not state.valid
    assert state.status == "invalid_initializer"
    assert state.timestamp_s == 0.
    assert fitter.history == []
    assert io.state_mapping(state, model.faces.numpy())["valid"] is False


def test_interrupted_export_is_atomic_and_resume_repairs_archive(numerical_model, tmp_path, monkeypatch):
    model_path = tmp_path / "numerical_fixture.npz"
    make_fixture_model(model_path)
    observation_path = tmp_path / "observations.npz"
    truth = _flat_state(numerical_model)
    io.save_observations(observation_path, [Observation(0., truth.joints_m, np.ones(21))],
                         metadata={"model_type": "numerical_fixture"})
    output = tmp_path / "fit"
    args = cli.build_parser().parse_args([
        "fit", "--model", str(model_path), "--allow-test-model", "--device", "cpu",
        "--input", str(observation_path), "--output", str(output), "--mode", "joint",
        "--iterations", "2",
    ])
    # Rendering is unrelated to persistence; exercise the actual fit/resume path.
    monkeypatch.setattr(visualization, "render_frame", lambda *args, **kwargs: None)
    result = cli._run_fit(args)
    path = Path(result["reconstruction"])
    original_bytes = path.read_bytes()
    keys = ("theta", "beta", "global_orient", "translation_m", "joints_m", "vertices_m",
            "timestamps_s", "valid", "faces")
    with np.load(path, allow_pickle=False) as archive:
        expected = {key: archive[key].copy() for key in keys}

    def interrupted_write(stream, **arrays):
        stream.write(b"PK\x03\x04interrupted zip member")
        stream.flush()
        raise OSError("injected interrupted NPZ write")

    fresh_output = tmp_path / "failed_first_export"
    with monkeypatch.context() as failure:
        failure.setattr(io.np, "savez_compressed", interrupted_write)
        for directory, overwrite in ((fresh_output, False), (output, True)):
            with pytest.raises(OSError, match="injected interrupted NPZ write"):
                io.save_reconstruction(directory, [truth], numerical_model,
                                       FitConfig(mode="joint"), overwrite=overwrite)
            if overwrite:
                assert path.read_bytes() == original_bytes, "A failed replacement must preserve the old archive"
            else:
                assert not (directory / "reconstruction.npz").exists()
                assert list(directory.iterdir()) == [], "Temporary partial archives must be cleaned"

    # Emulate a partial archive left by an interrupted older exporter. A complete,
    # matching checkpoint must repair it without fitting those frames again.
    path.write_bytes(b"PK\x03\x04truncated reconstruction")
    args.resume = True

    def unexpected_refit(*args, **kwargs):
        pytest.fail("Completed, verified checkpoint frames must not be refitted")

    monkeypatch.setattr(ManoFitter, "fit", unexpected_refit)
    resumed = cli._run_fit(args)
    assert resumed["frames"] == 1
    assert resumed["valid_frames"] == 1
    assert Path(resumed["reconstruction"]) == path
    with np.load(path, allow_pickle=False) as archive:
        for key, value in expected.items():
            np.testing.assert_array_equal(archive[key], value)
