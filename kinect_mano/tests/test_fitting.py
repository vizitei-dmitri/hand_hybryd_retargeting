"""Reconstruction tests on an independent numerical fixture, never real MANO."""
import os

import numpy as np
import pytest
import torch

from kinect_mano.fitting import (FitConfig, FitState, ManoFitter, Observation,
                                 calibrate_shape, rigid_initialize)
from kinect_mano.model import ManoModel
from kinect_mano.surface import SurfaceView
from kinect_mano.synthetic import generate_sequence, make_fixture_model


torch.set_num_threads(1)


@pytest.fixture
def model():
    return ManoModel(make_fixture_model(), allow_test_model=True)


def _mean_error(actual, expected):
    return np.linalg.norm(actual - expected, axis=-1).mean()


def _forward(model, theta, beta, orient, translation):
    with torch.no_grad():
        return tuple(value[0].cpu().numpy() for value in model(
            *(torch.as_tensor(value, dtype=model.dtype, device=model.device)
              for value in (theta, beta, orient, translation))))


def test_joint_fitting_recovers_articulation_and_improves_rigid_initialization(model):
    observations, truth = generate_sequence(model, frames=1, points_per_camera=0,
                                             joint_noise_m=0, seed=3)
    observation = observations[0]
    orient, translation = rigid_initialize(model, observation, torch.as_tensor(truth["beta"], dtype=model.dtype))
    _, initial_joints = _forward(model, np.zeros(45), truth["beta"], orient, translation)
    initial_error = _mean_error(initial_joints, truth["joints_m"][0])
    state = ManoFitter(model, FitConfig(mode="joint", iterations=100), beta=truth["beta"]).fit(observation)
    fitted_error = _mean_error(state.joints_m, truth["joints_m"][0])
    assert initial_error > .01, "The task must require articulation beyond a rigid alignment"
    assert fitted_error < .002
    assert fitted_error < .2 * initial_error
    assert state.valid and state.status == "tracked"
    assert state.quality["iterations"] > 10
    assert np.linalg.norm(state.theta) > .1
    np.testing.assert_allclose(state.translation_m, truth["translation_m"][0], atol=.003)
    vertices, joints = _forward(model, state.theta, state.beta, state.global_orient, state.translation_m)
    np.testing.assert_allclose(state.vertices_m, vertices, atol=1e-7)
    np.testing.assert_allclose(state.joints_m, joints, atol=1e-7)


def test_visible_surface_fitting_improves_noisy_joints_and_rejects_depth_outliers(model):
    observations, truth = generate_sequence(model, frames=1, points_per_camera=32,
                                             noise_m=.001, joint_noise_m=.004, seed=3)
    observation = observations[0]
    views = []
    for view in observation.views:
        # Same rays, but 20 cm behind their observed front surface. These must
        # be excluded by depth compatibility, rather than attracting the mesh.
        rays = view.points_m[:8] - view.camera_center_m
        outliers = view.points_m[:8] + .20 * rays / np.linalg.norm(rays, axis=1, keepdims=True)
        views.append(SurfaceView(np.vstack((view.points_m, outliers)), view.camera_center_m, view.camera_id))
    corrupted = Observation(observation.timestamp_s, observation.joints_m, observation.confidence, tuple(views))
    fitter = ManoFitter(model, FitConfig(mode="hybrid", iterations=100), beta=truth["beta"])
    state = fitter.fit(corrupted)
    joint_only = ManoFitter(model, FitConfig(mode="joint", iterations=100), beta=truth["beta"]).fit(observation)
    measured_error = _mean_error(observation.joints_m, truth["joints_m"][0])
    fitted_error = _mean_error(state.joints_m, truth["joints_m"][0])
    assert state.valid
    assert fitted_error < .005
    assert fitted_error < .8 * measured_error
    assert fitted_error < _mean_error(joint_only.joints_m, truth["joints_m"][0])
    assert state.quality["terms"]["surface"] > 0
    assert len(state.quality["surface_coverage"]) == 2
    assert all(.6 <= coverage <= .81 for coverage in state.quality["surface_coverage"])


@pytest.mark.parametrize("confidence", [1., .01])
def test_joint_outlier_has_bounded_influence(model, confidence):
    observations, truth = generate_sequence(model, frames=1, points_per_camera=0,
                                             joint_noise_m=.002, seed=4)
    joints, weights = observations[0].joints_m.copy(), observations[0].confidence.copy()
    joints[8] += [.16, -.12, .10]
    weights[8] = confidence
    observation = Observation(0, joints, weights)
    state = ManoFitter(model, FitConfig(mode="joint", iterations=100), beta=truth["beta"]).fit(observation)
    assert state.valid
    assert _mean_error(state.joints_m, truth["joints_m"][0]) < .007
    # An accepted result should still refuse to chase a single corrupted tip.
    assert np.linalg.norm(state.joints_m[8] - joints[8]) > .15


def test_missing_joints_are_masked_and_reconstructed_from_remaining_constraints(model):
    observations, truth = generate_sequence(model, frames=1, points_per_camera=0,
                                             joint_noise_m=.002, seed=4)
    missing = np.array([3, 7, 11, 15, 19])
    joints = observations[0].joints_m.copy()
    joints[missing] = np.nan
    observation = Observation(0, joints, np.ones(21))
    assert np.all(observation.confidence[missing] == 0)
    assert np.isfinite(observation.joints_m).all()
    state = ManoFitter(model, FitConfig(mode="joint", iterations=100), beta=truth["beta"]).fit(observation)
    assert state.valid
    assert _mean_error(state.joints_m, truth["joints_m"][0]) < .006
    assert _mean_error(state.joints_m[missing], truth["joints_m"][0, missing]) < .008


def test_insufficient_joints_clear_history_and_do_not_replay_a_valid_frame(model):
    observations, truth = generate_sequence(model, frames=2, points_per_camera=0,
                                             joint_noise_m=0)
    fitter = ManoFitter(model, FitConfig(mode="temporal", iterations=60), beta=truth["beta"])
    first = fitter.fit(observations[0])
    assert first.valid and len(fitter.history) == 1
    missing = observations[1].joints_m.copy()
    missing[5:] = np.nan
    observation = Observation(observations[1].timestamp_s, missing, np.ones(21))
    state = fitter.fit(observation)
    assert not state.valid and state.status == "insufficient_joints"
    assert fitter.history == []
    assert state.timestamp_s == observation.timestamp_s
    assert state.timestamp_s != first.timestamp_s
    assert state is not first
    assert not np.array_equal(state.joints_m, first.joints_m)
    assert np.isfinite(state.joints_m).all()
    np.testing.assert_array_equal(state.theta, np.zeros(45))
    all_missing = fitter.fit(Observation(.1, np.full((21, 3), np.nan), np.ones(21)))
    assert not all_missing.valid
    assert np.isfinite(all_missing.vertices_m).all()
    assert fitter.history == []


def test_rejected_fit_clears_history(model):
    observations, truth = generate_sequence(model, frames=2, points_per_camera=0,
                                             joint_noise_m=0)
    fitter = ManoFitter(model, FitConfig(mode="joint", iterations=60), beta=truth["beta"])
    assert fitter.fit(observations[0]).valid
    rng = np.random.default_rng(41)
    inconsistent = truth["joints_m"][1] + rng.normal(0, .15, (21, 3))
    state = fitter.fit(Observation(observations[1].timestamp_s, inconsistent, np.ones(21)))
    assert not state.valid and state.status == "rejected"
    assert state.quality["joint_residual_mm"] > 40
    assert fitter.history == []


def test_temporal_fit_reduces_static_hand_jitter_without_large_bias(model):
    _, truth = generate_sequence(model, frames=1, points_per_camera=0)
    rng = np.random.default_rng(19)
    observations = [Observation(index / 30, truth["joints_m"][0] + rng.normal(0, .004, (21, 3)), np.ones(21))
                    for index in range(7)]
    reconstructions = {}
    for mode in ("joint", "temporal"):
        fitter = ManoFitter(model, FitConfig(mode=mode, iterations=80), beta=truth["beta"])
        states = [fitter.fit(observation) for observation in observations]
        assert all(state.valid for state in states)
        reconstructions[mode] = np.stack([state.joints_m for state in states])
        assert len(fitter.history) == 2
        if mode == "temporal":
            assert states[0].quality["terms"]["temporal"] == 0
            assert states[-1].quality["terms"]["temporal"] > 0
            assert states[-1].quality["terms"]["acceleration"] > 0
    jitter = {mode: np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean()
              for mode, joints in reconstructions.items()}
    assert jitter["temporal"] < .95 * jitter["joint"]
    assert _mean_error(reconstructions["temporal"], truth["joints_m"][0]) < .006


def test_timestamps_are_strict_and_long_gaps_restart_without_temporal_penalty(model):
    observations, truth = generate_sequence(model, frames=2, points_per_camera=0,
                                             joint_noise_m=0)
    config = FitConfig(mode="temporal", iterations=60, max_gap_s=.25)
    fitter = ManoFitter(model, config, beta=truth["beta"])
    assert fitter.fit(observations[0]).valid
    for timestamp in (0., -.01):
        with pytest.raises(ValueError, match="timestamps.*increase"):
            fitter.fit(Observation(timestamp, observations[0].joints_m, np.ones(21)))
    after_gap = Observation(.7, observations[1].joints_m, np.ones(21))
    restarted = fitter.fit(after_gap)
    fresh = ManoFitter(model, config, beta=truth["beta"]).fit(after_gap)
    assert restarted.valid and len(fitter.history) == 1
    assert restarted.quality["terms"]["temporal"] == 0
    assert restarted.quality["terms"]["acceleration"] == 0
    np.testing.assert_allclose(restarted.joints_m, fresh.joints_m, atol=1e-7)
    np.testing.assert_allclose(restarted.theta, fresh.theta, atol=1e-7)
    fitter.reset()
    assert fitter.history == []


def test_temporal_terms_use_elapsed_time_and_irregular_constant_velocity_has_no_acceleration():
    model = ManoModel(make_fixture_model(), allow_test_model=True, dtype=torch.float64)
    observations, truth = generate_sequence(model, frames=1, points_per_camera=0,
                                             joint_noise_m=0)
    fitter = ManoFitter(model, FitConfig(mode="temporal"), beta=truth["beta"])
    current_time = .10
    observation = Observation(current_time, observations[0].joints_m, np.ones(21))
    arguments = [torch.as_tensor(truth[key] if key == "beta" else truth[key][0], dtype=model.dtype)
                 for key in ("theta", "beta", "global_orient", "translation_m")]
    prepared = fitter._prepare(observation)

    def previous(time, displacement):
        return FitState(time, truth["theta"][0], truth["beta"], truth["global_orient"][0],
                        truth["translation_m"][0] - displacement,
                        truth["joints_m"][0] - displacement,
                        truth["vertices_m"][0] - displacement, True, "tracked")

    penalties = []
    for elapsed in (.02, .04):
        history = [previous(current_time - elapsed, np.array([.002, 0, 0]))]
        _, terms, _, _ = fitter._loss(*arguments, observation, prepared, history=history)
        penalties.append(float(terms["temporal"]))
    assert penalties[0] == pytest.approx(4 * penalties[1], rel=1e-10)
    velocity = np.array([.02, -.01, .005])
    history = [previous(0., velocity * .10), previous(.06, velocity * .04)]
    _, terms, _, _ = fitter._loss(*arguments, observation, prepared, history=history)
    assert float(terms["temporal"]) > 0
    assert float(terms["acceleration"]) < 1e-20


def test_observation_rejects_invalid_inputs():
    joints = np.zeros((21, 3))
    for timestamp, weights in ((float("nan"), np.ones(21)), (0, np.full(21, -.1)),
                               (0, np.full(21, 1.1)), (0, np.full(21, np.inf))):
        with pytest.raises(ValueError):
            Observation(timestamp, joints, weights)
    with pytest.raises(ValueError, match="21"):
        Observation(0, np.zeros((20, 3)), np.ones(21))


@pytest.mark.parametrize("kwargs", [dict(mode="unknown"), dict(iterations=0), dict(learning_rate=0),
                                    dict(surface_scale_m=0), dict(temporal_weight=-1),
                                    dict(temporal_weight=-1.), dict(max_gap_s=float("inf"))])
def test_fit_config_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        FitConfig(**kwargs)


def test_shape_calibration_requires_observable_frames(model):
    missing = Observation(0, np.full((21, 3), np.nan), np.ones(21))
    for observations in ([], [missing]):
        with pytest.raises(ValueError, match="valid observations"):
            calibrate_shape(model, observations)


@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") != "1",
                    reason="Optional shared-shape calibration: set RUN_SLOW_TESTS=1 (numerical fixture, NOT MANO)")
def test_shape_calibration_recovers_one_common_shape_across_frames(model):
    observations, truth = generate_sequence(model, frames=3, points_per_camera=48,
                                             noise_m=.0005, joint_noise_m=.001, seed=24)
    missing = Observation(.5, np.full((21, 3), np.nan), np.ones(21))
    beta, quality = calibrate_shape(model, observations + [missing], iterations=180, mode="hybrid")
    assert beta.shape == (10,) and np.isfinite(beta).all()
    assert quality["frames"] == 3 and quality["iterations"] == 180
    assert np.isfinite(quality["loss"])
    zero_errors, calibrated_errors = [], []
    for frame in range(3):
        zero, _ = _forward(model, truth["theta"][frame], np.zeros(10),
                           truth["global_orient"][frame], truth["translation_m"][frame])
        calibrated, _ = _forward(model, truth["theta"][frame], beta,
                                 truth["global_orient"][frame], truth["translation_m"][frame])
        zero_errors.append(_mean_error(zero, truth["vertices_m"][frame]))
        calibrated_errors.append(_mean_error(calibrated, truth["vertices_m"][frame]))
    assert np.mean(calibrated_errors) < .8 * np.mean(zero_errors)
    fitter = ManoFitter(model, FitConfig(mode="joint", iterations=60), beta=beta)
    for observation in observations:
        state = fitter.fit(observation)
        assert state.valid
        np.testing.assert_allclose(state.beta, beta, atol=1e-7)
