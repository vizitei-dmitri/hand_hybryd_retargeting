"""Closest-surface gradients and first-hit visibility, including occlusion."""
import json

import numpy as np
import pytest
import torch

from kinect_mano.model import ManoModel
from kinect_mano.surface import SurfaceView, point_triangle_squared_distance, raycast_front_faces, robust_huber
from kinect_mano.synthetic import generate_sequence, make_fixture_model


torch.set_num_threads(1)


@pytest.mark.parametrize("point,expected", [
    ([.2, .3, .4], .16),          # Projection inside the triangle.
    ([.6, .6, .2], .06),          # Closest point on the diagonal edge.
    ([-.2, -.3, .4], .29),        # Closest point is the origin vertex.
])
def test_exact_point_triangle_distance_and_both_input_gradients(point, expected):
    points = torch.tensor([point], dtype=torch.float64, requires_grad=True)
    triangles = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]],
                             dtype=torch.float64, requires_grad=True)
    distance = point_triangle_squared_distance(points, triangles)
    torch.testing.assert_close(distance, torch.tensor([expected], dtype=points.dtype))
    assert torch.autograd.gradcheck(point_triangle_squared_distance, (points, triangles),
                                    eps=1e-6, atol=1e-6, rtol=1e-4)
    gradients = torch.autograd.grad(distance.sum(), (points, triangles))
    assert all(torch.isfinite(value).all() for value in gradients)
    # Translating both inputs cannot change the distance.
    torch.testing.assert_close(gradients[0] + gradients[1].sum(1), torch.zeros_like(points))


def _stacked_triangles():
    vertices = torch.tensor([[-1., -1., 1.], [1., -1., 1.], [0., 1., 1.],
                             [-1., -1., 1.02], [1., -1., 1.02], [0., 1., 1.02]], dtype=torch.float64)
    # Put the rear triangle first to catch nearest-index rather than first-hit logic.
    faces = torch.tensor([[3, 4, 5], [0, 1, 2]])
    return vertices, faces, torch.zeros(3, dtype=vertices.dtype)


def test_self_occlusion_selects_front_triangle_even_for_rear_surface_observation():
    vertices, faces, camera = _stacked_triangles()
    points = torch.tensor([[0., 0., 1.02], [.1, 0., 1.01]], dtype=vertices.dtype)
    selected = raycast_front_faces(points, vertices, faces, camera, max_distance_m=.04)
    assert selected.tolist() == [1, 1]
    assert not selected.requires_grad
    # Incompatible first-hit depth must not silently match the hidden back face.
    assert raycast_front_faces(points[:1], vertices, faces, camera,
                               max_distance_m=.005).tolist() == [-1]
    # Visibility is independent of mesh winding.
    torch.testing.assert_close(raycast_front_faces(points, vertices, faces.flip(1), camera), selected)


def test_raycast_rejects_misses_behind_camera_and_incompatible_depth_and_handles_chunks():
    vertices, faces, camera = _stacked_triangles()
    points = torch.tensor([[0., 0., 1.], [4., 0., 1.], [0., 0., -1.],
                            [0., 0., 1.2], [0., 0., 0.]], dtype=vertices.dtype)
    expected = torch.tensor([1, -1, -1, -1, -1])
    for chunk_size in (1, 2, 128):
        torch.testing.assert_close(raycast_front_faces(points, vertices, faces, camera,
                                                       chunk_size=chunk_size), expected)
    empty = raycast_front_faces(points[:0], vertices, faces, camera)
    assert empty.shape == (0,)
    assert empty.dtype == torch.long


def test_surface_view_requires_finite_world_points_and_camera_center():
    empty = SurfaceView(np.empty((0, 3)), [0, 0, 0], "empty")
    assert empty.points_m.shape == (0, 3)
    for points, center in (([[np.nan, 0, 0]], [0, 0, 0]), ([[0, 0]], [0, 0, 0]),
                           ([[0, 0, 1]], [0, np.inf, 0]), ([[0, 0, 1]], [0, 0])):
        with pytest.raises(ValueError):
            SurfaceView(points, center)


def test_huber_limits_outlier_influence_without_losing_inlier_gradient():
    residual = torch.tensor([.5, 2., 20.], dtype=torch.float64, requires_grad=True)
    loss = robust_huber(residual)
    torch.testing.assert_close(loss, residual.new_tensor([.125, 1.5, 19.5]))
    gradient, = torch.autograd.grad(loss.sum(), residual)
    torch.testing.assert_close(gradient, residual.new_tensor([.5, 1., 1.]))


def test_synthetic_views_are_actual_frontmost_mesh_hits_with_shared_shape():
    model = ManoModel(make_fixture_model(), allow_test_model=True, dtype=torch.float64)
    observations, truth = generate_sequence(model, frames=2, points_per_camera=48,
                                             noise_m=0, joint_noise_m=0, seed=7)
    assert json.loads(str(truth["metadata_json"]))["model_type"] == "numerical_fixture"
    assert truth["beta"].shape == (10,)
    assert truth["theta"].shape == (2, 45)
    assert truth["vertices_m"].shape == (2, 778, 3)
    assert truth["joints_m"].shape == (2, 21, 3)
    assert np.all(np.diff([observation.timestamp_s for observation in observations]) > 0)
    for frame, observation in enumerate(observations):
        np.testing.assert_array_equal(observation.joints_m, truth["joints_m"][frame])
        np.testing.assert_array_equal(observation.confidence, np.ones(21))
        assert len(observation.views) == 2
        assert len({view.camera_id for view in observation.views}) == 2
        vertices = torch.from_numpy(truth["vertices_m"][frame])
        for index, view in enumerate(observation.views):
            assert view.points_m.shape == (48, 3)
            np.testing.assert_array_equal(view.camera_center_m, truth["camera_centers_m"][index])
            points, camera = torch.from_numpy(view.points_m), torch.from_numpy(view.camera_center_m)
            faces = raycast_front_faces(points, vertices, model.faces, camera, max_distance_m=1e-7)
            assert (faces >= 0).all(), "Every noiseless sample must be at its ray's first mesh hit"
            distance2 = point_triangle_squared_distance(points, vertices[model.faces[faces]])
            assert distance2.max() < 1e-20
    reconstructed = model(*(torch.as_tensor(truth[key], dtype=model.dtype)
                            for key in ("theta", "beta", "global_orient", "translation_m")))
    for actual, key in zip(reconstructed, ("vertices_m", "joints_m")):
        np.testing.assert_allclose(actual.detach(), truth[key], atol=1e-14)


def test_synthetic_noise_and_seed_are_reproducible_and_do_not_change_truth():
    model = ManoModel(make_fixture_model(), allow_test_model=True, dtype=torch.float64)
    arguments = dict(frames=1, points_per_camera=64, noise_m=.001, joint_noise_m=.004, seed=9)
    noisy, truth = generate_sequence(model, **arguments)
    repeated, repeated_truth = generate_sequence(model, **arguments)
    clean, clean_truth = generate_sequence(model, **dict(arguments, noise_m=0, joint_noise_m=0))
    for key in truth:
        np.testing.assert_array_equal(truth[key], repeated_truth[key])
        np.testing.assert_array_equal(truth[key], clean_truth[key])
    np.testing.assert_array_equal(noisy[0].joints_m, repeated[0].joints_m)
    assert .002 < np.std(noisy[0].joints_m - clean[0].joints_m) < .006
    for first, second, exact in zip(noisy[0].views, repeated[0].views, clean[0].views):
        np.testing.assert_array_equal(first.points_m, second.points_m)
        assert .0006 < np.std(first.points_m - exact.points_m) < .0014


@pytest.mark.parametrize("kwargs", [dict(frames=0), dict(frames=1.5), dict(points_per_camera=-1),
                                    dict(points_per_camera=True), dict(noise_m=-.1),
                                    dict(joint_noise_m=float("nan"))])
def test_synthetic_argument_validation(kwargs):
    model = ManoModel(make_fixture_model(), allow_test_model=True)
    with pytest.raises(ValueError):
        generate_sequence(model, **kwargs)
