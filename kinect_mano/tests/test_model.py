"""Numerical LBS contracts; fixture checks are not MANO accuracy validation."""
import json
import os

import numpy as np
import pytest
import torch

from kinect_mano.model import JOINT_ORDER, ManoModel, axis_angle_to_matrix
from kinect_mano.synthetic import make_fixture_model


torch.set_num_threads(1)


@pytest.fixture
def arrays():
    return make_fixture_model(seed=12)


@pytest.fixture
def model(arrays):
    return ManoModel(arrays, allow_test_model=True, dtype=torch.float64)


def test_independent_fixture_schema_seed_and_explicit_gate(arrays, tmp_path):
    metadata = json.loads(str(arrays["metadata_json"]))
    assert metadata["model_type"] == "numerical_fixture"
    assert metadata["handedness"] == "right"
    assert metadata["units"] == "m"
    assert "NOT MANO" in metadata["authoring"]
    for key, value in make_fixture_model(seed=12).items():
        np.testing.assert_array_equal(arrays[key], value)
        assert value.dtype != object
    assert not np.array_equal(arrays["posedirs"], make_fixture_model(seed=13)["posedirs"])
    assert arrays["v_template"].shape == (778, 3)
    assert arrays["shapedirs"].shape == (778, 3, 10)
    assert arrays["posedirs"].shape == (778, 3, 135)
    assert np.linalg.matrix_rank(arrays["shapedirs"].reshape(-1, 10)) == 10
    np.testing.assert_allclose(arrays["weights"].sum(1), 1, atol=1e-14)
    np.testing.assert_allclose(arrays["J_regressor"].sum(1), 1, atol=1e-14)
    assert np.all(arrays["weights"] >= 0)
    triangles = arrays["v_template"][arrays["faces"]]
    areas2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0]), axis=1)
    assert np.all(areas2 > 1e-9)
    assert len(np.unique(arrays["faces"])) == 778
    with pytest.raises(ValueError, match="allow_test_model"):
        ManoModel(arrays)
    path = tmp_path / "nested" / "numerical_fixture.npz"
    saved = make_fixture_model(path, seed=12)
    with np.load(path, allow_pickle=False) as archive:
        for key in saved:
            np.testing.assert_array_equal(archive[key], arrays[key])
    with pytest.raises(ValueError, match="allow_test_model"):
        ManoModel(path)
    loaded = ManoModel(path, allow_test_model=True, dtype=torch.float64)
    assert loaded.model_type == "numerical_fixture"
    assert len(loaded.sha256) == 64
    torch.testing.assert_close(loaded.v_template, torch.from_numpy(arrays["v_template"]))
    with pytest.raises(FileExistsError):
        make_fixture_model(path)


def test_flat_pose_matches_rest_vertices_and_ordered_joints(model, arrays):
    vertices, joints = model(torch.zeros(45, dtype=model.dtype), torch.zeros(10, dtype=model.dtype),
                             torch.zeros(3, dtype=model.dtype), torch.zeros(3, dtype=model.dtype))
    rest = arrays["J_regressor"] @ arrays["v_template"]
    expected_vertices = arrays["v_template"] - rest[0]
    expected_joints = np.vstack((rest - rest[0], expected_vertices[arrays["tip_vertex_ids"]]))[JOINT_ORDER]
    np.testing.assert_allclose(vertices[0], expected_vertices, atol=1e-14)
    np.testing.assert_allclose(joints[0], expected_joints, atol=1e-14)
    assert vertices.shape == (1, 778, 3)
    assert joints.shape == (1, 21, 3)
    assert .10 < np.ptp(expected_vertices[:, 1]) < .25


def test_metric_scale_equivariance_and_millimetre_input_rejected(model, arrays):
    scale = 1.7
    scaled = dict(arrays)
    for key in ("v_template", "shapedirs", "posedirs"):
        scaled[key] = arrays[key] * scale
    larger = ManoModel(scaled, allow_test_model=True, dtype=model.dtype)
    theta = torch.linspace(-.15, .35, 45, dtype=model.dtype)
    beta = torch.linspace(-.3, .5, 10, dtype=model.dtype)
    orient = torch.tensor([.2, -.3, .1], dtype=model.dtype)
    translation = torch.tensor([.01, -.02, .55], dtype=model.dtype)
    for actual, expected in zip(larger(theta, beta, orient, translation * scale),
                                model(theta, beta, orient, translation)):
        torch.testing.assert_close(actual, expected * scale, atol=1e-12, rtol=1e-12)
    wrong_units = dict(arrays, v_template=arrays["v_template"] * 1000)
    with pytest.raises(ValueError, match="extent|units"):
        ManoModel(wrong_units, allow_test_model=True)


def test_nonzero_pose_rigid_equivariance_and_batch_broadcast(model):
    theta = torch.stack((torch.linspace(-.1, .4, 45, dtype=model.dtype),
                         torch.linspace(.2, -.1, 45, dtype=model.dtype)))
    beta = torch.linspace(-.2, .4, 10, dtype=model.dtype)
    orient = torch.tensor([[.3, -.2, .4], [-.2, .1, -.35]], dtype=model.dtype)
    translation = torch.tensor([[.1, -.2, .7], [-.1, .03, .5]], dtype=model.dtype)
    rotation = axis_angle_to_matrix(orient)
    rest_outputs = model(theta, beta, torch.zeros(3, dtype=model.dtype), torch.zeros(3, dtype=model.dtype))
    actual_outputs = model(theta, beta, orient, translation)
    for rest, actual in zip(rest_outputs, actual_outputs):
        torch.testing.assert_close(actual, rest @ rotation.transpose(-1, -2) + translation[:, None],
                                   atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_outputs[1][:, 0], translation, atol=1e-14, rtol=0)
    for frame in range(2):
        single = model(theta[frame], beta, orient[frame], translation[frame])
        for batched, separate in zip(actual_outputs, single):
            torch.testing.assert_close(batched[frame], separate[0])


def test_articulation_changes_fingertips_and_preserves_skeleton_bone_lengths(model):
    zeros = lambda n: torch.zeros(n, dtype=model.dtype)
    _, rest = model(zeros(45), zeros(10), zeros(3), zeros(3))
    theta = zeros(45).reshape(15, 3)
    theta[:, 0] = .45
    _, bent = model(theta, zeros(10), zeros(3), zeros(3))
    assert torch.linalg.vector_norm(bent[0, 8] - rest[0, 8]) > .02
    for start in (1, 5, 9, 13, 17):
        # Tip positions are skinned surface vertices; test the skeletal bones.
        for first in (start, start + 1):
            torch.testing.assert_close(torch.linalg.vector_norm(bent[:, first + 1] - bent[:, first]),
                                       torch.linalg.vector_norm(rest[:, first + 1] - rest[:, first]),
                                       atol=1e-12, rtol=1e-12)


def _check_flat_gradients(model):
    inputs = tuple(torch.zeros(size, dtype=model.dtype, device=model.device, requires_grad=True)
                   for size in (45, 10, 3, 3))

    def selected_outputs(*arguments):
        vertices, joints = model(*arguments)
        return torch.cat((vertices[:, model.tip_vertex_ids].reshape(-1), joints.reshape(-1)))

    assert torch.autograd.gradcheck(selected_outputs, inputs, eps=1e-6, atol=2e-6,
                                    rtol=2e-4, fast_mode=True)
    output = selected_outputs(*inputs)
    coefficients = torch.linspace(.1, 1.3, output.numel(), dtype=model.dtype, device=model.device)
    gradients = torch.autograd.grad((output * coefficients).sum(), inputs)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.linalg.vector_norm(gradient) > 1e-5 for gradient in gradients)


def test_flat_pose_has_correct_nonzero_gradients(model):
    _check_flat_gradients(model)


@pytest.mark.skipif(not os.environ.get("MANO_MODEL"),
                    reason="OFFICIAL MANO validation skipped: set MANO_MODEL to a licensed converted NPZ; numerical fixture is NOT MANO")
def test_official_mano_flat_pose_and_gradients_only_with_licensed_model():
    # Deliberately do not enable the fixture gate for official-model validation.
    official = ManoModel(os.environ["MANO_MODEL"], dtype=torch.float64)
    assert official.model_type == "MANO"
    _check_flat_gradients(official)
    zeros = lambda n: torch.zeros(n, dtype=official.dtype)
    vertices, joints = official(zeros(45), zeros(10), zeros(3), zeros(3))
    wrist = (official.J_regressor @ official.v_template)[0]
    torch.testing.assert_close(vertices[0], official.v_template - wrist, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(joints[0, 0], zeros(3), atol=1e-12, rtol=0)
