import os

import numpy as np
import pytest
from lerobot_robot_dg5f.constants import BROKEN_PINKY_INDEX

from conftest import REPO_ROOT
from lerobot_robot_ur10_dg5f.synergies import HandSynergies, hand_actions_from_dataset


def synthetic_postures(rng, n=500):
    """Postures from three known directions plus noise; the pinky base stays at zero."""
    basis = np.linalg.qr(rng.normal(size=(20, 3)))[0].T
    basis[:, BROKEN_PINKY_INDEX] = 0.0
    basis = np.linalg.qr(basis.T)[0].T
    weights = rng.normal(scale=[30.0, 15.0, 8.0], size=(n, 3))
    data = 10.0 + weights @ basis + rng.normal(scale=0.3, size=(n, 20))
    data[:, BROKEN_PINKY_INDEX] = 0.0
    return data, basis


def test_recovers_known_subspace():
    data, basis = synthetic_postures(np.random.default_rng(0))
    synergies = HandSynergies.fit(data, k=3)
    # Singular values of the overlap matrix are the cosines of the principal angles.
    cosines = np.linalg.svd(synergies.components @ basis.T, compute_uv=False)
    assert np.all(cosines > 0.999)
    assert synergies.explained_variance_ratio.sum() > 0.99


def test_offsets_roundtrip_and_pinky_has_no_weight():
    data, _ = synthetic_postures(np.random.default_rng(1))
    synergies = HandSynergies.fit(data, k=3)
    dz = np.array([1.0, -2.0, 0.5])
    assert np.allclose(synergies.project_offset(synergies.to_joint_offset(dz)), dz)
    assert np.allclose(synergies.components[:, BROKEN_PINKY_INDEX], 0.0, atol=1e-9)


def test_save_and_load(tmp_path):
    data, _ = synthetic_postures(np.random.default_rng(2))
    synergies = HandSynergies.fit(data, k=4)
    synergies.save(tmp_path / "synergies.npz")
    loaded = HandSynergies.load(tmp_path / "synergies.npz")
    assert np.allclose(loaded.components, synergies.components)
    assert loaded.k == 4


def test_reads_hand_actions_from_recorded_dataset():
    root = os.environ.get("DG5F_MOCK_DATASET", REPO_ROOT / "lerobot_datasets" / "mock_dataset_and_debug_20260908")
    if not (os.path.isdir(root)):
        pytest.skip(f"Dataset not found: {root}")
    actions = hand_actions_from_dataset(root)
    assert actions.ndim == 2 and actions.shape[1] == 20 and len(actions) > 0
    assert np.all(np.isfinite(actions))
