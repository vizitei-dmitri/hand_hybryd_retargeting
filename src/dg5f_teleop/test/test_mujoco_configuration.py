import numpy as np
import pytest

import dg5f_teleop.mujoco_bridge_node as bridge


class _Model:
    def __init__(self):
        self.actuator_gainprm = np.zeros((3, 10), dtype=float)
        self.actuator_biasprm = np.zeros((3, 10), dtype=float)
        self.geom_group = np.array([1, 1, 1, 2, 0])
        self.geom_contype = np.ones(5, dtype=int)
        self.geom_conaffinity = np.ones(5, dtype=int)


def test_configure_position_actuators_sets_affine_pd_coefficients():
    model = _Model()

    bridge.configure_position_actuators(model, np.array([0, 2]), 40.0, 0.5)

    assert np.all(model.actuator_gainprm[[0, 2], 0] == 40.0)
    assert np.all(model.actuator_biasprm[[0, 2], 1] == -40.0)
    assert np.all(model.actuator_biasprm[[0, 2], 2] == -0.5)
    assert model.actuator_gainprm[1, 0] == 0.0


@pytest.mark.parametrize("kp,kd", [(0.0, 0.5), (np.nan, 0.5), (40.0, -0.1)])
def test_configure_position_actuators_rejects_invalid_gains(kp, kd):
    with pytest.raises(ValueError):
        bridge.configure_position_actuators(_Model(), np.array([0]), kp, kd)


def test_tip_only_collision_keeps_environment_and_tip_contacts(monkeypatch):
    model = _Model()
    names = [
        "rl_dg_1_1_collision",
        "rl_dg_1_tip_collision",
        "rl_dg_2_tip_collision",
        "visual",
        "floor",
    ]
    monkeypatch.setattr(
        bridge.mujoco,
        "mj_id2name",
        lambda _model, _kind, geom_id: names[geom_id],
    )

    hand, tips = bridge.configure_hand_self_collision(model, "tip_only")

    assert hand.tolist() == [0, 1, 2]
    assert tips.tolist() == [1, 2]
    assert model.geom_contype.tolist() == [2, 4, 4, 1, 1]
    assert model.geom_conaffinity.tolist() == [1, 5, 5, 1, 1]


def test_full_collision_does_not_change_model():
    model = _Model()
    before = (model.geom_contype.copy(), model.geom_conaffinity.copy())

    bridge.configure_hand_self_collision(model, "full")

    assert np.array_equal(model.geom_contype, before[0])
    assert np.array_equal(model.geom_conaffinity, before[1])


def test_invalid_collision_mode_is_rejected():
    with pytest.raises(ValueError, match="self_collision_mode"):
        bridge.configure_hand_self_collision(_Model(), "sometimes")
