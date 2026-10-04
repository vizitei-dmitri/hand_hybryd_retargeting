import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from night_smoothing_rules import choose_smoke, early_stop, size_action_penalty
from smoothing_telemetry import Telemetry


def metric(near=.867, goals=9.7, drop=.617, held=.859):
    return {"episode_reward": 139.1, "held_success_rate": held, "drop_rate": drop,
            "goals_completed_per_episode": goals, "action_std": .5,
            "smoothing": {"command": {"near_limit_fraction": near}, "action_penalty_to_task": .18,
                          "mean_action_square": .9066, "task_abs": .255, "action_penalty_scale": .002,
                          "action_square_sum_per_episode": 893.0}}


def test_sizing_measured_trajectory_and_zero_sanity():
    selected = size_action_penalty(metric(), {"episode_reward": 5.5})
    assert selected["selected"]["scale"] == .05
    assert .15 < selected["selected"]["fraction_of_task_abs"] < .35
    assert size_action_penalty(metric(), {"episode_reward": 200})["selected"] is None


def test_low_std_preferred_only_when_competent():
    assert choose_smoke({"H": [metric()], "L": [metric(near=.85)]}, metric())[0] == "L"
    assert choose_smoke({"H": [metric()], "L": [metric(goals=1)]}, metric())[0] == "H"
    assert choose_smoke({"H": [metric(goals=1)], "L": [metric(goals=1)]}, metric())[0] is None


def test_persistence_and_saturation_failure():
    assert early_stop([metric(goals=2)], metric(), 500) is None
    assert early_stop([metric(goals=2), metric(goals=2)], metric(), 1000).startswith("goals_")
    assert early_stop([metric(), metric()], metric(), 1500) == "saturation_did_not_move_after_1500"
    assert early_stop([metric(near=.7), metric(near=.7)], metric(), 1500) == "action_smoothing_succeeded_grasp_survival_did_not"


def test_telemetry_excludes_reset_episodes_and_distinguishes_clipped_actions():
    raw = NS(num_envs=2, device="cpu", active_joint_ids=[0, 1], step_dt=.1,
             cfg=NS(active_action_joints=["a", "b"], sim=NS(dt=.05), action_penalty_scale=.002, action_rate_penalty_scale=.001),
             hand=NS(data=NS(joint_pos=torch.zeros(2, 2), joint_vel=torch.ones(2, 2))),
             cube=NS(data=NS(root_lin_vel_w=torch.zeros(2, 3))),
             tip_in_contact=torch.ones(2, 5, dtype=torch.bool), grasp_quality_value=torch.ones(2),
             palm_contact_force=torch.zeros(2), episode_length_buf=torch.ones(2, dtype=torch.long),
             reset_terminated=torch.tensor([True, False]), reset_time_outs=torch.zeros(2, dtype=torch.bool))
    t = Telemetry(raw)
    t.physics_sample()
    t.mu = torch.tensor([[4., -3.], [.5, -.5]])
    raw.actions = t.mu.clamp(-1, 1); raw.action_delta = raw.actions.clone()
    raw.reward_terms = {"orientation_state_reward": torch.ones(2)*.1,
                        "action_penalty": -.002*raw.actions.square().mean(-1)}
    t.reward_sample()
    raw.hand.data.joint_pos[0] = 1000  # Reset teleport of completed env must be ignored.
    t.physics_sample()
    t.mu = torch.tensor([[1000., 1000.], [.5, -.5]])
    raw.actions = t.mu.clamp(-1,1)
    t.reward_sample()
    result = t.summary()
    assert result["raw_mu"]["samples"] == 6
    assert result["raw_mu"]["max"] == 4
    assert result["command"]["max"] == 1
    assert result["qdot_fd"]["max"] == 0
    assert result["reward_terms"]["orientation_state_reward"]["sum_per_episode"] == pytest.approx(.15)
