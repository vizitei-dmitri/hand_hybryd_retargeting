from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

spec = spec_from_file_location("make_warm_start", Path(__file__).resolve().parents[1] / "scripts/make_warm_start.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("key", ["std", "log_std"])
def test_preserves_learned_value_policy_normalizers_and_optimizer(key):
    source = {
        "model_state_dict": {key: torch.ones(19) * 5,
                             "actor.0.weight": torch.randn(4, 3),
                             "critic.0.weight": torch.randn(4, 3),
                             "actor_obs_normalizer._mean": torch.randn(3),
                             "critic_obs_normalizer._var": torch.rand(3)},
        "optimizer_state_dict": {"param_groups": [{"lr": 0.0001, "params": [0]}],
                                 "state": {0: {"exp_avg": torch.randn(19), "step": torch.tensor(9000)}}},
        "iter": 3599, "infos": {"custom": "keep"},
    }
    result = module.reset_std_only(source, .5)
    for name, value in source["model_state_dict"].items():
        if name != key:
            assert torch.equal(result["model_state_dict"][name], value)
    assert torch.equal(source["model_state_dict"][key], torch.ones(19) * 5)
    actual = result["model_state_dict"][key]
    assert torch.allclose(actual.exp() if key == "log_std" else actual, torch.full((19,), .5))
    assert result["iter"] == 3599 and result["infos"] == source["infos"]
    assert result["optimizer_state_dict"]["param_groups"] == source["optimizer_state_dict"]["param_groups"]
    for name, value in source["optimizer_state_dict"]["state"][0].items():
        assert torch.equal(result["optimizer_state_dict"]["state"][0][name], value)


@pytest.mark.parametrize("std", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_std(std):
    with pytest.raises(ValueError):
        module.reset_std_only({"model_state_dict": {"std": torch.ones(19)}}, std)


@pytest.mark.parametrize("model", [{}, {"std": torch.ones(19), "log_std": torch.ones(19)}])
def test_rejects_ambiguous_noise_parameters(model):
    with pytest.raises(ValueError):
        module.reset_std_only({"model_state_dict": model}, .5)
