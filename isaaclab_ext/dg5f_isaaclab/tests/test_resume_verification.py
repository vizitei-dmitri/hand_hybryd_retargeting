from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = spec_from_file_location("resume_verification", Path(__file__).resolve().parents[1] / "scripts/rsl_rl/resume_verification.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_verifies_actual_state_and_rejects_reward_change(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    for folder in (source, output):
        (folder / "params").mkdir(parents=True)
        (folder / "params/env.yaml").write_text(f"reward: 0.3\nlog_dir: {folder}\n")
    model = {"std": torch.full((19,), .5), "critic.0.weight": torch.ones(2, 2)}
    optimizer = {"param_groups": [{"lr": .0002}], "state": {0: {"exp_avg": torch.ones(2)}}}
    torch.save({"model_state_dict": model, "optimizer_state_dict": optimizer, "iter": 3599}, source / "model.pt")
    runner = SimpleNamespace(current_learning_iteration=3600, alg=SimpleNamespace(
        policy=SimpleNamespace(state_dict=lambda: model), optimizer=SimpleNamespace(state_dict=lambda: optimizer),
        learning_rate=.0002, entropy_coef=.001))
    result = module.verify_resume_start(runner, source / "model.pt", output, .001)
    assert result["std_mean"] == .5 and result["next_iteration"] == 3600
    (output / "params/env.yaml").write_text("reward: 0.9\n")
    with pytest.raises(ValueError, match="environment_including_reward/reward"):
        module.verify_resume_start(runner, source / "model.pt", output, .001)


@pytest.mark.parametrize("part", ["critic", "Adam"])
def test_detects_tensor_mismatch(part):
    with pytest.raises(ValueError, match=part):
        module.assert_same_state({part: torch.ones(3)}, {part: torch.zeros(3)})
