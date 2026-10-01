"""Opt-in verification of loaded state BEFORE the first continuation update."""
import hashlib
import json
from pathlib import Path

import torch
import yaml


def assert_same_state(expected, actual, path="state"):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(expected.cpu(), actual.detach().cpu()):
            raise ValueError(f"Resume mismatch: {path}")
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"Resume keys mismatch: {path}")
        for key in expected:
            assert_same_state(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise ValueError(f"Resume length mismatch: {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_same_state(left, right, f"{path}/{index}")
    elif expected != actual:
        raise ValueError(f"Resume mismatch: {path}: {expected!r} != {actual!r}")


def verify_resume_start(runner, checkpoint_path, log_dir, entropy):
    source = Path(checkpoint_path).resolve()
    destination = Path(log_dir)
    saved = torch.load(source, map_location="cpu", weights_only=False)
    actual = runner.alg.policy.state_dict()
    assert_same_state(saved["model_state_dict"], actual, "model")
    assert_same_state(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict(), "Adam")
    expected_iteration = int(saved["iter"]) + 1
    if runner.current_learning_iteration != expected_iteration:
        raise ValueError("Next iteration was not restored")
    rates = [group["lr"] for group in saved["optimizer_state_dict"]["param_groups"]]
    if any(rate != runner.alg.learning_rate for rate in rates):
        raise ValueError("PPO adaptive learning_rate was not restored")
    if runner.alg.entropy_coef != entropy:
        raise ValueError("Runtime entropy differs from configured entropy")
    # BaseLoader preserves the serialized configuration without executing YAML tags.
    configs = []
    for path in (source.parent / "params/env.yaml", destination / "params/env.yaml"):
        value = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        value.pop("log_dir", None)
        configs.append(value)
    assert_same_state(configs[0], configs[1], "environment_including_reward")
    std = (actual["std"] if "std" in actual else actual["log_std"].exp()).detach().cpu()
    result = {
        "checkpoint": str(source), "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "all_model_tensors_exact": True, "critic_preserved": True, "normalizers_preserved": True,
        "optimizer_exact": True, "environment_and_reward_exact_except_log_dir": True,
        "source_iteration": int(saved["iter"]), "next_iteration": expected_iteration,
        "ppo_learning_rate": runner.alg.learning_rate, "Adam_lrs": rates,
        "entropy_coef": runner.alg.entropy_coef, "std_mean": float(std.mean()),
        "std_min": float(std.min()), "std_max": float(std.max()), "std_per_action": std.tolist(),
        "model_tensor_count": len(actual), "Adam_state_entries": len(saved["optimizer_state_dict"]["state"]),
    }
    (destination / "resume_verification.json").write_text(json.dumps(result, indent=2))
    print("[RESUME_VERIFIED] " + json.dumps(result), flush=True)
    return result
