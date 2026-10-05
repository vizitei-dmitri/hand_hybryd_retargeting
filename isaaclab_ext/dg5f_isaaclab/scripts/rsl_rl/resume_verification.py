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


def actor_fingerprint(policy):
    """A host-independent identity hash of the actor.

    The hash is taken over raw float32 bytes, so it must not depend on the backend flags of
    whichever script happens to call it. train.py sets torch.backends.cuda.matmul.allow_tf32 = True
    and eval_checkpoints.py leaves it at the PyTorch default of False, so the SAME weights hashed
    differently in the two processes -- reproducibly within each one, never equal across them. The
    H1 gait campaign aborted on exactly that: `Actor fingerprint mismatch BEFORE optimizing`, while
    the checkpoint sha256 matched and every tensor compared exact. TF32 is therefore pinned off
    here, which is also the state the recorded baseline fingerprints were taken in.
    """
    if not hasattr(policy, "actor"):
        return None
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode():
            parameter = next(policy.actor.parameters())
            width = policy.actor[0].in_features
            probe = torch.sin(torch.arange(16 * width, device=parameter.device, dtype=parameter.dtype).reshape(16, width) * .017)
            output = policy.actor(policy.actor_obs_normalizer(probe)).detach().cpu().contiguous()
            return hashlib.sha256(output.numpy().tobytes()).hexdigest()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def verify_resume_start(runner, checkpoint_path, log_dir, entropy, reward_overrides=None, reset_overrides=None, hypothesis_overrides=None, task_overrides=None):
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
    # These opt-in fields did not exist in older checkpoints; absent means disabled/default.
    defaults = {"gait_task_gate": False, "gait_observation_mode": "none", "track_gait_contact_points": False, "gait_transition_cache_path": None,
                "gait_transition_fraction": 0.0, "gait_phase_weights": [0.0,.125,.125,.35,.25,.15,0.0],
                "axis_velocity_reward_scale": 0.0, "axis_velocity_clip_rad_s": 0.5, "axis_velocity_min_tips": 3,
                "grasp_cache_orientation": False, "joint_limit_penalty_scale": 0.0, "joint_limit_margin_deg": 10.0,
                "relocation_bonus": 0.0, "relocation_min_release_steps": 4, "relocation_min_displacement_m": 0.0109,
                "relocation_min_support_tips": 2,
                # Absent in every env.yaml before 2026-10-05: those runs used the decomposition.
                "tip_collider_approximation": "convexDecomposition"}
    serialized_defaults = yaml.load(yaml.safe_dump(defaults), Loader=yaml.BaseLoader)
    for config in configs:
        for key, value in serialized_defaults.items():
            config.setdefault(key, value)
    reset_changes = {}
    for key, value in (reset_overrides or {}).items():
        if key not in ("gait_transition_cache_path", "gait_transition_fraction", "gait_phase_weights",
                       "grasp_cache_path", "grasp_cache_orientation"):
            raise ValueError(f"Unapproved reset override: {key}")
        expected = yaml.load(yaml.safe_dump({key: value}), Loader=yaml.BaseLoader)[key]
        assert_same_state(expected, configs[1][key], f"reset_override/{key}")
        reset_changes[key] = {"source": configs[0][key], "actual": configs[1][key]}
        configs[0][key] = configs[1][key]
    hypothesis_changes = {}
    for key, value in (hypothesis_overrides or {}).items():
        if key not in ("gait_task_gate", "gait_observation_mode", "observation_space"):
            raise ValueError(f"Unapproved hypothesis override: {key}")
        expected = yaml.load(yaml.safe_dump({key: value}), Loader=yaml.BaseLoader)[key]
        assert_same_state(expected, configs[1][key], f"hypothesis_override/{key}")
        hypothesis_changes[key] = {"source": configs[0][key], "actual": configs[1][key]}
        configs[0][key] = configs[1][key]
    # The continuous-rotation experiments change WHAT is asked (goal direction, rotation reward),
    # never physics or control; each such key must be named explicitly and must match exactly.
    task_changes = {}
    for key, value in (task_overrides or {}).items():
        if key not in ("goal_stream_stage", "axis_velocity_reward_scale", "axis_velocity_clip_rad_s",
                       "axis_velocity_min_tips", "joint_limit_penalty_scale", "joint_limit_margin_deg",
                       "relocation_bonus"):
            raise ValueError(f"Unapproved task override: {key}")
        expected = yaml.load(yaml.safe_dump({key: value}), Loader=yaml.BaseLoader)[key]
        assert_same_state(expected, configs[1][key], f"task_override/{key}")
        task_changes[key] = {"source": configs[0][key], "actual": configs[1][key]}
        configs[0][key] = configs[1][key]
    changes = {}
    for key, value in (reward_overrides or {}).items():
        if key not in ("action_penalty_scale", "action_rate_penalty_scale"):
            raise ValueError(f"Unapproved reward override: {key}")
        if float(configs[1][key]) != value:
            raise ValueError(f"Actual reward override differs: {key}")
        changes[key] = {"source": configs[0][key], "actual": configs[1][key]}
        configs[0][key] = configs[1][key]
    assert_same_state(configs[0], configs[1], "environment_including_reward")
    actor_hash = hashlib.sha256()
    for name, tensor in sorted(actual.items()):
        if name.startswith("actor.") or name.startswith("actor_obs_normalizer."):
            actor_hash.update(name.encode())
            actor_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    std = (actual["std"] if "std" in actual else actual["log_std"].exp()).detach().cpu()
    result = {
        "actor_parameters_sha256": actor_hash.hexdigest(), "checkpoint": str(source), "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "all_model_tensors_exact": True, "critic_preserved": True, "normalizers_preserved": True,
        "optimizer_exact": True, "environment_and_reward_exact_except_log_dir": not any(v["source"] != v["actual"] for group in (changes, reset_changes, hypothesis_changes, task_changes) for v in group.values()),
        "verified_hypothesis_overrides": hypothesis_changes, "verified_task_overrides": task_changes, "verified_reset_overrides": reset_changes, "actor_output_fingerprint": actor_fingerprint(runner.alg.policy),
        "verified_reward_overrides": changes, "all_other_environment_parameters_exact": True,
        "goal_stream_stage": configs[1].get("goal_stream_stage"),
        "goal_curriculum": configs[1].get("goal_curriculum"),
        "source_iteration": int(saved["iter"]), "next_iteration": expected_iteration,
        "ppo_learning_rate": runner.alg.learning_rate, "Adam_lrs": rates,
        "entropy_coef": runner.alg.entropy_coef, "std_mean": float(std.mean()),
        "std_min": float(std.min()), "std_max": float(std.max()), "std_per_action": std.tolist(),
        "model_tensor_count": len(actual), "Adam_state_entries": len(saved["optimizer_state_dict"]["state"]),
    }
    (destination / "resume_verification.json").write_text(json.dumps(result, indent=2))
    print("[RESUME_VERIFIED] " + json.dumps(result), flush=True)
    return result
