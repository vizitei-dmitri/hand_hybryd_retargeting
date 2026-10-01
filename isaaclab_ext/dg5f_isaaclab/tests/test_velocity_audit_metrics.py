from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

spec = spec_from_file_location("velocity_audit_metrics", Path(__file__).resolve().parents[1] / "scripts/velocity_audit_metrics.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_excludes_reset_jump_and_restarts_age_per_environment():
    q = torch.zeros(30, 2, 2)
    q[:, :, 0] = torch.arange(30)[:, None] * .1  # 1 rad/s
    q[15:, 0, 0] += 100  # teleport, not a velocity spike
    q[:, :, 1] = torch.arange(30)[:, None] * 10  # disabled joint excluded from active stats
    epoch = torch.zeros(30, 2, dtype=torch.long)
    epoch[15:, 0] = 1
    result = module.velocity_details(q, torch.zeros_like(q), epoch, .1, ["active", "disabled"], "disabled")
    stats = result["active_joints_only"]
    assert stats["all"]["joint_samples"] == 57
    assert abs(stats["all"]["max"] - 1) < .0001
    assert stats["after_1_second"]["joint_samples"] == 24
    assert all(not (event["env"] == 0 and event["physics_sample"] == 15)
               for event in result["largest_finite_differences"])
