"""The bounds-loss hook must equal adding rl_games' bound loss to the PPO loss, and nothing else."""
import importlib.util
from pathlib import Path

import torch
from tensordict import TensorDict

PATH = Path(__file__).resolve().parents[1] / "source/dg5f_isaaclab/dg5f_isaaclab/tasks/direct/dg5f_cube/agents/bounded_policy.py"
spec = importlib.util.spec_from_file_location("bounded_policy", PATH)
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)


def make(coef):
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.randn(64, 7)}, batch_size=[64])
    policy = bounded.BoundedActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 3,
                                        actor_hidden_dims=[16], critic_hidden_dims=[16],
                                        bounds_loss_coef=coef, bounds_soft_limit=1.0)
    with torch.no_grad():  # push the mean well outside [-1, 1] so the loss is active
        policy.actor[-1].bias.copy_(torch.tensor([4.0, -3.0, 0.2]))
    return policy, obs


def surrogate(policy, obs):
    policy.act(obs)
    actions = torch.zeros(64, 3)
    return -policy.get_actions_log_prob(actions).mean()


def test_hook_equals_explicit_bound_loss():
    coef = 0.37
    hooked, obs = make(coef)
    surrogate(hooked, obs).backward()
    explicit, obs = make(0.0)
    loss = surrogate(explicit, obs)
    loss = loss + coef * bounded.bounds_loss(explicit.distribution.loc, 1.0)
    loss.backward()
    for (name, a), (_, b) in zip(hooked.named_parameters(), explicit.named_parameters()):
        if a.grad is None and b.grad is None:
            continue
        torch.testing.assert_close(a.grad, b.grad, msg=name)


def test_no_hook_outside_the_update():
    policy, obs = make(0.37)
    with torch.inference_mode():
        policy.act(obs)  # rollouts run under inference mode; registering a hook there would raise
    with torch.no_grad():
        policy.act(obs)


def test_bound_loss_is_zero_inside_the_range():
    mean = torch.tensor([[0.5, -0.99, 1.0]])
    assert float(bounded.bounds_loss(mean, 1.0)) == 0.0
    torch.testing.assert_close(bounded.bounds_loss(torch.tensor([[3.0, -2.0]]), 1.0), torch.tensor(5.0))


def test_registration_adds_a_resolvable_name():
    bounded.register()
    import rsl_rl.runners.on_policy_runner as runner_module
    assert runner_module.BoundedActorCritic is bounded.BoundedActorCritic


def test_tasks_package_import_registers_the_class():
    # The night-run failure: eval_checkpoints.py builds the runner from agent.yaml after importing
    # only dg5f_isaaclab.tasks, so registration must happen in the agents package itself.
    init = (PATH.parent / "__init__.py").read_text()
    assert "_register_bounded_policy()" in init
