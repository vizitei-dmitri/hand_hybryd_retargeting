"""ActorCritic with rl_games' bounds loss, registered locally for RSL-RL 3.1.2.

Why: actions are clipped to [-1, 1] after sampling, so once the Gaussian mean leaves that range the
clip hides it and nothing pulls it back. Measured on this task: raw mean |mu| 5.9 at ROOT and 12.1
at FINAL_LONG, with 95% of delivered actions at +-1 (logs/gait_tree, delivered-action audit). The
existing action penalty cannot help: it is computed on the CLIPPED action, so its gradient with
respect to mu is zero exactly where mu is out of range.

rl_games (HORA, AnyRotate, DeXtreme, IsaacGymEnvs) adds bounds_loss_coef * mean_batch(sum_dims(
max(0, |mu| - soft_bound)^2 )) to the PPO loss; HORA uses coef 1e-4. RSL-RL has no such option,
and patching PPO.update is out of scope, so the same gradient is injected with a tensor hook on the
distribution mean. Every PPO term that depends on mu (log-prob, KL) reaches it through
distribution.loc, so adding d(bounds)/d(loc) there is exactly equivalent to adding the loss term.
The hook is only attached when the mean requires grad, i.e. inside the PPO update and never during
rollouts or inference.
"""

import torch
from rsl_rl.modules import ActorCritic


class BoundedActorCritic(ActorCritic):
    def __init__(self, *args, bounds_loss_coef: float = 0.0, bounds_soft_limit: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if bounds_loss_coef < 0 or bounds_soft_limit <= 0:
            raise ValueError("bounds_loss_coef must be >= 0 and bounds_soft_limit > 0")
        self.bounds_loss_coef = float(bounds_loss_coef)
        self.bounds_soft_limit = float(bounds_soft_limit)

    def _update_distribution(self, obs):
        super()._update_distribution(obs)
        mean = self.distribution.loc
        if self.bounds_loss_coef and mean.requires_grad:
            mean.register_hook(bounds_gradient_hook(mean.detach(), self.bounds_loss_coef, self.bounds_soft_limit))


def bounds_loss(mean: torch.Tensor, soft_limit: float) -> torch.Tensor:
    """rl_games' bound_loss, averaged over the batch: per-sample sum over action dimensions."""
    excess = (mean.abs() - soft_limit).clamp_min(0.0)
    return excess.square().sum(dim=-1).mean()


def bounds_gradient_hook(mean: torch.Tensor, coef: float, soft_limit: float):
    excess = (mean.abs() - soft_limit).clamp_min(0.0) * mean.sign()
    extra = coef * 2.0 * excess / mean.shape[0]
    return lambda grad: grad + extra


def register() -> None:
    """Make class_name="BoundedActorCritic" resolvable: OnPolicyRunner resolves the policy class
    with eval() in its own module namespace. Adds a name; changes nothing for existing ones."""
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.BoundedActorCritic = BoundedActorCritic
