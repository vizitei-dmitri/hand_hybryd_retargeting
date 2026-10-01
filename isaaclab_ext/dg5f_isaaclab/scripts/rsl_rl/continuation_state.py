"""Restore non-serialized runner/PPO bookkeeping after an exact training resume."""
import math


def restore_continuation_state(runner):
    rates = [group['lr'] for group in runner.alg.optimizer.param_groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError('Checkpoint learning rate must be finite and positive')
    if any(rate != rates[0] for rate in rates):
        raise ValueError('PPO scalar learning rate cannot represent different parameter-group rates')
    # RSL-RL load() restores Adam, but PPO.learning_rate remains the constructor's
    # value. The next adaptive update would overwrite the restored Adam rate.
    runner.alg.learning_rate = rates[0]
    # Checkpoints store the LAST completed iteration, not the next one.
    runner.current_learning_iteration += 1
    return rates[0], runner.current_learning_iteration
