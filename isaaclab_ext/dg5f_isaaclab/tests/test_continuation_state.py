from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = spec_from_file_location('continuation_state', Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/continuation_state.py')
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_restores_scheduler_without_resetting_adam_or_policy():
    groups = [{'lr': 0.00011390625}]
    optimizer = SimpleNamespace(param_groups=groups, state={'moments': object()})
    policy = object()
    runner = SimpleNamespace(alg=SimpleNamespace(optimizer=optimizer, policy=policy, learning_rate=.001),
                             current_learning_iteration=599)
    assert module.restore_continuation_state(runner) == (0.00011390625, 600)
    assert runner.alg.learning_rate == groups[0]['lr']
    assert runner.alg.optimizer is optimizer
    assert runner.alg.policy is policy
    assert runner.current_learning_iteration == 600


@pytest.mark.parametrize('rates', [[], [0], [float('nan')], [.001, .002]])
def test_rejects_invalid_or_ambiguous_lr(rates):
    runner = SimpleNamespace(alg=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{'lr': r} for r in rates])),
                             current_learning_iteration=599)
    with pytest.raises(ValueError):
        module.restore_continuation_state(runner)
