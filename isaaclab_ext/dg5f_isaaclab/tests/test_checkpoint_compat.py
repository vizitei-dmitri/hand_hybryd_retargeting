"""Legacy normalization statistics must survive the inference-only adapter exactly."""
import importlib.util
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/checkpoint_compat.py'
spec = importlib.util.spec_from_file_location('checkpoint_compat_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_legacy_conversion_preserves_weights_statistics_and_original(tmp_path):
    original = {'model_state_dict': {'actor.0.weight': torch.randn(3, 2)},
                'obs_norm_state_dict': {'_mean': torch.tensor([[2., 3.]]),
                                        '_var': torch.tensor([[4., 9.]]), '_std': torch.tensor([[2., 3.]])},
                'critic_obs_norm_state_dict': {'_mean': torch.tensor([[1., 4.]]),
                                               '_var': torch.tensor([[1., 16.]]), '_std': torch.tensor([[1., 4.]])}}
    path = tmp_path / 'checkpoint.pt'
    torch.save(original, path)
    before = path.read_bytes()
    converted = torch.load(module.prepare_inference_checkpoint(path), weights_only=False)
    assert path.read_bytes() == before
    state = converted['model_state_dict']
    assert torch.equal(state['actor.0.weight'], original['model_state_dict']['actor.0.weight'])
    sample = torch.tensor([[7., -5.]])
    for source, target in [('obs_norm_state_dict', 'actor_obs_normalizer'),
                           ('critic_obs_norm_state_dict', 'critic_obs_normalizer')]:
        old = (sample - original[source]['_mean']) / (original[source]['_std'] + .01)
        new = (sample - state[target + '._mean']) / (state[target + '._std'] + .01)
        assert torch.equal(old, new)
        assert state[target + '.count'].item() == 0
    assert converted['inference_only_conversion']
    copy_path = path.with_name('checkpoint_rsl3_inference.pt')
    assert module.prepare_inference_checkpoint(copy_path) == str(copy_path)


def test_current_checkpoint_is_not_rewritten(tmp_path):
    path = tmp_path / 'current.pt'
    torch.save({'model_state_dict': {'actor_obs_normalizer._mean': torch.zeros(1, 2)}}, path)
    before = path.read_bytes()
    assert module.prepare_inference_checkpoint(path) == str(path)
    assert path.read_bytes() == before
