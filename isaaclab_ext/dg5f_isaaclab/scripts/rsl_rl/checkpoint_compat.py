"""Adapt legacy separate observation normalizers for inference in RSL-RL 3.x."""
from pathlib import Path
import torch


def prepare_inference_checkpoint(path):
    """Preserve weights/statistics; missing sample counts are irrelevant in eval mode.

    The converted copy is ONLY for inference: legacy normalization sample counts
    were not serialized, so resuming normalization updates would be incorrect.
    """
    source = Path(path)
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if 'obs_norm_state_dict' not in checkpoint:
        return str(source)
    model = checkpoint['model_state_dict'].copy()
    for old, new in (('obs_norm_state_dict', 'actor_obs_normalizer'),
                     ('critic_obs_norm_state_dict', 'critic_obs_normalizer')):
        if any(key.startswith(new + '.') for key in model):
            raise ValueError('Ambiguous checkpoint: both embedded and separate normalizers')
        stats = checkpoint[old]
        if set(stats) != {'_mean', '_var', '_std'}:
            raise ValueError(f'Unsupported legacy normalizer fields: {set(stats)}')
        for name, value in stats.items():
            model[f'{new}.{name}'] = value.clone()
        model[f'{new}.count'] = torch.tensor(0, dtype=torch.long)
        checkpoint.pop(old)
    checkpoint['model_state_dict'] = model
    checkpoint['inference_only_conversion'] = {
        'source': str(source.resolve()), 'reason': 'Legacy normalizer buffers moved into policy',
        'count': 'Not present in original; zero placeholder, never update statistics'}
    destination = source.with_name(source.stem + '_rsl3_inference' + source.suffix)
    torch.save(checkpoint, destination)
    print(f'[PLAY] Converted legacy normalizers for inference: {destination}', flush=True)
    return str(destination)
