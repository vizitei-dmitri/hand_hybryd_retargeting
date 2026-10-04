"""Reject incomplete dynamic states before they can enter a reset distribution."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'source/dg5f_isaaclab/dg5f_isaaclab/assets/gait_cache.py'
spec=importlib.util.spec_from_file_location('gait_cache_under_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
CACHE=p.parent/'data/gait_transition_cache_robust.npz'

def test_real_cache_contains_dynamic_intermediate_states():
    with np.load(CACHE) as f:names=f['joint_names'].copy()
    a,sha=m.load_gait_cache(CACHE,names)
    assert len(sha)==64 and np.isin([1,2,3,4,5],a['phase']).all()
    assert np.any(abs(a['fifo'])>0) and np.any(abs(a['q_vel'])>0)
    assert not np.allclose(a['q'][:,:19],a['q_cmd'])

@pytest.mark.parametrize('key',['q_vel','fifo','cube_angvel'])
def test_rejects_nonfinite_dynamic_state(tmp_path,key):
    with np.load(CACHE) as f:a={k:f[k] for k in f.files}
    a[key][0].flat[0]=np.nan;p=tmp_path/'bad.npz';np.savez(p,**a)
    with pytest.raises(ValueError,match=key):m.load_gait_cache(p,a['joint_names'])
