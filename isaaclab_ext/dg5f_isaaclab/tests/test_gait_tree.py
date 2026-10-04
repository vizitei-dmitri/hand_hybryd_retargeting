"""Behavior preservation and mechanistic observation checks independent of simulator."""
import importlib.util, sys, types
from pathlib import Path
import pytest, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from expand_gait_checkpoint import expand, outputs
D=ROOT/'source/dg5f_isaaclab/dg5f_isaaclab/tasks/direct/dg5f_cube'
pkg=types.ModuleType('_gait_tree_test');pkg.__path__=[str(D)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location('_gait_tree_test.gait_features',D/'gait_features.py');features=importlib.util.module_from_spec(spec);spec.loader.exec_module(features)

@pytest.mark.parametrize('added',[5,12,20])
def test_expansion_preserves_policy_and_optimizer(added):
    saved=torch.load(ROOT/'logs/rsl_rl/dg5f_cube_direct/2026-10-02_18-15-45_gait_H1_B/model_3999.pt',map_location='cpu',weights_only=False)
    result,report=expand(saved,added)
    assert report['actor_max_abs_error']<1e-5
    for name,v in saved['model_state_dict'].items():
        new=result['model_state_dict'][name]
        if v.shape!=new.shape:
            assert torch.equal(v,new[...,:148])
        else:assert torch.equal(v,new)
    for key,state in saved['optimizer_state_dict']['state'].items():
        for name,value in state.items():
            new=result['optimizer_state_dict']['state'][key][name]
            assert torch.equal(value,new[...,:148] if value.shape!=new.shape else new)

def test_mechanics_release_support():
    x=torch.tensor([[[.03,0,0],[-.03,0,0],[0,.03,0],[0,0,.03],[0,-.03,0]]])
    c=torch.ones((1,5),dtype=torch.bool);f=features.mechanics_features(x,c)
    assert f.shape==(1,12) and torch.isfinite(f).all()
    assert (f[:,7:] <= f[:,:1]+1e-6).all()
    assert torch.equal(features.mechanics_features(x,torch.zeros_like(c)),torch.zeros_like(f))

def test_timer_edges_no_multiple_counts():
    c=torch.ones((1,5),dtype=torch.bool);p=c.clone();t=torch.zeros(1,5,3)
    c[0,0]=False;t=features.update_timers(t,p,c,.1)
    assert t[0,0,0]==0 and t[0,0,1]==.1
    p=c.clone();c[0,0]=True;t=features.update_timers(t,p,c,.1)
    assert t[0,0,2]==.1 and t[0,0,1]==.2
    t=features.update_timers(t,c,c,10.)
    assert t.max()==2.
