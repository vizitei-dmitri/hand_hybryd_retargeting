"""Contact flicker must not be mistaken for a supported relocation."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gait_metrics import analyze

def trajectory(distance=.02,support=True):
    t,n=100,1
    a={'dt':np.array(1/60),'valid':np.ones((t,n),bool),'contact':np.ones((t,n,5),bool),
       'contact_local':np.zeros((t,n,5,3)), 'tip_local':np.zeros((t,n,5,3)),
       'cube_quat':np.tile([1.,0,0,0],(t,n,1)),'cube_linvel':np.zeros((t,n,3)),
       'cube_angvel':np.zeros((t,n,3)), 'palm':np.zeros((t,n),bool), 'drop':np.zeros((t,n),bool),
       'goals':np.zeros((t,n),int),'quality':np.ones((t,n))}
    a['contact'][10:20,0,0]=False;a['contact_local'][20:,0,0,0]=distance
    a['goals'][30:]=1
    if not support:a['contact'][10:20,0,1:4]=False
    return a

THRESHOLDS={'release_min_s':4/60,'meaningful_displacement_m':.011}

def test_supported_relocation_and_continued_manipulation():
    m=analyze(trajectory(),THRESHOLDS)
    assert m['gait/meaningful_events_per_episode']==1
    assert m['gait/successful_gait_cycles_per_episode']==1
    assert m['gait/goals_with_meaningful_switch_fraction']==1

def test_small_recontact_or_one_finger_support_not_gait():
    assert analyze(trajectory(.001),THRESHOLDS)['gait/meaningful_events_per_episode']==0
    assert analyze(trajectory(support=False),THRESHOLDS)['gait/meaningful_events_per_episode']==0

def test_no_cycle_for_palm_or_drop_and_no_post_episode_leak():
    a=trajectory();a['palm'][15]=True
    assert analyze(a,THRESHOLDS)['gait/meaningful_events_per_episode']==0
    a=trajectory();a['drop'][25]=True;a['valid'][26:]=False
    m=analyze(a,THRESHOLDS)
    assert m['gait/successful_gait_cycles_per_episode']==0
    assert m['gait/goals_with_meaningful_switch_fraction']==0
