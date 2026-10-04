"""Episode bootstrap comparisons and compact tables from deterministic gait evaluations."""
import json
from pathlib import Path
import numpy as np

def episode_values(path):
    result=next(iter(json.loads(Path(path).read_text()).values()));g=result['gait'];n=g['episodes']
    trajectories=list(Path(path).parent.glob(Path(path).stem+'_*_trajectory.npz'))
    if len(trajectories)!=1:raise ValueError(f'Ambiguous trajectory for {path}')
    with np.load(trajectories[0]) as a:
        lengths=a['valid'].sum(0).astype(int);goals=a['goals'][lengths-1,np.arange(n)]
        drops=(a['drop']&a['valid']).any(0).astype(float)
    return {'goals':goals,'drop':drops,
            'meaningful_gait':np.bincount([e['env'] for e in g['events'] if e['meaningful']],minlength=n),
            'successful_cycles':np.bincount([e['env'] for e in g['events'] if e['successful_cycle']],minlength=n)}

def compare(a_path,b_path):
    a,b=episode_values(a_path),episode_values(b_path);rng=np.random.default_rng(20261002);out={}
    for key in a:
        x,y=a[key],b[key]
        # Independent episode resampling: simulator seed matching is not exact trajectory pairing.
        delta=y[rng.integers(len(y),size=(4000,len(y)))].mean(1)-x[rng.integers(len(x),size=(4000,len(x)))].mean(1)
        out[key]={'A':float(x.mean()),'B':float(y.mean()),'B_minus_A':float(y.mean()-x.mean()),'bootstrap_95_interval':np.percentile(delta,[2.5,97.5]).tolist()}
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');p.add_argument('--output',required=True);args=p.parse_args()
    Path(args.output).write_text(json.dumps(compare(args.a,args.b),indent=2))
