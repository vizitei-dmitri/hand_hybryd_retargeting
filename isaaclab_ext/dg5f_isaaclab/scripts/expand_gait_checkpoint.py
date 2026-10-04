"""Behavior-preserving observation expansion, including Adam and normalization state."""
import argparse,copy,hashlib,json,re,shutil
from pathlib import Path
import torch

WIDTHS={'contacts':5,'mechanics':12,'timers':20}

def outputs(state,x,network):
    n=network+'_obs_normalizer.'
    x=(x-state[n+'_mean'])/(state[n+'_std']+1e-2)
    for layer in (0,2,4):
        x=torch.nn.functional.linear(x,state[f'{network}.{layer}.weight'],state[f'{network}.{layer}.bias'])
        if layer!=4:x=torch.nn.functional.elu(x)
    return x

def expand(saved,added):
    result=copy.deepcopy(saved);s=result['model_state_dict'];old=s['actor.0.weight'].shape[1]
    # Parameter order is explicit in the saved optimizer and verified against expected shapes.
    params=[k for k in s if '_obs_normalizer.' not in k]
    ids=[p for g in result['optimizer_state_dict']['param_groups'] for p in g['params']]
    assert len(params)==len(ids)
    for net in ('actor','critic'):
        k=net+'.0.weight';w=s[k];s[k]=torch.cat((w,torch.zeros(w.shape[0],added)),1)
        for suffix in ('_mean','_var','_std'):
            k1=net+'_obs_normalizer.'+suffix;v=s[k1]
            s[k1]=torch.cat((v,torch.full((1,added),0. if suffix=='_mean' else 1.)),1)
        state=result['optimizer_state_dict']['state'][ids[params.index(k)]]
        for moment in ('exp_avg','exp_avg_sq'):
            assert state[moment].shape==w.shape
            state[moment]=torch.cat((state[moment],torch.zeros(w.shape[0],added)),1)
    probe=torch.sin(torch.arange(256*old).reshape(256,old)*.017)
    extra=torch.cos(torch.arange(256*added).reshape(256,added)*.07)
    errors={}
    for net in ('actor','critic'):
        a=outputs(saved['model_state_dict'],probe,net);b=outputs(s,torch.cat((probe,extra),1),net)
        torch.testing.assert_close(a,b,rtol=2e-6,atol=1e-5)
        errors[net+'_max_abs_error']=float((a-b).abs().max())
    return result,errors

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--mode',choices=WIDTHS,required=True);a=p.parse_args()
    saved=torch.load(a.source,map_location='cpu',weights_only=False);out,report=expand(saved,WIDTHS[a.mode])
    a.destination.mkdir(parents=True,exist_ok=False);shutil.copytree(a.source.parent/'params',a.destination/'params')
    env=a.destination/'params/env.yaml';s=env.read_text()
    s=re.sub(r'^observation_space: \d+$',f'observation_space: {148+WIDTHS[a.mode]}',s,flags=re.M)
    s+='\ngait_task_gate: false\ngait_observation_mode: '+a.mode+'\n';env.write_text(s)
    dest=a.destination/a.source.name;torch.save(out,dest)
    report.update(source=str(a.source.resolve()),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),checkpoint=str(dest.resolve()),mode=a.mode,new_features=WIDTHS[a.mode],old_columns_exact=True,new_columns_zero=True,normalizer_old_values_exact=True,Adam_old_values_exact=True,Adam_new_columns_zero=True,iteration=out['iter'])
    (a.destination/'expansion_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
