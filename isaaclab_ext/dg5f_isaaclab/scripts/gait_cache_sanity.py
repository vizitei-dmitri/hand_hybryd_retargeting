"""Cold reset each dynamic snapshot under exact/joint/position/orientation perturbations."""
import argparse,json,math,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=128);p.add_argument('--trials',type=int,default=2);p.add_argument('--hold_s',type=float,default=.5)
p.add_argument('--report',type=Path,default=Path('logs/gait_research/cache_sanity.json'))
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.headless=True;app=AppLauncher(args);simulation_app=app.app
import torch,numpy as np
from isaaclab_tasks.utils import parse_env_cfg
import dg5f_isaaclab.tasks
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env import DG5FCubeEnv
from dg5f_isaaclab.assets.gait_cache import load_gait_cache,restore,PHASES

def main():
    cfg=parse_env_cfg('DG5F-Cube-Stream-Direct-v0',device=args.device,num_envs=args.num_envs)
    env=DG5FCubeEnv(cfg);a,sha=load_gait_cache(args.cache,cfg.actuated_joint_names,env.action_queue.history_steps)
    total=len(a['q']);n=env.num_envs;ids=torch.arange(n,device=env.device);zero=torch.zeros((n,19),device=env.device)
    measured=[];perturbations=[('exact',0,0,0),('joint',math.radians(.25),0,0),('position',0,.0005,0),('orientation',0,0,math.radians(1))]
    for kind,joint,pos,angle in perturbations:
        for trial in range(args.trials):
            for start in range(0,total,n):
                real=min(n,total-start);picks=(torch.arange(n,device=env.device)+start)%total
                env.reset();restore(env,a,picks,ids,joint,pos,angle);env._compute_state()
                dead=torch.zeros(n,dtype=torch.bool,device=env.device);low=torch.zeros(n,device=env.device);palm=low.clone();lin=low.clone();ang=low.clone();acc=low.clone();impulse=low.clone();previous=None
                steps=round(args.hold_s/env.step_dt)
                for step in range(steps):
                    _,_,term,trunc,_=env.step(zero);dead|=term|trunc
                    if step<2:continue  # cold contact sensor reconstruction, explicitly excluded
                    low+=(env.tip_contact_count<2).float();palm+=(env.palm_contact_force>cfg.tip_contact_force_n).float()
                    lv=env.cube.data.root_lin_vel_w.clone();lin=torch.maximum(lin,lv.norm(dim=-1));ang=torch.maximum(ang,env.cube.data.root_ang_vel_w.norm(dim=-1))
                    if previous is not None:acc=torch.maximum(acc,((lv-previous)/env.step_dt).norm(dim=-1))
                    previous=lv
                    force=torch.stack([s.data.force_matrix_w[:,0,0].norm(dim=-1) for s in env.contact_sensors.values()]).sum(0)
                    impulse=torch.maximum(impulse,force*env.step_dt)
                # Force magnitudes are logged, not treated as calibrated SI acceptance evidence.
                ok=~dead&(low/(steps-2)<=.10)&(palm/(steps-2)<=.05)&(lin<.15)&(ang<4)&(acc<50)
                for e in range(real):measured.append({'snapshot':int(picks[e]),'kind':kind,'trial':trial,'survived':bool(ok[e]),'dropped':bool(dead[e]),'fraction_below_2':float(low[e]/(steps-2)),'palm_fraction':float(palm[e]/(steps-2)),'max_linvel':float(lin[e]),'max_angvel':float(ang[e]),'max_acceleration':float(acc[e]),'reported_contact_impulse':float(impulse[e])})
                print(f'[GAIT_SANITY] kind={kind} trial={trial} start={start} accepted={int(ok[:real].sum())}/{real}',flush=True)
    keep=[];survival=[]
    for i in range(total):
        rows=[x for x in measured if x['snapshot']==i];exact=[x for x in rows if x['kind']=='exact'];noisy=[x for x in rows if x['kind']!='exact']
        survival.append(np.mean([x['survived'] for x in rows]));keep.append(all(x['survived'] for x in exact) and np.mean([x['survived'] for x in noisy])>=.75)
    keep=np.array(keep);report={'source':str(args.cache.resolve()),'source_sha256':sha,'candidate_states':total,'robust_states':int(keep.sum()),'survival_rate':float(np.mean(survival)),'per_state_survival':survival,'kept':keep.tolist(),'trials':measured,'criteria':{'hold_s':args.hold_s,'joint_deg':.25,'position_mm':.5,'orientation_deg':1,'max_fraction_below_2':.1,'max_palm_fraction':.05,'exact_required':1.,'perturbed_required':.75},'states_by_phase':{PHASES[i]:int(((a['phase']==i)&keep).sum()) for i in range(7)}}
    args.report.write_text(json.dumps(report,indent=2))
    if keep.any():
        out={k:(v[keep] if v.ndim>0 and len(v)==total and k not in ('joint_names','phase_names') else v) for k,v in a.items()}
        out['reset_survival']=np.array(survival)[keep];np.savez_compressed(args.out,**out)
    env.close()
if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    except Exception:traceback.print_exc();raise
    finally:simulation_app.close()
