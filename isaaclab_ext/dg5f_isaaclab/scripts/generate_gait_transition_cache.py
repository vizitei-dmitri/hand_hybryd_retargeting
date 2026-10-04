"""Physically execute single-finger DLS detach/transfer/recontact primitives.

Only offline generation. Every action traverses the unchanged FIFO/integrated-delta/PD path.
"""
import argparse,json,math,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--num_envs',type=int,default=128);p.add_argument('--batches',type=int,default=4)
p.add_argument('--seed',type=int,default=2718);p.add_argument('--thresholds',type=Path,default=Path('logs/gait_research/gait_thresholds.json'))
p.add_argument('--out',type=Path,default=Path('source/dg5f_isaaclab/dg5f_isaaclab/assets/data/gait_transition_cache.npz'))
p.add_argument('--report',type=Path,default=Path('logs/gait_research/generation.json'))
p.add_argument('--speed',type=float,default=.4);p.add_argument('--transfer_s',type=float,default=1.2)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.headless=True
app=AppLauncher(args);simulation_app=app.app
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.controllers import DifferentialIKController,DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply,quat_apply_inverse
import dg5f_isaaclab.tasks
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env import DG5FCubeEnv
from dg5f_isaaclab.assets.gait_cache import snapshot,PHASES

class GenerationEnv(DG5FCubeEnv):
    def _pick_grasps(self,ids):
        return self.forced_picks[ids] if hasattr(self,'forced_picks') else super()._pick_grasps(ids)


def main():
    thresholds=json.loads(args.thresholds.read_text());minimum=thresholds['meaningful_displacement_m']
    cfg=parse_env_cfg('DG5F-Cube-Stream-Direct-v0',device=args.device,num_envs=args.num_envs)
    cfg.track_gait_contact_points=True;cfg.seed=args.seed
    env=GenerationEnv(cfg);torch.manual_seed(args.seed);n=env.num_envs;device=env.device
    env.forced_picks=torch.arange(n,device=device)%env.primary_grasp_count
    indices=torch.arange(n,device=device);zero=torch.zeros((n,19),device=device)
    ik=DifferentialIKController(DifferentialIKControllerCfg(command_type='position',ik_method='dls',ik_params={'lambda_val':.003}),n,device)
    all_states=[];attempts=[];successful=[];actions_saved=[];trajectories=[]
    detach_steps=round(.6/env.step_dt);transfer_steps=round(args.transfer_s/env.step_dt);approach_steps=round(.7/env.step_dt);hold_steps=round(.4/env.step_dt)
    total=detach_steps+transfer_steps+approach_steps+hold_steps
    finger_action=[torch.tensor([i for i,name in enumerate(cfg.active_action_joints) if name.startswith(f'rj_dg_{f+1}_')],device=device) for f in range(5)]
    def geometry():
        points=torch.stack([s.data.contact_pos_w[:,0,0] for s in list(env.contact_sensors.values())[:5]],1)
        cq=env.cube.data.root_quat_w[:,None].expand(-1,5,-1).reshape(-1,4)
        return quat_apply_inverse(cq,(points-env.cube.data.root_pos_w[:,None]).reshape(-1,3)).reshape(n,5,3)
    for batch in range(args.batches):
        env.forced_picks=(torch.arange(n,device=device)+batch*n)%env.primary_grasp_count
        env.reset();dead=torch.zeros(n,dtype=torch.bool,device=device);settle_masks=[];settle_palms=[]
        for _ in range(45):
            _,_,term,trunc,_=env.step(zero);dead|=term|trunc
            settle_masks.append(env.tip_in_contact.clone());settle_palms.append(env.palm_contact_force>cfg.tip_contact_force_n)
        persistent=torch.stack(settle_masks[-12:]).float().mean(0)>=.9
        valid=(persistent.sum(-1)>=3)&~torch.stack(settle_palms[-12:]).any(0)&~dead
        # Cycle all fingers across sources/batches; require chosen finger initially contacting.
        finger=(indices+batch)%5
        selected_contact=persistent[indices,finger];valid &=selected_contact
        support=persistent.clone();support[indices,finger]=False
        cp=geometry();old=cp[indices,finger].clone()
        old=torch.nan_to_num(old)
        tips=env.hand.data.body_pos_w[indices,torch.tensor(env.tip_ids,device=device)[finger]]
        localtip=quat_apply_inverse(env.cube.data.root_quat_w,tips-env.cube.data.root_pos_w)
        offset=localtip-old
        axis=old.abs().argmax(-1);normal=torch.zeros_like(old);normal[indices,axis]=torch.sign(old[indices,axis])
        detach=torch.tensor([.005,.010,.015],device=device)[(indices//5+batch)%3]
        # Same face, target inside edges. Sample two tangential directions independently.
        tangent=torch.randn_like(old);tangent[indices,axis]=0;tangent/=tangent.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        distance=minimum*torch.tensor([1.25,1.75,2.25],device=device)[(indices//15+batch)%3]
        target=(old+tangent*distance[:,None]).clamp(-.027,.027);target[indices,axis]=normal[indices,axis]*.03
        requested_distance=(target-old).norm(dim=-1);valid&=requested_distance>minimum
        start=snapshot(env);stage_states={0:start,1:{k:v.copy() for k,v in start.items()},5:{k:v.copy() for k,v in start.items()}};released=torch.zeros_like(valid);release_run=torch.zeros(n,device=device,dtype=torch.long)
        recorded_release=torch.zeros_like(valid);recorded_recontact=torch.zeros_like(valid)
        max_release=torch.zeros_like(release_run);support_ok=torch.ones_like(valid);palm_bad=torch.zeros_like(valid)
        max_lin=torch.zeros(n,device=device);max_ang=max_lin.clone();commands=[];frames=[];recontact_step=torch.full_like(release_run,-1)
        sustained=torch.zeros_like(release_run);settled_count=torch.zeros_like(release_run)
        for step in range(total):
            if step<detach_steps:
                phase=1;alpha=(step+1)/detach_steps;desired=old+offset+normal*detach[:,None]*alpha
            elif step<detach_steps+transfer_steps:
                phase=2;alpha=(step-detach_steps+1)/transfer_steps;desired=old+(target-old)*alpha+offset+normal*detach[:,None]
            elif step<detach_steps+transfer_steps+approach_steps:
                phase=3;alpha=(step-detach_steps-transfer_steps+1)/approach_steps
                desired=target+offset+normal*(detach*(1-alpha)-.002*alpha)[:,None]
            else:
                phase=4;desired=target+offset-normal*.002
            tip_ids=torch.tensor(env.tip_ids,device=device)[finger]
            current=env.hand.data.body_pos_w[indices,tip_ids]
            world_target=env.cube.data.root_pos_w+quat_apply(env.cube.data.root_quat_w,desired)
            jac_all=env.hand.root_physx_view.get_jacobians()
            bodyindex=tip_ids-1 if env.hand.is_fixed_base else tip_ids
            J=jac_all[indices,bodyindex,:3][:,:,env.active_joint_ids].clone()
            mask=torch.stack([torch.tensor([name.startswith(f'rj_dg_{int(f)+1}_') for name in cfg.active_action_joints],device=device) for f in finger.tolist()])
            J*=mask[:,None,:]
            ik.set_command(world_target,ee_quat=env.hand.data.body_quat_w[indices,tip_ids])
            desired_q=ik.compute(current,env.hand.data.body_quat_w[indices,tip_ids],J,env.hand.data.joint_pos[:,env.active_joint_ids])
            predicted=env.joint_command+env.action_queue.history.sum(1)*cfg.delta_action_scale
            action=((desired_q-predicted)/cfg.delta_action_scale).clamp(-args.speed,args.speed)*mask
            action[~valid|dead]=0
            if phase==4: action[env.tip_in_contact[indices,finger]]=0
            commands.append(action.cpu().numpy().copy())
            _,_,term,trunc,_=env.step(action);dead|=term|trunc
            contact=env.tip_in_contact[indices,finger]
            release_run=torch.where(~contact,release_run+1,torch.zeros_like(release_run));max_release=torch.maximum(max_release,release_run)
            released|=max_release>=thresholds['release_min_steps']
            just_release=released&~recorded_release
            just_recontact=released&contact&~recorded_recontact&(step>=detach_steps+transfer_steps)
            if just_release.any() or just_recontact.any():
                snap=snapshot(env)
                for phase_id,mask_now in ((1,just_release),(5,just_recontact)):
                    selected=mask_now.cpu().numpy()
                    for k,v in snap.items():stage_states[phase_id][k][selected]=v[selected]
                recorded_release|=just_release;recorded_recontact|=just_recontact
            other=env.tip_in_contact&support;moving=~contact&released
            support_ok &=~moving|(other.sum(-1)>=2)
            palm_bad |=env.palm_contact_force>cfg.tip_contact_force_n
            max_lin=torch.maximum(max_lin,env.cube.data.root_lin_vel_w.norm(dim=-1));max_ang=torch.maximum(max_ang,env.cube.data.root_ang_vel_w.norm(dim=-1))
            sustained=torch.where(contact,sustained+1,torch.zeros_like(sustained))
            if phase>=3:
                just=(recontact_step<0)&released&contact;recontact_step[just]=step
            if step==detach_steps+transfer_steps//4: stage_states[2]=snapshot(env)
            if step==detach_steps+transfer_steps//2: stage_states[3]=snapshot(env)
            if step==detach_steps+transfer_steps-1: stage_states[4]=snapshot(env)
            if step==total-1: stage_states[6]=snapshot(env)
            frames.append(np.concatenate((env.hand.data.body_pos_w[:,env.tip_ids].cpu().numpy(),env.cube.data.root_pos_w.cpu().numpy()[:,None,:]),axis=1))
        new=geometry()[indices,finger];displacement=(new-old).norm(dim=-1)
        accepted=valid&released&support_ok&~palm_bad&~dead&(sustained>=hold_steps//2)&(displacement>minimum)&torch.isfinite(displacement)
        accepted&=recorded_recontact&(env.tip_contact_count>=3)
        # Low-ballistic bounds: below baseline policy's p95, declared before generator results.
        base=json.loads(Path('logs/gait_research/baseline_event_analysis.json').read_text())['velocity_limits_for_cache']
        accepted&=(max_lin<=base['linear_p95'])&(max_ang<=base['angular_p95'])
        for e in range(n):
            row={'attempt':batch*n+e,'source_grasp':int(env.forced_picks[e]),'finger':int(finger[e]),'detach_m':float(detach[e]),
                 'requested_distance_m':float(requested_distance[e]),'valid_start':bool(valid[e]),'released':bool(released[e]),
                 'support_ok':bool(support_ok[e]),'palm':bool(palm_bad[e]),'drop':bool(dead[e]),
                 'max_linear_velocity':float(max_lin[e]),'max_angular_velocity':float(max_ang[e]),
                 'contact_displacement_m':float(displacement[e]) if torch.isfinite(displacement[e]) else None,
                 'stable_contact_steps':int(sustained[e]),'accepted':bool(accepted[e])}
            attempts.append(row)
            if accepted[e]:
                tid=len(successful);successful.append(row);actions_saved.append(np.stack(commands)[:,e]);trajectories.append(np.stack(frames)[:,e])
                for phase,snap in stage_states.items():
                    if phase in (1,2,3,4) and snap['contact_mask'][e,int(finger[e])]:continue
                    entry={k:v[e] for k,v in snap.items()};entry.update(moving_finger=int(finger[e]),phase=phase,transition_id=tid,
                        source_grasp_id=int(env.forced_picks[e]),old_contact=old[e].cpu().numpy(),new_contact=new[e].cpu().numpy(),support_set=support[e].cpu().numpy(),source_type='scripted')
                    all_states.append(entry)
        report={'attempted':len(attempts),'successful':len(successful),'thresholds':thresholds,'attempts':attempts,
                'successful_transitions':successful,'velocity_bounds':base,'phase_names':PHASES,
                'controller':'IsaacLab DifferentialIKController DLS lambda=.003; selected finger only; predictive FIFO; no physics/authority edits'}
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2))
        print(f'[GAIT_GENERATOR] batch={batch} valid={int(valid.sum())} released={int((valid&released).sum())} support={int((valid&released&support_ok).sum())} accepted={int(accepted.sum())} total={len(successful)}',flush=True)
    if all_states:
        arrays={k:np.stack([x[k] for x in all_states]) for k in all_states[0]}
        arrays.update(schema_version=np.array(1),joint_names=np.array(cfg.actuated_joint_names),phase_names=np.array(PHASES))
        args.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out,**arrays)
        np.savez_compressed(args.report.with_suffix('.replay.npz'),actions=np.stack(actions_saved),positions=np.stack(trajectories))
        print(f'[GAIT_GENERATOR] saved {len(all_states)} snapshots to {args.out}',flush=True)
    else: print('[GAIT_GENERATOR] NO ACCEPTED TRANSITIONS; no usable cache written.',flush=True)
    env.close()

if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    except Exception:traceback.print_exc();raise
    finally:simulation_app.close()
