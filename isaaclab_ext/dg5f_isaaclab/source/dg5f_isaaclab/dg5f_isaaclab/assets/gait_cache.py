"""Versioned dynamic reset snapshots. Stable caches are never modified."""
from pathlib import Path
import hashlib
import numpy as np

PHASES=('pre_release','released','early_transfer','mid_transfer','near_contact','recontact','settled')
DYNAMIC_FIELDS=('q','q_vel','q_cmd','cube_pos','cube_quat','cube_linvel','cube_angvel','fifo',
                'grasp_command','cube_anchor','applied_actions','previous_joint_command')


def load_gait_cache(path, joint_names, history_steps=3):
    path=Path(path)
    with np.load(path,allow_pickle=False) as f: a={k:f[k] for k in f.files}
    if int(a['schema_version'])!=1: raise ValueError('Unsupported gait cache version')
    if tuple(a['joint_names'])!=tuple(joint_names): raise ValueError('Gait cache joint order mismatch')
    n=len(a['q'])
    if not n: raise ValueError('Empty gait cache cannot seed PPO')
    sizes={'q':(20,),'q_vel':(20,),'q_cmd':(19,),'cube_pos':(3,),'cube_quat':(4,),
           'cube_linvel':(3,),'cube_angvel':(3,),'fifo':(history_steps,19),
           'grasp_command':(20,),'cube_anchor':(3,),'applied_actions':(19,),'previous_joint_command':(19,)}
    for k,shape in sizes.items():
        if a[k].shape!=(n,*shape) or not np.isfinite(a[k]).all(): raise ValueError(f'Invalid gait field {k}')
    if not np.allclose(np.linalg.norm(a['cube_quat'],axis=-1),1,atol=1e-4): raise ValueError('Nonunit quaternion')
    if (abs(a['fifo'])>1.00001).any(): raise ValueError('Invalid queued action')
    if not np.isin(a['phase'],range(7)).all(): raise ValueError('Unknown phase')
    return a,hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(env):
    from isaaclab.utils.math import quat_apply_inverse
    data={'q':env.hand.data.joint_pos[:,env.joint_ids], 'q_vel':env.hand.data.joint_vel[:,env.joint_ids],
          'q_cmd':env.joint_command,'cube_pos':env.cube_pos,'cube_quat':env.cube_quat,
          'cube_linvel':quat_apply_inverse(env.palm_quat_w,env.cube.data.root_lin_vel_w),
          'cube_angvel':quat_apply_inverse(env.palm_quat_w,env.cube.data.root_ang_vel_w),
          'fifo':env.action_queue.history,'grasp_command':env.grasp_command,'cube_anchor':env.cube_anchor,
          'applied_actions':env.applied_actions,'previous_joint_command':env.previous_joint_command,
          'contact_mask':env.tip_in_contact,'grasp_quality':env.grasp_quality_value,
          'palm_force':env.palm_contact_force}
    return {k:v.detach().cpu().numpy().copy() for k,v in data.items()}


def restore(env, arrays, picks, ids, joint_noise=0.,pos_noise=0.,orientation_noise=0.):
    """Restore after normal episode reset, with a fresh goal relative to CURRENT orientation.

    No cached task goal/counter is restored. Physics contact/solver warm starts cannot be
    serialized; the separate perturbation sanity test validates cold-contact reconstruction.
    GraspQualityReward.scale is batch-global, not per-env dynamic state; remains unchanged.
    """
    import torch
    from isaaclab.utils.math import quat_apply,quat_mul,quat_error_magnitude,quat_from_angle_axis
    dev=env.device
    def t(k): return torch.as_tensor(arrays[k],device=dev)[picks].clone()
    q=t('q');dq=t('q_vel');cmd=t('q_cmd'); cp=t('cube_pos');cq=t('cube_quat')
    if joint_noise: q+=(torch.rand_like(q)*2-1)*joint_noise
    if pos_noise: cp+=(torch.rand_like(cp)*2-1)*pos_noise
    if orientation_noise:
        axis=torch.randn_like(cp);axis/=axis.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        cq=quat_mul(quat_from_angle_axis((torch.rand(len(ids),device=dev)*2-1)*orientation_noise,axis),cq)
    limits=env.hand.data.joint_pos_limits[ids][:,env.joint_ids]
    q=q.clamp(limits[...,0],limits[...,1]);q[:,env.disabled_index]=env.cfg.disabled_joint_position
    dq[:,env.disabled_index]=0
    env.hand.write_joint_state_to_sim(q,dq,joint_ids=env.joint_ids,env_ids=ids)
    env.joint_command[ids]=cmd.clamp(env.lower[ids],env.upper[ids])
    env.previous_joint_command[ids]=t('previous_joint_command')
    env.grasp_command[ids]=t('grasp_command');env.cube_anchor[ids]=t('cube_anchor')
    env.joint_targets[ids]=t('grasp_command');env.joint_targets[ids[:,None],env.active_indices]=env.joint_command[ids]
    env.joint_targets[ids,env.disabled_index]=env.cfg.disabled_joint_position
    env.hand.set_joint_position_target(env.joint_targets[ids],joint_ids=env.joint_ids,env_ids=ids)
    env.action_queue.history[ids]=t('fifo');env.actions[ids]=t('fifo')[:,0]
    env.previous_actions[ids]=env.actions[ids];env.applied_actions[ids]=t('applied_actions');env.action_delta[ids]=0
    env.measured_at_command[ids]=q[:,env.active_indices]
    pose=torch.cat((env.palm_pos_w[ids]+quat_apply(env.palm_quat_w[ids],cp),quat_mul(env.palm_quat_w[ids],cq)),dim=-1)
    velocity=torch.cat((quat_apply(env.palm_quat_w[ids],t('cube_linvel')),quat_apply(env.palm_quat_w[ids],t('cube_angvel'))),dim=-1)
    env.cube.write_root_pose_to_sim(pose,env_ids=ids);env.cube.write_root_velocity_to_sim(velocity,env_ids=ids)
    env.goal_quat[ids]=env._sample_goals_for(ids,cq)
    error=quat_error_magnitude(cq,env.goal_quat[ids])
    env.previous_orientation_error[ids]=error;env.initial_orientation_error[ids]=error
    env.success_eligible[ids]=True;env.success.reset(ids)
    env.goals_completed[ids]=0;env.goals_issued[ids]=1;env.error_sum[ids]=0
    env.error_min[ids]=torch.inf;env.tip_contact_sum[ids]=0;env.steps_without_tip_contact[ids]=0;env.episode_returns[ids]=0
