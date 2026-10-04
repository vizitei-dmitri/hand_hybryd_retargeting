"""First-episode contact geometry and event metrics. No reward/control mutation."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def stats(values):
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if not len(v):
        return {'count': 0}
    return {'count': len(v), 'mean': float(v.mean()), 'max': float(v.max()),
            **{f'p{p}': float(np.percentile(v,p)) for p in (50,90,95,99)}}


class GaitTelemetry:
    def __init__(self, raw):
        self.raw = raw
        self.frames = defaultdict(list)
        self.done = np.zeros(raw.num_envs, bool)

    def sample(self):
        import torch
        from isaaclab.utils.math import quat_apply_inverse
        r = self.raw
        def local(x):
            return quat_apply_inverse(r.cube.data.root_quat_w[:,None].expand(-1,5,-1).reshape(-1,4),
                       (x-r.cube.data.root_pos_w[:,None]).reshape(-1,3)).reshape(r.num_envs,5,3)
        points = torch.stack([s.data.contact_pos_w[:,0,0] for s in list(r.contact_sensors.values())[:5]],1)
        frame = {'contact':r.tip_in_contact, 'contact_local':local(points),
                 'tip_local':local(r.hand.data.body_pos_w[:,r.tip_ids]),
                 'tip_velocity':r.hand.data.body_lin_vel_w[:,r.tip_ids],
                 'q':r.hand.data.joint_pos[:,r.joint_ids], 'cube_quat':r.cube.data.root_quat_w,
                 'cube_linvel':r.cube.data.root_lin_vel_w, 'cube_angvel':r.cube.data.root_ang_vel_w,
                 'palm':r.palm_contact_force>r.cfg.tip_contact_force_n,
                 'quality':r.grasp_quality_value, 'goals':r.goals_completed,
                 'drop':r.reset_terminated, 'terminal':r.reset_terminated|r.reset_time_outs}
        for key,val in frame.items():
            self.frames[key].append(val.detach().cpu().numpy().copy())
        self.frames['valid'].append(~self.done.copy())
        self.done |= frame['terminal'].cpu().numpy()

    def save(self, path, thresholds=None):
        a = {k:np.stack(v) for k,v in self.frames.items()}
        a['dt'] = np.array(self.raw.step_dt)
        np.savez_compressed(path, **a)
        return analyze(a, thresholds)


def raw_events(a):
    dt=float(a['dt']); events=[]
    for e in range(a['contact'].shape[1]):
        length=int(a['valid'][:,e].sum())
        c=a['contact'][:length,e]
        for f in range(5):
            starts=np.flatnonzero(c[:-1,f]&~c[1:,f])+1
            ends=np.flatnonzero(~c[:-1,f]&c[1:,f])+1
            for start in starts:
                later=ends[ends>start]
                if not len(later): continue
                end=int(later[0]); old=a['contact_local'][start-1,e,f]; new=a['contact_local'][end,e,f]
                if not (np.isfinite(old).all() and np.isfinite(new).all()): continue
                other=c[start:end+1].copy(); other[:,f]=False
                # Same support fingers must persist; two changing contacts is weaker evidence.
                persistent=int((other.mean(0)>=.9).sum())
                support=(other.sum(-1)>=2)
                q0=a['cube_quat'][start-1,e]; q1=a['cube_quat'][end,e]
                angle=2*np.arccos(np.clip(abs(np.dot(q0,q1)),0,1))
                positions=a['tip_local'][start:end+1,e,f]
                outside=np.maximum(abs(positions)-.03,0)
                after=min(length,end+1+round(.5/dt))
                lv=a['cube_linvel'][start-1:after,e]; av=a['cube_angvel'][start-1:after,e]
                events.append({'env':e,'finger':f,'start':int(start),'end':end,'duration_s':(end-start)*dt,
                    'old_contact_local':old.tolist(),'new_contact_local':new.tolist(),
                    'displacement_m':float(np.linalg.norm(new-old)),
                    'max_tip_surface_distance_m':float(np.linalg.norm(outside,axis=-1).max()),
                    'other_support_fraction':float(support.mean()), 'persistent_support_fingers':persistent,
                    'support_mask':(other.mean(0)>=.9).tolist(), 'min_other_tips':int(other.sum(-1).min()),
                    'cube_rotation_rad':float(angle),
                    'cube_linear_acceleration_max':float(np.linalg.norm(np.diff(lv,axis=0)/dt,axis=-1).max(initial=0)),
                    'cube_angular_acceleration_max':float(np.linalg.norm(np.diff(av,axis=0)/dt,axis=-1).max(initial=0)),
                    'palm':bool(a['palm'][start:after,e].any()),
                    'drop_soon':bool(a['drop'][end:after,e].any()),
                    'followup_complete':after-end-1>=round(.5/dt),
                    'recovered_3plus':bool((c[end:after].sum(-1)>=3).any()),
                    'continued_goals':int(a['goals'][after-1,e]-a['goals'][end,e]),
                    'contact_stable_s':float(next((i for i,v in enumerate(c[end:,f]) if not v),length-end)*dt)})
    return events


def calibrate(events, dt):
    if not events: raise ValueError('No release/recontact events: cannot calibrate jitter')
    steps=np.array([round(x['duration_s']/dt) for x in events]); log=np.log(steps)
    centers=np.percentile(log,[20,80])
    for _ in range(30):
        labels=np.argmin(abs(log[:,None]-centers),axis=1)
        new=np.array([log[labels==i].mean() if (labels==i).any() else centers[i] for i in range(2)])
        if np.allclose(new,centers): break
        centers=new
    boundary=max(1,int(np.floor(np.exp(np.mean(centers)))))
    # Shortest-duration mode, separated in log duration; freeze before generating cache/A-B.
    jitter=[x['displacement_m'] for x,s in zip(events,steps) if s<=boundary]
    p95,p99=np.percentile(jitter,[95,99])
    margin=max(float(p99-p95), float(np.finfo(np.float32).eps*.06))
    return {'release_min_steps':boundary+1,'release_min_s':(boundary+1)*dt,
            'baseline_contact_jitter_distance':float(p99),'safety_margin_m':margin,
            'meaningful_displacement_m':float(p99+margin),
            'method':'two clusters of log release duration; shortest cluster p99 displacement plus p99-p95 margin',
            'duration_cluster_centers_steps':np.exp(centers).tolist(),
            'release_duration_hist_steps':{str(s):int((steps==s).sum()) for s in np.unique(steps)},
            'jitter_sample_count':len(jitter)}


def analyze(a, thresholds=None):
    events=raw_events(a); dt=float(a['dt']); n=a['contact'].shape[1]
    thresholds=thresholds or calibrate(events,dt)
    valid=a['valid']; count=a['contact'].sum(-1); masks=(a['contact']*(1<<np.arange(5))).sum(-1)
    for ev in events:
        ev['meaningful']=(ev['duration_s']+1e-8>=thresholds['release_min_s'] and
            ev['displacement_m']>thresholds['meaningful_displacement_m'] and
            ev['other_support_fraction']>=.9 and ev['persistent_support_fingers']>=2 and not ev['palm'])
        ev['successful_cycle']=(ev['meaningful'] and ev['recovered_3plus'] and
            ev['contact_stable_s']>=thresholds['release_min_s'] and not ev['drop_soon'] and
            ev['followup_complete'] and ev['continued_goals']>0)
    meaningful=[x for x in events if x['meaningful']]; cycles=[x for x in events if x['successful_cycle']]
    graph=defaultdict(int); distinct=[]; recovery=[]; support_goals=0; goals_with_switch=0; goals_total=0
    for e in range(n):
        length=int(valid[:,e].sum()); m=masks[:length,e]; distinct.append(len(np.unique(m)))
        for u,v in zip(m[:-1],m[1:]):
            if u!=v: graph[f'{u:05b}->{v:05b}']+=1
        cnt=count[:length,e]; entries=np.flatnonzero((cnt[1:]==2)&(cnt[:-1]>=3))+1
        for t in entries:
            ahead=cnt[t:min(length,t+round(.5/dt))]; recovery.append(bool((ahead>=3).any()))
        successes=np.flatnonzero(np.diff(a['goals'][:length,e],prepend=0)>0)
        previous=0
        for t in successes:
            goals_total+=1
            dwell=slice(max(0,t-round(.3/dt)+1),t+1)
            support_goals+=int((cnt[dwell]>=3).all() and not a['palm'][dwell,e].any())
            goals_with_switch+=int(any(x['env']==e and previous<=x['end']<=t for x in meaningful)); previous=t+1
    out={'thresholds':thresholds,'episodes':n,'events':events,'contact_geometry':'PhysX ContactSensor contact_pos_w centroid in cube coordinates; no proxy and no force weighting',
         'gait/events_per_episode':len(events)/n,'gait/meaningful_events_per_episode':len(meaningful)/n,
         'gait/successful_gait_cycles_per_episode':len(cycles)/n,
         'gait/successful_gait_cycle_rate':len(cycles)/max(1,len(meaningful)),
         'gait/contact_switch_distance_mean':stats([x['displacement_m'] for x in events]).get('mean',0),
         'gait/contact_switch_distance_p50':stats([x['displacement_m'] for x in events]).get('p50',0),
         'gait/contact_switch_distance_p90':stats([x['displacement_m'] for x in events]).get('p90',0),
         'gait/release_duration_mean':stats([x['duration_s'] for x in events]).get('mean',0),
         'gait/two_tip_support_duration':float(((count==2)&valid).sum()*dt/n),
         'gait/recovery_to_3plus_rate':float(np.mean(recovery)) if recovery else 0,
         'gait/distinct_fingers_switched_per_episode':float(np.mean([len({x['finger'] for x in meaningful if x['env']==e}) for e in range(n)])),
         'gait/goals_with_meaningful_switch_fraction':goals_with_switch/max(1,goals_total),
         'gait/distinct_masks_per_episode':float(np.mean(distinct)), 'contact_graph':dict(graph),
         'support_qualified_goals_per_episode':support_goals/n,
         'tip_count_fractions':{str(k):float(((count==k)&valid).sum()/valid.sum()) for k in range(6)},
         'palm_contact_fraction':float(a['palm'][valid].mean()),'grasp_quality':stats(a['quality'][valid]),
         'distributions':{k:stats([x[k] for x in events]) for k in ('duration_s','displacement_m','max_tip_surface_distance_m','cube_rotation_rad','cube_linear_acceleration_max','cube_angular_acceleration_max')},
         'per_finger':{str(f):{'events':sum(x['finger']==f for x in events),'meaningful':sum(x['finger']==f for x in meaningful),
             'displacement':stats([x['displacement_m'] for x in events if x['finger']==f]),
             'successful_cycles':sum(x['finger']==f for x in cycles),
             'release_count':int(((a['contact'][:-1,:,f]&~a['contact'][1:,:,f])&valid[1:]).sum()),
             'support_frequency':float(a['contact'][:,:,f][valid].mean())} for f in range(5)}}
    out['contact_mask_occupancy']={f'{i:05b}':float(((masks==i)&valid).sum()/valid.sum()) for i in range(32)}
    loops=defaultdict(int)
    for e in range(n):
        m=masks[:int(valid[:,e].sum()),e]
        sequence=m[np.r_[True,m[1:]!=m[:-1]]]
        for u,v,w in zip(sequence[:-2],sequence[1:-1],sequence[2:]):
            if u==w:loops[f'{u:05b}->{v:05b}->{w:05b}']+=1
    out['contact_mask_return_loops']=dict(loops)
    out['event_filters']={'closed_recontacts':len(events),'long_enough':sum(x['duration_s']+1e-8>=thresholds['release_min_s'] for x in events),
        'beyond_jitter':sum(x['displacement_m']>thresholds['meaningful_displacement_m'] for x in events),
        'persistent_support':sum(x['persistent_support_fingers']>=2 and x['other_support_fraction']>=.9 for x in events),
        'meaningful':len(meaningful),'stable_recontact_among_meaningful':sum(x['contact_stable_s']>=thresholds['release_min_s'] for x in meaningful),
        'continued_goal_among_meaningful':sum(x['continued_goals']>0 for x in meaningful),'successful_cycles':len(cycles)}
    out['velocity_limits_for_cache']={'linear_p95':stats(np.linalg.norm(a['cube_linvel'][valid],axis=-1)).get('p95',0),
                                      'angular_p95':stats(np.linalg.norm(a['cube_angvel'][valid],axis=-1)).get('p95',0)}
    if events:
        # Same isotropic Gramian as H4, evaluated just BEFORE release, leaving released finger out.
        pos=np.stack([a['tip_local'][x['start']-1,x['env']] for x in events])/.03
        mask=np.stack([a['contact'][x['start']-1,x['env']] for x in events]).copy()
        for i,ev in enumerate(events):mask[i,ev['finger']]=False
        skew=np.zeros((*pos.shape[:2],3,3));x,y,z=np.moveaxis(pos,-1,0)
        skew[:,:,0,1]=-z;skew[:,:,0,2]=y;skew[:,:,1,0]=z;skew[:,:,1,2]=-x;skew[:,:,2,0]=-y;skew[:,:,2,1]=x
        basis=np.concatenate((np.broadcast_to(np.eye(3),skew.shape),skew),axis=-2)
        gram=((basis@np.swapaxes(basis,-1,-2))*mask[:,:,None,None]).sum(1)
        loo=np.maximum(np.linalg.eigvalsh(gram)[:,0],0)/5
        out['leave_one_out_quality_at_release']={str(f):stats(loo[[i for i,ev in enumerate(events) if ev['finger']==f]]) for f in range(5)}
    return json.loads(json.dumps(out, default=lambda value: value.item()))
