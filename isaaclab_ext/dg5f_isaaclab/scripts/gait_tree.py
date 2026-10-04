"""Persistent, sequential decision-tree campaign. All inference evaluations use stable resets."""
import datetime, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import gait_h1 as base
from gait_multiseed import flatten, paired_bootstrap
ROOT=base.ROOT;OUT=ROOT/'logs/gait_tree';base.OUT=OUT
PY=sys.executable;LR=base.LR;CACHE=base.CACHE
SOURCE=ROOT/'logs/rsl_rl/dg5f_cube_direct/2026-10-02_18-15-45_gait_H1_B/model_3999.pt'
SEEDS=[1234,4321,2026,31415,27182,9876,5555,7777]

class Tree(base.Campaign):
    def __init__(self):
        self.s=json.loads((OUT/'night_state.json').read_text());self.source=SOURCE
        self.deadline=datetime.datetime.fromisoformat(self.s['training_deadline']).timestamp()
        self.end=datetime.datetime.fromisoformat(self.s['deadline']).timestamp()
        self.lease=base.reserve_gui_gpu(ROOT,wait=True)
        self.s.update(controller_pid=os.getpid(),tmux_session='gait_tree',root_checkpoint=str(SOURCE))
        for path,digest in json.loads((OUT/'FIXED_INPUT_HASHES.json').read_text()).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Fixed input changed: {path}')
        self.s['cache_hash']=hashlib.sha256(CACHE.read_bytes()).hexdigest()
        (OUT/'reset.json').write_text(json.dumps({'gait_transition_cache_path':str(CACHE),'gait_transition_fraction':.3}))
        self.save()
    def config(self,name,gate=False,mode='none'):
        p=OUT/(name+'_config.json');p.write_text(json.dumps({'gait_task_gate':gate,'gait_observation_mode':mode}));return p
    def evaluate(self,label,source,seed=1234,gate=False,mode='none'):
        path=OUT/(label+'.json');cfg=self.config(label,gate,mode)
        if not path.exists():
            cmd=[PY,'-u','scripts/eval_checkpoints.py','--task',base.TASK,'--headless','--num_envs','128','--seed',str(seed),'--baselines','','--stage','A','--goal_angle_deg','20','--gait_telemetry','--smoothing_telemetry','--gait_thresholds',str(ROOT/'logs/gait_research/gait_thresholds.json'),'--gait_hypothesis_config',str(cfg),'--output',str(path),'--run',f'{label}={source.parent}:{source.stem[6:]}']
            self.run(label,cmd)
        row=flatten(path)
        if row.get('episodes')!=128 or not row.get('actor_output_fingerprint'):
            raise RuntimeError('Incomplete deterministic evaluation or missing fingerprint')
        row.update(checkpoint=str(source),seed=seed,reset_distribution='stable',gate=gate,mode=mode)
        self.s['evaluations'][label]=row;self.save(latest_metrics=row)
        return row
    def panel(self,name,checkpoint,seeds=SEEDS[:3],gate=False,mode='none'):
        return [self.evaluate(f'{name}_s{seed}',checkpoint,seed,gate,mode) for seed in seeds]
    def train(self,name,source=SOURCE,updates=300,gate=False,mode='none',offsets=(150,300)):
        for path,digest in json.loads((OUT/'FIXED_INPUT_HASHES.json').read_text()).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Fixed input changed: {path}')
        if name in self.s['training']:return Path(self.s['training'][name]['checkpoint'])
        free=shutil.disk_usage(ROOT).free
        self.s.setdefault('disk_checks',[]).append({'phase':name,'free_bytes':free,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        if free < 8e9:
            removed=json.loads((OUT/'DELETIONS.json').read_text())
            for dump in sorted(OUT.glob('*_trajectory.npz'),key=lambda p:p.stat().st_mtime):
                if shutil.disk_usage(ROOT).free>=8e9:break
                aggregates=[p for p in OUT.glob('*.json') if dump.name.startswith(p.stem+'_') and '_config' not in p.stem]
                valid=[]
                for p in aggregates:
                    data=json.loads(p.read_text())
                    if isinstance(data,dict) and any(isinstance(v,dict) and v.get('episodes')==128 and v.get('gait') for v in data.values()):valid.append(p)
                if not valid:continue
                removed.append({'path':str(dump),'bytes':dump.stat().st_size,'sha256':hashlib.sha256(dump.read_bytes()).hexdigest(),'preserved_metrics':[str(p) for p in valid]})
                (OUT/'DELETIONS.json').write_text(json.dumps(removed,indent=2));dump.unlink()
        if shutil.disk_usage(ROOT).free < 3e9:raise RuntimeError('Disk reserve <3GB')
        cfg=self.config(name,gate,mode);tag='gait_tree_'+name
        cmd=[PY,'-u','scripts/rsl_rl/train.py','--task',base.TASK,'--headless','--num_envs','1024','--seed','42','--max_iterations',str(updates),'--resume','--load_run',source.parent.name,'--checkpoint',source.name,'--restore_continuation_state','--verify_resume_state','--verify_fixed_lr',repr(LR),'--verify_reset_config',str(OUT/'reset.json'),'--verify_hypothesis_config',str(cfg),'--checkpoint_offsets',','.join(map(str,offsets)),f'agent.run_name={tag}','agent.algorithm.schedule=fixed',f'agent.algorithm.learning_rate={LR!r}','agent.algorithm.entropy_coef=0.005','env.goal_stream_stage=A','agent.save_interval=500',f'env.gait_transition_cache_path={CACHE}','env.gait_transition_fraction=0.3',f'env.gait_task_gate={str(gate).lower()}',f'env.gait_observation_mode={mode}']
        fingerprints={v["actor_output_fingerprint"] for v in self.s["evaluations"].values() if Path(v["checkpoint"]).resolve()==source.resolve()}
        if len(fingerprints)!=1:raise RuntimeError(f"Need one verified source fingerprint before training: {fingerprints}")
        cmd += ["--expected_resume_checkpoint",str(source.resolve()),"--expected_actor_fingerprint",next(iter(fingerprints))]
        self.run(name,cmd,source,updates)
        dirs=[d for d in (ROOT/'logs/rsl_rl/dg5f_cube_direct').glob('*_'+tag) if (d/f'model_{int(source.stem[6:])+updates}.pt').exists()]
        if len(dirs)!=1:raise RuntimeError(f'Ambiguous final checkpoints: {dirs}')
        end=dirs[0]/f'model_{int(source.stem[6:])+updates}.pt'
        self.s['training'][name]={'source':str(source),'checkpoint':str(end),'updates':updates,'gate':gate,'mode':mode};self.save();return end
    def control(self):
        rows=self.panel('ROOT',SOURCE)
        self.s['baseline_actor_fingerprint']=rows[0]['actor_output_fingerprint'];self.save()
        end=self.train('CONTROL_LONGISH')
        for age in (150,300):self.panel(f'CONTROL_{age}',end.parent/f'model_{3999+age}.pt')
        self.save(status='CONTROL_COMPLETE',next_command='Run H2_GATE smoke from ROOT and compare matched ages.')

    def expanded(self,mode):
        destination=ROOT/'logs/rsl_rl/dg5f_cube_direct'/('gait_tree_root_'+mode)
        if not destination.exists():
            self.run('expand_'+mode,[PY,'scripts/expand_gait_checkpoint.py',str(SOURCE),str(destination),'--mode',mode])
        return destination/SOURCE.name
    def smoke(self,name,gate=False,mode='none'):
        source=SOURCE if mode=='none' else self.expanded(mode)
        self.panel(name+'_0',source,gate=gate,mode=mode)
        end=self.train(name,source,gate=gate,mode=mode)
        for age in (150,300):self.panel(f'{name}_{age}',end.parent/f'model_{3999+age}.pt',gate=gate,mode=mode)
        return end
    def comparison(self,name,control='CONTROL',seeds=SEEDS[:3]):
        a=[self.s['evaluations'][f'{control}_300_s{s}'] for s in seeds]
        b=[self.s['evaluations'][f'{name}_300_s{s}'] for s in seeds]
        metrics=['gait/successful_gait_cycles_per_episode','gait/successful_gait_cycle_rate','gait/recovery_to_3plus_rate','gait/meaningful_events_per_episode','gait/contact_switch_distance_p90','gait/two_tip_support_duration','goals_completed_per_episode','drop_rate']
        pairs={m:paired_bootstrap([y[m]-x[m] for x,y in zip(a,b)],draws=4000) for m in metrics}
        mean=lambda rows,key:sum(x[key] for x in rows)/len(rows)
        guard=mean(b,'goals_completed_per_episode')>=.75*mean(a,'goals_completed_per_episode') and mean(b,'drop_rate')<=mean(a,'drop_rate')+.10
        if control!='CONTROL':
            standard=[self.s['evaluations'][f'CONTROL_300_s{s}'] for s in seeds if f'CONTROL_300_s{s}' in self.s['evaluations']]
            guard=guard and mean(b,'goals_completed_per_episode')>=.75*mean(standard,'goals_completed_per_episode') and mean(b,'drop_rate')<=mean(standard,'drop_rate')+.10
        cyc='gait/successful_gait_cycles_per_episode';rec='gait/recovery_to_3plus_rate'
        positive=(guard and pairs[cyc]['mean_difference']>=max(.05,.20*mean(a,cyc)) and pairs[cyc]['seeds_positive']>= (6 if len(seeds)==8 else 2) and pairs['gait/successful_gait_cycle_rate']['mean_difference']>0)
        if name=='H2_GATE':positive=positive and pairs[rec]['mean_difference']>0 and pairs[rec]['seeds_positive']>=(6 if len(seeds)==8 else 2)
        result={'branch':name,'matched_control':control,'seeds':seeds,'guardrails_pass':guard,'positive':positive,'paired':pairs,'means_control':{m:mean(a,m) for m in metrics},'means_branch':{m:mean(b,m) for m in metrics}}
        (OUT/(name+('_full' if len(seeds)==8 else '_smoke')+'_comparison.json')).write_text(json.dumps(result,indent=2))
        return result
    def decision(self,name,result,next_branch,why):
        info=self.s['training'][name]
        fields={'HYPOTHESIS':name,'MECHANISM':{'gate':info['gate'],'observations':info['mode']},'CHECKPOINT':info['checkpoint'],'MATCHED CONTROL':result['matched_control']+' at +300','RESULT':result['means_branch'],'PRIMARY METRICS':result['paired'],'TASK GUARDRAILS':result['guardrails_pass'],'VERDICT':'PROMISING' if result['positive'] else 'NOT QUALIFIED','NEXT BRANCH':next_branch,'WHY':why}
        with (OUT/'DECISION_LOG.md').open('a') as f:
            for key,value in fields.items():f.write(key+': '+(json.dumps(value) if isinstance(value,(dict,list,bool)) else str(value))+'\n\n')
        self.s['decisions'].append(fields);self.save()
    def validate(self,name,control='CONTROL'):
        if control!='CONTROL':
            self.panel('CONTROL_300',Path(self.s['training']['CONTROL_LONGISH']['checkpoint']),SEEDS)
        cp=Path(self.s['training'][name]['checkpoint']);info=self.s['training'][name]
        control_cp=Path(self.s['training']['CONTROL_LONGISH' if control=='CONTROL' else control]['checkpoint'])
        ci=self.s['training']['CONTROL_LONGISH' if control=='CONTROL' else control]
        self.panel(control+'_300',control_cp,SEEDS,ci['gate'],ci['mode'])
        self.panel(name+'_300',cp,SEEDS,info['gate'],info['mode'])
        result=self.comparison(name,control,SEEDS)
        self.decision(name,result,'LONG' if result['positive'] else 'NEXT HYPOTHESIS','Full 8-seed paired replication before promotion; same predeclared practical effect and task guardrails.')
        return result['positive']
    def execute(self):
        self.control()
        candidates=[]
        # Initial hypotheses are isolated; interaction follows only a measured promising parent.
        h2=self.smoke('H2_GATE',gate=True);r2=self.comparison('H2_GATE')
        self.decision('H2_GATE',r2,'H2_H3' if r2['positive'] else 'H3_CONTACT_MASK','Gate must improve cycles and recovery; reducing two-tip time alone does not qualify.')
        if r2['positive']:
            self.smoke('H2_H3',gate=True,mode='contacts');combo=self.comparison('H2_H3')
            self.decision('H2_H3',combo,'FULL PANEL','Cheap contact-awareness interaction explicitly required after positive H2.')
            candidates=['H2_GATE']+(['H2_H3'] if combo['positive'] else [])
        else:
            self.smoke('H3_CONTACT_MASK',mode='contacts');r3=self.comparison('H3_CONTACT_MASK')
            self.decision('H3_CONTACT_MASK',r3,'H2_H3' if r3['positive'] else 'H4_GRASP_MECHANICS','Five binary flags isolate contact awareness. Geometry branch requires evidence of placement-specific failure; absent such evidence prioritize leave-one-out support mechanics.')
            if r3['positive']:
                self.smoke('H2_H3',gate=True,mode='contacts');combo=self.comparison('H2_H3')
                self.decision('H2_H3',combo,'FULL PANEL','Test whether knowledge of contacts needs task reward eligibility to complete recontact.')
                candidates=['H3_CONTACT_MASK']+(['H2_H3'] if combo['positive'] else [])
        selected=self.choose(candidates)
        elapsed=time.time()-datetime.datetime.fromisoformat(self.s['started_at']).timestamp()
        if selected is None and elapsed<7*3600:
            self.smoke('H4_GRASP_MECHANICS',mode='mechanics');r4=self.comparison('H4_GRASP_MECHANICS')
            self.decision('H4_GRASP_MECHANICS',r4,'H4_H2' if r4['positive'] and r4['paired']['gait/two_tip_support_duration']['mean_difference']>0 else 'FULL PANEL / H5','Global spectrum and five leave-one-out qualities test release selection; improved drop alone is insufficient.')
            candidates=['H4_GRASP_MECHANICS'] if r4['positive'] else []
            if candidates and r4['paired']['gait/two_tip_support_duration']['mean_difference']>0:
                self.smoke('H4_H2',gate=True,mode='mechanics');rc=self.comparison('H4_H2')
                self.decision('H4_H2',rc,'FULL PANEL','H4 gait improvement coexists with increased two-contact time, matching H2 mechanism.')
                if rc['positive']:candidates.append('H4_H2')
            selected=self.choose(candidates)
        elapsed=time.time()-datetime.datetime.fromisoformat(self.s['started_at']).timestamp()
        if selected is None and elapsed<7*3600:
            if 'H3_CONTACT_MASK' not in self.s['training']:
                self.smoke('H3_CONTACT_MASK',mode='contacts')
            self.smoke('H5_CONTACT_MASK_TIMERS',mode='timers');r5=self.comparison('H5_CONTACT_MASK_TIMERS','H3_CONTACT_MASK')
            self.decision('H5_CONTACT_MASK_TIMERS',r5,'FULL PANEL' if r5['positive'] else 'DIAGNOSTICS','Timers compared to flags at equal training age; no imposed frequency or periodic reward.')
            if r5['positive'] and self.validate('H5_CONTACT_MASK_TIMERS','H3_CONTACT_MASK'):selected='H5_CONTACT_MASK_TIMERS'
        self.s['selected_candidate']=selected;self.save()
        if selected:self.long_run(selected)
        # Report and videos are mandatory even when no hypothesis qualifies.
        self.finalize()
    def choose(self,candidates):
        candidates=sorted(candidates,key=lambda n:self.comparison(n)['means_branch']['gait/successful_gait_cycles_per_episode'],reverse=True)
        for name in candidates:
            if self.validate(name):return name
        return None
    def long_run(self,name):
        info=self.s['training'][name];source=Path(info['checkpoint']);bad=0
        for additional in range(500,3001,500):
            if self.deadline-time.time()<2400:break
            end=self.train(f'LONG_{name}_{additional}',source,500,info['gate'],info['mode'],(500,))
            rows=self.panel(f'LONG_{additional}',end,gate=info['gate'],mode=info['mode'])
            controls=[self.s['evaluations'][f'CONTROL_300_s{s}'] for s in SEEDS[:3]]
            avg=lambda rr,k:sum(x[k] for x in rr)/len(rr)
            useful=avg(rows,'goals_completed_per_episode')>=.75*avg(controls,'goals_completed_per_episode') and avg(rows,'drop_rate')<=avg(controls,'drop_rate')+.1 and avg(rows,'gait/successful_gait_cycles_per_episode')>avg(controls,'gait/successful_gait_cycles_per_episode')
            bad=0 if useful else bad+1
            self.s['long_curve']=[x for x in self.s.get('long_curve',[]) if x['additional']!=additional]
            self.s['long_curve'].append({'additional':additional,'checkpoint':str(end),'rows':rows,'useful':useful});self.save()
            source=end
            if bad>=2:break
    def finalize(self):
        self.save(status='FINALIZING',next_command='Full panel, videos, final report.')
        # Pick best measured long checkpoint by cycles under task guardrails; no weight tuning.
        selected=self.s.get('selected_candidate')
        if selected:
            eligible=[x for x in self.s.get('long_curve',[]) if x['useful']]
            if eligible:
                best=max(eligible,key=lambda x:sum(r['gait/successful_gait_cycles_per_episode'] for r in x['rows']))
                self.s['best_long_checkpoint']=best['checkpoint']
                i=self.s['training'][selected];self.panel('FINAL_LONG',Path(best['checkpoint']),SEEDS,i['gate'],i['mode'])
        self.panel('ROOT',SOURCE,SEEDS)
        control=Path(self.s['training']['CONTROL_LONGISH']['checkpoint']);self.panel('CONTROL_300',control,SEEDS)
        names=[n for n in self.s['training'] if n not in ('CONTROL_LONGISH',) and not n.startswith('LONG_')]
        feasible=[n for n in names if self.comparison(n,'H3_CONTACT_MASK' if n=='H5_CONTACT_MASK_TIMERS' else 'CONTROL')['guardrails_pass']]
        if feasible:
            best=selected or max(feasible,key=lambda n:self.comparison(n)['means_branch']['gait/successful_gait_cycles_per_episode'])
            self.s['best_short']=best;i=self.s['training'][best];self.panel(best+'_300',Path(i['checkpoint']),SEEDS,i['gate'],i['mode'])
        self.save()
        videos=[('ROOT',SOURCE,False,'none'),('CONTROL',control,False,'none')]
        if self.s.get('best_short'):
            i=self.s['training'][self.s['best_short']];videos.append(('BEST_SHORT',Path(i['checkpoint']),i['gate'],i['mode']))
        if self.s.get('best_long_checkpoint'):
            i=self.s['training'][selected];videos.append(('BEST_LONG',Path(self.s['best_long_checkpoint']),i['gate'],i['mode']))
        for label,checkpoint,gate,mode in videos:
            cmd=[PY,'scripts/gait_policy_video.py','--checkpoint',str(checkpoint),'--output',str(OUT/'videos'/f'{label}.mp4'),'--label',label,'--mode',mode]
            if gate:cmd.append('--gate')
            self.run('video_'+label,cmd)
        self.run('final_artifacts',[PY,'scripts/gait_tree_report.py'])
        self.save(status='AWAITING_VIDEO_REVIEW',final_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),next_command='Record and inspect deterministic comparison videos, then mark campaign complete.')

if __name__=='__main__':
    c=None
    try:
        c=Tree();c.execute()
    except BaseException:
        if c:c.save(status='ERROR',error=traceback.format_exc())
        raise
    finally:
        if c:c.lease.close()
