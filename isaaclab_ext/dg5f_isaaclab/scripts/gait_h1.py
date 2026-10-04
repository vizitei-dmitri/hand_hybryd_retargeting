"""Bounded controlled reset-distribution experiment; sequential GPU ownership, durable state."""
import datetime,hashlib,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/gait_research';PY=sys.executable
sys.path.insert(0,str(ROOT/'scripts/rsl_rl'))
from play_resource_guard import reserve_gui_gpu
TASK='DG5F-Cube-Stream-Direct-v0'
LR=.0002562890625000001
CACHE=ROOT/'source/dg5f_isaaclab/dg5f_isaaclab/assets/data/gait_transition_cache_robust.npz'

class Campaign:
    def __init__(self):
        self.s=json.loads((OUT/'night_state.json').read_text());self.source=Path(self.s['baseline_policy'])
        self.s.setdefault('evaluations',{});self.s.setdefault('training',{})
        self.deadline=datetime.datetime.fromisoformat(self.s['training_deadline']).timestamp()
        self.end=datetime.datetime.fromisoformat(self.s['deadline']).timestamp()
        self.lease=reserve_gui_gpu(ROOT,wait=True)
        self.s['cache_hash']=hashlib.sha256(CACHE.read_bytes()).hexdigest()
        self.s['controller_pid']=os.getpid();self.s['tmux_session']='gait_research'
    def save(self,**kw):
        self.s.update(kw,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),remaining_seconds=max(0,self.deadline-time.time()))
        self.s['git_diff']=subprocess.check_output(['git','diff','--stat'],cwd=ROOT,text=True)
        tmp=OUT/'state.tmp';tmp.write_text(json.dumps(self.s,indent=2));tmp.replace(OUT/'night_state.json')
        (OUT/'NIGHT_STATE.md').write_text('# Gait research\n\n```json\n'+json.dumps(self.s,indent=2)+'\n```\n')
    def run(self,label,cmd,source=None,updates=None):
        job=self.s['jobs'].get(label,{})
        if job.get('returncode')==0:return job
        path=OUT/(label+'.log');start=time.time()
        with path.open('w') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            job={'command':cmd,'log':str(path),'pid':child.pid,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            self.s['jobs'][label]=job
            self.save(status='RUNNING_IN_TMUX',phase=label,active_process=child.pid,checkpoint=str(source) if source else None,next_command='Continue the predeclared controlled gait experiment queue.')
            try:
                deadline=min(self.deadline if source else self.end,start+(max(1800,updates*12) if updates else 1800))
                while child.poll() is None:
                    if time.time()>deadline:raise TimeoutError(label)
                    text=path.read_text(errors='replace')
                    if 'Traceback (most recent call last)' in text or 'CUDA out of memory' in text:raise RuntimeError('Child failure: '+label)
                    lines=re.findall(r'\[FIXED_LR\] iteration=(\d+) lr=([^ ]+) std=([^ ]+)',text)
                    if lines:
                        it,lr,std=lines[-1];self.s['current_iteration']=int(it);self.s['current_std']=float(std)
                        if float(lr)!=LR or not 0<float(std)<10:raise RuntimeError('Invalid LR/std')
                    loaded=re.search(r'Loading model checkpoint from:\s*(.+)',text)
                    if source and loaded and Path(loaded[1].strip()).resolve()!=source.resolve():raise RuntimeError('Wrong resumed checkpoint')
                    verified=re.search(r'\[RESUME_VERIFIED\] (\{[^\n]+\})',text)
                    if source and verified:
                        report=json.loads(verified[1]);job['resume_verification']=report
                        if source==self.source and report['actor_output_fingerprint']!=self.s['baseline_actor_fingerprint']:raise RuntimeError('Actor fingerprint mismatch BEFORE optimizing')
                    self.save();time.sleep(5)
                if child.returncode:raise RuntimeError(f'{label} exited {child.returncode}')
                if source:
                    text=path.read_text(errors='replace')
                    if str(source.resolve()) not in text or '[RESUME_VERIFIED]' not in text:raise RuntimeError('Missing resume verification')
                    lines=re.findall(r'\[FIXED_LR\] iteration=(\d+) lr=([^ ]+) std=([^ ]+)',text)
                    if len(lines)!=updates or any(float(x[1])!=LR for x in lines):raise RuntimeError('Incomplete fixed LR trace')
                    job['lr_std_trace']=[{'iteration':int(i),'lr':float(lr),'std':float(std)} for i,lr,std in lines]
                job.update(returncode=0,seconds=time.time()-start)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                job['returncode']=child.returncode;raise
            finally:self.save(active_process=None)
        return job
    def evaluate(self,label,source,seed=1234,mixed=False):
        path=OUT/(label+'.json')
        if not path.exists():
            cmd=[PY,'-u','scripts/eval_checkpoints.py','--task',TASK,'--headless','--num_envs','128','--seed',str(seed),'--baselines','','--stage','A','--goal_angle_deg','20','--gait_telemetry','--smoothing_telemetry','--gait_thresholds',str(OUT/'gait_thresholds.json'),'--output',str(path),'--run',f'{label}={source.parent}:{source.stem[6:]}']
            if mixed:cmd+=['--gait_reset_config',str(OUT/'reset_B.json')]
            self.run(label,cmd)
        m=next(iter(json.loads(path.read_text()).values()))
        if m['episodes']!=128:raise RuntimeError('Incomplete deterministic evaluation')
        g=m['gait'];summary={k:m[k] for k in ('goals_completed_per_episode','target_completion_rate','drop_rate','mean_tip_contacts','action_std','actor_output_fingerprint')}
        summary.update({k:v for k,v in g.items() if k.startswith('gait/')});summary.update(support_qualified_goals_per_episode=g['support_qualified_goals_per_episode'],tip_count_fractions=g['tip_count_fractions'],palm_contact_fraction=g['palm_contact_fraction'],raw_mu=m['smoothing']['raw_mu']['mean'],saturation=m['smoothing']['command']['near_limit_fraction'],checkpoint=str(source),seed=seed,reset_distribution='mixed' if mixed else 'stable')
        self.s['evaluations'][label]=summary;self.save(latest_metrics=summary)
        return summary
    def train(self,label,source,updates,mixed):
        cfg='B' if mixed else 'A';tag='gait_'+label
        cmd=[PY,'-u','scripts/rsl_rl/train.py','--task',TASK,'--headless','--num_envs','1024','--seed','42','--max_iterations',str(updates),'--resume','--load_run',source.parent.name,'--checkpoint',source.name,'--restore_continuation_state','--verify_resume_state','--verify_fixed_lr',repr(LR),'--verify_reset_config',str(OUT/f'reset_{cfg}.json'),'--checkpoint_offsets',','.join(str(v) for v in (100,300,updates) if v<=updates),f'agent.run_name={tag}','agent.algorithm.schedule=fixed',f'agent.algorithm.learning_rate={LR!r}','agent.algorithm.entropy_coef=0.005','env.goal_stream_stage=A','agent.save_interval=500']
        if mixed:cmd += [f'env.gait_transition_cache_path={CACHE}','env.gait_transition_fraction=0.3']
        job=self.run(label,cmd,source,updates)
        dirs=list((ROOT/'logs/rsl_rl/dg5f_cube_direct').glob('*_'+tag))
        # A crashed earlier attempt leaves an EMPTY run directory with the same tag, and erroring on
        # the glob then throws away a 400-iteration job that actually finished. Keep the safety
        # intent -- never guess between two real runs -- but ignore directories with no checkpoints.
        if len(dirs)>1:dirs=[d for d in dirs if any(d.glob('model_*.pt'))]
        if len(dirs)!=1:raise RuntimeError(f'Ambiguous run dirs {dirs}')
        start=int(source.stem[6:]);end=dirs[0]/f'model_{start+updates}.pt'
        if not end.exists():raise RuntimeError('Missing final checkpoint')
        self.s['training'][label]={'source':str(source),'checkpoint':str(end),'updates':updates,'mixed':mixed,'seed':42};self.save()
        return end
    def execute(self):
        (OUT/'reset_A.json').write_text('{}\n')
        (OUT/'reset_B.json').write_text(json.dumps({'gait_transition_cache_path':str(CACHE),'gait_transition_fraction':.3}))
        design={'baseline':str(self.source),'baseline_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),'cache_sha256':self.s['cache_hash'],'num_envs':1024,'training_seeds':[42],'evaluation_seeds':[1234,4321],'updates':400,'common_PPO':{'schedule':'fixed','learning_rate':LR,'entropy_coef':.005},'reward':'unchanged baseline .002 action magnitude, .001 action rate; no support gate, no raw-mu term','reset_A':'100% robust stable','reset_B':'70% robust stable + 30% robust gait','phase_weights':[0,.125,.125,.35,.25,.15,0],'primary_evaluation_distribution':'common stable resets, to avoid counting scripted initial transitions as learned skill','H1_signal_rule':'B meaningful events >= max(1.5*A, A+0.5), successful cycles >= A, goals >= 50% baseline, drop <= A+0.10; verify on second seed before extending','long_stop':'two consecutive evals without gait advantage or drop > A+0.10; maximum 3000 extra updates initially','followups':'H2 only if real gait signal but persistent 2-tip occupancy; H3 conditional on H1 and H2 outcomes'}
        (OUT/'EXPERIMENT_DESIGN.json').write_text(json.dumps(design,indent=2))
        a0=self.evaluate('H1_A_0',self.source);b0=self.evaluate('H1_B_0',self.source)
        if a0['actor_output_fingerprint']!=b0['actor_output_fingerprint']:raise RuntimeError('Iteration-0 actor mismatch')
        self.s['baseline_actor_fingerprint']=a0['actor_output_fingerprint'];self.save()
        self.evaluate('H1_B_0_mixed',self.source,mixed=True)
        endpoints={}
        for branch,mixed in [('A',False),('B',True)]:
            end=self.train('H1_'+branch,self.source,400,mixed);endpoints[branch]=end
            for updates in (100,300,400):self.evaluate(f'H1_{branch}_{updates}',end.parent/f'model_{3599+updates}.pt')
        a=self.s['evaluations']['H1_A_400'];b=self.s['evaluations']['H1_B_400']
        for name,source in [('BASELINE',self.source),('STABLE_ONLY',endpoints['A']),('MIXED_RESETS',endpoints['B'])]:
            self.evaluate('FINAL_'+name+'_4321',source,4321)
        def signal(a,b):
            return b['gait/meaningful_events_per_episode']>=max(1.5*a['gait/meaningful_events_per_episode'],a['gait/meaningful_events_per_episode']+.5) and b['gait/successful_gait_cycles_per_episode']>=a['gait/successful_gait_cycles_per_episode'] and b['goals_completed_per_episode']>=.5*a0['goals_completed_per_episode'] and b['drop_rate']<=a['drop_rate']+.10
        supported=signal(a,b) and signal(self.s['evaluations']['FINAL_STABLE_ONLY_4321'],self.s['evaluations']['FINAL_MIXED_RESETS_4321'])
        self.s['hypothesis_status']['H1']='supported_on_two_eval_seeds' if supported else 'not_supported_by_400_iteration_smoke'
        self.save(status='H1_SMOKE_COMPLETE',next_command='Review H1 measurements before any conditional continuation/H2/H3.',endpoints={k:str(v) for k,v in endpoints.items()})
        # Deliberately leave scientific follow-up choice to the active agent after the measured smoke.
        # No unattended unrelated jobs are launched. The root agent continues from this durable phase.

if __name__=='__main__':
    c=None
    try:c=Campaign();c.execute()
    except BaseException:
        if c:c.save(status='H1_ERROR',error=traceback.format_exc(),next_command='Inspect error; repair only the failed phase and resume without repeating completed jobs.')
        raise
    finally:
        if c:c.lease.close()
