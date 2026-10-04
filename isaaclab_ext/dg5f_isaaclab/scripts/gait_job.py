"""One durable, bounded GPU job under the shared campaign lease."""
import argparse, datetime, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'logs/gait_research'
sys.path.insert(0,str(ROOT/'scripts/rsl_rl'))
from play_resource_guard import reserve_gui_gpu
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--timeout',type=int,default=3600);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
state=json.loads((OUT/'night_state.json').read_text())
def save(**kw):
    state.update(kw,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state['remaining_seconds']=max(0,datetime.datetime.fromisoformat(state['deadline']).timestamp()-time.time())
    state['git_diff']=subprocess.check_output(['git','diff','--stat'],cwd=ROOT,text=True)
    tmp=OUT/'state.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(OUT/'night_state.json')
    (OUT/'NIGHT_STATE.md').write_text('# Gait research\n\n```json\n'+json.dumps(state,indent=2)+'\n```\n')
lease=reserve_gui_gpu(ROOT,wait=True)
log=OUT/(a.label+'.log'); deadline=min(time.time()+a.timeout,datetime.datetime.fromisoformat(state['deadline']).timestamp())
with log.open('w') as out:
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    state['jobs'][a.label]={'command':command,'pid':child.pid,'log':str(log),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(status='RUNNING_IN_TMUX',phase=a.label,active_process=child.pid,tmux_session='gait_research',next_command='Inspect this job result, then continue the prescribed experiment queue.')
    try:
        while child.poll() is None:
            if time.time()>deadline: raise TimeoutError(a.label)
            # Kit can hang during shutdown after an exception; never leave that holding GPU.
            if 'Traceback (most recent call last)' in log.read_text(errors='replace')[-30000:]:
                raise RuntimeError('Child traceback; inspect '+str(log))
            save();time.sleep(15)
    except BaseException:
        os.killpg(child.pid,signal.SIGTERM)
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        state['jobs'][a.label]['returncode']=child.poll()
        save(status='JOB_FINISHED' if child.returncode==0 else 'JOB_FAILED',active_process=None)
        lease.close()
sys.exit(child.returncode)
