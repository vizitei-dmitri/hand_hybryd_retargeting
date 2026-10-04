"""Identical deterministic, stable-reset videos for gait-tree comparisons."""
import argparse,json,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--mode',default='none');p.add_argument('--gate',action='store_true');p.add_argument('--seed',type=int,default=31415);p.add_argument('--steps',type=int,default=3600)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.enable_cameras=True
app=AppLauncher(a);simulation_app=app.app
import gymnasium as gym,numpy as np,torch,imageio.v2 as imageio
from PIL import Image,ImageDraw
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import dg5f_isaaclab.tasks

def main():
    cfg=parse_env_cfg('DG5F-Cube-Stream-Direct-v0',device=a.device,num_envs=1);cfg.seed=a.seed
    cfg.gait_task_gate=a.gate;cfg.gait_observation_mode=a.mode;cfg.gait_transition_fraction=0.;cfg.goal_stream_stage='A';cfg.goal_marker=True;cfg.viewer.resolution=(960,720);cfg.resolve_control_config()
    env=RslRlVecEnvWrapper(gym.make('DG5F-Cube-Stream-Direct-v0',cfg=cfg,render_mode='rgb_array'));raw=env.unwrapped
    runner=OnPolicyRunner(env,load_yaml(str(a.checkpoint.parent/'params/agent.yaml')),log_dir=None,device=raw.device)
    print(f'Loading model checkpoint from: {a.checkpoint.resolve()}',flush=True);runner.load(str(a.checkpoint));policy=runner.get_inference_policy(device=raw.device)
    obs=env.get_observations();a.output.parent.mkdir(exist_ok=True,parents=True);episodes=0;rows=[]
    for _ in range(5):raw.sim.render()
    with imageio.get_writer(a.output,fps=30,codec='libx264',quality=7) as writer:
        for step in range(a.steps):
            mu=policy(obs);obs,_,done,_=env.step(mu)
            if step%2==0:
                im=Image.fromarray(raw.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,45),fill=(0,0,0));draw.text((12,12),f'{a.label} | deterministic | seed {a.seed} | ep {episodes+1} | t={step/60:.1f}s | tips={int(raw.tip_contact_count[0])} | goals={int(raw.goals_completed[0])}',fill='white');writer.append_data(np.asarray(im))
            if bool(done[0]):episodes+=1
            rows.append({'step':step,'tips':int(raw.tip_contact_count[0]),'goals':int(raw.goals_completed[0]),'done':bool(done[0]),'mean_abs_mu':float(mu.abs().mean())})
            if episodes>=3 and step>=1800:break
    a.output.with_suffix('.json').write_text(json.dumps({'checkpoint':str(a.checkpoint.resolve()),'stable_resets':True,'seed':a.seed,'label':a.label,'episodes_completed':episodes,'steps':len(rows),'trace':rows},indent=2));env.close()
if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    except Exception:traceback.print_exc();raise
    finally:simulation_app.close()
