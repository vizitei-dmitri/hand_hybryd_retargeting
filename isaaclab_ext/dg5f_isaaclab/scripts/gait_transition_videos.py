"""Render physical action replays, and independently classify replay success."""
import argparse,json,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--count',type=int,default=8)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.headless=True;args.enable_cameras=True
app=AppLauncher(args);simulation_app=app.app
import numpy as np,torch,imageio.v2 as imageio
from PIL import Image,ImageDraw
from isaaclab_tasks.utils import parse_env_cfg
import dg5f_isaaclab.tasks
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env import DG5FCubeEnv
from dg5f_isaaclab.assets.gait_cache import load_gait_cache,restore
from gait_metrics import GaitTelemetry

def main():
    cfg=parse_env_cfg('DG5F-Cube-Stream-Direct-v0',device=args.device,num_envs=1);cfg.track_gait_contact_points=True
    cfg.viewer.resolution=(960,720)
    env=DG5FCubeEnv(cfg,render_mode='rgb_array');a,sha=load_gait_cache(args.cache,cfg.actuated_joint_names)
    replay=np.load(args.replay);thresholds=json.loads(Path('logs/gait_research/gait_thresholds.json').read_text())
    out=Path('logs/gait_research/videos');out.mkdir(exist_ok=True);results=[];accepted=0
    starts=np.flatnonzero(a['phase']==0)
    # Round robin by moving finger so the first eight do not all show the same finger.
    groups={int(f):list(starts[a['moving_finger'][starts]==f]) for f in np.unique(a['moving_finger'])};ordered=[]
    while any(groups.values()):
        for g in groups.values():
            if g:ordered.append(g.pop(0))
    for index in ordered:
        env.reset();restore(env,a,torch.tensor([index],device=env.device),torch.tensor([0],device=env.device));env._compute_state()
        for _ in range(5):env.sim.render()
        tid=int(a['transition_id'][index]);finger=int(a['moving_finger'][index]);steps=replay['actions'][tid]
        telemetry=GaitTelemetry(env);original=env._get_rewards
        def measured():
            value=original();telemetry.sample();return value
        env._get_rewards=measured
        path=out/f'transition_{tid:03d}_finger{finger+1}.mp4'
        with imageio.get_writer(path,fps=30,codec='libx264',quality=7) as writer:
            for s,action in enumerate(steps):
                env.step(torch.as_tensor(action[None],device=env.device))
                if s%2==0:
                    frame=env.render();im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
                    phase='DETACH' if s<36 else 'TRANSFER' if s<108 else 'RECONTACT' if s<150 else 'HOLD'
                    draw.rectangle((0,0,960,45),fill=(0,0,0));draw.text((12,12),f'Physical replay | finger {finger+1} | {phase} | t={s/60:.2f}s | tips={int(env.tip_contact_count[0])}',fill='white')
                    writer.append_data(np.asarray(im))
        env._get_rewards=original
        metric=telemetry.save(out/f'transition_{tid:03d}_trajectory.npz',thresholds)
        valid=[x for x in metric['events'] if x['finger']==finger and x['meaningful'] and x['recovered_3plus'] and not x['drop_soon']]
        row={'transition':tid,'finger':finger,'video':str(path),'replay_meaningful':bool(valid),'events':valid};results.append(row)
        if valid:accepted+=1
        (out/'manifest.json').write_text(json.dumps({'cache_sha256':sha,'successful_replays':accepted,'videos':results},indent=2))
        print(f'[GAIT_VIDEO] id={tid} finger={finger+1} replay_ok={bool(valid)} successes={accepted}',flush=True)
        if accepted>=args.count:break
    env.close()
if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    except Exception:traceback.print_exc();raise
    finally:simulation_app.close()
