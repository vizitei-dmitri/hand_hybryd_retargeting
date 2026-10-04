"""Launch a named bounded experiment in the project's portable tmux."""
import os,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]; tools=root/'logs/gait_research/tools/extracted'
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(tools/'usr/lib/x86_64-linux-gnu')
py='/home/yoba/Documents/work/IsaacLab/env_isaaclab/bin/python'
cmd=['env','-u','LD_LIBRARY_PATH','systemd-inhibit','--what=sleep','--mode=block',py,'scripts/gait_job.py',*sys.argv[1:]]
subprocess.run([str(tools/'usr/bin/tmux'),'new-session','-d','-s','gait_research','-c',str(root),shlex.join(cmd)],env=env,check=True)
print('Launched in tmux gait_research')
