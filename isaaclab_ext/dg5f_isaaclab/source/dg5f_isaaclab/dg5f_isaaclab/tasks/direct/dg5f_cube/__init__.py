"""Register the state-only DG5F cube PPO teacher."""

import gymnasium as gym

from . import agents

gym.register(
    id="DG5F-Cube-Direct-v0",
    entry_point=f"{__name__}.dg5f_cube_env:DG5FCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dg5f_cube_env_cfg:DG5FCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
