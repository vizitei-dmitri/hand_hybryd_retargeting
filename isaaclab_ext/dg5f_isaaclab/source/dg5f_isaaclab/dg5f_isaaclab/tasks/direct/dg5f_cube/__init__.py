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

gym.register(
    id="DG5F-Cube-Fingertip-Direct-v0",
    entry_point=f"{__name__}.dg5f_cube_env:DG5FCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dg5f_cube_env_cfg:DG5FCubeFingertipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Reward v4: a stream of fixed 20 deg goals that must be reached, braked into and held.
# Registered separately from the task above so that task and its night run stay reproducible.
gym.register(
    id="DG5F-Cube-Stream-Direct-v0",
    entry_point=f"{__name__}.dg5f_cube_env:DG5FCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dg5f_cube_env_cfg:DG5FCubeStreamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOStreamRunnerCfg",
    },
)

# The continuous-rotation reference (Khandate 2022 / AnyRotate): same env config class, so the
# physics and control are identical; the bounded-mean PPO runner is what differs at registration.
gym.register(
    id="DG5F-Cube-Stream-Bounded-Direct-v0",
    entry_point=f"{__name__}.dg5f_cube_env:DG5FCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dg5f_cube_env_cfg:DG5FCubeStreamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOStreamBoundedRunnerCfg",
    },
)
