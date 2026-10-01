"""DG5F asset configuration for the upstream solved in-hand task."""
import gymnasium as gym

gym.register(
    id="Isaac-Repose-Cube-DG5F-Direct-v0",
    entry_point="isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dg5f_repose_env_cfg:DG5FReposeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.dg5f_repose_env_cfg:DG5FReposePPORunnerCfg",
    },
)
