# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Re-exported for convenience; it lives in a module with no Isaac imports because the training
# orchestrator has to read the thresholds without starting Kit.
from ..curriculum import STREAM_CURRICULUM, StreamCurriculumCfg  # noqa: F401


# Experiment B (reward v2) changes vs the first 500-it run (2026-09-16_23-16-44):
#   num_steps_per_env 16 -> 32, init_noise_std 1.0 -> 0.5, entropy_coef 0.005 -> 0.001,
#   actor/critic [32, 32] -> [128, 128], actor/critic_obs_normalization False -> True
#   (RSL-RL 3.1.2 EmpiricalNormalization, stored in the checkpoint). Everything else unchanged.
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 150
    save_interval = 50
    experiment_name = "dg5f_cube_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPOStreamRunnerCfg(PPORunnerCfg):
    """Reward-v4 goal-stream runner. Architecture deliberately identical to PPORunnerCfg.

    The reward, the goal generator and the reset distribution are the experiment; running an
    architecture sweep at the same time would make the result unattributable. So only the two
    exploration settings change, and both for a measured reason: the reward-v3 night run inflated
    action_std from 0.20 to 0.795 while deterministic performance did not improve, and its rare
    genuine over-limit joint velocities sat in exactly those over-excited checkpoints.

    RSL-RL 3.1.2 has no supported maximum-std option (only init_noise_std and noise_std_type), and
    patching its internals to add one is out of scope, so the ceiling is enforced as a watchdog in
    scripts/train_stream.py instead of silently in the network.
    """

    def __post_init__(self):
        # Guarded: configclass may inject its own __post_init__ for MISSING checks.
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        # A Gaussian policy at std 0.2 already explores; paying for extra entropy is what let the
        # previous run drift, since the advantage signal was weak and noisy.
        self.policy.init_noise_std = 0.2
        self.algorithm.entropy_coef = 0.0
