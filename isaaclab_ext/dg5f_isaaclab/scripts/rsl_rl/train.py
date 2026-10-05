# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--freeze_actor_iterations", type=int, default=0,
                    help="Train only the critic for this many iterations first. Needed after an "
                         "actor-only warm start: the fresh critic would otherwise move a competent "
                         "actor along advantages from a value function that has never seen this "
                         "reward. Counts towards --max_iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--restore_continuation_state", action="store_true",
                    help="Resume the saved adaptive LR and advance the last-completed iteration index")
parser.add_argument("--verify_resume_state", action="store_true",
                    help="Before training verify exact model/Adam/iteration/LR and unchanged env/reward config")
parser.add_argument("--verify_action_penalty_scale", type=float, default=None,
                    help="Explicit expected reward-only difference permitted by resume verification")
parser.add_argument("--verify_action_rate_penalty_scale", type=float, default=None)
parser.add_argument("--verify_reset_config", type=str, default=None,
                    help="JSON declaring the only allowed gait reset distribution differences")
parser.add_argument("--expected_resume_checkpoint", type=str, default=None)
parser.add_argument("--expected_actor_fingerprint", type=str, default=None)
parser.add_argument("--verify_hypothesis_config", type=str, default=None)
parser.add_argument("--reset_action_std", type=float, default=None,
                    help="Set the policy's exploration std after loading (and after resume verification). "
                         "For the bounds-loss arm: a mean pulled back into [-1, 1] under the inherited "
                         "std ~5.5 would make every rollout action a coin flip at +-1.")
parser.add_argument("--verify_task_config", type=str, default=None,
                    help="JSON of explicitly approved task changes (goal stage, rotation reward)")
parser.add_argument("--checkpoint_offsets", type=str, default="",
                    help="Extra checkpoint offsets after this resume, e.g. 100,300,400")
parser.add_argument("--verify_fixed_lr", type=float, default=None,
                    help="Assert fixed schedule and this actual PPO/Adam LR before and at every update")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.restore_continuation_state and not args_cli.resume:
    parser.error("--restore_continuation_state requires --resume")
if (args_cli.expected_resume_checkpoint or args_cli.expected_actor_fingerprint) and not args_cli.verify_resume_state:
    parser.error("Expected source/actor checks require --verify_resume_state")
if args_cli.verify_resume_state and not args_cli.restore_continuation_state:
    parser.error("--verify_resume_state requires --restore_continuation_state")
if any(v is not None for v in (args_cli.verify_action_penalty_scale, args_cli.verify_action_rate_penalty_scale,
                              args_cli.verify_fixed_lr)) and not args_cli.verify_resume_state:
    parser.error("Night verification options require --verify_resume_state")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import dg5f_isaaclab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.expected_resume_checkpoint:
            from pathlib import Path
            if Path(resume_path).resolve() != Path(args_cli.expected_resume_checkpoint).resolve():
                raise RuntimeError("Resolved checkpoint differs from explicitly expected source")
        # load previously trained model
        runner.load(resume_path)
        if args_cli.restore_continuation_state:
            from continuation_state import restore_continuation_state
            lr, next_iteration = restore_continuation_state(runner)
            agent_cfg.algorithm.learning_rate = lr
            print(f"[CONTINUE] restored adaptive lr={lr:.12g}; next iteration={next_iteration}", flush=True)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if args_cli.verify_resume_state:
        from resume_verification import verify_resume_start
        overrides = ({"action_penalty_scale": args_cli.verify_action_penalty_scale}
                     if args_cli.verify_action_penalty_scale is not None else None)
        if args_cli.verify_action_rate_penalty_scale is not None:
            overrides = {**(overrides or {}), "action_rate_penalty_scale": args_cli.verify_action_rate_penalty_scale}
        import json
        reset_overrides = json.load(open(args_cli.verify_reset_config)) if args_cli.verify_reset_config else None
        hypothesis_overrides = json.load(open(args_cli.verify_hypothesis_config)) if args_cli.verify_hypothesis_config else None
        task_overrides = json.load(open(args_cli.verify_task_config)) if args_cli.verify_task_config else None
        verified = verify_resume_start(runner, resume_path, log_dir, agent_cfg.algorithm.entropy_coef, overrides, reset_overrides, hypothesis_overrides, task_overrides)
        if args_cli.expected_actor_fingerprint and verified["actor_output_fingerprint"] != args_cli.expected_actor_fingerprint:
            raise RuntimeError("Deterministic actor fingerprint differs BEFORE optimization")
    if args_cli.reset_action_std is not None:
        import math
        if args_cli.reset_action_std <= 0:
            raise ValueError("reset_action_std must be positive")
        policy = runner.alg.policy
        before = float(policy.std.mean()) if policy.noise_std_type == "scalar" else float(policy.log_std.exp().mean())
        with torch.no_grad():
            if policy.noise_std_type == "scalar":
                policy.std.fill_(args_cli.reset_action_std)
            else:
                policy.log_std.fill_(math.log(args_cli.reset_action_std))
        print(f"[STD_RESET] action std {before:.4f} -> {args_cli.reset_action_std:.4f} (after verification)", flush=True)
    if args_cli.verify_fixed_lr is not None:
        expected_lr = args_cli.verify_fixed_lr
        def check_fixed_lr():
            if runner.alg.schedule != "fixed" or runner.alg.learning_rate != expected_lr:
                raise RuntimeError("Fixed PPO learning rate/schedule does not match expectation")
            if any(group["lr"] != expected_lr for group in runner.alg.optimizer.param_groups):
                raise RuntimeError("Fixed Adam learning rate does not match expectation")
        check_fixed_lr()
        original_log = runner.log
        def verified_log(locs, *args, **kwargs):
            check_fixed_lr()
            print(f"[FIXED_LR] iteration={locs['it']} lr={runner.alg.learning_rate:.17g} "
                  f"std={float(runner.alg.policy.std.mean()):.7f} stage={env_cfg.goal_stream_stage}", flush=True)
            return original_log(locs, *args, **kwargs)
        runner.log = verified_log
        print(f"[FIXED_LR_START] iteration={runner.current_learning_iteration} lr={expected_lr:.17g} "
              f"stage={env_cfg.goal_stream_stage}", flush=True)

    if args_cli.checkpoint_offsets:
        start_iteration = runner.current_learning_iteration
        requested_iterations = {start_iteration + int(x) - 1 for x in args_cli.checkpoint_offsets.split(',')}
        offset_log = runner.log
        def save_requested_checkpoints(locs, *args, **kwargs):
            result = offset_log(locs, *args, **kwargs)
            if locs['it'] in requested_iterations:
                runner.save(os.path.join(log_dir, f"model_{locs['it']}.pt"))
            return result
        runner.log = save_requested_checkpoints

    # run training
    remaining = agent_cfg.max_iterations
    if args_cli.freeze_actor_iterations:
        # Critic warm-up. Warm-starting the actor alone leaves the critic estimating the value of a
        # reward it has never seen, and PPO then moves a competent actor along advantages from a
        # systematically wrong value function. Measured on this task: the warm-start actor has a
        # deterministic drop_rate of 0.078 over full episodes, and 50 ordinary PPO iterations take
        # it to 0.977 while Loss/value_function is still 17 and falling (logs/stream/*_eval.json).
        # Freezing the actor's parameters simply leaves their .grad at None, so Adam skips them and
        # clip_grad_norm_ ignores them; only the critic moves.
        policy = runner.alg.policy
        frozen = [p for name, p in policy.named_parameters()
                  if name.startswith("actor.") or name in ("std", "log_std")]
        for parameter in frozen:
            parameter.requires_grad_(False)
        # The adaptive schedule keys off the policy KL, which is ~0 while the actor is frozen, so it
        # would drive the learning rate to its ceiling and hand that rate to the actor on unfreeze.
        schedule, learning_rate = runner.alg.schedule, agent_cfg.algorithm.learning_rate
        runner.alg.schedule = "fixed"
        runner.alg.learning_rate = learning_rate
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = learning_rate
        warmup = min(args_cli.freeze_actor_iterations, remaining)
        print(f"[INFO]: Critic warm-up: {warmup} iterations with {len(frozen)} actor tensors frozen,"
              f" schedule fixed at lr {learning_rate}")
        runner.learn(num_learning_iterations=warmup, init_at_random_ep_len=True)
        for parameter in frozen:
            parameter.requires_grad_(True)
        runner.alg.schedule = schedule
        runner.alg.learning_rate = learning_rate
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = learning_rate
        remaining -= warmup
        print(f"[INFO]: Critic warm-up done; actor unfrozen, {remaining} iterations to go")
    if remaining > 0:
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
