"""Validate DG5F's upstream-task layout and physical asset before its PPO smoke."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
import dg5f_isaaclab.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg
from dg5f_isaaclab.tasks.direct.dg5f_cube.dg5f_cube_env_cfg import DG5FCubeEnvCfg


def main():
    cfg = parse_env_cfg('Isaac-Repose-Cube-DG5F-Direct-v0', device=args.device, num_envs=args.num_envs)
    upstream = ShadowHandEnvCfg()
    same = ('dist_reward_scale', 'rot_reward_scale', 'rot_eps', 'action_penalty_scale', 'reach_goal_bonus',
            'fall_penalty', 'fall_dist', 'success_tolerance', 'max_consecutive_success', 'av_factor',
            'decimation', 'episode_length_s', 'act_moving_average', 'reset_position_noise',
            'reset_dof_pos_noise', 'reset_dof_vel_noise', 'asymmetric_obs', 'obs_type')
    for name in same:
        assert getattr(cfg, name) == getattr(upstream, name), name
    original = DG5FCubeEnvCfg()
    assert cfg.object_cfg.spawn.size == (0.06, 0.06, 0.06)
    assert cfg.object_cfg.spawn.mass_props.mass == 0.05
    for attribute in ('stiffness', 'damping', 'armature', 'friction', 'effort_limit_sim', 'velocity_limit_sim'):
        actual = getattr(cfg.robot_cfg.actuators['fingers'], attribute)
        expected = getattr(original.robot_cfg.actuators['fingers'], attribute)
        for name in cfg.actuated_joint_names:
            assert actual[name] == expected[name], (attribute, name)
    env = gym.make('Isaac-Repose-Cube-DG5F-Direct-v0', cfg=cfg)
    raw = env.unwrapped
    assert raw.hand.num_joints == 19, raw.hand.joint_names
    assert original.disabled_joint not in raw.hand.joint_names
    assert len(raw.finger_bodies) == 5
    obs, _ = env.reset()
    assert obs['policy'].shape == (args.num_envs, 146), obs['policy'].shape
    assert torch.isfinite(obs['policy']).all()
    # Fixed initial position targets, in upstream sorted articulation order.
    targets = raw.hand.data.default_joint_pos[:, raw.actuated_dof_indices]
    low = raw.hand_dof_lower_limits[:, raw.actuated_dof_indices]
    high = raw.hand_dof_upper_limits[:, raw.actuated_dof_indices]
    actions = 2 * (targets - low) / (high - low) - 1
    resets = 0
    for _ in range(600):
        obs, reward, terminated, truncated, extras = env.step(actions)
        assert torch.isfinite(obs['policy']).all(), 'Nonfinite observation'
        assert torch.isfinite(reward).all(), 'Nonfinite reward'
        resets += int((terminated | truncated).sum())
    result = {'num_envs': args.num_envs, 'steps': 600, 'joint_names': raw.hand.joint_names,
              'action_space': cfg.action_space, 'observation_space': cfg.observation_space,
              'state_space': cfg.state_space, 'cube_edge_m': .06, 'cube_mass_kg': .05,
              'upstream_fields_verified': {name: getattr(cfg, name) for name in same},
              'finite_observations_rewards': True, 'resets': resets,
              'consecutive_successes': float(raw.consecutive_successes.mean())}
    args.output.write_text(json.dumps(result, indent=2))
    print('[PROBE] ' + json.dumps(result), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
