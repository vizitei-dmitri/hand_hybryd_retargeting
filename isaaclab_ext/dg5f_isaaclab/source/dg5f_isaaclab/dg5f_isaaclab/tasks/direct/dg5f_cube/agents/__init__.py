# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Register the local BoundedActorCritic on ANY import of the tasks package. Scripts that rebuild a
# runner from a saved params/agent.yaml (eval_checkpoints.py, play.py) never import the agent config
# module, and without this a bounded checkpoint fails with NameError: 'BoundedActorCritic'.
from .bounded_policy import register as _register_bounded_policy

_register_bounded_policy()
