"""Stage-advancement thresholds for the reward-v4 goal stream (pure Python, no Isaac Sim).

Kept free of Isaac imports on purpose: scripts/train_stream.py is an orchestrator that must read
these thresholds WITHOUT starting Kit, and importing the package pulls in pxr, which only exists
inside the simulator app.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamCurriculumCfg:
    """When the directional stage changes, when a stage is stalled, and when to stop.

    Every threshold here is judged on a separate DETERMINISTIC evaluation that the trainer never
    sees. That is not a stylistic choice: the reward-v3 night run reported held_success 0.22 from
    its training rollouts while deterministic evaluation of the same checkpoints gave 0.000, because
    those successes were passes through the tolerance driven by exploration noise.
    """

    # Segment length between evaluations, and how many episodes each evaluation runs.
    eval_every_iterations: int = 250
    eval_episodes: int = 128
    # A stage advances only after this many CONSECUTIVE passing evaluations, never on one lucky one.
    consecutive_passes: int = 3
    # A stage that has not passed after this many of its own iterations is STALLED: the run stops and
    # reports, rather than quietly lowering the bar.
    stage_max_iterations: int = 2000
    # The gate.
    first_goal_held_success_rate: float = 0.70
    max_drop_rate: float = 0.15
    goals_completed_per_episode: float = 1.5
    # RSL-RL 3.1.2 has no supported maximum-std option, and patching its internals for one is out of
    # scope, so the ceiling is a watchdog: std above this WITH no deterministic improvement over the
    # recent evaluations stops the run. The pairing is the point -- a high std while the policy is
    # still improving is exploration, not divergence.
    action_std_watchdog: float = 0.35
    watchdog_window: int = 3
    # --- bootstrap phase: a SMALLER goal magnitude before the prescribed 20 deg.
    #
    # The spec forbids an angle-magnitude curriculum, and the reason was sound: a policy trained on
    # easy goals learns those instead of the target. What the stage-A run measured is the cost of the
    # other side of that choice. Deterministically the policy reaches min_error 12.2 deg from a 20 deg
    # start against a 5 deg tolerance -- it covers ~8 of the 15 deg needed and stalls -- so
    # success_bonus fires on 0.047 goals per episode against a gate of 1.5. The critic therefore has
    # almost no positive examples of rotation paying off, and three separate reward-weight changes
    # could not create any (see JOURNAL 2026-09-30).
    #
    # 10 deg sits INSIDE the 8 deg the policy already demonstrates, so the bonus starts firing. This
    # is a predecessor phase, not a curriculum over the whole run: it happens once, inside stage A,
    # and the SAME policy must then pass the same gate again at the full 20 deg before any
    # directional stage advances. Nothing downstream is evaluated at the reduced angle.
    bootstrap_angle_deg: float = 10.0
    bootstrap_max_iterations: int = 1500


STREAM_CURRICULUM = StreamCurriculumCfg()
