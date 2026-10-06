import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.teleoperators.utils import TeleopEvents
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ur10_dg5f.residual import (
    ResidualInterventionEnv,
    ResidualScale,
    apply_residual,
    residual_label,
)
from lerobot_robot_ur10_dg5f.synergies import HandSynergies


HOME = np.r_[-0.10, 0.50, 0.45, 2.2, -2.2, 0.0, np.zeros(20)]


def make_synergies(k=3):
    components = np.linalg.qr(np.random.default_rng(0).normal(size=(20, k)))[0].T
    return HandSynergies(np.zeros(20), components, np.full(k, 1.0 / k))


class FakeRobot:
    def __init__(self):
        self.last_command = HOME.copy()


class FakeEnv(gym.Env):
    """Executes exactly what it is given, so the wrapper is tested in isolation."""

    period_s = 0.1
    observation_space = spaces.Box(-np.inf, np.inf, shape=(26,))
    action_space = spaces.Box(-np.inf, np.inf, shape=(26,))

    def __init__(self):
        self.robot = FakeRobot()
        self.sent = []

    def reset(self, *, seed=None, options=None):
        self.robot.last_command = HOME.copy()
        return HOME.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.sent.append(action)
        self.robot.last_command = action.copy()
        return action.copy(), 0.0, False, False, {"sent_action": action.copy()}


class ScriptedOperator:
    def __init__(self, takeover_steps=(), success_step=None, wrist=None, hand=None):
        self.takeover_steps = set(takeover_steps)
        self.success_step = success_step
        self.step = -1
        self.wrist = np.r_[5.0, 5.0, 5.0, 0.0, 0.0, 0.0] if wrist is None else wrist
        self.hand = np.full(20, 40.0) if hand is None else hand

    def get_teleop_events(self):
        self.step += 1
        return {
            TeleopEvents.IS_INTERVENTION: self.step in self.takeover_steps,
            TeleopEvents.SUCCESS: self.step == self.success_step,
        }

    def get_action(self):
        return {"wrist_pose": self.wrist.copy(), "hand_deg": self.hand.copy()}


def hold_policy(observation):
    return HOME.copy()


def same_pose(a, b, atol=1e-9):
    return (np.allclose(a[:3], b[:3], atol=atol)
            and np.allclose(R.from_rotvec(a[3:6]).as_matrix(), R.from_rotvec(b[3:6]).as_matrix(), atol=atol)
            and np.allclose(a[6:], b[6:], atol=atol))


def make_env(operator, scale=ResidualScale(), policy=hold_policy):
    env = ResidualInterventionEnv(FakeEnv(), policy, make_synergies(), operator, scale=scale)
    env.reset()
    return env


def test_beta_zero_reproduces_base_policy():
    rng = np.random.default_rng(3)
    env = make_env(ScriptedOperator(), scale=ResidualScale(0.0, 0.0, 0.0))
    for _ in range(50):
        _, _, _, _, info = env.step(rng.uniform(-1, 1, size=9))
        assert same_pose(info["executed_action"], info["base_action"])


def test_autonomous_label_is_the_clipped_residual():
    env = make_env(ScriptedOperator())
    residual = np.r_[2.0, -0.5, 0.1, 0.0, 0.0, 0.3, -3.0, 0.2, 0.0]
    _, reward, _, _, info = env.step(residual)
    assert np.allclose(info["residual_label"], np.clip(residual, -1, 1))
    assert reward == 0.0 and not info["is_intervention"]


def test_rlif_penalty_only_on_takeover_start():
    env = make_env(ScriptedOperator(takeover_steps=range(10, 30)))
    rewards, flags, starts = [], [], []
    for _ in range(40):
        _, reward, _, _, info = env.step(np.zeros(9))
        rewards.append(reward)
        flags.append(info["is_intervention"])
        starts.append(info["intervention_started"])
    assert [i for i, r in enumerate(rewards) if r != 0] == [10]
    assert rewards[10] == -1.0
    assert [i for i, f in enumerate(flags) if f] == list(range(10, 30))
    assert [i for i, s in enumerate(starts) if s] == [10]


def test_takeover_does_not_jump_the_arm():
    env = make_env(ScriptedOperator(takeover_steps=range(3, 10)))  # operator hand is 8 m away
    for _ in range(3):
        env.step(np.full(9, 0.5))
    before = env.env.robot.last_command.copy()
    _, _, _, _, info = env.step(np.zeros(9))
    assert info["intervention_started"]
    assert np.allclose(info["executed_action"][:3], before[:3])


def test_fingers_blend_in_and_back_out():
    operator = ScriptedOperator(takeover_steps=range(0, 10), hand=np.full(20, 40.0))
    env = make_env(operator)
    hands = [env.step(np.zeros(9))[4]["executed_action"][6:].mean() for _ in range(20)]
    assert np.isclose(hands[0], 10.0)  # 0.4 s blend at 10 Hz: a quarter of the way on the first step
    assert np.all(np.diff(hands[:4]) > 0) and np.isclose(hands[3], 40.0)
    assert np.isclose(hands[10], 30.0) and np.isclose(hands[13], 0.0)  # and back to the policy


def test_operator_label_recovers_small_corrections():
    synergies = make_synergies()
    scale = ResidualScale()
    base = HOME.copy()
    true_residual = np.r_[0.3, -0.2, 0.1, 0.05, -0.4, 0.2, 0.5, -0.7, 0.1]
    executed = apply_residual(base, true_residual, synergies, scale)
    label, saturated = residual_label(executed, base, synergies, scale)
    assert np.allclose(label, true_residual) and not saturated


def test_large_operator_correction_saturates():
    synergies = make_synergies()
    executed = HOME.copy()
    executed[0] += 0.10  # 10 cm against lin_m = 2 cm
    label, saturated = residual_label(executed, HOME, synergies, ResidualScale())
    assert saturated and label[0] == 1.0


def test_success_terminates_with_reward():
    env = make_env(ScriptedOperator(success_step=5))
    for step in range(6):
        _, reward, terminated, _, _ = env.step(np.zeros(9))
    assert reward == 1.0 and terminated


def test_wraps_the_simulated_cell(make_sim_robot):
    from lerobot_robot_ur10_dg5f.env import Ur10Dg5fEnv

    env = Ur10Dg5fEnv(make_sim_robot())
    wrapper = ResidualInterventionEnv(
        env, lambda observation: env.robot.last_command.copy(), make_synergies(),
        ScriptedOperator(), scale=ResidualScale(0.0, 0.0, 0.0))
    wrapper.reset()
    for _ in range(10):
        _, _, _, _, info = wrapper.step(np.ones(9))
        assert same_pose(info["executed_action"], info["base_action"], atol=1e-9)
