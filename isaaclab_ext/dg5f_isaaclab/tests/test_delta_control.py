"""Small CPU-only contracts; no Kit startup and no physical hand."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/dg5f_isaaclab/dg5f_isaaclab"


def load_module(name, path):
    # Package __init__ registers Kit tasks; these two pure modules need no Kit.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sysid = load_module("dg5f_test_sysid", PACKAGE / "assets/sysid.py")
control = load_module("dg5f_test_control", PACKAGE / "tasks/direct/dg5f_cube/control.py")
goals = load_module("dg5f_test_goals", PACKAGE / "tasks/direct/dg5f_cube/goals.py")
success = load_module("dg5f_test_success", PACKAGE / "tasks/direct/dg5f_cube/success.py")
CONTROL_DT = 2 / 120  # DG5FCubeEnvCfg: sim.dt=1/120, decimation=2
JOINTS = tuple(j.get("name") for j in ET.parse(ROOT.parents[1] / "models/dg5f/urdf/dg5f_right.urdf").getroot()
               .findall("joint") if j.get("type") != "fixed")


class SysIDTests(unittest.TestCase):
    def test_real_file_covers_urdf(self):
        data = sysid.load_sysid(sysid.DEFAULT_SYSID_PATH, JOINTS)
        self.assertEqual(data.joint_order, JOINTS)
        self.assertEqual(data.disabled_joint, "rj_dg_5_1")
        active = [name for name in JOINTS if name != data.disabled_joint]
        self.assertEqual(len(active), 19)
        self.assertEqual(data.delays_for(active), [3] * 19)
        self.assertEqual(set(data.parameters), {"stiffness", "damping", "armature", "friction"})

    def test_reject_bad_metadata_and_parameters(self):
        original = json.loads(sysid.DEFAULT_SYSID_PATH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sysid.json"
            for case in ("version", "order", "missing", "nonfinite", "negative_delay", "disabled"):
                with self.subTest(case=case):
                    data = json.loads(json.dumps(original))
                    if case == "version":
                        data["format_version"] = 2
                    elif case == "order":
                        data["joint_order"].reverse()
                    elif case == "missing":
                        del data["parameters"]["armature"][JOINTS[0]]
                    elif case == "nonfinite":
                        data["parameters"]["damping"][JOINTS[0]] = float("nan")
                    elif case == "negative_delay":
                        data["delay_control_steps_by_finger"]["1"] = -1
                    else:
                        data["disabled_joint"] = "not_a_joint"
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        sysid.load_sysid(path, JOINTS)


class ControlTests(unittest.TestCase):
    def test_impulse_arrives_after_three_control_steps(self):
        queue = control.ActionDelayQueue(1, [3] * 19, "cpu")
        outputs = []
        for step in range(7):
            action = torch.zeros(1, 19)
            action[0, 4] = float(step == 0)
            outputs.append(queue.push(action)[0, 4].item())
        self.assertEqual(outputs, [0, 0, 0, 1, 0, 0, 0])

    def test_history_fully_describes_next_delivery_and_rate(self):
        queue = control.ActionDelayQueue(2, [0, 1, 3], "cpu")
        for value in (0.2, 0.4, 0.6):
            queue.push(torch.full((2, 3), value))
        restored = control.ActionDelayQueue(2, [0, 1, 3], "cpu")
        restored.history.copy_(queue.history)
        action = torch.full((2, 3), 0.8)
        torch.testing.assert_close(queue.history[:, 0], torch.full((2, 3), 0.6))
        delivered = queue.push(action)
        torch.testing.assert_close(delivered, restored.push(action))
        torch.testing.assert_close(delivered, torch.tensor([[0.8, 0.6, 0.2]]).expand(2, -1))

    def test_partial_reset_drops_only_that_episodes_commands(self):
        queue = control.ActionDelayQueue(2, [3], "cpu")
        queue.push(torch.ones(2, 1))
        queue.reset(torch.tensor([0]))
        self.assertEqual(queue.history[0].abs().sum(), 0)
        values = [queue.push(torch.zeros(2, 1)) for _ in range(3)]
        self.assertEqual(values[-1][0, 0], 0)
        self.assertEqual(values[-1][1, 0], 1)
        queue.reset()
        self.assertEqual(queue.history.abs().sum(), 0)

    def test_zero_delay_still_exposes_previous_action_for_rate(self):
        queue = control.ActionDelayQueue(1, [0, 0], "cpu")
        action = torch.tensor([[0.2, -0.3]])
        torch.testing.assert_close(queue.push(action), action)
        torch.testing.assert_close(queue.history[:, 0], action)

    def test_integrated_delta_is_delayed_then_accumulated_and_held(self):
        scale = math.radians(3)
        lower, upper, grasp = torch.full((1, 3), -1.), torch.ones(1, 3), torch.full((1, 3), 0.2)
        queue = control.ActionDelayQueue(1, [3] * 3, "cpu")
        command, commands = grasp.clone(), []
        for step in range(7):
            action = torch.tensor([[1., -1., 0.]]) if step in (0, 1) else torch.zeros(1, 3)
            # Measurements must not matter: an external push does not move the command.
            measured = torch.full((1, 3), -0.7 + step)
            command = control.position_targets(queue.push(action), measured, command, lower, upper, grasp,
                                               "integrated_delta_position", scale)
            commands.append(command)
        for step in range(3):
            torch.testing.assert_close(commands[step], grasp)
        torch.testing.assert_close(commands[3], grasp + torch.tensor([[scale, -scale, 0.]]))
        torch.testing.assert_close(commands[4], grasp + torch.tensor([[2 * scale, -2 * scale, 0.]]))
        for step in (5, 6):
            torch.testing.assert_close(commands[step], commands[4])
        at_limit = control.position_targets(torch.tensor([[1., -1., 1.]]), torch.zeros(1, 3),
                                            torch.tensor([[0.99, -0.99, 0.3]]), lower, upper, grasp,
                                            "integrated_delta_position", scale)
        torch.testing.assert_close(at_limit, torch.tensor([[1., -1., 0.3 + scale]]))

    def test_measured_delta_uses_delivery_measurement_and_clamps(self):
        scale = math.radians(3)
        lower, upper, grasp = torch.full((1, 3), -1.), torch.ones(1, 3), torch.zeros(1, 3)
        measured = torch.tensor([[0.99, -0.99, 0.3]])
        target = control.position_targets(torch.tensor([[1., -1., 0.]]), measured, torch.zeros(1, 3),
                                          lower, upper, grasp, "measured_delta_position", scale)
        torch.testing.assert_close(target, torch.tensor([[1., -1., 0.3]]))

    def test_legacy_absolute_keeps_grasp_and_full_range(self):
        lower, upper, grasp = torch.full((1, 19), -1.), torch.full((1, 19), 2.), torch.full((1, 19), 0.4)
        for action, expected in ((-1., lower), (0., grasp), (1., upper)):
            target = control.position_targets(torch.full((1, 19), action), torch.zeros_like(grasp),
                                              torch.zeros_like(grasp), lower, upper, grasp,
                                              "absolute_position", 0.05)
            torch.testing.assert_close(target, expected)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            control.position_targets(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), -torch.ones(1, 1),
                                     torch.ones(1, 1), torch.zeros(1, 1), "delta_position", 0.05)

class GoalSamplingTests(unittest.TestCase):
    def test_ten_thousand_goals_respect_minimum_error(self):
        generator = torch.Generator().manual_seed(0)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(10_000, -1)
        min_error = math.radians(10.0)
        for mode in ("axis", "uniform"):
            with self.subTest(mode=mode):
                quats = goals.sample_goal_quats(identity, mode, (1.0, 0.0, 0.0), (-20.0, 20.0), min_error,
                                                generator=generator)
                torch.testing.assert_close(quats.norm(dim=-1), torch.ones(10_000))
                error = goals.quat_angle(quats, identity)
                self.assertGreaterEqual(error.min().item(), min_error)
                if mode == "axis":
                    self.assertLessEqual(error.max().item(), math.radians(20.0) + 1e-5)
                    # Rotations stay about the palm X axis and use both signs.
                    self.assertLess(quats[:, 2:].abs().max().item(), 1e-6)
                    self.assertTrue((quats[:, 1] > 0).any() and (quats[:, 1] < 0).any())

    def test_nonidentity_start_and_impossible_range(self):
        start = goals.axis_angle_quat(torch.full((1000,), 0.3), (0.0, 1.0, 0.0))
        quats = goals.sample_goal_quats(start, "uniform", (1.0, 0.0, 0.0), (-20.0, 20.0), math.radians(10.0))
        self.assertGreaterEqual(goals.quat_angle(quats, start).min().item(), math.radians(10.0))
        with self.assertRaises(ValueError):
            goals.sample_goal_quats(start, "axis", (1.0, 0.0, 0.0), (-5.0, 5.0), math.radians(10.0))


class RewardV2Tests(unittest.TestCase):
    def test_a_state_reward_peaks_at_zero_and_decreases(self):
        errors = torch.tensor([math.radians(e) for e in (0, 2.5, 5, 10, 15, 20, 30, 45, 90, 180)])
        reward = success.orientation_state_reward(errors, 0.1, math.radians(10))
        self.assertAlmostEqual(reward[0].item(), 0.1, places=7)
        self.assertTrue(torch.all(reward[1:] < reward[:-1]))
        self.assertTrue(torch.all(reward >= 0))
        # Documented anchors: 5 deg -> 0.078, 10 deg -> 0.037, 20 deg -> 0.002.
        torch.testing.assert_close(reward[[2, 3, 5]], torch.tensor([0.1 * math.exp(-0.25), 0.1 * math.exp(-1),
                                                                    0.1 * math.exp(-4)]))

    def tracker(self, num_envs=1):
        steps = success.hold_steps(0.30, CONTROL_DT)
        return success.HeldSuccessTracker(num_envs, steps, "cpu"), steps

    def feed(self, tracker, pattern, start=1):
        bonuses = []
        for i, inside in enumerate(pattern):
            newly = tracker.update(torch.tensor([bool(inside)]), torch.tensor([start + i]))
            bonuses.append(bool(newly[0]))
        return bonuses

    def test_hold_steps_follow_control_dt(self):
        self.assertEqual(success.hold_steps(0.30, CONTROL_DT), 18)
        self.assertEqual(success.hold_steps(0.30, 1 / 30), 9)

    def test_b_single_step_inside_is_not_success(self):
        tracker, _ = self.tracker()
        self.assertEqual(self.feed(tracker, [1, 0, 0]), [False] * 3)
        self.assertFalse(tracker.succeeded[0])
        self.assertTrue(tracker.entered[0])

    def test_c_required_consecutive_steps_give_success(self):
        tracker, steps = self.tracker()
        bonuses = self.feed(tracker, [1] * steps)
        self.assertEqual(bonuses, [False] * (steps - 1) + [True])
        self.assertTrue(tracker.succeeded[0])
        self.assertEqual(tracker.success_step[0].item(), steps)

    def test_d_leaving_resets_the_counter(self):
        tracker, steps = self.tracker()
        self.feed(tracker, [1] * 10 + [0])
        self.assertEqual(tracker.consecutive[0].item(), 0)
        self.assertEqual(self.feed(tracker, [1] * (steps - 1)), [False] * (steps - 1))
        self.assertFalse(tracker.succeeded[0])
        self.assertEqual(tracker.steps_in_tolerance[0].item(), 10 + steps - 1)

    def test_e_bonus_only_once(self):
        tracker, steps = self.tracker()
        bonuses = self.feed(tracker, [1] * (3 * steps) + [0] + [1] * (2 * steps))
        self.assertEqual(sum(bonuses), 1)

    def test_f_reset_clears_only_selected_envs(self):
        tracker = success.HeldSuccessTracker(2, 3, "cpu")
        for step in range(4):
            tracker.update(torch.tensor([True, True]), torch.tensor([step, step]))
        tracker.reset(torch.tensor([0]))
        self.assertEqual((tracker.consecutive[0].item(), tracker.succeeded[0].item(), tracker.entered[0].item(),
                          tracker.steps_in_tolerance[0].item(), tracker.success_step[0].item()), (0, False, False, 0, -1))
        self.assertTrue(tracker.succeeded[1])
        self.assertEqual(tracker.consecutive[1].item(), 4)

    def test_g_no_success_right_after_reset(self):
        tracker, steps = self.tracker()
        self.feed(tracker, [1] * steps)
        tracker.reset()
        # Even a cube already inside the tolerance needs a full new hold after reset.
        self.assertEqual(self.feed(tracker, [1] * (steps - 1)), [False] * (steps - 1))
        self.assertFalse(tracker.succeeded[0])


if __name__ == "__main__":
    unittest.main()
