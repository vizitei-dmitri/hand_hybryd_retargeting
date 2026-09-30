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
grasp = load_module("dg5f_test_grasp", PACKAGE / "tasks/direct/dg5f_cube/grasp.py")
grasp_cache = load_module("dg5f_test_grasp_cache", PACKAGE / "assets/grasp_cache.py")
curriculum = load_module("dg5f_test_curriculum", PACKAGE / "tasks/direct/dg5f_cube/curriculum.py")
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


class GraspQualityTests(unittest.TestCase):
    """lambda_min(G G^T) must separate a real grip from fingertips merely cradling the cube."""

    OPPOSED = [[0.03, 0, 0], [-0.03, 0.02, 0], [-0.03, -0.02, 0], [0, 0, 0], [0, 0, 0]]
    BOWL = [[0.0, 0, -0.03], [0.02, 0, -0.03], [-0.02, 0, -0.03], [0, 0, 0], [0, 0, 0]]
    THREE = [[True, True, True, False, False]]

    def quality(self, offsets, mask):
        gramian = grasp.grasp_gramian(torch.tensor([offsets]), torch.tensor(mask))
        return grasp.gramian_min_eigenvalue(gramian)[0].item()

    def test_a_opposed_beats_bowl(self):
        # A one-sided contact set cannot resist a wrench along its open direction.
        self.assertGreater(self.quality(self.OPPOSED, self.THREE), 1e3 * self.quality(self.BOWL, self.THREE))

    def test_b_too_few_contacts_is_singular(self):
        # eigvalsh leaves ~1e-9 of numerical noise on a singular Gramian; the reward's margin
        # (grasp_quality_margin, 1e-4) is five orders of magnitude above that, so it is harmless.
        for mask in ([[True, False, False, False, False]], [[True, True, False, False, False]]):
            self.assertLess(self.quality(self.OPPOSED, mask), 1e-7)

    def test_c_gramian_is_symmetric_psd(self):
        gramian = grasp.grasp_gramian(torch.tensor([self.OPPOSED]), torch.tensor(self.THREE))
        torch.testing.assert_close(gramian, gramian.transpose(-1, -2))
        self.assertGreaterEqual(torch.linalg.eigvalsh(gramian).amin().item(), -1e-12)

    def test_d_reward_is_bounded_and_ema_tracks(self):
        reward = grasp.GraspQualityReward(1e-4, 0.5, 0.97)
        offsets = torch.tensor([self.OPPOSED, self.BOWL])
        mask = torch.tensor(self.THREE * 2)
        for _ in range(50):
            value, lambda_min = reward(offsets, mask)
        self.assertTrue(torch.all(value.abs() <= 0.5 + 1e-9))
        # The opposed grasp must still rank above the bowl after normalisation.
        self.assertGreater(value[0].item(), value[1].item())
        self.assertGreater(lambda_min[0].item(), lambda_min[1].item())

    def test_e_rejects_bad_parameters(self):
        for margin, clip, ema in ((1e-4, 0.5, 0.0), (1e-4, 0.5, 1.0), (1e-4, 0.0, 0.97), (-1.0, 0.5, 0.97)):
            with self.assertRaises(ValueError):
                grasp.GraspQualityReward(margin, clip, ema)


class GoalCurriculumTests(unittest.TestCase):
    def test_a_haar_band_recovers_uniform_so3(self):
        # Widening the band to [0, pi] must reproduce uniform SO(3), whose mean angle is 126.47 deg.
        angles = goals.sample_angles_haar(200000, 0.0, math.pi, "cpu")
        self.assertAlmostEqual(math.degrees(angles.mean().item()), 126.47, delta=0.6)

    def test_b_band_is_respected(self):
        angles = goals.sample_angles_haar(20000, math.radians(20), math.radians(30), "cpu")
        self.assertGreaterEqual(math.degrees(angles.amin().item()), 20.0 - 1e-6)
        self.assertLessEqual(math.degrees(angles.amax().item()), 30.0 + 1e-6)

    def test_c_rejects_invalid_band(self):
        for lo, hi in ((0.5, 0.2), (-0.1, 1.0), (0.0, 4.0)):
            with self.assertRaises(ValueError):
                goals.sample_angles_haar(8, lo, hi, "cpu")

    def test_d_goals_obey_limit_and_minimum(self):
        current = torch.zeros(4096, 4)
        current[:, 0] = 1
        quats, frontier = goals.sample_curriculum_goals(
            current, math.radians(10), math.radians(40), math.radians(10), 0.6, 0.3)
        error = torch.rad2deg(goals.quat_angle(quats, current))
        self.assertGreaterEqual(error.amin().item(), 10.0 - 1e-3)
        self.assertLessEqual(error.amax().item(), 40.0 + 1e-3)
        torch.testing.assert_close(quats.norm(dim=-1), torch.ones(4096), atol=1e-5, rtol=0)
        self.assertAlmostEqual(frontier.float().mean().item(), 0.6, delta=0.03)
        # Frontier goals are the ones whose success rate drives promotion, so they must be hard.
        self.assertGreaterEqual(math.degrees(torch.deg2rad(error[frontier]).amin().item()), 30.0 - 1e-3)

    def test_e_goals_are_relative_to_the_current_orientation(self):
        current = goals.uniform_quat(2048, "cpu")
        quats, _ = goals.sample_curriculum_goals(
            current, math.radians(10), math.radians(40), math.radians(10), 0.6, 0.3)
        error = torch.rad2deg(goals.quat_angle(quats, current))
        self.assertGreaterEqual(error.amin().item(), 10.0 - 1e-3)
        self.assertLessEqual(error.amax().item(), 40.0 + 1e-3)

    def test_f_degenerate_band_does_not_crash(self):
        current = torch.zeros(64, 4)
        current[:, 0] = 1
        quats, _ = goals.sample_curriculum_goals(
            current, math.radians(10), math.radians(10), math.radians(10), 0.6, 0.3)
        self.assertEqual(tuple(quats.shape), (64, 4))


class GraspCacheTests(unittest.TestCase):
    NAMES = JOINTS

    def cache_arrays(self, count=4):
        import numpy as np
        return {
            "q": np.zeros((count, len(self.NAMES)), dtype=np.float32),
            "q_cmd": np.zeros((count, len(self.NAMES)), dtype=np.float32),
            "cube_pos": np.zeros((count, 3), dtype=np.float32),
            "cube_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (count, 1)),
            "joint_names": np.array(self.NAMES),
        }

    def write(self, arrays):
        import numpy as np
        handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        handle.close()
        np.savez(handle.name, **arrays)
        return handle.name

    def test_a_round_trip(self):
        cache = grasp_cache.load_grasp_cache(self.write(self.cache_arrays()), self.NAMES)
        self.assertEqual(len(cache), 4)
        self.assertEqual(cache.joint_names, tuple(self.NAMES))
        self.assertEqual(len(cache.sha256), 64)

    def test_b_rejects_wrong_joint_order(self):
        arrays = self.cache_arrays()
        import numpy as np
        arrays["joint_names"] = np.array(tuple(reversed(self.NAMES)))
        with self.assertRaises(ValueError):
            grasp_cache.load_grasp_cache(self.write(arrays), self.NAMES)

    def test_c_rejects_missing_array_and_empty_cache(self):
        arrays = self.cache_arrays()
        del arrays["cube_quat"]
        with self.assertRaises(ValueError):
            grasp_cache.load_grasp_cache(self.write(arrays), self.NAMES)
        with self.assertRaises(ValueError):
            grasp_cache.load_grasp_cache(self.write(self.cache_arrays(0)), self.NAMES)

    def test_d_rejects_non_finite_and_unnormalised_quat(self):
        arrays = self.cache_arrays()
        arrays["q"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            grasp_cache.load_grasp_cache(self.write(arrays), self.NAMES)
        arrays = self.cache_arrays()
        arrays["cube_quat"][:] = 0.3  # norm 0.6; note [0.5]*4 would be unit-norm
        with self.assertRaises(ValueError):
            grasp_cache.load_grasp_cache(self.write(arrays), self.NAMES)

    def test_e_rejects_out_of_limit_joints(self):
        arrays = self.cache_arrays()
        arrays["q"][0, 0] = 100.0
        limits = [(-1.0, 1.0)] * len(self.NAMES)
        with self.assertRaises(ValueError):
            grasp_cache.load_grasp_cache(self.write(arrays), self.NAMES, limits)


if __name__ == "__main__":
    unittest.main()


class GoalStreamTests(unittest.TestCase):
    """The fixed-20-degree goal stream (reward v4). Section 26 of the experiment spec."""

    ANGLE = math.radians(20.0)
    AXIS = (1.0, 0.0, 0.0)

    def delta_axis_angle(self, reference, goal):
        """Rotation taking reference to goal, as (axis, angle)."""
        conjugate = reference * torch.tensor([1.0, -1.0, -1.0, -1.0])
        delta = goals.quat_multiply(conjugate, goal)
        # Canonical hemisphere, so the axis sign is the rotation's and not the quaternion's.
        delta = torch.where(delta[:, :1] < 0, -delta, delta)
        angle = 2.0 * torch.acos(delta[:, 0].clamp(-1.0, 1.0))
        axis = delta[:, 1:] / torch.sin(0.5 * angle).clamp_min(1e-9)[:, None]
        return axis, angle

    def references(self, count, seed=0):
        torch.manual_seed(seed)
        return goals.uniform_quat(count, "cpu")

    def test_a_every_goal_is_exactly_the_stream_angle_away(self):
        for stage in goals.STREAM_STAGES:
            reference = self.references(2048, seed=hash(stage) % 1000)
            goal = goals.sample_fixed_angle_goals(reference, self.ANGLE, stage, self.AXIS)
            distance = goals.quat_angle(reference, goal)
            self.assertLess(float((distance - self.ANGLE).abs().max()), 1e-5,
                            f"stage {stage} does not hold the fixed angle")

    def test_b_quaternion_sign_does_not_change_the_distance(self):
        reference = self.references(512, seed=7)
        goal = goals.sample_fixed_angle_goals(reference, self.ANGLE, "C", self.AXIS)
        base = goals.quat_angle(reference, goal)
        # q and -q are the same rotation, so every sign combination must give the same geodesic.
        for reference_sign, goal_sign in ((1, -1), (-1, 1), (-1, -1)):
            flipped = goals.quat_angle(reference_sign * reference, goal_sign * goal)
            torch.testing.assert_close(flipped, base)

    def test_c_goal_is_referenced_from_the_pose_passed_in(self):
        # The env passes the cube's ACTUAL orientation, so tracking error cannot accumulate into a
        # target that is no longer the requested distance away. A goal measured from identity
        # instead would be 20 deg from identity, not from the achieved pose.
        reference = self.references(256, seed=11)
        goal = goals.sample_fixed_angle_goals(reference, self.ANGLE, "C", self.AXIS)
        identity = torch.zeros_like(reference)
        identity[:, 0] = 1.0
        from_identity = goals.quat_angle(identity, goal)
        self.assertGreater(float((from_identity - self.ANGLE).abs().mean()), math.radians(5.0))

    def test_d_stage_a_uses_the_primary_axis_with_both_signs(self):
        reference = self.references(1024, seed=3)
        goal = goals.sample_fixed_angle_goals(reference, self.ANGLE, "A", self.AXIS)
        axis, angle = self.delta_axis_angle(reference, goal)
        torch.testing.assert_close(angle, torch.full((1024,), self.ANGLE), atol=1e-5, rtol=0)
        primary = torch.tensor(self.AXIS)
        alignment = (axis * primary).sum(dim=-1)
        torch.testing.assert_close(alignment.abs(), torch.ones(1024), atol=1e-4, rtol=0)
        self.assertTrue(bool((alignment > 0).any()) and bool((alignment < 0).any()), "sign is not random")

    def test_e_stage_b_uses_the_three_principal_axes(self):
        reference = self.references(3072, seed=5)
        goal = goals.sample_fixed_angle_goals(reference, self.ANGLE, "B", self.AXIS)
        axis, _ = self.delta_axis_angle(reference, goal)
        magnitude = axis.abs()
        # Exactly one component is +-1 and the other two vanish.
        torch.testing.assert_close(magnitude.max(dim=-1).values, torch.ones(3072), atol=1e-4, rtol=0)
        self.assertLess(float(magnitude.sum(dim=-1).max() - 1.0), 1e-4)
        used = magnitude.argmax(dim=-1)
        counts = torch.bincount(used, minlength=3).float() / 3072
        self.assertTrue(torch.all(counts > 0.25), f"axes are not all used: {counts.tolist()}")

    def test_f_stage_c_axes_are_approximately_uniform(self):
        reference = self.references(8192, seed=13)
        goal = goals.sample_fixed_angle_goals(reference, self.ANGLE, "C", self.AXIS)
        axis, _ = self.delta_axis_angle(reference, goal)
        # Uniform on the sphere: every component has mean 0, and E|cos| to any fixed axis is 0.5.
        self.assertLess(float(axis.mean(dim=0).abs().max()), 0.05)
        for component in range(3):
            self.assertAlmostEqual(float(axis[:, component].abs().mean()), 0.5, delta=0.03)

    def test_g_unknown_stage_is_rejected(self):
        reference = self.references(4)
        with self.assertRaises(ValueError):
            goals.sample_fixed_angle_goals(reference, self.ANGLE, "D", self.AXIS)

    def test_h_new_goal_clears_the_hold_but_keeps_episode_statistics(self):
        # This is what "a new goal starts immediately, without resetting the episode" means for the
        # success tracker: the dwell restarts, the per-episode counters do not.
        steps = success.hold_steps(0.30, CONTROL_DT)
        tracker = success.HeldSuccessTracker(1, steps, "cpu")
        inside = torch.tensor([True])
        for step in range(steps):
            newly = tracker.update(inside, torch.tensor([step + 1]))
        self.assertTrue(bool(newly[0]))
        first_success_step = int(tracker.success_step[0])
        tracker.new_goal(torch.tensor([0]))
        self.assertEqual(int(tracker.consecutive[0]), 0)
        self.assertFalse(bool(tracker.succeeded[0]))
        self.assertTrue(bool(tracker.entered[0]), "episode statistics must survive a new goal")
        self.assertEqual(int(tracker.steps_in_tolerance[0]), steps)
        # A second success in the same episode must not overwrite the FIRST success time.
        for step in range(steps):
            tracker.update(inside, torch.tensor([steps + step + 1]))
        self.assertEqual(int(tracker.success_step[0]), first_success_step)


class OrientationBaselineRewardTests(unittest.TestCase):
    """Section 6: the baseline-centred dense orientation term."""

    SCALE, SIGMA, REFERENCE = 0.30, math.radians(15.0), math.radians(20.0)

    def reward(self, degrees):
        errors = torch.tensor([math.radians(d) for d in degrees])
        return success.orientation_baseline_reward(errors, self.SCALE, self.SIGMA, self.REFERENCE)

    def test_a_zero_exactly_at_the_stream_angle(self):
        # "Do nothing at 20 deg" must be worth nothing, which is the whole purpose of the form.
        self.assertAlmostEqual(float(self.reward([20.0])[0]), 0.0, places=7)

    def test_b_sign_follows_being_closer_or_farther(self):
        closer = self.reward([0.0, 2.5, 5.0, 10.0, 15.0])
        farther = self.reward([25.0, 30.0, 45.0, 90.0])
        self.assertTrue(torch.all(closer > 0), closer)
        self.assertTrue(torch.all(farther < 0), farther)

    def test_c_monotonically_decreasing_in_error(self):
        values = self.reward([0, 2.5, 5, 10, 15, 20, 25, 30, 45, 90])
        self.assertTrue(torch.all(values[1:] < values[:-1]))

    def test_d_documented_anchors(self):
        peak = self.SCALE * (1.0 - math.exp(-self.REFERENCE / self.SIGMA))
        self.assertAlmostEqual(float(self.reward([0.0])[0]), peak, places=7)
        # 0.30 * (1 - exp(-20/15)) with sigma = 15 deg and a 20 deg stream angle.
        self.assertAlmostEqual(peak, 0.2209, places=4)

    def test_e_table_covers_the_requested_errors(self):
        table = success.orientation_reward_table(self.SCALE, self.SIGMA, self.REFERENCE)
        self.assertEqual([deg for deg, _ in table], [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 90.0])


class GraspDeficitPenaltyTests(unittest.TestCase):
    """The one-sided grasp-defence penalties added after stage A degraded.

    The formulas live inline in _get_rewards (they are two lines and need no helper), so these
    tests restate them and pin the sizing that logs/reward_preflight/preflight_grasp_ref.json
    measured. The point of pinning it: reward v3 failed because a weight was sized from an ASSUMED
    deviation, and the stage-A failure happened because nobody had compared the dense terms'
    magnitudes. If a scale is retuned, the arithmetic in the config comment must be retuned with it.
    """

    MIN_TIPS = 3
    TIP_SCALE, QUALITY_SCALE, QUALITY_FLOOR = 0.015, 0.06, 0.25

    def tip_penalty(self, counts):
        counts = torch.tensor(counts)
        deficit = (self.MIN_TIPS - counts).clamp_min(0).float()
        return -self.TIP_SCALE * deficit

    def quality_penalty(self, qualities):
        shortfall = (self.QUALITY_FLOOR - torch.tensor(qualities)).clamp_min(0.0)
        return -self.QUALITY_SCALE * shortfall

    def test_a_intact_grasp_is_free(self):
        # 3 or more tips and the warm-start quality must cost exactly nothing, or the term is a tax
        # on the grasp itself -- reward v3's failure.
        self.assertTrue(torch.all(self.tip_penalty([3, 4, 5]) == 0.0))
        self.assertTrue(torch.all(self.quality_penalty([0.25, 0.255, 0.361, 0.5]) == 0.0))

    def test_b_one_sided_never_pays(self):
        self.assertTrue(torch.all(self.tip_penalty([0, 1, 2, 3, 4, 5]) <= 0.0))
        self.assertTrue(torch.all(self.quality_penalty([-0.5, 0.0, 0.3, 1.0]) <= 0.0))

    def test_c_graded_in_the_count_not_binary(self):
        # The slide from 3 to 1 tip is what predicts the drop; the >=3 indicator would price the
        # whole flicker band the same as a collapsing grasp.
        values = self.tip_penalty([3, 2, 1, 0])
        self.assertTrue(torch.all(values[1:] < values[:-1]))
        self.assertAlmostEqual(float(values[1]), -0.015, places=7)
        self.assertAlmostEqual(float(values[3]), -0.045, places=7)

    def test_d_floor_sits_in_the_measured_bimodal_gap(self):
        # grasp_quality is bimodal: warm-start p01..p25 = -0.245..-0.175, p50..p90 = +0.425..+0.500.
        # The floor must select the degenerate mode and nothing in the intact mode, and it must do so
        # for any value inside the gap -- that insensitivity is why the choice is safe.
        degenerate = [-0.245, -0.220, -0.197, -0.175]
        intact = [0.425, 0.450, 0.500]
        for floor in (0.05, 0.15, 0.25, 0.35):
            penalty = -self.QUALITY_SCALE * (floor - torch.tensor(degenerate)).clamp_min(0.0)
            self.assertTrue(torch.all(penalty < 0), f"floor {floor} misses the degenerate mode")
            free = -self.QUALITY_SCALE * (floor - torch.tensor(intact)).clamp_min(0.0)
            self.assertTrue(torch.all(free == 0.0), f"floor {floor} taxes the intact mode")

    def test_e_sizing_must_not_be_read_off_the_mean(self):
        # The mistake this term was re-sized after: the penalty is clipped, hence convex, so the
        # mean understates it whenever the value fluctuates. Jensen, on the measured bimodal sample.
        sample = torch.tensor([-0.20] * 35 + [0.45] * 65)   # ~35% degenerate, as warm-start measures
        from_distribution = float((-self.QUALITY_SCALE
                                   * (self.QUALITY_FLOOR - sample).clamp_min(0.0)).mean())
        from_mean = float(-self.QUALITY_SCALE
                          * max(0.0, self.QUALITY_FLOOR - float(sample.mean())))
        # The measured understatement was 6x (-7 per episode predicted from the mean, -43 measured).
        self.assertLess(from_distribution, from_mean)
        self.assertGreater(from_distribution / from_mean, 4.0)

    def test_f_measured_balance_against_the_task_term(self):
        # Per-step defence measured at each policy's occupancy, against orientation_state.
        for defence, task, lower, upper in ((0.003, 0.017, 0.10, 0.25),    # zero, drop 0.031
                                            (0.013, 0.035, 0.30, 0.45),    # warm, drop 0.039
                                            (0.046, 0.051, 0.80, 1.00)):   # probe, drop 0.984
            self.assertTrue(lower <= defence / task <= upper, f"{defence}/{task}")

    def test_e_quality_deficit_requires_the_quality_term(self):
        # grasp_quality_value only exists inside the grasp_quality_scale branch.
        self.assertIn("grasp_quality_deficit_scale needs grasp_quality_scale", CFG_SOURCE)


CFG_SOURCE = (Path(__file__).resolve().parents[1]
              / "source/dg5f_isaaclab/dg5f_isaaclab/tasks/direct/dg5f_cube/dg5f_cube_env_cfg.py"
              ).read_text()


class BootstrapPhaseTests(unittest.TestCase):
    """The 10 deg predecessor phase, and the invariants that keep it from becoming a curriculum."""

    def setUp(self):
        self.cfg = curriculum.STREAM_CURRICULUM
        self.driver = (Path(__file__).resolve().parents[1] / "scripts/train_stream.py").read_text()

    def test_a_bootstrap_angle_is_inside_demonstrated_capability(self):
        # Measured: min_error 12.2 deg from a 20 deg start, i.e. ~8 deg of rotation, against a 5 deg
        # tolerance. The bootstrap angle must be reachable, or the bonus still never fires.
        demonstrated_rotation_deg = 20.0 - 12.2
        tolerance_deg = 5.0
        self.assertLessEqual(self.cfg.bootstrap_angle_deg - tolerance_deg, demonstrated_rotation_deg)
        self.assertGreater(self.cfg.bootstrap_angle_deg, tolerance_deg)

    def test_b_same_gate_at_both_angles(self):
        # There is exactly one gate() and one set of thresholds; a separate, looser bootstrap gate is
        # what would turn this into "trained on easy goals".
        self.assertEqual(self.driver.count("def gate("), 1)
        for field in ("first_goal_held_success_rate", "max_drop_rate", "goals_completed_per_episode"):
            self.assertEqual(self.driver.count(f"cfg.{field}"), 1, field)

    def test_c_bootstrap_must_return_to_the_target_angle(self):
        # Passing the bootstrap sets phase to target and re-arms the gate; it must not advance a stage.
        self.assertIn('state["phase"] = "target"', self.driver)
        self.assertIn('state["consecutive_passes"] = 0', self.driver)
        head = self.driver[:self.driver.index('state["phase"] = "target"')]
        self.assertNotIn("STREAM_STAGES.index(stage)", head,
                         "the bootstrap branch must come before any stage advance")

    def test_d_both_angle_fields_move_together(self):
        # The env asserts initial_error >= min_initial_goal_error_deg, so the angle alone trips it.
        self.assertIn("env.goal_stream_angle_deg={angle}", self.driver)
        self.assertIn("env.min_initial_goal_error_deg={angle}", self.driver)
        evaluator = (Path(__file__).resolve().parents[1] / "scripts/eval_checkpoints.py").read_text()
        self.assertIn("env_cfg.goal_stream_angle_deg = args_cli.goal_angle_deg", evaluator)
        self.assertIn("env_cfg.min_initial_goal_error_deg = args_cli.goal_angle_deg", evaluator)

    def test_e_evaluation_happens_at_the_training_angle(self):
        self.assertIn('"--goal_angle_deg", str(angle)', self.driver)

    def test_f_watchdog_does_not_compare_across_angles(self):
        # A different angle is a different task; comparing deterministic progress across the switch
        # would fire the watchdog on the angle change rather than on divergence.
        self.assertIn('e.get("angle_deg", angle) == angle', self.driver)

    def test_g_bootstrap_can_stall(self):
        self.assertIn("cfg.bootstrap_max_iterations", self.driver)
        self.assertGreater(self.cfg.bootstrap_max_iterations, 0)
        self.assertLessEqual(self.cfg.bootstrap_max_iterations, self.cfg.stage_max_iterations)


class NearGoalVelocityWindowTests(unittest.TestCase):
    """The anti-overshoot window has to mean "near the goal" at every goal magnitude.

    Found by changing the goal angle: with the window fixed at an absolute 10 deg and a 10 deg goal,
    goal_velocity_penalty measured -0.0025 per step against -0.0001..-0.0006 at a 20 deg goal, i.e.
    the term was braking the whole approach instead of the arrival.
    """

    ABSOLUTE_WINDOW_DEG = 10.0

    def window(self, goal_angle_deg, goal_stream=True):
        if not goal_stream:
            return self.ABSOLUTE_WINDOW_DEG
        return min(self.ABSOLUTE_WINDOW_DEG, 0.5 * goal_angle_deg)

    def test_a_unchanged_at_the_prescribed_angle(self):
        self.assertEqual(self.window(20.0), self.ABSOLUTE_WINDOW_DEG)

    def test_b_shrinks_with_the_bootstrap_angle(self):
        self.assertEqual(self.window(10.0), 5.0)

    def test_c_never_covers_more_than_half_the_approach(self):
        for angle in (6.0, 10.0, 15.0, 20.0, 45.0, 90.0):
            self.assertLessEqual(self.window(angle), 0.5 * angle)

    def test_d_still_leaves_room_outside_the_tolerance(self):
        # A window at or below the 5 deg tolerance would make the term fire only where the goal is
        # already reached, which is not anti-overshoot any more.
        self.assertGreaterEqual(self.window(curriculum.STREAM_CURRICULUM.bootstrap_angle_deg), 5.0)

    def test_e_not_applied_without_the_goal_stream(self):
        # Without the stream the magnitude comes from target_angle_range_deg, so keying off
        # goal_stream_angle_deg would be a wrong coupling.
        self.assertEqual(self.window(10.0, goal_stream=False), self.ABSOLUTE_WINDOW_DEG)
        source = (Path(__file__).resolve().parents[1]
                  / "source/dg5f_isaaclab/dg5f_isaaclab/tasks/direct/dg5f_cube/dg5f_cube_env_cfg.py"
                  ).read_text()
        self.assertIn("if self.goal_stream:\n            self.goal_velocity_window_deg = min(", source)
