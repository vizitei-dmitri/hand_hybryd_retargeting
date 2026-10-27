"""Contract tests with tiny local NPZ fixtures and real Python subprocesses."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from kinect_mano.data import JointEstimate
from kinect_mano.hotrack import (
    HOTrackInitializer, HOTrackProcessError, OFFICIAL_INFERENCE_IMPLEMENTED,
    load_joint_estimate,
)


class HOTrackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hotrack-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.joints = np.arange(63, dtype=float).reshape(21, 3) / 1000
        self.points = np.array([[0., 0., 0.4], [0.02, 0.03, 0.41]])

    def archive(self, name="joints.npz", **arrays):
        path = self.root / name
        np.savez(path, **arrays)
        return path

    def worker(self, source, **kwargs):
        # Keep the worker file outside the adapter's temporary working directory.
        path = self.root / "user supplied worker.py"
        path.write_text("import sys\nimport numpy as np\n" + source)
        return HOTrackInitializer(command=[sys.executable, str(path), "{input}", "{output}"], **kwargs)

    def test_precomputed_returns_shared_dataclass_and_copies(self):
        path = self.archive(joints_m=self.joints)
        with self.assertLogs("kinect_mano.hotrack", level="INFO") as logs:
            initializer = HOTrackInitializer(joints_npz=path)
        self.assertIn("Official HOTrack inference integration is not implemented", " ".join(logs.output))
        self.assertFalse(OFFICIAL_INFERENCE_IMPLEMENTED)
        estimate = initializer.estimate_joints(None)
        self.assertIsInstance(estimate, JointEstimate)
        np.testing.assert_array_equal(estimate.joints_m, self.joints)
        np.testing.assert_array_equal(estimate.confidence, np.ones(21))
        self.assertTrue(estimate.valid)
        estimate.joints_m[:] = 99
        np.testing.assert_array_equal(initializer.estimate_joints(None).joints_m, self.joints)

    def test_explicit_sequence_selection_and_confidence(self):
        confidence = np.stack([np.ones(21), np.full(21, .6)])
        path = self.archive(joints_m=np.stack([self.joints, self.joints + .1]),
                            confidence=confidence, valid=np.array([True, False]))
        initializer = HOTrackInitializer(joints_npz=path)
        estimate = initializer.estimate_joints(SimpleNamespace(frame_index=1))
        np.testing.assert_array_equal(estimate.joints_m, self.joints + .1)
        np.testing.assert_array_equal(estimate.confidence, confidence[1])
        self.assertFalse(estimate.valid)
        np.testing.assert_array_equal(load_joint_estimate(path, frame_index=0).joints_m, self.joints)
        with self.assertRaisesRegex(ValueError, "requires.*frame_index"):
            initializer.estimate_joints(None)
        with self.assertRaisesRegex(IndexError, "outside precomputed"):
            initializer.estimate_joints({"frame_index": 2})
        for index in (-1, 1., True, "0", [0]):
            with self.subTest(index=index), self.assertRaises(ValueError):
                initializer.estimate_joints({"frame_index": index})

    def test_sequence_does_not_broadcast_confidence(self):
        path = self.archive(joints_m=np.stack([self.joints, self.joints]), confidence=np.ones(21))
        with self.assertRaisesRegex(ValueError, "no implicit broadcast"):
            HOTrackInitializer(joints_npz=path)

    def test_sequence_validates_later_frames_at_configuration(self):
        frames = np.stack([self.joints, self.joints])
        frames[1, 5, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            HOTrackInitializer(joints_npz=self.archive(joints_m=frames))

    def test_missing_and_empty_resources(self):
        missing = self.root / "missing.pt"
        empty = self.root / "empty.pt"
        empty.touch()
        for path, error in ((missing, FileNotFoundError), (empty, ValueError), (self.root, FileNotFoundError)):
            with self.subTest(path=path):
                with self.assertRaises(error):
                    HOTrackInitializer(joints_npz=path)
                with self.assertRaises(error):
                    HOTrackInitializer(command=[sys.executable, "{input}", "{output}", "{weights}"], weights_path=path)

    def test_constructor_rejects_ambiguous_or_incomplete_modes(self):
        path = self.archive(joints_m=self.joints)
        configurations = [
            {}, {"joints_npz": path, "command": ["x"]},
            {"command": "python worker.py {input} {output}"}, {"command": []},
            {"command": ["x", "{input}"]}, {"command": ["x", "{output}"]},
            {"command": ["x", "{input}", "{output}", "{weights}"]},
            {"command": ["x", "{input}", "{output}", ""]},
            {"command": ["x", "{input}", "{output}", 7]},
            {"command": ["x\0", "{input}", "{output}"]},
            {"joints_npz": path, "weights_path": path},
        ]
        for kwargs in configurations:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HOTrackInitializer(**kwargs)
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                HOTrackInitializer(joints_npz=path, timeout_s=timeout)
        with self.assertRaises(FileNotFoundError):
            HOTrackInitializer(joints_npz=path, cwd=self.root / "missing")

    def test_invalid_joint_results(self):
        nan_joints = self.joints.copy()
        nan_joints[4] = np.nan
        inf_joints = self.joints.copy()
        inf_joints[4] = np.inf
        cases = [
            {}, {"joints_m": np.zeros((20, 3))}, {"joints_m": np.zeros((21, 2))},
            {"joints_m": np.zeros((0, 21, 3))}, {"joints_m": nan_joints},
            {"joints_m": inf_joints, "valid_mask": np.zeros(21, bool)},
            {"joints_m": self.joints.astype(complex)},
            {"joints_m": self.joints.astype(str)}, {"joints_m": self.joints.astype(object)},
            {"joints_m": self.joints, "confidence": np.ones(20)},
            {"joints_m": self.joints, "confidence": np.full(21, 1.1)},
            {"joints_m": self.joints, "confidence": np.full(21, -.1)},
            {"joints_m": self.joints, "confidence": np.full(21, np.nan)},
            {"joints_m": self.joints, "valid_mask": np.ones(20, bool)},
            {"joints_m": self.joints, "valid_mask": np.full(21, 2)},
            {"joints_m": self.joints, "valid_mask": np.full(21, "false")},
            {"joints_m": self.joints, "confidence": np.zeros(21), "valid_mask": np.ones(21, bool)},
            {"joints_m": self.joints, "valid": "false"},
            {"joints_m": self.joints, "valid": np.ones(1, bool)},
            {"joints_m": self.joints, "valid_mask": np.zeros(21, bool), "valid": True},
            {"joints_m": self.joints, "schema_version": 2},
            {"joints_m": self.joints, "schema_version": 1.},
        ]
        for index, arrays in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_joint_estimate(self.archive(**arrays))

    def test_nan_is_allowed_only_for_missing_joints(self):
        joints = self.joints.copy()
        joints[4] = np.nan
        mask = np.ones(21, bool)
        mask[4] = False
        estimate = load_joint_estimate(self.archive(joints_m=joints, valid_mask=mask))
        self.assertTrue(estimate.valid)
        self.assertFalse(estimate.valid_mask[4])
        self.assertEqual(estimate.confidence[4], 0)
        self.assertTrue(np.isnan(estimate.joints_m[4]).all())

    def test_zero_confidence_and_explicit_invalid_estimate(self):
        estimate = load_joint_estimate(self.archive(joints_m=np.full((21, 3), np.nan),
                                                  confidence=np.zeros(21), valid=False))
        self.assertFalse(estimate.valid)
        self.assertFalse(estimate.valid_mask.any())

    def test_corrupt_npz_and_npy_are_rejected(self):
        path = self.root / "broken.npz"
        path.write_bytes(b"not a numpy archive")
        with self.assertRaisesRegex(ValueError, "Invalid joint NPZ"):
            load_joint_estimate(path)
        path = self.root / "array.npy"
        np.save(path, self.joints)
        with self.assertRaisesRegex(ValueError, "Invalid joint NPZ"):
            load_joint_estimate(path)

    def test_worker_receives_validated_cloud_previous_state_and_index(self):
        source = '''
from pathlib import Path
with np.load(sys.argv[1], allow_pickle=False) as inputs:
    assert inputs['schema_version'].item() == 1
    assert inputs['frame_index'].item() == 7
    np.testing.assert_array_equal(inputs['points_m'], [[0.,0.,.4],[.02,.03,.41]])
    assert inputs['previous_valid'].item()
    mask = inputs['previous_valid_mask']
    assert not mask[4]
    assert np.isnan(inputs['previous_joints_m'][4]).all()
    np.savez(sys.argv[2], schema_version=np.array(1), joints_m=inputs['previous_joints_m'],
             confidence=inputs['previous_confidence'], valid_mask=mask, valid=True)
Path(sys.argv[2]).with_suffix('.unused').write_text('cleaned with temporary directory')
'''
        previous = JointEstimate(self.joints, np.ones(21), np.arange(21) != 4, True)
        estimate = self.worker(source).estimate_joints({"points_m": self.points, "frame_index": 7}, previous)
        np.testing.assert_allclose(estimate.joints_m, previous.joints_m, equal_nan=True)
        np.testing.assert_array_equal(estimate.valid_mask, previous.valid_mask)

    def test_worker_accepts_raw_previous_joints_and_cloud_object(self):
        source = '''
with np.load(sys.argv[1], allow_pickle=False) as inputs:
    np.savez(sys.argv[2], joints_m=inputs['previous_joints_m'])
'''
        estimate = self.worker(source).estimate_joints(SimpleNamespace(points_m=self.points), self.joints)
        np.testing.assert_array_equal(estimate.joints_m, self.joints)

    def test_worker_without_previous_and_with_detection_failure(self):
        source = '''
with np.load(sys.argv[1], allow_pickle=False) as inputs:
    assert not any(key.startswith('previous_') for key in inputs.files)
np.savez(sys.argv[2], joints_m=np.full((21,3), np.nan), confidence=np.zeros(21), valid=False)
'''
        estimate = self.worker(source).estimate_joints(self.points)
        self.assertFalse(estimate.valid)

    def test_worker_paths_with_spaces_weights_cwd_and_cleanup(self):
        weights = self.root / "local weights.pt"
        weights.write_bytes(b"dummy test file; adapter does not parse checkpoints")
        record = self.root / "observed_paths.txt"
        worker = self.root / "worker.py"
        worker.write_text('''
import sys
from pathlib import Path
import numpy as np
assert Path.cwd() == Path(sys.argv[5])
assert Path(sys.argv[3]).is_file()
Path(sys.argv[4]).write_text(sys.argv[1] + '\\n' + sys.argv[2])
with np.load(sys.argv[1], allow_pickle=False) as inputs:
    np.savez(sys.argv[2], joints_m=np.tile(inputs['points_m'][0], (21,1)))
''')
        initializer = HOTrackInitializer(
            command=[sys.executable, str(worker), "{input}", "{output}", "{weights}", str(record), str(self.root)],
            weights_path=weights, cwd=self.root,
        )
        estimate = initializer.estimate_joints(self.points)
        np.testing.assert_array_equal(estimate.joints_m[0], self.points[0])
        for path in record.read_text().splitlines():
            self.assertFalse(Path(path).exists())
        weights.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "weights.*local weights.pt"):
            initializer.estimate_joints(self.points)

    def test_invalid_observation_and_previous_are_rejected_before_subprocess(self):
        initializer = self.worker("raise AssertionError('worker must not run')\n")
        for observation in (None, {}, [], np.empty((0, 3)), np.ones((3, 2)),
                            np.full((1, 3), np.nan), np.full((1, 3), np.inf),
                            np.full((1, 3), "1"), {"points_m": self.points, "frame_index": -1}):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                initializer.estimate_joints(observation)
        for previous in (np.ones((20, 3)), np.full((21, 3), np.nan)):
            with self.subTest(previous=previous), self.assertRaises(ValueError):
                initializer.estimate_joints(self.points, previous)

    def test_nonzero_exit_preserves_error_and_does_not_fall_back(self):
        initializer = self.worker("sys.stderr.write('checkpoint architecture mismatch'); sys.exit(9)\n")
        with self.assertRaisesRegex(HOTrackProcessError, "code 9: checkpoint architecture mismatch"):
            initializer.estimate_joints(self.points, self.joints)

    def test_error_output_is_bounded(self):
        initializer = self.worker("sys.stderr.write('x'*20000 + ' error tail');sys.exit(1)\n")
        with self.assertRaises(HOTrackProcessError) as context:
            initializer.estimate_joints(self.points)
        self.assertLess(len(str(context.exception)), 4300)
        self.assertIn("error tail", str(context.exception))

    def test_missing_executable(self):
        initializer = HOTrackInitializer(command=[str(self.root / "missing-python"), "{input}", "{output}"])
        with self.assertRaisesRegex(HOTrackProcessError, "Cannot start"):
            initializer.estimate_joints(self.points)

    def test_timeout(self):
        initializer = self.worker("import time; time.sleep(30)\n", timeout_s=.1)
        with self.assertRaisesRegex(HOTrackProcessError, "timed out"):
            initializer.estimate_joints(self.points)

    def test_missing_empty_malformed_and_sequence_worker_output(self):
        sources = [
            "pass\n", "open(sys.argv[2], 'wb').close()\n",
            "open(sys.argv[2], 'wb').write(b'not npz')\n",
            "np.savez(sys.argv[2], wrong_key=np.zeros((21,3)))\n",
            "np.savez(sys.argv[2], joints_m=np.full((21,3), np.nan))\n",
            "np.savez(sys.argv[2], joints_m=np.zeros((1,21,3)))\n",
        ]
        for source in sources:
            with self.subTest(source=source), self.assertRaisesRegex(HOTrackProcessError, "output"):
                self.worker(source).estimate_joints(self.points)

    def test_literal_braces_are_not_python_format_templates(self):
        source = "np.savez(sys.argv[2], **{'joints_m': np.zeros((21,3))})"
        initializer = HOTrackInitializer(command=[sys.executable, "-c", "import sys,numpy as np;" + source,
                                                 "{input}", "{output}"])
        self.assertTrue(initializer.estimate_joints(self.points).valid)


if __name__ == "__main__":
    unittest.main()
