"""Contracts guarding against the costly silent-resume and orphan-process failures."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from overnight_process import ResumeError, check_resume, run_checked


def test_resume_requires_actual_path(tmp_path):
    wanted = tmp_path / 'warm/model_0.pt'
    assert not check_resume('initializing simulator', wanted)
    assert check_resume(f'[INFO]: Loading model checkpoint from: {wanted}\nLearning iteration 0/600', wanted)
    with pytest.raises(ResumeError, match='Wrong warm'):
        check_resume(f'Loading model checkpoint from: {tmp_path}/other/model_0.pt', wanted)
    with pytest.raises(ResumeError, match='without'):
        check_resume('Learning iteration 0/600', wanted)


def test_successful_child_without_resume_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr('overnight_process.gpu_processes', lambda: {})
    log = tmp_path / 'child.log'
    with pytest.raises(ResumeError, match='No checkpoint'):
        run_checked([sys.executable, '-c', 'print("no load")'], log, tmp_path,
                    expected=tmp_path / 'warm.pt', timeout=20)
    assert log.with_suffix('.process.json').exists()


def test_nonzero_exit_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr('overnight_process.gpu_processes', lambda: {})
    with pytest.raises(RuntimeError, match='exit=7'):
        run_checked([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path / 'fail.log', tmp_path)


def test_foreign_gpu_job_blocks_launch(tmp_path, monkeypatch):
    monkeypatch.setattr('overnight_process.gpu_processes', lambda: {1234: '1024'})
    with pytest.raises(RuntimeError, match='GPU occupied'):
        run_checked([sys.executable, '-c', 'print("must not run")'], tmp_path / 'blocked.log', tmp_path)
    assert not (tmp_path / 'blocked.log').exists()


def test_nested_sessions_are_owned_but_unrelated_processes_are_not(monkeypatch):
    from types import SimpleNamespace
    from overnight_process import descendant_groups
    listing = '100 1 100\n101 100 100\n200 101 200\n201 200 200\n999 1 999\n'
    monkeypatch.setattr('overnight_process.subprocess.run', lambda *a, **kw: SimpleNamespace(stdout=listing))
    assert descendant_groups(100) == {100, 200}
