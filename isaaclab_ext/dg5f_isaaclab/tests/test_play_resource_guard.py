import fcntl
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('play_guard', Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/play_resource_guard.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_active_campaign_blocks_before_gpu_query(tmp_path, monkeypatch):
    folder = tmp_path / 'logs/overnight_ab'
    folder.mkdir(parents=True)
    with (folder / 'driver.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match='campaign owns'):
            module.reserve_gui_gpu(tmp_path)


def test_unrelated_gpu_process_is_not_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='123, 4362 MiB'))
    with pytest.raises(RuntimeError, match='active compute'):
        module.reserve_gui_gpu(tmp_path)


def test_free_gpu_lease_is_held_until_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=''))
    lease = module.reserve_gui_gpu(tmp_path)
    with pytest.raises(RuntimeError, match='campaign owns'):
        module.reserve_gui_gpu(tmp_path)
    lease.close()
    module.reserve_gui_gpu(tmp_path).close()


def test_verified_small_telegram_does_not_block_gui(tmp_path, monkeypatch):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='4209, 23 MiB'))
    monkeypatch.setattr(Path, 'readlink', lambda self: next(iter(module._DESKTOP_EXECUTABLES)))
    module.reserve_gui_gpu(tmp_path).close()


@pytest.mark.parametrize('row,executable', [
    ('4209, 129 MiB', next(iter(module._DESKTOP_EXECUTABLES))),
    ('4209, 23 MiB', Path('/usr/bin/python3')),
    ('4209, N/A', next(iter(module._DESKTOP_EXECUTABLES))),
])
def test_exception_does_not_hide_other_workloads(tmp_path, monkeypatch, row, executable):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=row))
    monkeypatch.setattr(Path, 'readlink', lambda self: executable)
    with pytest.raises(RuntimeError, match='active compute'):
        module.reserve_gui_gpu(tmp_path)
