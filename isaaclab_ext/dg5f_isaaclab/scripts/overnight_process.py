"""Supervise one GPU child; never kill processes belonging to another job."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


class ResumeError(RuntimeError):
    pass


def gpu_processes():
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True)
    return {int(parts[0]): parts[1].strip() for line in result.stdout.splitlines()
            if len(parts := line.split(',')) == 2}


def assert_gpu_free():
    processes = gpu_processes()
    if processes:
        raise RuntimeError(f'GPU occupied; refusing concurrent simulation: {processes}')


def descendant_groups(root_pid):
    """Nested stream supervisors start their own sessions; track those children too."""
    listing = subprocess.run(['ps', '-eo', 'pid=,ppid=,pgid='], capture_output=True,
                             text=True, check=True).stdout
    rows = [tuple(map(int, line.split())) for line in listing.splitlines()]
    descendants = {root_pid}
    while True:
        children = {pid for pid, parent, _ in rows if parent in descendants}
        enlarged = descendants | children
        if enlarged == descendants:
            break
        descendants = enlarged
    return {group for pid, _, group in rows if pid in descendants}


def stop_group(pgid):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        time.sleep(2)


def check_resume(text, expected):
    prefix = 'Loading model checkpoint from:'
    paths = [line.split(prefix, 1)[1].strip() for line in text.splitlines() if prefix in line]
    if paths and any(Path(path).resolve() != Path(expected).resolve() for path in paths):
        raise ResumeError(f'Wrong warm start: {paths}; expected {expected}')
    if not paths and 'Learning iteration' in text:
        raise ResumeError('Training began without Loading model checkpoint from')
    return bool(paths)


def run_checked(command, log_path, cwd, expected=None, timeout=7200, require_free=True):
    """Validate resume before updates, catch hung traceback/shutdown, record peak VRAM."""
    if require_free:
        assert_gpu_free()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'[JOB] {command}\n[JOB] log={log_path}', flush=True)
    started = time.monotonic()
    seen_resume = expected is None
    failure_since = None
    finished_since = None
    peak = 0
    peak_total = 0
    samples = []
    with log_path.open('w') as handle:
        child = subprocess.Popen(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        owned_groups = {child.pid}
        try:
            while True:
                owned_groups.update(descendant_groups(child.pid))
                status = child.poll()
                content = log_path.read_text(errors='replace')
                elapsed = time.monotonic() - started
                if expected is not None:
                    seen_resume = check_resume(content, expected)
                    if not seen_resume and elapsed > 600:
                        raise ResumeError('No checkpoint load confirmation within 600 s')
                for pid, memory in gpu_processes().items():
                    try:
                        owned = os.getpgid(pid) in owned_groups
                    except ProcessLookupError:
                        continue
                    if owned and memory.isdigit():
                        peak = max(peak, int(memory))
                        samples.append({'seconds': round(elapsed, 1), 'pid': pid, 'MiB': int(memory)})
                total = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True).stdout.strip()
                if total.isdigit():
                    peak_total = max(peak_total, int(total))
                if 'ResumeError:' in content:
                    raise ResumeError(f'Nested training failed resume verification: {log_path}')
                if status is not None:
                    break
                if 'Traceback (most recent call last)' in content or 'Error executing job with overrides' in content:
                    failure_since = failure_since or time.monotonic()
                    if time.monotonic() - failure_since > 20:
                        raise RuntimeError(f'Child failed and did not exit: {log_path}')
                if 'Training time:' in content:
                    finished_since = finished_since or time.monotonic()
                    if time.monotonic() - finished_since > 120:
                        raise RuntimeError(f'Child shutdown hung after training: {log_path}')
                if elapsed > timeout:
                    raise TimeoutError(f'Child exceeded {timeout}s: {log_path}')
                time.sleep(5)
            if not seen_resume:
                raise ResumeError(f'No checkpoint load confirmation: {log_path}')
            if status:
                raise RuntimeError(f'Child exit={status}: {log_path}')
        finally:
            for group in sorted(owned_groups, reverse=True):
                stop_group(group)
            child.wait()
            log_path.with_suffix('.process.json').write_text(json.dumps({
                'command': command, 'expected_checkpoint': str(expected) if expected else None,
                'resume_verified': seen_resume, 'returncode': child.returncode,
                'owned_process_groups': sorted(owned_groups),
                'seconds': time.monotonic() - started, 'peak_compute_MiB': peak,
                'peak_total_MiB': peak_total,
                'gpu_samples': samples}, indent=2))
    assert_gpu_free()
    return {'seconds': time.monotonic() - started, 'peak_compute_MiB': peak, 'peak_total_MiB': peak_total}
