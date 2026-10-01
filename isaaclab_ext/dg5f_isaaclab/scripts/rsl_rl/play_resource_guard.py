"""Reserve the campaign GPU before starting a memory-hungry GUI/Kit instance."""
import fcntl
from pathlib import Path
import subprocess
import time


def reserve_gui_gpu(root: Path, wait: bool = False):
    directory = root / 'logs/overnight_ab'
    directory.mkdir(parents=True, exist_ok=True)
    lease = (directory / 'driver.lock').open('a')
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                raise RuntimeError('Training/evaluation campaign owns the GPU. Use --wait_for_gpu to wait before opening GUI.')
            print('[PLAY] Waiting for training AND evaluation to release the GPU...', flush=True)
            fcntl.flock(lease, fcntl.LOCK_EX)
        announced = False
        while True:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
                capture_output=True, text=True, check=True)
            processes = result.stdout.strip()
            if not processes:
                return lease  # Keep the descriptor alive until play exits.
            if not wait:
                raise RuntimeError(f'GPU has active compute processes: {processes}. Use --wait_for_gpu.')
            if not announced:
                print(f'[PLAY] Waiting for GPU processes: {processes}', flush=True)
                announced = True
            time.sleep(5)
    except BaseException:
        lease.close()
        raise
