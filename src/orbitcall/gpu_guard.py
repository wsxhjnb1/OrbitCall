"""Read-only whole-GPU occupancy checks; never terminate another project."""

import os
import subprocess


def gpu_pids():
    text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True, timeout=15)
    return {int(row.strip()) for row in text.splitlines() if row.strip()}


def require_idle_gpu():
    active = gpu_pids()
    if active:
        raise RuntimeError(f"GPU already occupied; no model launched: {sorted(active)}")


def require_owned_gpu():
    unrelated = gpu_pids() - {os.getpid()}
    if unrelated:
        raise RuntimeError(f"Unrelated GPU contexts invalidate this run: {sorted(unrelated)}")
