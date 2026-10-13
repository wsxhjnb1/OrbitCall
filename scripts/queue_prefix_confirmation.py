"""Start the second frozen model only after the existing first owned run exits."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first-pid", type=int, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    first = root / "runs/prefix_approximate_confirmation_3b_20261013_v1"
    second = root / "runs/prefix_approximate_confirmation_1p5b_20261013_v1"
    if second.exists():
        raise FileExistsError("Reconcile the existing second-model run")
    deadline = time.monotonic() + 3700
    while True:
        process = Path(f"/proc/{args.first_pid}/cmdline")
        try:
            command = process.read_bytes()
        except FileNotFoundError:
            break
        if b"run_approximate_scoring.py" not in command or first.name.encode() not in command:
            break  # An exited/reused PID is not an instruction to control it.
        if time.monotonic() > deadline:
            raise RuntimeError("Bounded first-model wait expired")
        time.sleep(5)
    complete = json.loads((first / "complete.json").read_text())
    if complete.get("valid") is not True:
        raise RuntimeError("First model failed; no queued second model launched")
    command = [str(root / ".venv/bin/python"), "scripts/run_approximate_scoring.py",
               "--model", "models/Qwen2.5-1.5B-Instruct",
               "--ranking", "runs/rank_qwen1p5b_20261013_reference_v2",
               "--candidates", "runs/orbitcall_qwen1p5b_20261013_v1/candidates.json",
               "--selection", "runs/prefix_approximate_discovery_3b_20261013_v1/selection.json",
               "--run", str(second.relative_to(root))]
    print(json.dumps({"action": "launch_second_frozen_model", "first_pid": args.first_pid,
                      "first_valid": True, "second_run": second.name}), flush=True)
    log = root / "logs" / (second.name + ".log")
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Second model driver failed; retained log and run directory")
    print(json.dumps({"action": "second_model_finished", "returncode": result.returncode}), flush=True)


if __name__ == "__main__":
    main()
