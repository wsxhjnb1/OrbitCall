"""Wait for the sole serving-study owner, then audit both full Triton score sets."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    if sys.platform != "linux":
        raise RuntimeError("Audits run only on Spark")
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-driver-pid", type=int, required=True)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    completion = Path("runs/normalizer_qwen1p5b_20261013_budget_v2/completed.json")
    deadline = time.monotonic() + 3000
    while not completion.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Retain/reconcile the original study")
        os.kill(args.wait_driver_pid, 0)
        time.sleep(5)
    time.sleep(3)
    for model, label in (("Qwen2.5-3B-Instruct", "qwen3b"), ("Qwen2.5-1.5B-Instruct", "qwen1p5b")):
        run = f"runs/audit_full_triton_{label}_20261013_v1"
        if Path(run).exists():
            raise FileExistsError(run)
        with Path(f"logs/audit_full_triton_{label}_v1.log").open("x") as output:
            subprocess.run([sys.executable, "scripts/audit_full_triton.py", "--model", f"models/{model}",
                            "--ranking", f"runs/rank_{label}_20261013_reference_v2", "--run", run],
                           env={**os.environ, "PYTHONPATH": "src"}, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
    Path("runs/full_triton_audits_complete.txt").write_text("Both immutable 119-item audit runs returned success.\n")


if __name__ == "__main__":
    main()
