"""Sequential Spark study driver; resumes completed stages, never duplicates GPU work."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    if sys.platform != "linux":
        raise RuntimeError("Study execution is authorized only on Spark")
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-generation-pid", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    checkpoint = root / "runs/orbitcall_qwen1p5b_20261013_v1/candidates.json"
    deadline = time.monotonic() + 1200
    while not checkpoint.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Retain the generation run; do not launch replacements")
        os.kill(args.wait_generation_pid, 0)
        time.sleep(5)
    time.sleep(3)  # Allow the completed generator's own CUDA context to exit.
    environment = {**os.environ, "PYTHONPATH": "src"}
    stages = [
        ("gate3b_views", ["scripts/check_scoring.py", "--model", "models/Qwen2.5-3B-Instruct",
                          "--output", "runs/scoring_gate_3b_fp32_v3.json"],
         Path("runs/scoring_gate_3b_fp32_v3.json")),
    ]
    for model, label in (("Qwen2.5-3B-Instruct", "qwen3b"), ("Qwen2.5-1.5B-Instruct", "qwen1p5b")):
        generation = f"runs/orbitcall_{label}_20261013_v1"
        ranking = f"runs/rank_{label}_20261013_reference_v2"
        benchmark = f"runs/normalizer_{label}_20261013_budget_v2"
        stages.extend([
            (f"rank_{label}_reference_v2", ["scripts/rank_candidates.py", "--model", f"models/{model}",
                               "--generation", generation, "--run", ranking],
             Path(ranking) / "pre_gold_freeze.json"),
            (f"normalizer_{label}_budget_v2", ["scripts/benchmark_scoring.py", "--model", f"models/{model}",
                                    "--candidates", generation + "/candidates.json", "--run", benchmark,
                                    "--normalizer-only", "--max-seconds", "1200"],
             Path(benchmark) / "completed.json"),
        ])
    for name, command, completion in stages:
        if completion.exists():
            print(json.dumps({"stage": name, "status": "retained_completion"}), flush=True)
            continue
        log = root / "logs" / f"{name}.log"
        if log.exists():
            raise RuntimeError(f"Unreconciled prior attempt: {name}; retain it for audit")
        print(json.dumps({"stage": name, "status": "started", "epoch": time.time()}), flush=True)
        with log.open("x") as output:
            result = subprocess.run([sys.executable, *command], env=environment,
                                    stdout=output, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not completion.exists():
            raise RuntimeError(f"Stage {name} failed; preserve its exact run/log")
        print(json.dumps({"stage": name, "status": "completed", "epoch": time.time()}), flush=True)
    print(json.dumps({"status": "both_models_scored_frozen_and_timed"}), flush=True)


if __name__ == "__main__":
    main()
