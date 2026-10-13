#!/usr/bin/env python3
"""Freeze original input-only Agent tasks on Spark; never evaluate a response."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("Task construction and split computation run only on authorized Spark")
    identity = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"
    ], text=True, timeout=15).strip().splitlines()
    if len(identity) != 1 or "GB10" not in identity[0] or GPU_UUID not in identity[0]:
        raise RuntimeError("Wrong live task-freeze host/GPU")
    from orbitcall.agent_tasks import FORMAT_VERSION, SPLIT_SALT, input_provenance, tasks

    output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise FileExistsError("Input freeze must be a fresh path inside the OrbitCall repository")
    records = tasks()
    groups = {}
    for record in records:
        previous = groups.setdefault(record["group_id"], record["split"])
        if previous != record["split"]:
            raise ValueError("An object/policy group leaked across splits")
    if (len(records) != 48 or len({row["id"] for row in records}) != 48
            or Counter(row["split"] for row in records) != {"development": 12, "test": 36}
            or Counter(row["family"] for row in records) != {"catalog": 24, "calendar": 24}):
        raise ValueError("Input-only fixture population changed")
    fixture_ids = [row["id"] for family in ("catalog", "calendar")
                   for row in sorted((row for row in records if row["family"] == family and row["split"] == "test"),
                                     key=lambda row: row["id"])[:4]]
    inventory = [Path(__file__), ROOT / "src/orbitcall/agent_tasks.py", ROOT / "src/orbitcall/serialization.py"]
    payload = {"format_version": FORMAT_VERSION, "stage": "agent-inputs-frozen", "epoch": time.time(),
               "answers_accessed": False, "tool_execution_performed": False, "model_inference_performed": False,
               "live_gpu_identity": identity[0], "split_salt": SPLIT_SALT,
               "provenance": input_provenance(), "record_count": 48,
               "split_counts": dict(Counter(row["split"] for row in records)),
               "family_counts": dict(Counter(row["family"] for row in records)),
               "group_count": len(groups), "timing_fixture_ids": fixture_ids,
               "timing_selection_rule": "first four lexicographic test IDs within each family; no model-output eligibility filter",
               "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in inventory}, "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "input_only_agent_tasks_frozen", "records": len(records),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "timing_fixture_ids": fixture_ids}, indent=2))


if __name__ == "__main__":
    main()
