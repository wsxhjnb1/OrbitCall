"""Freeze answer-free acceleration timing fixtures on Spark before discovery."""
import json
from pathlib import Path
import platform
import time

from benchmark_scoring import sha256, write_new
from orbitcall.bfcl import load_frozen_inputs
from run_accelerated_scoring import choose_timing_fixtures

if platform.system() != "Linux" or platform.machine() != "aarch64":
    raise RuntimeError("Fixture freezing runs on Spark")
root = Path(__file__).resolve().parents[1]
rows = load_frozen_inputs(root / "data/frozen_inputs.json")
records = []
for model, name in (("Qwen2.5-3B-Instruct", "orbitcall_qwen3b_20261013_v1"),
                    ("Qwen2.5-1.5B-Instruct", "orbitcall_qwen1p5b_20261013_v1")):
    path = root / "runs" / name / "candidates.json"
    data = json.loads(path.read_text())
    fixtures, dispositions = choose_timing_fixtures(data, rows)
    records.append({"model": model, "candidate_run": name, "candidates_sha256": sha256(path),
                    "timing_fixture_ids": [x["id"] for x in fixtures], "test_dispositions": dispositions})
write_new(root / "metadata/acceleration_input_freeze_v1.json", {
    "answers_accessed": False, "epoch": time.time(), "records": records,
    "inputs_sha256": sha256(root / "data/frozen_inputs.json"),
    "selection_rule": "first up to8 lexical test IDs with >=2 primary calls",
    "stage": "before tensorlinear or CUDAgraph discovery; initial12-arm packing screen retired",
    "script_sha256": sha256(Path(__file__))})
print(json.dumps(records))
