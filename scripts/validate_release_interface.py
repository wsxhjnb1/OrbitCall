"""Verify the post-study API-only default correction on the original Spark."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("metadata/release_interface_validation.json"))
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("Release validation runs on Spark")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                                  text=True, timeout=15).strip()
    if "GB10" not in gpu or "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4" not in gpu:
        raise RuntimeError("Original experiment host required")
    current = Path("src/orbitcall/scoring.py")
    current_bytes = current.read_bytes()
    old = b'def score(self, prompt, sequences, method="tree", reducer="torch", batch_size=8):'
    new = b'def score(self, prompt, sequences, method="full", reducer="torch", batch_size=8):'
    if current_bytes.count(new) != 1:
        raise ValueError("Full scorer is not the sole default dispatch")
    sources = []
    for run in ("rank_qwen3b_20261013_reference_v2", "rank_qwen1p5b_20261013_reference_v2"):
        root = Path("runs") / run
        manifest = json.loads((root / "ranking_manifest.json").read_text())
        snapshot = root / "source" / current
        original = snapshot.read_bytes()
        if sha(snapshot) != manifest["source_sha256"][current.as_posix()]:
            raise ValueError("Original scorer snapshot changed")
        if original.count(old) != 1 or original.replace(old, new) != current_bytes:
            raise ValueError("Release differs by more than the one default parameter")
        sources.append({"run": run, "original_scoring_sha256": sha(snapshot),
                        "manifest_sha256": sha(root / "ranking_manifest.json")})
    test_log = Path("logs/unit_release_default_full_v1.log")
    if "230 passed" not in test_log.read_text() or "failed" in test_log.read_text():
        raise ValueError("Final release test suite did not pass")
    receipt = {"valid": True, "host": "original DGX Spark GB10", "current_scoring_sha256": sha(current),
               "validator_sha256": sha(Path(__file__)), "test_log_sha256": sha(test_log),
               "change": "Only SequenceScorer.score default method: tree -> full",
               "numerical_implementation_unchanged": True, "experiment_sources_preserved": sources,
               "tests": 230, "quality_or_timing_rerun": False,
               "scope": "Post-study interface correction; retained experiments use explicit method arguments"}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
