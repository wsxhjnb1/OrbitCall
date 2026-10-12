"""Record the experiment runtime without exposing credentials or other projects."""

import importlib.metadata
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--analysis", action="store_true")
parser.add_argument("--output", type=Path, default=root / "metadata/runtime_current.json")
args = parser.parse_args()
packages = ["torch", "transformers", "triton", "numpy", "pytest"]
if args.analysis:
    packages.extend(["scipy", "matplotlib"])
record = {"host_architecture": platform.machine(), "python": platform.python_version(),
          "packages": {name: importlib.metadata.version(name)
                       for name in packages},
          "gpu": subprocess.check_output([
              "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"
          ], text=True).strip()}
if record["host_architecture"] != "aarch64" or "GB10" not in record["gpu"]:
    raise RuntimeError("This experiment is authorized for Spark GB10")
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as output:
    output.write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
