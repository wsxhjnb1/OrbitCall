"""Keep the explicitly pinned public runtime ahead of public fallback packages."""
import argparse
from pathlib import Path
import platform
if platform.system()!="Linux" or platform.machine()!="aarch64": raise RuntimeError("Spark only")
p=argparse.ArgumentParser(); p.add_argument("--runtime",type=Path,required=True); args=p.parse_args()
root=Path(__file__).resolve().parents[1]
site=args.runtime.resolve()/"lib/python3.12/site-packages"
if not (site/"vllm-0.29.0.dist-info").is_dir(): raise RuntimeError("Not the pinned public runtime")
target=root/".venv-vllm/lib/python3.12/site-packages/zz-vllm-runtime-first.pth"
with target.open("x") as stream:
    stream.write('import sys; sys.path.insert(0,'+repr(str(site))+')\n')
from importlib.metadata import distribution
print("Pinned priority file created; verify in a fresh interpreter")
