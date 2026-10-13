"""Expose this project's already installed public test dependencies offline."""
from pathlib import Path
import platform
if platform.system() != "Linux" or platform.machine() != "aarch64":
    raise RuntimeError("Offline environment preparation runs on Spark")
root=Path(__file__).resolve().parents[1]
target=root/".venv-vllm/lib/python3.12/site-packages/project-test-runtime.pth"
if target.exists(): raise FileExistsError(target)
target.write_text('import site; site.addsitedir('+repr(str(root/".venv/lib/python3.12/site-packages"))+')\n')
