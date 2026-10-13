#!/usr/bin/env bash
set -euo pipefail
# Reuse only the preinstalled public-library runtime; no private project code.
task_root=$(cd "$(dirname "$0")/.." && pwd)
source_runtime=${1:?Pass the already provisioned public vLLM runtime directory}
cd "$task_root"
test -x "$source_runtime/bin/python"
test ! -e .venv-vllm
.venv/bin/python -m venv --without-pip .venv-vllm
task_site="$task_root/.venv-vllm/lib/python3.12/site-packages"
printf 'import site; site.addsitedir("%s/lib/python3.12/site-packages")\n' "$source_runtime" > "$task_site/public-runtime.pth"
.venv-vllm/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv-vllm/bin/python -m pip show vllm torch transformers
