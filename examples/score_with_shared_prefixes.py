"""Score caller-supplied tool calls on Spark, with explicit approximate opt-in.

Input JSON: {"messages": [...], "calls": [{"name": ..., "arguments": {...}}, ...]}.
The finite family supports two/three flat scalar arguments. This example does
not generate candidates or read task labels. Run from the checkout with
PYTHONPATH=src; --tensor-core explicitly chooses the studied TF32x3 retrofit.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch

from audit_full_triton import GPU_UUID, verify_model
from benchmark_scoring import ce_selected_adapter
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
from orbitcall.linear_acceleration import accelerated_linears
from orbitcall.ranking import logsumexp
from orbitcall.scoring import SequenceScorer
from orbitcall.serialization import semantic_key, serialization_orbit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tensor-core", action="store_true")
    p.add_argument("--method", choices=("full", "cached", "tree"), default="tree")
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Numerical use runs on the authorized Spark")
    require_idle_gpu()
    identity = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                       text=True, timeout=15).strip()
    if identity != GPU_UUID:
        raise RuntimeError("Unexpected experiment GPU")
    receipt, _ = verify_model(args.model, lambda path: json.loads(path.read_text()))
    data = json.loads(args.input.read_text(encoding="utf-8"))
    calls = {semantic_key(call): call for call in data["calls"]}
    if not calls:
        raise ValueError("At least one caller-supplied candidate is required")
    families = {key: serialization_orbit(call) for key, call in sorted(calls.items())}
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    scorer = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=32)
    require_owned_gpu()
    prompt = scorer.prompt_ids(data["messages"])
    sequences = scorer.sequence_ids([text for family in families.values() for text in family])
    context = accelerated_linears(scorer.model, scope="all", precision="tf32x3") if args.tensor_core else nullcontext()
    with context, patch("orbitcall.scoring.selected_logprobs", ce_selected_adapter):
        scores = scorer.score(prompt, sequences, args.method, reducer="torch_ce", batch_size=len(sequences))
    offset, mass = 0, []
    for key, family in families.items():
        mass.append({"semantic_key": key, "call": calls[key], "serialization_count": len(family),
                     "log_mass": logsumexp(scores.sequence_logprobs[offset:offset + len(family)])})
        offset += len(family)
    winner = sorted(mass, key=lambda x: (-x["log_mass"], x["semantic_key"]))[0]
    print(json.dumps({"model": receipt["repo_id"], "method": args.method,
                      "tensor_core": args.tensor_core, "scores": asdict(scores),
                      "candidates": mass, "winner": winner,
                      "scope": "finite family score; new-input fidelity is not certified"},
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
