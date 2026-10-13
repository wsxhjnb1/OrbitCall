"""Spark-only development discovery; no confirmatory speed claim from this run."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from benchmark_scoring import (freeze_fixtures, invoke, measure, payload_hash,
                               sha256, write_new, append_observation)
from audit_full_triton import verify_model
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
from orbitcall.full_acceleration import PackedFullScorer
from orbitcall.scoring import SequenceScorer

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b):
    if any(a[key] != b[key] for key in ("id", "sequence_ids", "prompt_ids", "families")):
        raise ValueError("Changed paths or family")
    left, right = a["scores"], b["scores"]
    errors = [[abs(x - y) for x, y in zip(p, q, strict=True)]
              for p, q in zip(left["token_logprobs"], right["token_logprobs"], strict=True)]
    winners = {k: a["predictions"][k]["semantic_key"] == b["predictions"][k]["semantic_key"]
               for k in a["predictions"]}
    maximum = max(x for path in errors for x in path)
    return {"max_token_error": maximum, "winner_agreement": winners,
            "strict_gate": maximum <= 3e-5 and all(winners.values()),
            "token_errors": errors}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Discovery runs only on Spark")
    if args.run.exists():
        raise FileExistsError(args.run)
    require_idle_gpu()
    args.run.mkdir(parents=True)
    start = time.monotonic()
    paths = [Path(__file__), ROOT / "AGENTS.md"] + sorted((ROOT / "src/orbitcall").glob("*.py"))
    for path in paths:
        target = args.run / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        model_receipt, _ = verify_model(args.model, lambda path: json.loads(path.read_text()))
        freeze = freeze_fixtures(args.candidates, ROOT / "data/frozen_inputs.json", args.model.name)
        write_new(args.run / "plan.json", {"stage": "discovery only", "answers_accessed": False,
                  "model": model_receipt, "fixture_freeze": freeze,
                  "source_sha256": {str(x.relative_to(ROOT)): sha256(x) for x in paths},
                  "token_tolerance": 3e-5, "max_seconds": 1800,
                  "arms": [f"{kind}_{attention}{chunk}" for kind in ("native", "packed")
                           for attention in ("eager", "sdpa") for chunk in (32, 128, 256)],
                  "repeats": 2})
        scorer = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=32)
        # Shared weights/tokenizer; class switch changes only the explicitly
        # declared dispatch and never duplicates a checkpoint in memory.
        references = {}
        for fixture in freeze["fixtures"]:
            references[fixture["id"]] = invoke(scorer, fixture, "full", "torch_ce")
        write_new(args.run / "references.json", references)
        arms = [(f"{kind}_{attention}{chunk}", attention, chunk, kind == "packed")
                for kind in ("native", "packed") for attention in ("eager", "sdpa")
                for chunk in (32, 128, 256)]
        observations = []
        for name, attention, chunk, packed in arms:
            scorer.__class__ = PackedFullScorer if packed else SequenceScorer
            scorer.omit_padding_mask = True
            scorer.head_chunk = chunk
            scorer.model.set_attn_implementation(attention)
            require_owned_gpu()
            # Capture CPU/GPU operator names to establish actual SDPA dispatch.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                gate = [invoke(scorer, fixture, "full", "triton" if packed else "torch_ce")
                        for fixture in freeze["fixtures"]]
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.run / (name + ".trace.json")))
            operators = [{"key": x.key, "calls": x.count,
                          "cpu_us": x.cpu_time_total, "device_us": x.device_time_total}
                         for x in prof.key_averages()]
            checks = {row["id"]: compare(references[row["id"]], row) for row in gate}
            write_new(args.run / (name + "_gate.json"), {"results": gate, "checks": checks,
                                                        "operators": operators})
            if not all(x["strict_gate"] for x in checks.values()):
                print(json.dumps({"arm": name, "valid": False,
                                  "max_error": max(x["max_token_error"] for x in checks.values())}), flush=True)
                continue
            for repeat in range(2):
                if time.monotonic() - start > 1800:
                    raise RuntimeError("Discovery budget exhausted")
                timing, outputs = measure(lambda: [invoke(scorer, fixture, "full",
                        "triton" if packed else "torch_ce") for fixture in freeze["fixtures"]], 1)
                row = {"arm": name, "repeat": repeat, "timing": timing,
                       "checks": {out["id"]: compare(references[out["id"]], out) for out in outputs}}
                append_observation(args.run / "observations.jsonl", row)
                observations.append(row)
                print(json.dumps({"arm": name, "seconds": timing["wall_seconds"]}), flush=True)
        write_new(args.run / "complete.json", {"elapsed_seconds": time.monotonic() - start,
                                                "observations": observations,
                                                "claim": "development discovery only"})
    except Exception as exc:
        write_new(args.run / "failure.json", {"exception": str(exc), "traceback": traceback.format_exc(),
                                               "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
