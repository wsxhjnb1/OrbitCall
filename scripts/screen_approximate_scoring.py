"""Prospectively bounded Tensor Core discovery; never reclassify strict runs."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from audit_full_triton import verify_model
from benchmark_scoring import freeze_fixtures, invoke, measure, sha256, write_new, append_observation
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
from orbitcall.scoring import SequenceScorer
from orbitcall.approximation import LIMITS, TORCH_CONTROL, precision_context, evaluate_accuracy, summarize_accuracy

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [{"kind": "linear", "precision": precision, "scope": scope}
           for precision in ("tf32x3", "tf32x6") for scope in ("head", "backbone", "all")] + [TORCH_CONTROL]


def checks_for(references, results):
    checks = []
    for result in results:
        reference = references[result["id"]]
        if any(reference[k] != result[k] for k in ("id", "prompt_ids", "sequence_ids", "families")):
            raise ValueError("Changed token paths or family")
        checks.append({"id": result["id"], **evaluate_accuracy(reference, result)})
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Experiments run only on Spark")
    if args.run.exists():
        raise FileExistsError(args.run)
    require_idle_gpu()
    args.run.mkdir(parents=True)
    started, current = time.monotonic(), None
    paths = [Path(__file__), ROOT / "docs/APPROXIMATE_SCORING_PROTOCOL.md",
             ROOT / "metadata/acceleration_input_freeze_v1.json"]
    paths += [ROOT / "scripts" / name for name in ("audit_full_triton.py", "benchmark_scoring.py")]
    paths += sorted((ROOT / "src/orbitcall").glob("*.py"))
    try:
        for path in paths:
            target = args.run / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        model, model_stats = verify_model(args.model, lambda path: json.loads(path.read_text()))
        fixture = freeze_fixtures(args.candidates, ROOT / "data/frozen_inputs.json", args.model.name)
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                                      "--format=csv,noheader"], text=True, timeout=15).strip()
        if "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4" not in gpu or "GB10" not in gpu:
            raise ValueError("Incorrect GPU")
        inventory = {str(x.relative_to(ROOT)): sha256(x) for x in paths}
        write_new(args.run / "plan.json", {"stage": "new approximate development discovery",
                  "answers_accessed": False, "model": model, "fixture_freeze": fixture,
                  "gpu": gpu, "torch": torch.__version__, "pid": os.getpid(),
                  "source_sha256": inventory, "configs": CONFIGS, "head_chunk": 32,
                  "attention": "eager", "limits": LIMITS, "max_seconds": 1800, "repeats": 3,
                  "native_precision": "IEEE FP32, TF32 off", "normalizer": "torch_ce",
                  "timing_rule": "fastest valid fixed arm by 3 development warm calls; no held-out claim"})
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        scorer = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=32)
        references = {x["id"]: invoke(scorer, x, "full", "torch_ce") for x in fixture["fixtures"]}
        write_new(args.run / "references.json", references)
        timing, _ = measure(lambda: [invoke(scorer, x, "full", "torch_ce") for x in fixture["fixtures"]], 1)
        write_new(args.run / "native_timing.json", timing)
        accepted = []
        for config in CONFIGS:
            current = config
            require_owned_gpu()
            if time.monotonic() - started > 1800:
                raise RuntimeError("Discovery budget exhausted")
            name = config["kind"] + "_" + config["precision"] + "_" + config["scope"]
            with precision_context(scorer.model, config):
                setup_started = time.monotonic()
                results = [invoke(scorer, x, "full", "torch_ce") for x in fixture["fixtures"]]
                setup_seconds = time.monotonic() - setup_started
                # Preserve source outputs before evaluating gates.
                write_new(args.run / (name + "_outputs.json"), results)
                checks = checks_for(references, results)
                summary = summarize_accuracy(checks, require_all_winners=True)
                write_new(args.run / (name + "_gate.json"), {"config": config, "summary": summary,
                          "checks": checks, "first_call_setup_seconds": setup_seconds})
                print(json.dumps({"config": config, "summary": summary}), flush=True)
                if not summary["valid"]:
                    continue
                times = []
                for n in range(3):
                    timing, result = measure(lambda: [invoke(scorer, x, "full", "torch_ce")
                                                       for x in fixture["fixtures"]], 1)
                    append_observation(args.run / "raw_observations.jsonl", {"config": config,
                                       "repeat": n, "timing": timing, "outputs": result})
                    check = checks_for(references, result)
                    summary = summarize_accuracy(check, require_all_winners=True)
                    append_observation(args.run / "observations.jsonl", {"config": config, "repeat": n,
                                         "timing": timing, "checks": check, "summary": summary})
                    if not summary["valid"]:
                        raise RuntimeError("Repeated discovery gate changed")
                    times.append(timing["wall_seconds"])
                accepted.append({"configuration": config, "median_seconds": statistics.median(times),
                                 "seconds": times, "first_call_setup_seconds": setup_seconds})
        accepted.sort(key=lambda x: (x["median_seconds"], json.dumps(x["configuration"], sort_keys=True)))
        if any(sha256(ROOT / name) != digest for name, digest in inventory.items()):
            raise RuntimeError("Frozen source changed")
        if sha256(args.candidates) != fixture["candidates_sha256"]:
            raise RuntimeError("Candidates changed")
        if any((args.model / name).stat().st_size != size or (args.model / name).stat().st_mtime_ns != mtime
               for name, (size, mtime) in model_stats.items()):
            raise RuntimeError("Pinned model changed")
        if accepted:
            write_new(args.run / "selection.json", {"valid": True, "answers_accessed": False,
                         **accepted[0], "run_id": args.run.name, "plan_sha256": sha256(args.run / "plan.json"),
                         "eligible": accepted, "limits": LIMITS,
                         "rule": "fastest valid fixed development arm; frozen before119/test timing"})
        write_new(args.run / "complete.json", {"valid_arm_count": len(accepted), "eligible": accepted,
                  "elapsed_seconds": time.monotonic() - started, "scope": "development only"})
    except Exception as exc:
        write_new(args.run / "failure.json", {"exception": str(exc), "traceback": traceback.format_exc(),
                                               "current": current, "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
