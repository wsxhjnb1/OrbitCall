"""Fixed Tensor Core discovery grid on the original 3B development inputs."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from audit_full_triton import verify_model
from benchmark_scoring import freeze_fixtures, invoke, measure, sha256, write_new, append_observation
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
from orbitcall.ranking import compare_scoring_results
from orbitcall.scoring import SequenceScorer
from orbitcall.linear_acceleration import accelerated_linears

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [{"kind": "linear", "precision": precision, "scope": scope}
           for precision in ("tf32x3", "tf32x6") for scope in ("head", "backbone", "all")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Tensor Core experiments run only on Spark")
    if args.run.exists():
        raise FileExistsError(args.run)
    require_idle_gpu()
    args.run.mkdir(parents=True)
    started = time.monotonic()
    current = None
    paths = [Path(__file__), ROOT / "docs/SCORER_ACCELERATION_PROTOCOL.md",
             ROOT / "docs/TENSORLINEAR_PROTOCOL.md", ROOT / "metadata/acceleration_input_freeze_v1.json"]
    paths += sorted((ROOT / "src/orbitcall").glob("*.py"))
    try:
        for path in paths:
            target = args.run / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        model, _ = verify_model(args.model, lambda path: json.loads(path.read_text()))
        fixture = freeze_fixtures(args.candidates, ROOT / "data/frozen_inputs.json", args.model.name)
        write_new(args.run / "plan.json", {"stage": "separate tensorlinear development discovery",
                  "answers_accessed": False, "model": model, "fixture_freeze": fixture,
                  "source_sha256": {str(x.relative_to(ROOT)): sha256(x) for x in paths},
                  "configs": CONFIGS, "head_chunk": 32, "attention": "eager", "token_tolerance": 3e-5,
                  "max_seconds": 1800, "repeats": 3, "native_precision": "IEEE FP32, TF32 off",
                  "timing_rule": "no timing accepted for invalid arm; all scope/precision configs fixed before launch"})
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
            name = config["precision"] + "_" + config["scope"]
            current = name
            require_owned_gpu()
            if time.monotonic() - started > 1800:
                raise RuntimeError("Discovery budget exhausted")
            with accelerated_linears(scorer.model, scope=config["scope"], precision=config["precision"]):
                results = [invoke(scorer, x, "full", "torch_ce") for x in fixture["fixtures"]]
                checks = {x["id"]: compare_scoring_results(references[x["id"]], x, tolerance=3e-5)
                          for x in results}
                valid = all(x["valid"] for x in checks.values())
                write_new(args.run / (name + "_gate.json"), {"config": config, "valid": valid,
                          "results": results, "checks": checks})
                print(json.dumps({"config": config, "valid": valid,
                           "max_error": max(x["maximum_token_error"] for x in checks.values())}), flush=True)
                if not valid:
                    continue
                times = []
                for n in range(3):
                    timing, result = measure(lambda: [invoke(scorer, x, "full", "torch_ce")
                                                       for x in fixture["fixtures"]], 1)
                    check = {x["id"]: compare_scoring_results(references[x["id"]], x, tolerance=3e-5)
                             for x in result}
                    if not all(x["valid"] for x in check.values()):
                        raise RuntimeError("Repeated discovery gate changed")
                    append_observation(args.run / "observations.jsonl", {"config": config, "repeat": n,
                                         "timing": timing, "outputs": result, "checks": check})
                    times.append(timing["wall_seconds"])
                accepted.append({"configuration": config, "median_seconds": statistics.median(times)})
        accepted.sort(key=lambda x: (x["median_seconds"], json.dumps(x["configuration"], sort_keys=True)))
        if accepted:
            selection = {"valid": True, "answers_accessed": False, **accepted[0],
                         "run_id": args.run.name, "plan_sha256": sha256(args.run / "plan.json"),
                         "eligible": accepted, "rule": "fastest valid fixed-development arm; frozen before whole119/test timing"}
            write_new(args.run / "selection.json", selection)
        write_new(args.run / "complete.json", {"valid_arm_count": len(accepted), "eligible": accepted,
                  "elapsed_seconds": time.monotonic() - started, "scope": "development discovery, no held-out claim"})
    except Exception as exc:
        write_new(args.run / "failure.json", {"exception": str(exc), "traceback": traceback.format_exc(),
                                               "current": current, "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
