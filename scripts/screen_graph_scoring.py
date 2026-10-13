"""Spark-only exact CUDAgraph development screen with a matched native control."""

import argparse
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

from audit_full_triton import verify_model
from benchmark_scoring import freeze_fixtures, invoke, measure, sha256, write_new, append_observation
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
from orbitcall.graph_scoring import GraphFullScorer
from orbitcall.ranking import compare_scoring_results
from orbitcall.scoring import SequenceScorer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"kind": "graph", "warmup_iterations": 3, "max_pool_bytes": 24 * 1024**3,
          "audit_max_graphs": 2, "max_graphs": 16}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("All numerical work runs on Spark")
    if args.run.exists():
        raise FileExistsError(args.run)
    require_idle_gpu()
    args.run.mkdir(parents=True)
    started = time.monotonic()
    current = None
    try:
        paths = [Path(__file__), ROOT / "docs/GRAPH_SCORING_PROTOCOL.md",
                 ROOT / "metadata/acceleration_input_freeze_v1.json",
                 ROOT / "scripts/audit_full_triton.py", ROOT / "scripts/benchmark_scoring.py"]
        paths += sorted((ROOT / "src/orbitcall").glob("*.py"))
        for path in paths:
            target = args.run / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        checkpoint, _ = verify_model(args.model, lambda p: json.loads(p.read_text()))
        freeze = freeze_fixtures(args.candidates, ROOT / "data/frozen_inputs.json", args.model.name)
        write_new(args.run / "plan.json", {"configs": [CONFIG], "answers_accessed": False,
                  "model": checkpoint, "fixture_freeze": freeze,
                  "source_sha256": {str(x.relative_to(ROOT)): sha256(x) for x in paths},
                  "max_seconds": 1200, "token_tolerance": 3e-5,
                  "normalizer": "torch_ce", "head_chunk": 32, "attention": "eager"})
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        original = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=32)
        native = GraphFullScorer.from_scorer(original, max_graphs=8,
                    max_pool_bytes=CONFIG["max_pool_bytes"], graphs_enabled=False)
        graph = GraphFullScorer.from_scorer(original, max_graphs=8,
                    max_pool_bytes=CONFIG["max_pool_bytes"], graphs_enabled=True)
        references = {x["id"]: invoke(original, x, "full", "torch_ce") for x in freeze["fixtures"]}
        write_new(args.run / "references.json", references)
        gates = {}
        for name, scorer in (("native", native), ("graph", graph)):
            current = name
            gate_start = time.monotonic()
            results = [invoke(scorer, x, "full", "torch_ce") for x in freeze["fixtures"]]
            checks = {x["id"]: compare_scoring_results(references[x["id"]], x, tolerance=3e-5) for x in results}
            gates[name] = all(x["valid"] for x in checks.values())
            write_new(args.run / (name + "_gate.json"), {"results": results, "checks": checks,
                      "elapsed_seconds": time.monotonic() - gate_start, "stats": scorer.graph_stats})
            print(json.dumps({"arm": name, "valid": gates[name],
                              "max_error": max(x["maximum_token_error"] for x in checks.values())}), flush=True)
        if not all(gates.values()):
            raise RuntimeError("Fixed graph development gate failed")
        observations = []
        for repeat in range(3):
            order = (("native", native), ("graph", graph)) if repeat % 2 == 0 else (("graph", graph), ("native", native))
            for name, scorer in order:
                if time.monotonic() - started > 1200:
                    raise RuntimeError("Graph discovery budget exhausted")
                require_owned_gpu()
                before = scorer.graph_stats
                timing, outputs = measure(lambda: [invoke(scorer, x, "full", "torch_ce") for x in freeze["fixtures"]], 1)
                after = scorer.graph_stats
                checks = {x["id"]: compare_scoring_results(references[x["id"]], x, tolerance=3e-5) for x in outputs}
                if (not all(x["valid"] for x in checks.values())
                        or after["captures_attempted"] != before["captures_attempted"]
                        or after["evictions"] != before["evictions"]
                        or after["fallback_batches"] != before["fallback_batches"]):
                    raise RuntimeError("Warm discovery window invalid or cache not retained")
                observation = {"arm": name, "repeat": repeat, "timing": timing,
                               "outputs": outputs, "checks": checks, "stats_before": before, "stats_after": after}
                append_observation(args.run / "observations.jsonl", observation)
                observations.append(observation)
                print(json.dumps({"arm": name, "seconds": timing["wall_seconds"]}), flush=True)
        write_new(args.run / "selection.json", {"valid": True, "configuration": CONFIG,
                  "answers_accessed": False, "run_id": args.run.name,
                  "plan_sha256": sha256(args.run / "plan.json"),
                  "rule": "single predeclared exact-shape graph configuration, no tuning"})
        write_new(args.run / "complete.json", {"valid": True, "elapsed_seconds": time.monotonic() - started,
                  "stats_native": native.graph_stats, "stats_graph": graph.graph_stats,
                  "scope": "development screen only; no accepted held-out gain"})
    except Exception as exc:
        write_new(args.run / "failure.json", {"exception": str(exc), "traceback": traceback.format_exc(),
                  "current": current, "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
