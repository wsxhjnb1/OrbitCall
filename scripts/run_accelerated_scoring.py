"""Fresh Spark-only full-population audit followed by held-out paired timing.

Configuration selection is a separate immutable development run. This driver
never changes a failed configuration, reads labels or regenerates candidates.
"""

import argparse
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from audit_full_triton import (verify_model, load_reference, validate_item,
                                validate_vectors, INPUT_SHA256)
from benchmark_scoring import (invoke, measure, sha256, write_new, append_observation)
from orbitcall.bfcl import load_frozen_inputs, render_messages
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
from orbitcall.ranking import (SCORE_METHODS, FIXED_METHODS, aggregate_scores,
                               candidate_pool, validate_sample_plan, compare_scoring_results)
from orbitcall.scoring import SequenceScorer
from orbitcall.serialization import semantic_key, serialization_orbit, validate_call

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 3e-5


def choose_timing_fixtures(candidates, rows):
    """Answer-free first-eight rule; full dispositions retained."""
    by_id = {row["id"]: row for row in rows}
    if len(candidates["records"]) != 119 or {x["id"] for x in candidates["records"]} != set(by_id):
        raise ValueError("Candidate population mismatch")
    chosen, dispositions = [], []
    for record in sorted(candidates["records"], key=lambda x: x["id"]):
        validate_sample_plan(record)
        source = by_id[record["id"]]
        if (record["split"] != source["split"] or record["function"] != source["function"][0]
                or tuple(record["function"]["parameters"]["properties"])
                != tuple(source["function"][0]["parameters"]["properties"])):
            raise ValueError("Candidate schema or split differs from input freeze")
        if source["split"] != "test":
            continue
        n = len(candidate_pool(record))
        dispositions.append({"id": record["id"], "primary_candidate_count": n, "eligible": n >= 2})
        if n >= 2 and len(chosen) < 8:
            chosen.append({"id": record["id"], "input_record": source, "candidate_record": record})
    if not chosen:
        raise ValueError("No eligible test timing fixtures")
    return chosen, dispositions


def checked_reference(ranking, model):
    manifest = json.loads((ranking / "ranking_manifest.json").read_text())
    archived = {name: sha256(ranking / "source" / name)
                for name in manifest["source_sha256"] if name.startswith("src/orbitcall/")}
    # The retained numerical implementation is immutable. Verify it, rather
    # than falsely claiming the new modules existed in the earlier freeze.
    for name, digest in archived.items():
        if digest != manifest["source_sha256"][name]:
            raise ValueError("Historical source snapshot mismatch")
        before, now = (ranking / "source" / name).read_bytes(), (ROOT / name).read_bytes()
        if name == "src/orbitcall/scoring.py":
            if before.count(b'method="tree", reducer="torch", batch_size=8') != 1:
                raise ValueError("Unexpected historical default correction")
            before = before.replace(b'method="tree", reducer="torch", batch_size=8',
                                    b'method="full", reducer="torch", batch_size=8')
        if before != now:
            raise ValueError("Historical numerical implementation changed")
    return load_reference(ranking, model, archived, lambda p: json.loads(p.read_text()), load_frozen_inputs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--ranking", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("All experiments run on Spark")
    if args.run.exists():
        raise FileExistsError(args.run)
    require_idle_gpu()
    args.run.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + 3600
    current = None
    try:
        selection = json.loads(args.selection.read_text())
        if selection["valid"] is not True or selection.get("answers_accessed") is not False:
            raise ValueError("No valid frozen configuration")
        config = selection["configuration"]
        selection_plan = json.loads((args.selection.parent / "plan.json").read_text())
        if (selection["plan_sha256"] != sha256(args.selection.parent / "plan.json")
                or config not in selection_plan["configs"]
                or selection_plan["model"]["repo_id"].split("/")[-1] != "Qwen2.5-3B-Instruct"
                or selection_plan["fixture_freeze"]["inputs_sha256"] != INPUT_SHA256):
            raise ValueError("Invalid development configuration freeze")
        for name, digest in selection_plan["source_sha256"].items():
            if (name.startswith("src/orbitcall/") or name == "metadata/acceleration_input_freeze_v1.json") and sha256(ROOT / name) != digest:
                raise ValueError("Selected numerical implementation changed after freeze")
        inventory = [Path(__file__), ROOT / "docs/SCORER_ACCELERATION_PROTOCOL.md",
                     ROOT / "docs/GRAPH_SCORING_PROTOCOL.md", ROOT / "docs/TENSORLINEAR_PROTOCOL.md"]
        inventory += sorted((ROOT / "src/orbitcall").glob("*.py"))
        inventory += [ROOT / "scripts" / name for name in ("audit_full_triton.py", "benchmark_scoring.py")]
        for path in inventory:
            target = args.run / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for source, relative in ((args.selection, "selection.json"),
                                 (args.selection.parent / "plan.json", "selection_plan.json"),
                                 (ROOT / "metadata/acceleration_input_freeze_v1.json", "input_freeze.json")):
            shutil.copyfile(source, args.run / relative)
        original, rows, items, hashes = checked_reference(args.ranking, args.model.name)
        candidates = json.loads(args.candidates.read_text())
        if (sha256(args.candidates) != original["candidates_sha256"]
                or candidates.get("answers_accessed") is not False):
            raise ValueError("Candidate hash or answer isolation changed")
        fixtures, dispositions = choose_timing_fixtures(candidates, rows)
        input_freeze = json.loads((ROOT / "metadata/acceleration_input_freeze_v1.json").read_text())
        frozen_model = [x for x in input_freeze["records"] if x["model"] == args.model.name]
        if (len(frozen_model) != 1 or input_freeze.get("answers_accessed") is not False
                or frozen_model[0]["candidates_sha256"] != sha256(args.candidates)
                or frozen_model[0]["timing_fixture_ids"] != [x["id"] for x in fixtures]
                or frozen_model[0]["test_dispositions"] != dispositions):
            raise ValueError("Timing fixture freeze changed")
        checkpoint, model_stats = verify_model(args.model, lambda p: json.loads(p.read_text()))
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                     "--format=csv,noheader"], text=True, timeout=15).strip()
        if "GB10" not in gpu or "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4" not in gpu:
            raise RuntimeError("Incorrect study GPU")
        rng = random.Random(271828)
        plan = []
        comparators = ["native", "legacy"] if config["kind"] == "graph" else ["native"]
        for control in comparators:
            for block in range(6):
                order = [control, "optimized", "optimized", control]
                if rng.randrange(2):
                    order = ["optimized", control, control, "optimized"]
                plan.append({"pair": control + "_vs_optimized", "control": control,
                             "block": block, "order": order})
        rng.shuffle(plan)
        manifest = {"stage": "whole119 audit and fresh held-out complete-scoring timing",
                    "answers_accessed": False, "model": checkpoint, "pid": os.getpid(),
                    "gpu": gpu, "torch": torch.__version__, "dtype": "float32",
                    "attention": "eager", "head_chunk": 32, "normalizer": "torch_ce",
                    "selection_sha256": sha256(args.selection), "selection": selection,
                    "record_ids": [x["id"] for x in rows], "reference_sha256": hashes,
                    "fixture_ids": [x["id"] for x in fixtures], "test_dispositions": dispositions,
                    "candidates_sha256": sha256(args.candidates), "inputs_sha256": INPUT_SHA256,
                    "source_sha256": {str(x.relative_to(ROOT)): sha256(x) for x in inventory},
                    "token_tolerance": TOLERANCE, "audit_batch_size": 8,
                    "timing_batch_size": "whole-family", "timing_plan": plan,
                    "max_seconds": 3600, "started_epoch": time.time()}
        write_new(args.run / "manifest.json", manifest)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        native = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=32)
        if config["kind"] == "linear":
            from orbitcall.linear_acceleration import accelerated_linears
            def context():
                return accelerated_linears(native.model, scope=config["scope"], precision=config["precision"])
            optimized = native
        elif config["kind"] == "graph":
            from orbitcall.graph_scoring import GraphFullScorer
            original = native
            native = GraphFullScorer.from_scorer(original, max_graphs=config["audit_max_graphs"],
                     max_pool_bytes=config["max_pool_bytes"], graphs_enabled=False,
                     warmup_iterations=config["warmup_iterations"])
            optimized = GraphFullScorer.from_scorer(original, max_graphs=config["audit_max_graphs"],
                     max_pool_bytes=config["max_pool_bytes"], graphs_enabled=True,
                     warmup_iterations=config["warmup_iterations"])
            from contextlib import nullcontext
            context = nullcontext
        else:
            raise ValueError("Unknown frozen mechanism")
        methods = (*SCORE_METHODS, *FIXED_METHODS)
        valid, scored, zeros, max_error = True, 0, 0, 0.0
        for index, (source, item) in enumerate(zip(rows, items, strict=True)):
            current = {"stage": "audit", "index": index, "id": item["id"]}
            if time.monotonic() > deadline:
                raise RuntimeError("Stage budget exhausted")
            require_owned_gpu()
            prompt, sequences, families = validate_item(item, source, native, aggregate_scores,
                semantic_key, serialization_orbit, validate_call, methods)
            if native.prompt_ids(render_messages(source)) != prompt:
                raise ValueError("Prompt changed")
            if not sequences:
                zeros += 1
                observation = {"id": item["id"], "zero_family": True, "valid": True,
                               "predictions": item["scoring"]["full"]["predictions"]}
            else:
                with context():
                    scores = asdict(optimized.score(prompt, sequences, "full", reducer="torch_ce", batch_size=8))
                validate_vectors(scores, prompt, sequences)
                predictions = aggregate_scores(families, scores["sequence_logprobs"], list(map(len, sequences)))
                alternative = {"scores": scores, "predictions": predictions}
                comparison = compare_scoring_results(item["scoring"]["full"], alternative, tolerance=TOLERANCE)
                max_error = max(max_error, comparison["maximum_token_error"])
                valid &= comparison["valid"]
                scored += 1
                observation = {"id": item["id"], "zero_family": False, "valid": comparison["valid"],
                               "source_item_sha256": hashes[f"items/{index:04d}.json"],
                               "prompt_token_ids": prompt, "sequence_token_ids": sequences,
                               "alternative": alternative, "comparison": comparison}
            write_new(args.run / "audit" / f"{index:04d}.json", observation)
            if index % 10 == 0 or not observation["valid"]:
                print(json.dumps({**current, "valid": observation["valid"], "max_error": max_error}), flush=True)
        write_new(args.run / "audit_summary.json", {"valid": bool(valid), "scored": scored,
                  "zero_family": zeros, "population": len(rows), "max_token_error": max_error,
                  "elapsed_seconds": time.monotonic() - started,
                  "graph_stats": getattr(optimized, "graph_stats", None)})
        if not valid:
            raise RuntimeError("Whole-population numerical gate failed; no timing claim")
        # Fresh reference at whole-family shapes. Graph setup/capture occurs
        # here; retain its actual cost and report warm reuse separately.
        if config["kind"] == "graph":
            optimized.clear_graph_cache()
            native.clear_graph_cache()
            optimized.max_graphs = native.max_graphs = config["max_graphs"]
        gate_started, fresh = time.monotonic(), {}
        for fixture in fixtures:
            native_result = invoke(native, fixture, "full", "torch_ce")
            legacy = invoke(original if config["kind"] == "graph" else native,
                            fixture, "full", "torch_ce")
            with context():
                fast = invoke(optimized, fixture, "full", "torch_ce")
            if any(native_result[k] != fast[k] for k in ("id", "prompt_ids", "sequence_ids", "families")):
                raise ValueError("Timing gate changed token paths or families")
            comparison = compare_scoring_results(native_result, fast, tolerance=TOLERANCE)
            legacy_comparison = compare_scoring_results(legacy, fast, tolerance=TOLERANCE)
            write_new(args.run / "timing_gates" / (fixture["id"] + ".json"),
                      {"native": native_result, "legacy": legacy, "optimized": fast,
                       "comparison": comparison, "legacy_comparison": legacy_comparison})
            if not comparison["valid"] or not legacy_comparison["valid"]:
                raise RuntimeError("Whole-family numerical gate failed")
            fresh[fixture["id"]] = native_result
        write_new(args.run / "timing_setup.json", {"elapsed_seconds": time.monotonic() - gate_started,
                  "graph_stats": getattr(optimized, "graph_stats", None), "includes_cold_capture": True})
        observations = []
        for block in plan:
            for position, arm in enumerate(block["order"]):
                current = {"stage": "timing", "pair": block["pair"], "block": block["block"],
                           "position": position, "arm": arm}
                if time.monotonic() > deadline:
                    raise RuntimeError("Stage budget exhausted")
                def pool():
                    if arm == "native":
                        return [invoke(native, fixture, "full", "torch_ce") for fixture in fixtures]
                    if arm == "legacy":
                        return [invoke(original, fixture, "full", "torch_ce") for fixture in fixtures]
                    with context():
                        return [invoke(optimized, fixture, "full", "torch_ce") for fixture in fixtures]
                stats_before = getattr(optimized, "graph_stats", None)
                timing, outputs = measure(pool, 1)
                stats_after = getattr(optimized, "graph_stats", None)
                append_observation(args.run / "raw_windows.jsonl", {**current, **timing,
                         "outputs": outputs, "graph_stats_before": stats_before,
                         "graph_stats_after": stats_after})
                stats_valid = stats_before is None or not any(stats_before[k] != stats_after[k]
                       for k in ("captures_attempted", "evictions", "fallback_batches"))
                checks = {out["id"]: compare_scoring_results(fresh[out["id"]], out,
                                      tolerance=TOLERANCE) for out in outputs}
                if any(fresh[out["id"]][k] != out[k] for out in outputs
                       for k in ("id", "prompt_ids", "sequence_ids", "families")):
                    raise ValueError("Measured window changed token paths or families")
                observation = {**current, **timing, "outputs": outputs, "checks": checks,
                               "graph_stats_before": stats_before, "graph_stats_after": stats_after,
                               "warm_graph_counters_valid": stats_valid}
                append_observation(args.run / "windows.jsonl", observation)
                if not stats_valid:
                    raise RuntimeError("Timed graph window includes cold setup, eviction or fallback")
                if not all(x["valid"] for x in checks.values()):
                    raise RuntimeError("Measured window numerical gate failed")
                observations.append(observation)
                print(json.dumps({**current, "seconds": timing["wall_seconds"]}), flush=True)
        pairs = []
        for control in comparators:
            blocks = []
            for n in range(6):
                times = {arm: statistics.mean(x["wall_seconds"] for x in observations
                             if x["pair"] == control + "_vs_optimized" and x["block"] == n and x["arm"] == arm)
                         for arm in (control, "optimized")}
                blocks.append({"block": n, "seconds": times,
                               "control_over_optimized": times[control] / times["optimized"]})
            ratios = [x["control_over_optimized"] for x in blocks]
            resampled = sorted(statistics.median(values) for values in itertools.product(ratios, repeat=6))
            pairs.append({"pair": control + "_vs_optimized", "blocks": blocks,
                  "median_ratio": statistics.median(ratios), "range": [min(ratios), max(ratios)],
                  "descriptive_bootstrap_median_95": [resampled[int(.025 * len(resampled))],
                                                       resampled[int(.975 * len(resampled))]]})
        if any(sha256(ROOT / x.relative_to(ROOT)) != manifest["source_sha256"][str(x.relative_to(ROOT))]
               for x in inventory):
            raise RuntimeError("Source changed during immutable run")
        if sha256(args.candidates) != manifest["candidates_sha256"] or any(
                sha256(args.ranking / relative) != digest for relative, digest in hashes.items()):
            raise RuntimeError("Reference/candidate changed during run")
        if any((args.model / relative).stat().st_size != size
                or (args.model / relative).stat().st_mtime_ns != mtime
                for relative, (size, mtime) in model_stats.items()):
            raise RuntimeError("Pinned model files changed during run")
        write_new(args.run / "complete.json", {"valid": True, "pairs": pairs,
                  "fixture_count": len(fixtures), "elapsed_seconds": time.monotonic() - started,
                  "graph_stats": getattr(optimized, "graph_stats", None),
                  "scope": "complete score stage; not candidate generation or serving throughput"})
    except Exception as exc:
        write_new(args.run / "failure.json", {"exception": str(exc), "traceback": traceback.format_exc(),
                  "current": current, "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
