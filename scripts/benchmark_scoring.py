"""Answer-free Spark benchmark of matched complete scorers and real-logit reducers.

Example (run only on the authorized Spark):
  PYTHONPATH=src python scripts/benchmark_scoring.py --model /path/to/model \
      --candidates runs/generation_v1/candidates.json --run runs/benchmark_v1

Scorer wall time includes reparsing primary samples, orbit enumeration, prompt
rendering/tokenization, packing, transfers, model work, extraction, and ranking.
It excludes checkpoint loading, candidate generation, disk I/O, and validation.
The complete-scorer CUDA interval includes host-induced idle gaps; it is not a
measurement of active GPU compute. Reducer CUDA intervals use device-resident
actual model logits and no host reads inside their repeated launch loop.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import time
import traceback
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from orbitcall.bfcl import load_frozen_inputs, render_messages
from orbitcall.gpu_guard import gpu_pids, require_idle_gpu
from orbitcall.kernels import selected_logprobs
from orbitcall.ranking import (
    PRIMARY_SEEDS, aggregate_scores, build_families, candidate_pool,
    validate_sample_plan,
)
from orbitcall.scoring import SequenceScorer


GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
TOKEN_TOLERANCE = 1e-3
REDUCER_TOLERANCE = 3e-5
FIXTURE_COUNT = 4
BLOCKS = 4
SCORER_REPEATS = 1
REDUCER_REPEATS = 100
REDUCER_WARMUP = 10
HEAD_CHUNK = 32
RANDOM_SEED = 20261013
MAX_SECONDS = 600
ROOT = Path(__file__).resolve().parents[1]
SCORER_ARMS = {"full": ("full", "torch"), "cached": ("cached", "torch"),
               "tree": ("tree", "torch"), "tree_lse": ("tree", "torch_lse"),
               "tree_ce": ("tree", "torch_ce"), "tree_triton": ("tree", "triton")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_hash(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    """Never replace an earlier completed artifact or a retained failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def append_observation(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_inventory():
    paths = [Path(__file__), ROOT / "PROTOCOL.md", ROOT / "AGENTS.md"]
    paths.extend(sorted((ROOT / "src" / "orbitcall").glob("*.py")))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in paths}


def owned_snapshot():
    active = gpu_pids()
    unrelated = active - {os.getpid()}
    if unrelated:
        raise RuntimeError(f"Unrelated GPU contexts invalidate this run: {sorted(unrelated)}")
    return {"epoch": time.time(), "compute_pids": sorted(active)}


def check_budget(deadline):
    if time.monotonic() > deadline:
        raise RuntimeError("Declared benchmark runtime budget exhausted; retain partial run")


def freeze_fixtures(candidates_path, inputs_path, model_name):
    """Select by ID and unlabeled parse validity, never by score or timing."""
    inputs = load_frozen_inputs(inputs_path)
    by_id = {row["id"]: row for row in inputs}
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if candidates.get("answers_accessed") is not False:
        raise ValueError("Candidates must declare answer isolation")
    rows = candidates["records"]
    if len(rows) != len(by_id) or {row["id"] for row in rows} != set(by_id):
        raise ValueError("Incomplete/duplicate candidates or changed frozen input population")
    manifest_path = candidates_path.parent / "generation_manifest.json"
    generation = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (generation["model"] != model_name
            or generation["inputs_sha256"] != sha256(inputs_path)
            or generation.get("answers_accessed") is not False
            or generation["direct_seeds"] != list(PRIMARY_SEEDS)):
        raise ValueError("Generation model, input hash, seeds, or isolation mismatch")
    selected, dispositions = [], []
    for row in sorted(rows, key=lambda value: value["id"]):
        source = by_id[row["id"]]
        validate_sample_plan(row)
        if (row["split"] != source["split"] or row["function"] != source["function"][0]
                or tuple(row["function"]["parameters"]["properties"])
                != tuple(source["function"][0]["parameters"]["properties"])):
            raise ValueError("Candidate split/schema/property order differs from frozen input")
        if source["split"] != "development":
            continue
        pool = candidate_pool(row)
        reason = "eligible" if len(pool) >= 2 else "fewer_than_two_valid_primary_calls"
        dispositions.append({"id": row["id"], "distinct_primary_calls": len(pool),
                             "disposition": reason})
        if len(pool) >= 2 and len(selected) < FIXTURE_COUNT:
            selected.append({"id": row["id"], "input_record": source,
                             "candidate_record": row,
                             "parameter_keys": list(source["function"][0]["parameters"]["properties"])})
    if not selected:
        raise ValueError("No eligible development fixture; no test/additional-seed fallback")
    eligible = [row["id"] for row in dispositions if row["disposition"] == "eligible"]
    return {"selection": "first up to four lexicographic development IDs with >=2 distinct valid primary-seed calls; all qualifying IDs if fewer than four",
            "primary_seeds": list(PRIMARY_SEEDS), "answers_accessed": False,
            "candidates_sha256": sha256(candidates_path), "inputs_sha256": sha256(inputs_path),
            "generation_manifest_sha256": sha256(manifest_path),
            "development_dispositions": dispositions, "eligible_ids": eligible,
            "eligible_count": len(eligible), "selected_count": len(selected),
            "fallback": "none", "fixtures": selected}


def prepare(scorer, fixture):
    # build_families reparses the original four primary sample texts, checks
    # semantic identities, and uses input schema order for complete orbits.
    families = build_families(fixture["candidate_record"])
    texts = [text for family in families for text in family["texts"]]
    prompt = scorer.prompt_ids(render_messages(fixture["input_record"]))
    sequences = scorer.sequence_ids(texts)
    return families, prompt, sequences


def invoke(scorer, fixture, method, reducer="torch"):
    families, prompt, sequences = prepare(scorer, fixture)
    # A whole-family batch is the strong prompt-cache control. It avoids an
    # artificial win from the default scorer's smaller sequence batches.
    scores = scorer.score(prompt, sequences, method, reducer=reducer,
                          batch_size=len(sequences))
    predictions = aggregate_scores(families, scores.sequence_logprobs,
                                   list(map(len, sequences)))
    return {"id": fixture["id"], "scores": asdict(scores), "predictions": predictions,
            "prompt_ids": prompt, "sequence_ids": sequences, "families": families}


def ce_selected_adapter(logits, targets, row_ids=None, *, backend="auto"):
    """Full-V native CE with shared parents, without repeating vocabulary rows.

    The scorer supplies valid IDs and at least one real edge for every parent.
    Device-side anchor planning is part of complete-scorer timing. This adapter
    is a benchmark control, not an additional production kernel API.
    """
    if backend != "torch_ce" or row_ids is None:
        return selected_logprobs(logits, targets, row_ids=row_ids, backend=backend)
    edges = torch.arange(len(targets), device=logits.device, dtype=torch.long)
    parents = torch.arange(logits.shape[0], device=logits.device)
    # Integer row/edge planning is deterministic and costs O(N*E), not O(E*V).
    first = torch.where(row_ids[None, :] == parents[:, None],
                        edges[None, :], len(targets)).amin(dim=-1)
    anchors = targets[first]
    anchor_values = selected_logprobs(logits, anchors, backend="torch_ce")
    return anchor_values[row_ids] + (logits[row_ids, targets] - logits[parents, anchors][row_ids])


def validate_result(result):
    sequences = result["sequence_ids"]
    scores = result["scores"]
    tokens, totals = scores["token_logprobs"], scores["sequence_logprobs"]
    if len(tokens) != len(sequences) or len(totals) != len(sequences):
        raise RuntimeError("Missing or additional scored sequences")
    for sequence, values, total in zip(sequences, tokens, totals, strict=True):
        if len(values) != len(sequence) or not values:
            raise RuntimeError("Scored token path has a different length")
        if not all(math.isfinite(value) for value in [*values, total]):
            raise RuntimeError("Non-finite score")
        if abs(total - math.fsum(values)) > 1e-8:
            raise RuntimeError("Sequence score is not the complete token sum")


def compare(reference, alternative):
    validate_result(reference)
    validate_result(alternative)
    for field in ("id", "prompt_ids", "sequence_ids", "families"):
        if reference[field] != alternative[field]:
            raise RuntimeError(f"Comparison changes {field}")
    left, right = reference["scores"], alternative["scores"]
    edge_errors = [[abs(a - b) for a, b in zip(x, y, strict=True)]
                   for x, y in zip(left["token_logprobs"], right["token_logprobs"], strict=True)]
    sequence_errors = [abs(a - b) for a, b in zip(left["sequence_logprobs"],
                                                right["sequence_logprobs"], strict=True)]
    token_error = max(error for path in edge_errors for error in path)
    sequence_error = max(sequence_errors)
    top_left = max(range(len(left["sequence_logprobs"])), key=left["sequence_logprobs"].__getitem__)
    top_right = max(range(len(right["sequence_logprobs"])), key=right["sequence_logprobs"].__getitem__)
    agreements = {method: prediction["semantic_key"] == alternative["predictions"][method]["semantic_key"]
                  for method, prediction in reference["predictions"].items()}
    record = {"max_token_error": token_error, "max_sequence_error": sequence_error,
              "per_sequence_token_errors": edge_errors, "sequence_errors": sequence_errors,
              "same_top_sequence": top_left == top_right, "winner_agreement": agreements,
              "reference_margins": {method: prediction["margin"] for method, prediction in reference["predictions"].items()},
              "alternative_margins": {method: prediction["margin"] for method, prediction in alternative["predictions"].items()},
              "margin_exceeds_twice_measured_sequence_error": {
                  method: prediction["margin"] is not None and prediction["margin"] > 2 * sequence_error
                  for method, prediction in reference["predictions"].items()},
              "valid": token_error < TOKEN_TOLERANCE and top_left == top_right and all(agreements.values())}
    if not record["valid"]:
        raise RuntimeError("Numerical/winner gate failed: " + json.dumps(record, allow_nan=False))
    return record


def measure(function, repeats):
    """Measure allocations and both complete wall and device-timeline brackets."""
    before = owned_snapshot()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started_epoch = time.time()
    started = time.perf_counter()
    start.record()
    output = None
    for _ in range(repeats):
        output = function()
    end.record()
    end.synchronize()
    wall = time.perf_counter() - started
    result = {"started_epoch": started_epoch, "ended_epoch": time.time(),
              "repeats": repeats, "wall_seconds": wall,
              "cuda_interval_ms": start.elapsed_time(end),
              "baseline_allocated_bytes": baseline_allocated,
              "baseline_reserved_bytes": baseline_reserved,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline_allocated,
              "guard_before": before, "guard_after": owned_snapshot()}
    return result, output


def plan_pair(rng, pair_name, a, b):
    for block in range(BLOCKS):
        orientation = rng.randrange(2)
        order = [a, b, b, a] if orientation == 0 else [b, a, a, b]
        yield {"pair": pair_name, "block": block, "order": order}


def pair_summary(observations, pairs):
    summaries = []
    for name, a, b in pairs:
        selected = [row for row in observations if row["pair"] == name]
        block_rows = []
        for block in range(BLOCKS):
            arms = {arm: [row for row in selected if row["block"] == block and row["arm"] == arm]
                    for arm in (a, b)}
            if any(len(values) != 2 for values in arms.values()):
                raise RuntimeError("An ABBA block is incomplete")
            medians = {arm: {metric: statistics.median(row[metric] / row["repeats"] for row in values)
                             for metric in ("wall_seconds", "cuda_interval_ms")}
                       for arm, values in arms.items()}
            block_rows.append({"block": block, "medians_per_call": medians,
                               "wall_speed_ratio_a_over_b": medians[a]["wall_seconds"] / medians[b]["wall_seconds"],
                               "cuda_ratio_a_over_b": medians[a]["cuda_interval_ms"] / medians[b]["cuda_interval_ms"]})
        summaries.append({"pair": name, "arm_a": a, "arm_b": b, "blocks": block_rows,
                          "median_block_wall_ratio_a_over_b": statistics.median(row["wall_speed_ratio_a_over_b"] for row in block_rows),
                          "median_block_cuda_ratio_a_over_b": statistics.median(row["cuda_ratio_a_over_b"] for row in block_rows),
                          "interpretation": "four paired blocks on the fixed up-to-four development fixture pool; descriptive screen, not a population speedup guarantee"})
    return summaries


def scorer_study(scorer, fixtures, run, deadline, rng):
    references, gate_records = {}, []
    for fixture in fixtures:
        check_budget(deadline)
        rows = {arm: invoke(scorer, fixture, method, reducer)
                for arm, (method, reducer) in SCORER_ARMS.items()}
        write_new(run / "gate_raw" / (fixture["id"] + ".json"), rows)
        references[fixture["id"]] = rows["full"]
        gate_records.append({"id": fixture["id"], "results": rows,
                             "comparisons": {method: compare(rows["full"], row) for method, row in rows.items()}})
    write_new(run / "scorer_gate.json", {"token_tolerance": TOKEN_TOLERANCE, "records": gate_records, "passed": True})
    warmup = []
    for arm, (method, reducer) in SCORER_ARMS.items():
        for fixture in fixtures:
            check_budget(deadline)
            row = invoke(scorer, fixture, method, reducer)
            warmup.append({"id": fixture["id"], "arm": arm,
                           "comparison": compare(references[fixture["id"]], row)})
    write_new(run / "scorer_warmup.json", warmup)
    if "full_triton" in SCORER_ARMS:
        pairs = [("full_torch_vs_lse", "full", "full_lse"),
                 ("full_torch_vs_ce", "full", "full_ce"),
                 ("full_torch_vs_triton", "full", "full_triton"),
                 ("full_lse_vs_triton", "full_lse", "full_triton"),
                 ("full_ce_vs_triton", "full_ce", "full_triton")]
    else:
        pairs = [("cached_vs_tree", "cached", "tree"), ("full_vs_cached", "full", "cached"),
                 ("tree_torch_vs_triton", "tree", "tree_triton"),
                 ("tree_lse_vs_triton", "tree_lse", "tree_triton"),
                 ("tree_ce_vs_triton", "tree_ce", "tree_triton")]
    plans = [block for name, a, b in pairs for block in plan_pair(rng, name, a, b)]
    # Randomize block order as well as ABBA orientation before any timed result.
    rng.shuffle(plans)
    write_new(run / "scorer_timing_plan.json", plans)
    observations = []
    for block in plans:
        for position, arm in enumerate(block["order"]):
            check_budget(deadline)
            def complete_pool():
                method, reducer = SCORER_ARMS[arm]
                return [invoke(scorer, fixture, method, reducer) for fixture in fixtures]
            timing, outputs = measure(complete_pool, SCORER_REPEATS)
            append_observation(run / "scorer_raw_windows.jsonl", {
                **timing, "pair": block["pair"], "block": block["block"],
                "position": position, "arm": arm, "outputs": outputs})
            checks = {row["id"]: compare(references[row["id"]], row) for row in outputs}
            row = {**timing, "pair": block["pair"], "block": block["block"],
                   "position": position, "arm": arm, "fixture_count": len(fixtures),
                   "scope": "complete parsing/tokenization/scorer/ranking; one call = full fixed fixture pool",
                   "outputs": outputs, "validity": checks}
            append_observation(run / "scorer_observations.jsonl", row)
            observations.append(row)
            print(json.dumps({"stage": "scorer", "pair": row["pair"], "block": row["block"],
                              "position": position, "arm": arm, "wall_seconds": row["wall_seconds"]}), flush=True)
    result = {"scope": "score stage, not generation plus ranking or serving throughput",
              "cuda_scope": "device timeline bracket including host launch/parsing/extraction gaps; not active GPU compute",
              "pairs": pair_summary(observations, pairs), "valid": True}
    write_new(run / "scorer_summary.json", result)
    return result


def capture_real_logits(scorer, fixture, run, method="tree"):
    captured = {}
    def capture(logits, targets, row_ids=None, *, backend="auto"):
        if not captured:
            captured.update(logits=logits.detach().clone(), targets=targets.detach().clone(),
                            row_ids=None if row_ids is None else row_ids.detach().clone())
        return selected_logprobs(logits, targets, row_ids=row_ids, backend=backend)
    # This hook observes the actual model's first tree projection chunk. It is
    # removed before any measured call and never changes a source file.
    with patch("orbitcall.scoring.selected_logprobs", side_effect=capture):
        result = invoke(scorer, fixture, method)
    validate_result(result)
    if not captured:
        raise RuntimeError("Expected a real parent/edge logit fixture")
    logits, targets, rows = captured["logits"], captured["targets"], captured["row_ids"]
    if rows is None:
        if len(targets) != logits.shape[0]:
            raise RuntimeError("Independent full parent/target count differs")
        rows = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Real captured logits are non-finite")
    row_list, target_list = rows.cpu().tolist(), targets.cpu().tolist()
    anchors = [None] * logits.shape[0]
    for row, target in zip(row_list, target_list, strict=True):
        if anchors[row] is None:
            anchors[row] = target
    if any(target is None for target in anchors):
        raise RuntimeError("Captured parent row has no real target edge")
    anchor_ids = torch.tensor(anchors, device=logits.device, dtype=torch.long)
    tensor_path = run / "real_reducer_fixture.pt"
    if tensor_path.exists():
        raise FileExistsError(tensor_path)
    torch.save({"logits": logits.cpu(), "target_ids": targets.cpu(), "row_ids": rows.cpu(),
                "anchor_ids": anchor_ids.cpu()}, tensor_path)
    metadata = {"origin": f"first actual {method} LM-head chunk of first frozen fixture",
                "backbone_method": method,
                "fixture_id": fixture["id"], "fixture_sha256": payload_hash(fixture),
                "shape": list(logits.shape), "dtype": str(logits.dtype), "stride": list(logits.stride()),
                "requested_edges": len(targets), "row_ids": row_list, "target_ids": target_list,
                "anchor_ids": anchors, "tensor_sha256": sha256(tensor_path),
                "synthetic": False, "scope": "one declared actual projection chunk; not all inference logits"}
    write_new(run / "real_reducer_fixture.json", metadata)
    return logits, targets, rows, anchor_ids


def reducer_study(scorer, fixture, run, deadline, rng, method="tree"):
    check_budget(deadline)
    logits, targets, rows, anchors = capture_real_logits(scorer, fixture, run, method)
    parent_indices = torch.arange(logits.shape[0], device=logits.device)
    def ce_shared():
        # Native CE normalizes each parent once using one actual anchor edge.
        # P(target)/P(anchor) = exp(logit_target-logit_anchor); this adapter
        # handles repeated rows without making an [E,V] repeated-logit tensor.
        if method == "full":
            return selected_logprobs(logits, targets, backend="torch_ce")
        anchor_values = selected_logprobs(logits, anchors, backend="torch_ce")
        return anchor_values[rows] + (logits[rows, targets] - logits[parent_indices, anchors][rows])
    functions = {
        "torch": lambda: selected_logprobs(logits, targets, rows, backend="torch"),
        "torch_lse": lambda: selected_logprobs(logits, targets, rows, backend="torch_lse"),
        "torch_ce_anchor": ce_shared,
        "triton": lambda: selected_logprobs(logits, targets, rows, backend="triton"),
    }
    with torch.inference_mode():
        reference = functions["torch"]().clone()
        def validate(output):
            if output.shape != reference.shape or output.dtype != torch.float32:
                raise RuntimeError("Reducer output shape/dtype mismatch")
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError("Non-finite reducer output")
            error = float((output - reference).abs().max().cpu())
            if error >= REDUCER_TOLERANCE:
                raise RuntimeError(f"Reducer error {error} exceeds {REDUCER_TOLERANCE}")
            return {"max_error": error, "tolerance": REDUCER_TOLERANCE,
                    "values": output.cpu().tolist(), "valid": True}
        gate = {}
        for arm, function in functions.items():
            output = function()
            write_new(run / "reducer_gate_raw" / (arm + ".json"), {
                "shape": list(output.shape), "dtype": str(output.dtype),
                "values": output.cpu().tolist()})
            gate[arm] = validate(output)
        # Also check the direct one-target-per-parent CE API on actual anchors.
        direct_reference = selected_logprobs(logits, anchors, backend="torch")
        direct_ce = selected_logprobs(logits, anchors, backend="torch_ce")
        direct_error = float((direct_reference - direct_ce).abs().max().cpu())
        if not bool(torch.isfinite(direct_ce).all()) or direct_error >= REDUCER_TOLERANCE:
            raise RuntimeError("Direct native cross-entropy control failed")
        write_new(run / "reducer_gate.json", {"arms": gate,
                  "direct_ce_anchor_max_error": direct_error, "passed": True,
                  "ce_adapter": "direct native full-V CE, one target per parent" if method == "full"
                                else "one native full-V CE per parent plus actual target-minus-anchor logit gather"})
        for function in functions.values():
            for _ in range(REDUCER_WARMUP):
                output = function()
            validate(output)
        torch.cuda.synchronize()
        pairs = [("torch_vs_lse", "torch", "torch_lse"),
                 ("torch_vs_ce", "torch", "torch_ce_anchor"),
                 ("torch_vs_triton", "torch", "triton"),
                 ("lse_vs_triton", "torch_lse", "triton"),
                 ("ce_vs_triton", "torch_ce_anchor", "triton")]
        plans = [block for name, a, b in pairs for block in plan_pair(rng, name, a, b)]
        rng.shuffle(plans)
        write_new(run / "reducer_timing_plan.json", plans)
        observations = []
        for block in plans:
            for position, arm in enumerate(block["order"]):
                check_budget(deadline)
                timing, output = measure(functions[arm], REDUCER_REPEATS)
                append_observation(run / "reducer_raw_windows.jsonl", {
                    **timing, "pair": block["pair"], "block": block["block"],
                    "position": position, "arm": arm, "values": output.cpu().tolist()})
                row = {**timing, "pair": block["pair"], "block": block["block"],
                       "position": position, "arm": arm, "validity": validate(output),
                       "scope": "actual resident logits, allocation and launches included; model projection excluded"}
                append_observation(run / "reducer_observations.jsonl", row)
                observations.append(row)
        summary = {"scope": "selected-token reduction on one actual model projection chunk; not backbone or end-to-end gain",
                   "pairs": pair_summary(observations, pairs), "valid": True}
        write_new(run / "reducer_summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data" / "frozen_inputs.json")
    parser.add_argument("--expected-gpu-uuid", default=GPU_UUID,
                        help="Original study GPU by default; declare a different GB10 for a new study")
    parser.add_argument("--normalizer-only", action="store_true",
                        help="Matched full backbone; no cache/tree speed or equality claim")
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS,
                        help="Declare a runtime budget before a new run; never replace a failed run")
    args = parser.parse_args()
    if args.max_seconds <= 0:
        raise ValueError("Runtime budget must be positive")
    if args.normalizer_only:
        SCORER_ARMS.clear()
        SCORER_ARMS.update(full=("full", "torch"), full_lse=("full", "torch_lse"),
                           full_ce=("full", "torch_ce"), full_triton=("full", "triton"))
    if args.run.exists():
        raise FileExistsError(args.run)
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("All numerical work must run on the authorized Spark, never Windows")
    require_idle_gpu()
    gpu_info = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"
    ], text=True, timeout=15).strip().splitlines()
    if len(gpu_info) != 1 or args.expected_gpu_uuid not in gpu_info[0] or "GB10" not in gpu_info[0]:
        raise RuntimeError("GPU identity differs from the authorized Spark")
    if not torch.cuda.is_available():
        raise RuntimeError("Spark CUDA is unavailable")
    candidates_path = args.candidates / "candidates.json" if args.candidates.is_dir() else args.candidates
    args.run.mkdir(parents=True, exist_ok=False)
    started_epoch, deadline = time.time(), time.monotonic() + args.max_seconds
    inventory = source_inventory()
    for relative in inventory:
        destination = args.run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    try:
        frozen = freeze_fixtures(candidates_path, args.inputs, Path(args.model).name)
        write_new(args.run / "fixtures.json", frozen)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(31337)
        torch.use_deterministic_algorithms(True)
        import transformers
        import triton
        manifest = {"started_epoch": started_epoch, "pid": os.getpid(), "hostname": platform.node(),
                    "gpu": gpu_info[0], "model": Path(args.model).name,
                    "expected_gpu_uuid": args.expected_gpu_uuid,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "triton": triton.__version__, "python": platform.python_version(),
                    "dtype": "float32", "attention": "eager", "tf32": False,
                    "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                    "head_chunk": HEAD_CHUNK, "batch_policy": "full and cached: entire actual sequence family",
                    "blocks": BLOCKS, "scorer_repeats_per_window": SCORER_REPEATS,
                    "reducer_repeats_per_window": REDUCER_REPEATS,
                    "reducer_warmup_calls": REDUCER_WARMUP, "random_seed": RANDOM_SEED,
                    "max_seconds": args.max_seconds, "fixture_sha256": payload_hash(frozen),
                    "fixture_file_sha256": sha256(args.run / "fixtures.json"),
                    "fixture_ids": [row["id"] for row in frozen["fixtures"]],
                    "eligible_fixture_count": frozen["eligible_count"], "selected_fixture_count": frozen["selected_count"],
                    "fixture_fallback": "none", "scorer_arms": SCORER_ARMS,
                    "source_sha256": inventory, "answers_accessed": False,
                    "study_kind": "full_reference_normalizer_study" if args.normalizer_only else "packed_scorer_study",
                    "claim_scope": "development scorer and actual-logit reduction only; no generation or serving throughput claim"}
        config_path = Path(args.model) / "config.json"
        manifest["model_config_sha256"] = sha256(config_path)
        for name in ("model_3b_verified.json", "model_1p5b_verified.json", "public_models.json"):
            path = ROOT / "metadata" / name
            if path.exists():
                manifest.setdefault("retained_model_receipts_sha256", {})[name] = sha256(path)
        write_new(args.run / "manifest.json", manifest)
        scorer = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=HEAD_CHUNK)
        if scorer.model.config.model_type != "qwen2" or getattr(scorer.model.config, "use_sliding_window", False):
            raise RuntimeError("Only the declared non-sliding Qwen2 checkpoint architecture is supported")
        if scorer.model.dtype != torch.float32:
            raise RuntimeError("Scoring checkpoint is not FP32")
        owned_snapshot()
        rng = random.Random(RANDOM_SEED)
        with patch("orbitcall.scoring.selected_logprobs", new=ce_selected_adapter):
            scorers = scorer_study(scorer, frozen["fixtures"], args.run, deadline, rng)
        reducers = reducer_study(scorer, frozen["fixtures"][0], args.run, deadline, rng,
                                 "full" if args.normalizer_only else "tree")
        if source_inventory() != inventory or sha256(candidates_path) != frozen["candidates_sha256"]:
            raise RuntimeError("Source or candidate artifact changed during measurement")
        owned_snapshot()
        write_new(args.run / "completed.json", {"started_epoch": started_epoch,
                  "ended_epoch": time.time(), "total_wall_seconds": time.time() - started_epoch,
                  "fixture_sha256": payload_hash(frozen), "scorer": scorers, "reducer": reducers,
                  "source_unchanged": True, "candidates_unchanged": True, "answers_accessed": False,
                  "study_kind": manifest["study_kind"],
                  "kind": manifest["study_kind"],
                  "status": "valid_development_benchmark"})
        print(json.dumps({"status": "complete", "run": args.run.name,
                          "total_wall_seconds": time.time() - started_epoch}), flush=True)
    except Exception as exc:
        write_new(args.run / "failure.json", {"started_epoch": started_epoch, "ended_epoch": time.time(),
                  "status": "invalid_or_incomplete", "error": str(exc), "traceback": traceback.format_exc(),
                  "answers_accessed": False, "partial_artifacts_retained": True})
        raise


if __name__ == "__main__":
    main()
