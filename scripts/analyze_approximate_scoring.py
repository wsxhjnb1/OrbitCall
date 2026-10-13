#!/usr/bin/env python3
"""Independent Spark CPU analysis of frozen approximate-scoring artifacts.

This file never imports Torch, loads a checkpoint, reads gold, or runs inference.
Every numerical summary is reconstructed from retained vectors/windows. Winner
fidelity is not task accuracy. Outputs are exclusive and never overwrite a run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
INPUT_SHA = "1e8edf278caa6066c1b4c1fab07599cc8acb03fdf0037d3447603ff75916060c"
LIMITS = {"max_token_absolute_nats": 0.01, "mean_token_absolute_nats": 0.0005,
          "max_sequence_absolute_nats": 0.1, "minimum_winner_agreement": 0.99}
FIXED = tuple("fixed_" + "".join(map(str, order)) for order in itertools.permutations(range(3)))
METHODS = ("mass", "max_order", "mean_logprob", "length_normalized", "canonical_schema", *FIXED)
MODELS = {"Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct"}
MODEL_REFERENCE = {"Qwen2.5-3B-Instruct": "rank_qwen3b_20261013_reference_v2",
                   "Qwen2.5-1.5B-Instruct": "rank_qwen1p5b_20261013_reference_v2"}
SOURCE_HASHES: dict[str, str] = {}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def loads(value: str):
    return json.loads(value, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read(path: Path):
    value = loads(path.read_text(encoding="utf-8"))
    SOURCE_HASHES[str(path.relative_to(ROOT))] = sha(path)
    return value


def lines(path: Path) -> list[dict]:
    rows = [loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    SOURCE_HASHES[str(path.relative_to(ROOT))] = sha(path)
    return rows


def child(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    need(path.is_relative_to(base.resolve()), "Artifact escaped its declared directory")
    return path


def near(actual: float, expected: float, name: str, tolerance: float = 1e-12) -> None:
    need(type(actual) in (float, int) and math.isfinite(actual)
         and math.isfinite(expected) and abs(actual - expected) <= tolerance,
         "Stored/recomputed numeric mismatch: " + name)


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def describe(values: list[float]) -> dict:
    return {"count": len(values), "mean": math.fsum(values) / len(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "p50": quantile(values, .5), "p90": quantile(values, .9),
            "p95": quantile(values, .95), "p99": quantile(values, .99),
            "quantile_definition": "linear interpolation at (N-1)*p; diagnostic, no threshold selection"}


def validate_vectors(scores: dict, sequences: list[list[int]], prompt: list[int]) -> None:
    need(scores["method"] in {"full", "cached", "tree"} and scores["prompt_tokens"] == len(prompt),
         "Wrong scoring mode or prompt count")
    paths, totals = scores["token_logprobs"], scores["sequence_logprobs"]
    need(len(paths) == len(totals) == len(sequences), "Incomplete token/sequence vector")
    for path, total, tokens in zip(paths, totals, sequences, strict=True):
        need(len(path) == len(tokens) and len(tokens) > 0, "Token path shape mismatch")
        need(tokens[-1] == 151645 and tokens.count(151645) == 1, "Missing/repeated Qwen EOS")
        need(all(type(value) in (float, int) and math.isfinite(value) for value in (*path, total)),
             "Non-finite score vector")
        near(total, math.fsum(path), "EOS-inclusive sequence sum", 1e-5)


def expected_layout(prompt: list[int], sequences: list[list[int]], method: str, batch_size: int) -> dict:
    edges = sum(map(len, sequences))
    if method == "tree":
        children, parents = {-1: {}}, []
        for sequence in sequences:
            parent = -1
            for token in sequence:
                branch = children.setdefault(parent, {})
                if token not in branch:
                    branch[token] = len(parents)
                    parents.append(parent)
                parent = branch[token]
        return {"independent_suffix_tokens": edges, "scored_edges": len(parents),
                "processed_prompt_positions": len(prompt),
                "processed_query_positions": len({parent for parent in parents if parent >= 0}),
                "actual_layout": "one packed ancestor-masked complete-family tree; batch_size argument unused"}
    queries = sum(len(group) * (max(map(len, group)) - 1)
                  for begin in range(0, len(sequences), batch_size)
                  for group in [sequences[begin:begin + batch_size]])
    return {"independent_suffix_tokens": edges, "scored_edges": edges,
            "processed_prompt_positions": len(prompt) * (len(sequences) if method == "full" else 1),
            "processed_query_positions": queries,
            "actual_layout": "independent sequence batches" if method == "full" else "immutable prompt cache and independent suffix batches"}


def validate_layout(scores: dict, sequences: list[list[int]], prompt: list[int], batch_size: int) -> dict:
    validate_vectors(scores, sequences, prompt)
    layout = expected_layout(prompt, sequences, scores["method"], batch_size)
    for field in ("independent_suffix_tokens", "scored_edges", "processed_prompt_positions", "processed_query_positions"):
        need(scores[field] == layout[field], "Scorer changed frozen work/layout: " + field)
    return layout


def independent_scores(result: dict, families: list[dict], sequences: list[list[int]], prompt: list[int]) -> None:
    validate_vectors(result["scores"], sequences, prompt)
    need(set(result["predictions"]) == set(METHODS), "Changed likelihood methods")
    totals = result["scores"]["sequence_logprobs"]
    values = {method: [] for method in METHODS}
    offset = 0
    for family in families:
        lo, hi = family["sequence_begin"], family["sequence_end"]
        need(lo == offset and hi - lo == len(family["texts"]) in (2, 6), "Incomplete finite family")
        paths = totals[lo:hi]
        lengths = list(map(len, sequences[lo:hi]))
        need(len(paths) == hi - lo, "Family scores omitted")
        largest = max(paths)
        values["mass"].append(largest + math.log(math.fsum(math.exp(value - largest) for value in paths)))
        values["max_order"].append(max(paths))
        values["mean_logprob"].append(math.fsum(paths) / len(paths))
        values["length_normalized"].append(math.fsum(value / length for value, length in zip(paths, lengths, strict=True)) / len(paths))
        values["canonical_schema"].append(paths[0])
        for method in FIXED:
            values[method].append(paths[family["fixed_indices"][method]])
        offset = hi
    need(offset == len(sequences), "Finite families do not cover all paths")
    for method, candidates in values.items():
        saved = result["predictions"][method]
        if not families:
            need(saved["call"] is None and saved["semantic_key"] is None and saved["score"] is None
                 and saved["candidate_scores"] == [], "Zero family repaired")
            continue
        need(len(saved["candidate_scores"]) == len(families), "Incomplete candidate scores")
        for family, value, recorded in zip(families, candidates, saved["candidate_scores"], strict=True):
            need(recorded["semantic_key"] == family["semantic_key"], "Candidate score identity changed")
            near(recorded["score"], value, "likelihood aggregate", 1e-10)
        ordered = sorted(range(len(families)), key=lambda index: (-candidates[index], families[index]["semantic_key"]))
        first, second = ordered[0], ordered[1] if len(ordered) > 1 else None
        need(saved["call"] == families[first]["call"] and saved["semantic_key"] == families[first]["semantic_key"],
             "Stored winner differs from independent aggregation")
        near(saved["score"], candidates[first], "winner score", 1e-10)
        if second is None:
            need(saved["margin"] is None and saved["runner_up_key"] is None, "Singleton margin invented")
        else:
            near(saved["margin"], candidates[first] - candidates[second], "winner margin", 1e-10)
            need(saved["runner_up_key"] == families[second]["semantic_key"], "Runner-up changed")


def comparison(reference: dict, alternative: dict, families: list[dict], sequences: list[list[int]],
               prompt: list[int], item_id: str, split: str) -> dict:
    independent_scores(reference, families, sequences, prompt)
    independent_scores(alternative, families, sequences, prompt)
    need(sequences, "Null families have no numerical denominator")
    left, right = reference["scores"], alternative["scores"]
    errors = [[abs(a - b) for a, b in zip(x, y, strict=True)]
              for x, y in zip(left["token_logprobs"], right["token_logprobs"], strict=True)]
    sequence_errors = [abs(a - b) for a, b in zip(left["sequence_logprobs"], right["sequence_logprobs"], strict=True)]
    flat = [value for path in errors for value in path]
    winners = {method: reference["predictions"][method]["semantic_key"] == alternative["predictions"][method]["semantic_key"]
               for method in METHODS}
    maximum_sequence = max(sequence_errors)
    return {"id": item_id, "split": split, "token_count": len(flat),
            "token_errors": errors, "sequence_errors": sequence_errors,
            "sum_absolute_token_error": math.fsum(flat), "mean_absolute_token_error": math.fsum(flat) / len(flat),
            "maximum_token_error": max(flat), "maximum_sequence_error": maximum_sequence,
            "numerical_valid": max(flat) <= LIMITS["max_token_absolute_nats"]
            and maximum_sequence <= LIMITS["max_sequence_absolute_nats"],
            "winner_agreement": winners,
            "changed_methods": [method for method in METHODS if not winners[method]],
            "reference_winner_margins": {method: reference["predictions"][method]["margin"] for method in METHODS},
            "alternative_winner_margins": {method: alternative["predictions"][method]["margin"] for method in METHODS},
            "reference_winner_keys": {method: reference["predictions"][method]["semantic_key"] for method in METHODS},
            "alternative_winner_keys": {method: alternative["predictions"][method]["semantic_key"] for method in METHODS},
            "measured_margin_exceeds_twice_sequence_error": {
                method: reference["predictions"][method]["margin"] is not None
                and reference["predictions"][method]["margin"] > 2 * maximum_sequence for method in METHODS}}


def summarize(checks: list[dict], zero_ids: list[str], population: int, all_winners: bool = False) -> dict:
    if not checks:
        return {"valid": False, "population": population, "scored_records": 0,
                "zero_family_ids": zero_ids, "fidelity_scope": "no numerical observations"}
    n = len(checks)
    tokens = sum(row["token_count"] for row in checks)
    mean = math.fsum(row["sum_absolute_token_error"] for row in checks) / tokens
    agreements = {method: sum(row["winner_agreement"][method] for row in checks) for method in METHODS}
    joint = sum(all(row["winner_agreement"].values()) for row in checks)
    percentage = 100 if all_winners else 99
    valid = (all(row["numerical_valid"] for row in checks) and mean <= LIMITS["mean_token_absolute_nats"]
             and all(100 * count >= percentage * n for count in agreements.values())
             and 100 * joint >= percentage * n)
    return {"valid": valid, "limits": LIMITS.copy(), "population": population,
            "scored_records": n, "zero_family": len(zero_ids), "zero_family_ids": zero_ids,
            "require_all_winners": all_winners, "token_count": tokens, "mean_absolute_token_error": mean,
            "maximum_token_error": max(row["maximum_token_error"] for row in checks),
            "maximum_sequence_error": max(row["maximum_sequence_error"] for row in checks),
            "winner_agreeing_counts": agreements, "winner_agreement": {method: count / n for method, count in agreements.items()},
            "winner_changed_counts": {method: n - count for method, count in agreements.items()},
            "joint_agreeing_count": joint, "joint_all_controls_agreement": joint / n,
            "union_changed_ids": [row["id"] for row in checks if not all(row["winner_agreement"].values())],
            "changed_ids_by_method": {method: [row["id"] for row in checks if not row["winner_agreement"][method]] for method in METHODS},
            "token_absolute_errors": describe([value for row in checks for path in row["token_errors"] for value in path]),
            "sequence_absolute_errors": describe([value for row in checks for value in row["sequence_errors"]]),
            "per_item_token_mean_errors": describe([row["mean_absolute_token_error"] for row in checks]),
            "changed_items": [{key: value for key, value in row.items() if key not in {"token_errors", "sequence_errors"}}
                              for row in checks if not all(row["winner_agreement"].values())],
            "fidelity_scope": "EOS-inclusive stored paths; empty pools excluded only from numerical denominator; not task accuracy"}


def assert_summary(saved: dict, actual: dict) -> None:
    for key in ("valid", "scored_records", "token_count", "require_all_winners", "union_changed_ids", "winner_changed_counts"):
        need(saved[key] == actual[key], "Stored fidelity summary mismatch: " + key)
    need(saved["limits"] == LIMITS, "Prospective limits changed")
    for key in ("mean_absolute_token_error", "maximum_token_error", "maximum_sequence_error", "joint_all_controls_agreement"):
        near(saved[key], actual[key], "fidelity summary " + key)
    need(set(saved["winner_agreement"]) == set(METHODS), "Summary control set changed")
    for method in METHODS:
        near(saved["winner_agreement"][method], actual["winner_agreement"][method], "winner agreement " + method)


def checked_timing(row: dict, pid: int) -> None:
    for name in ("wall_seconds", "cuda_interval_ms"):
        need(type(row[name]) in (int, float) and math.isfinite(row[name]) and row[name] > 0,
             "Non-positive/non-finite timed measurement")
    need(row["repeats"] == 1 and row["ended_epoch"] >= row["started_epoch"], "Changed timing repeat/clock")
    for name in ("guard_before", "guard_after"):
        need(set(row[name]["compute_pids"]) <= {pid}, "Unrelated GPU context invalidates window")
    for name in ("baseline_allocated_bytes", "baseline_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"):
        need(type(row[name]) is int and row[name] >= 0, "Invalid memory measurement")
    need(row["peak_allocated_bytes"] >= row["baseline_allocated_bytes"]
         and row["peak_reserved_bytes"] >= row["baseline_reserved_bytes"], "Memory high-watermark malformed")


def paired_blocks(windows: list[dict], a: str, b: str) -> dict:
    records = []
    for block in range(6):
        rows = [row for row in windows if row["block"] == block]
        need(len(rows) == 4 and sorted(row["position"] for row in rows) == list(range(4)), "Incomplete four-window block")
        ordered = sorted(rows, key=lambda row: row["position"])
        order = [row["arm"] for row in ordered]
        need(order in ([a, b, b, a], [b, a, a, b]), "Not an ABBA/BAAB block")
        times = {arm: statistics.mean(row["wall_seconds"] for row in rows if row["arm"] == arm) for arm in (a, b)}
        records.append({"block": block, "order": order, "seconds": times, "ratio_a_over_b": times[a] / times[b]})
    ratios = [row["ratio_a_over_b"] for row in records]
    states = sorted(statistics.median(values) for values in itertools.product(ratios, repeat=6))
    return {"a": a, "b": b, "block_count": 6, "window_count": len(windows), "blocks": records,
            "median_ratio_a_over_b": statistics.median(ratios), "range": [min(ratios), max(ratios)],
            "producer_empirical_bootstrap_median_95": [states[int(.025 * len(states))], states[int(.975 * len(states))]],
            "interpolated_empirical_bootstrap_median_95": [quantile(states, .025), quantile(states, .975)],
            "bootstrap_resample_states": len(states), "bootstrap_scope": "six paired blocks, not independent resample trials"}


def method_for(config: dict | None) -> str:
    return config["method"] if config and config["kind"] == "pipeline" else "full"


def expected_controls(config: dict) -> tuple[dict, dict, dict, dict, list[str]]:
    """Reconstruct required comparators instead of trusting a declared subset."""
    if config["kind"] == "pipeline":
        need(set(config) == {"kind", "method", "linear"} and config["method"] in {"cached", "tree"}
             and config["linear"] in (None, {"kind": "linear", "precision": "tf32x3", "scope": "all"}),
             "Selected prefix configuration is outside the frozen four-arm grid")
        controls = {"cached": {"kind": "pipeline", "method": "cached", "linear": None},
                    "structure": {"kind": "pipeline", "method": config["method"], "linear": None}}
    else:
        need(config in [{"kind": "linear", "precision": precision, "scope": scope}
                        for precision in ("tf32x3", "tf32x6") for scope in ("head", "backbone", "all")]
             or config == {"kind": "torch_tf32", "precision": "high", "scope": "all"},
             "Selected arithmetic configuration is outside the frozen seven-arm grid")
        controls = {"torch_tf32": {"kind": "torch_tf32", "precision": "high", "scope": "all"}}
    arms, mapped = {"optimized": config}, {}
    for name, configuration in controls.items():
        matching = [arm for arm, previous in arms.items() if previous == configuration]
        if matching:
            mapped[name] = matching[0]
        else:
            arms[name] = configuration
            mapped[name] = name
    aliases = {name: arm for name, arm in mapped.items() if name != arm}
    comparators = ["native", *(name for name, arm in mapped.items() if name == arm)]
    return controls, arms, mapped, aliases, comparators


def check_precision(snapshot: dict, config: dict | None) -> None:
    arithmetic = config.get("linear") if config and config["kind"] == "pipeline" else config
    high = arithmetic is not None and arithmetic["kind"] == "torch_tf32"
    need(snapshot["float32_matmul_precision"] == ("high" if high else "highest")
         and snapshot["cuda_matmul_allow_tf32"] is high and snapshot["cudnn_allow_tf32"] is False
         and snapshot["cuda_autocast_enabled"] is False, "Effective arithmetic policy snapshot changed")


def assert_check(saved: dict, actual: dict) -> None:
    need(saved["id"] == actual["id"] and saved.get("split", actual["split"]) == actual["split"],
         "Accuracy check ID/split changed")
    need(saved["limits"] == LIMITS and saved["token_count"] == actual["token_count"]
         and saved["winner_agreement"] == actual["winner_agreement"]
         and saved["numerical_valid"] == actual["numerical_valid"]
         and saved["valid"] == actual["numerical_valid"], "Accuracy check acceptance changed")
    for field in ("maximum_token_error", "maximum_sequence_error", "mean_absolute_token_error", "sum_absolute_token_error"):
        near(saved[field], actual[field], "raw check " + field)
    need(saved["per_sequence_token_errors"] == actual["token_errors"]
         and saved["sequence_errors"] == actual["sequence_errors"], "Raw error arrays changed")
    for field in ("reference_winner_margins", "alternative_winner_margins"):
        need(saved[field] == actual[field], "Stored margins changed")


def check_scoped_summary(saved: dict, checks: list[dict], zero_ids: list[str], rows: list[dict],
                         all_winners: bool = False) -> dict:
    actual = summarize(checks, zero_ids, len(rows), all_winners)
    assert_summary(saved, actual)
    actual["by_split"] = {}
    for split in ("development", "test"):
        subset = [row for row in checks if row["split"] == split]
        population = [row for row in rows if row["split"] == split]
        zeros = [row["id"] for row in population if row["id"] in zero_ids]
        result = summarize(subset, zeros, len(population), all_winners)
        if subset:
            assert_summary(saved["by_split"][split], result)
        else:
            need(saved["by_split"][split]["scored_records"] == 0, "Empty subgroup changed")
        result["descriptive_only"] = True
        actual["by_split"][split] = result
    need(saved["changed_record_ids"] == actual["changed_ids_by_method"], "Changed-record IDs omitted")
    return actual


def verify_source(run: Path, inventory: dict[str, str], current: bool = False) -> None:
    for relative, digest in inventory.items():
        path = child(run / "source", relative)
        need(sha(path) == digest, "Archived source hash changed: " + relative)
        SOURCE_HASHES[str(path.relative_to(ROOT))] = digest
        if current:
            need(sha(child(ROOT, relative)) == digest, "Current source differs from frozen run: " + relative)


def primary_calls(record: dict) -> dict[str, dict]:
    selected = [sample for sample in record["samples"] if sample["mode"] == "direct" and sample["seed"] in (17, 29, 43, 71)]
    need(len(selected) == 4 and {sample["seed"] for sample in selected} == {17, 29, 43, 71}, "Primary seeds changed")
    calls = {}
    for sample in selected:
        if sample["call"] is not None:
            semantic = json.dumps(sample["call"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            need(semantic == sample["semantic_key"], "Generated semantic key changed")
            calls[semantic] = sample["call"]
    return calls


def validate_families(item: dict, source: dict, generated: dict) -> None:
    families = item["families"]
    calls = primary_calls(generated)
    need([family["semantic_key"] for family in families] == sorted(calls), "Inference-only primary pool changed")
    names = tuple(source["function"][0]["parameters"]["properties"])
    orders = list(itertools.permutations(names))
    offset = 0
    for family in families:
        call = calls[family["semantic_key"]]
        need(family["call"] == call and set(call["arguments"]) == set(names), "Candidate call/schema changed")
        texts = [json.dumps({"name": call["name"], "arguments": {name: call["arguments"][name] for name in order}},
                            ensure_ascii=False, separators=(",", ":"), allow_nan=False) for order in orders]
        fixed = {method: orders.index(tuple(names[index] for index in policy if index < len(names)))
                 for method, policy in zip(FIXED, itertools.permutations(range(3)), strict=True)}
        need(family["texts"] == texts and family["orders"] == list(map(list, orders))
             and family["fixed_indices"] == fixed and family["sequence_begin"] == offset
             and family["sequence_end"] == offset + len(texts), "Complete source-order orbit changed")
        offset += len(texts)
    need(offset == len(item["sequence_token_ids"]) and item["candidate_count"] == len(families), "Orbit/token population changed")


def compare_output(reference: dict, alternative: dict, item_id: str, split: str,
                   expected_method: str, batch_size: int) -> dict:
    for field in ("id", "prompt_ids", "sequence_ids", "families"):
        need(reference[field] == alternative[field], "Scoring changed " + field)
    need(alternative["scores"]["method"] == expected_method, "Output scoring method mislabeled")
    validate_layout(alternative["scores"], alternative["sequence_ids"], alternative["prompt_ids"], batch_size)
    return comparison(reference, alternative, reference["families"], reference["sequence_ids"],
                      reference["prompt_ids"], item_id, split)


def discovery_name(config: dict) -> str:
    if config["kind"] == "pipeline":
        return config["method"] + ("_native" if config["linear"] is None else "_tf32x3_all")
    return config["kind"] + "_" + config["precision"] + "_" + config["scope"]


def audit_discovery(discovery: Path, selection: dict, plan: dict) -> dict:
    references = read(discovery / "references.json")
    fixtures = plan["fixture_freeze"]["fixtures"]
    ids = [row["id"] for row in fixtures]
    need(len(ids) == 4 and set(references) == set(ids) and plan["repeats"] == 3
         and plan["limits"] == LIMITS and plan["answers_accessed"] is False,
         "Development population or selection policy changed")
    raw = lines(discovery / "raw_observations.jsonl")
    observed = lines(discovery / "observations.jsonl")
    need(len(raw) == len(observed)
         and all(row["config"] in plan["configs"] for row in [*raw, *observed]),
         "Development raw/checked count or frozen configuration set differs")
    accepted, summaries = [], []
    for config in plan["configs"]:
        name = discovery_name(config)
        outputs = read(discovery / (name + "_outputs.json"))
        gate = read(discovery / (name + "_gate.json"))
        need(gate["config"] == config and [out["id"] for out in outputs] == ids, "Development config/fixture changed")
        checks = [compare_output(references[out["id"]], out, out["id"], "development", method_for(config),
                                 len(out["sequence_ids"])) for out in outputs]
        need(len(gate["checks"]) == len(checks), "Development check count changed")
        for stored, actual in zip(gate["checks"], checks, strict=True):
            assert_check(stored, actual)
        actual_summary = summarize(checks, [], len(ids), True)
        assert_summary(gate["summary"], actual_summary)
        selected_raw = [row for row in raw if row["config"] == config]
        selected_observed = [row for row in observed if row["config"] == config]
        times = []
        if actual_summary["valid"]:
            need(len(selected_raw) == len(selected_observed) == 3
                 and [row["repeat"] for row in selected_raw] == [0, 1, 2], "Missing/extra development repetitions")
            for raw_row, row in zip(selected_raw, selected_observed, strict=True):
                need(raw_row["timing"] == row["timing"] and raw_row["config"] == row["config"]
                     and raw_row["repeat"] == row["repeat"], "Development timing changed after gate")
                checked_timing(raw_row["timing"], plan["pid"])
                repeat_checks = [compare_output(references[out["id"]], out, out["id"], "development",
                                                method_for(config), len(out["sequence_ids"]))
                                 for out in raw_row["outputs"]]
                need([out["id"] for out in raw_row["outputs"]] == ids, "Development repeat paths reordered")
                for saved, actual in zip(row["checks"], repeat_checks, strict=True):
                    assert_check(saved, actual)
                repeated = summarize(repeat_checks, [], len(ids), True)
                assert_summary(row["summary"], repeated)
                need(repeated["valid"], "Development selection includes a failed repeat")
                times.append(raw_row["timing"]["wall_seconds"])
            accepted.append({"configuration": config, "median_seconds": statistics.median(times),
                             "seconds": times, "first_call_setup_seconds": gate["first_call_setup_seconds"]})
        else:
            need(not selected_raw and not selected_observed, "Failed development arm contributed timing")
        summaries.append({"configuration": config, "summary": actual_summary,
                          "measured_seconds": times, "first_call_setup_seconds": gate["first_call_setup_seconds"]})
    accepted.sort(key=lambda row: (row["median_seconds"], json.dumps(row["configuration"], sort_keys=True)))
    completion = read(discovery / "complete.json")
    need(accepted and accepted == selection["eligible"] == completion["eligible"]
         and completion["valid_arm_count"] == len(accepted)
         and selection["configuration"] == accepted[0]["configuration"], "Fastest valid development selection not reproduced")
    near(selection["median_seconds"], accepted[0]["median_seconds"], "Selected development median")
    return {"run": discovery.name, "configuration": selection["configuration"], "fixed_config_count": len(plan["configs"]),
            "eligible_config_count": len(accepted), "fixture_ids": ids, "arms": summaries,
            "elapsed_seconds": completion["elapsed_seconds"],
            "selection_rule_verified": "fastest valid development median, fixed lexical tie break; no held-out selection",
            "timing_scope": "three development calls; discovery only, not accepted confirmation"}


def source_and_inputs(run: Path, manifest: dict) -> tuple[str, list[dict], list[dict], dict, dict]:
    model = manifest["model"]["repo_id"].split("/")[-1]
    need(model in MODELS and manifest["gpu"].count(GPU_UUID) == 1 and "GB10" in manifest["gpu"], "Wrong recorded model/GPU")
    need(manifest["expected_gpu_uuid"] == GPU_UUID and manifest["limits"] == LIMITS
         and manifest["answers_accessed"] is False and manifest["audit_batch_size"] == 8
         and manifest["timing_batch_size"] == "whole-family" and manifest["head_chunk"] == 32
         and manifest["attention"] == "eager" and manifest["normalizer"] == "torch_ce"
         and set(manifest["methods"]) == set(METHODS), "Frozen scope/configuration changed")
    controls, arms, mapped, aliases, _ = expected_controls(manifest["configuration"])
    need(manifest["control_configurations"] == controls and manifest["audit_arms"] == arms
         and manifest["control_audit_arms"] == mapped and manifest["control_aliases"] == aliases
         and manifest["selected_scoring_method"] == method_for(manifest["configuration"]),
         "Required native/cached/same-structure or Torch-TF32 comparators changed")
    if manifest["configuration"]["kind"] == "pipeline":
        need(manifest["confirmation_control_amendment"] == "docs/PREFIX_CONFIRMATION_CONTROL_AMENDMENT.md",
             "Prospective same-structure comparator amendment is missing")
    required_sources = {"scripts/run_approximate_scoring.py", "scripts/audit_full_triton.py",
                        "scripts/benchmark_scoring.py", "scripts/run_accelerated_scoring.py",
                        "docs/APPROXIMATE_SCORING_PROTOCOL.md", "metadata/acceleration_input_freeze_v1.json",
                        "metadata/public_models.json", "metadata/model_3b_verified.json", "metadata/model_1p5b_verified.json"}
    if manifest["configuration"]["kind"] == "pipeline":
        required_sources.update({"docs/PREFIX_APPROXIMATE_PROTOCOL.md", "docs/PREFIX_CONFIRMATION_CONTROL_AMENDMENT.md"})
    need(required_sources <= manifest["source_sha256"].keys(), "Confirmation source inventory omitted a required dependency")
    verify_source(run, manifest["source_sha256"], current=True)
    selection, plan = read(run / "selection.json"), read(run / "selection_plan.json")
    need(sha(run / "selection.json") == manifest["selection_sha256"] and selection == manifest["selection"]
         and selection["valid"] is True and selection["answers_accessed"] is False
         and selection["configuration"] == manifest["configuration"]
         and sha(run / "selection_plan.json") == manifest["selection_plan_sha256"] == selection["plan_sha256"]
         and plan["limits"] == LIMITS and selection["configuration"] in plan["configs"], "Development selection freeze changed")
    expected_grid = ([{"kind": "pipeline", "method": method, "linear": linear}
                      for linear in (None, {"kind": "linear", "precision": "tf32x3", "scope": "all"})
                      for method in ("cached", "tree")]
                     if manifest["configuration"]["kind"] == "pipeline" else
                     [{"kind": "linear", "precision": precision, "scope": scope}
                      for precision in ("tf32x3", "tf32x6") for scope in ("head", "backbone", "all")]
                     + [{"kind": "torch_tf32", "precision": "high", "scope": "all"}])
    need(plan["configs"] == expected_grid, "Prospective development configuration grid/order changed")
    discovery = child(ROOT / "runs", selection["run_id"])
    need(sha(discovery / "selection.json") == manifest["selection_sha256"]
         and sha(discovery / "plan.json") == manifest["selection_plan_sha256"], "Retained development receipt changed")
    verify_source(discovery, plan["source_sha256"])
    for relative, digest in plan["source_sha256"].items():
        need(manifest["source_sha256"].get(relative) == digest, "Selected implementation changed in confirmation: " + relative)
    discovery_report = audit_discovery(discovery, selection, plan)
    freeze = read(run / "input_freeze.json")
    freeze_relative = "metadata/acceleration_input_freeze_v1.json"
    need(sha(run / "input_freeze.json") == manifest["source_sha256"][freeze_relative]
         == plan["source_sha256"][freeze_relative] and freeze["answers_accessed"] is False,
         "Timing fixture byte freeze changed")
    input_path = ROOT / "data/frozen_inputs.json"
    inputs = read(input_path)
    need(sha(input_path) == INPUT_SHA == manifest["inputs_sha256"] and inputs["answer_accessed"] is False,
         "Input-only population changed")
    rows = inputs["records"]
    need(len(rows) == 119 and Counter(row["split"] for row in rows) == {"development": 20, "test": 99}
         and [row["id"] for row in rows] == manifest["record_ids"] and manifest["record_count"] == 119,
         "Complete119 split/count changed")
    development_model = [record for record in freeze["records"] if record["model"] == "Qwen2.5-3B-Instruct"]
    need(len(development_model) == 1 and plan["model"]["repo_id"] == "Qwen/Qwen2.5-3B-Instruct"
         and plan["fixture_freeze"]["inputs_sha256"] == INPUT_SHA, "Development used another model or input freeze")
    development_candidates_path = child(ROOT / "runs", development_model[0]["candidate_run"] + "/candidates.json")
    development_candidates = read(development_candidates_path)
    need(sha(development_candidates_path) == development_model[0]["candidates_sha256"]
         == plan["fixture_freeze"]["candidates_sha256"], "Development candidate freeze changed")
    by_id = {row["id"]: row for row in rows}
    development_generated = {row["id"]: row for row in development_candidates["records"]}
    expected_development = [row["id"] for row in sorted(rows, key=lambda row: row["id"])
                            if row["split"] == "development" and len(primary_calls(development_generated[row["id"]])) >= 2][:4]
    need([fixture["id"] for fixture in plan["fixture_freeze"]["fixtures"]] == expected_development,
         "Development first-four input-only selection changed")
    for fixture in plan["fixture_freeze"]["fixtures"]:
        need(fixture["input_record"] == by_id[fixture["id"]]
             and fixture["candidate_record"] == development_generated[fixture["id"]], "Development fixture bytes/content changed")
    matching = [record for record in freeze["records"] if record["model"] == model]
    need(len(matching) == 1, "Duplicate/missing model input freeze")
    candidate_path = child(ROOT / "runs", matching[0]["candidate_run"] + "/candidates.json")
    candidates = read(candidate_path)
    need(sha(candidate_path) == manifest["candidates_sha256"] == matching[0]["candidates_sha256"]
         and candidates["answers_accessed"] is False
         and [row["id"] for row in candidates["records"]] == manifest["record_ids"], "Candidates changed")
    expected_dispositions, expected_ids = [], []
    by_id = {row["id"]: row for row in rows}
    for candidate in sorted(candidates["records"], key=lambda row: row["id"]):
        source = by_id[candidate["id"]]
        need(candidate["split"] == source["split"] and candidate["schema_group"] == source["schema_group"]
             and candidate["function"] == source["function"][0], "Candidate input metadata changed")
        if source["split"] == "test":
            count = len(primary_calls(candidate))
            expected_dispositions.append({"id": candidate["id"], "primary_candidate_count": count, "eligible": count >= 2})
            if count >= 2 and len(expected_ids) < 8:
                expected_ids.append(candidate["id"])
    need(len(expected_ids) == 8 and expected_ids == matching[0]["timing_fixture_ids"] == manifest["fixture_ids"]
         and expected_dispositions == matching[0]["test_dispositions"] == manifest["test_dispositions"], "First-eight input-only timing rule changed")
    ranking = child(ROOT / "runs", manifest["ranking_run"])
    need(ranking.name == MODEL_REFERENCE[model], "Wrong full-reference run")
    for relative, digest in manifest["reference_sha256"].items():
        path = child(ranking, relative)
        need(sha(path) == digest, "Historical reference artifact changed: " + relative)
        SOURCE_HASHES[str(path.relative_to(ROOT))] = digest
    historical = read(ranking / "ranking_manifest.json")
    receipt = read(ranking / "pre_gold_freeze.json")
    predictions = read(ranking / "predictions.json")
    need(historical["model"] == model and historical["candidates_sha256"] == manifest["candidates_sha256"]
         and historical["inputs_sha256"] == INPUT_SHA and historical["batch_size"] == 8
         and historical["head_chunk"] == 32 and historical["selected_method"] == "full"
         and historical["reducer"] == "torch_lse" and historical["attention"] == "eager"
         and historical["dtype"] == "float32" and historical["tf32"] is False
         and receipt["stage"] == "pre-gold" and receipt["answers_accessed"] is False
         and historical["answers_accessed"] is False and predictions["answers_accessed"] is False
         and predictions["selected_scoring_method"] == "full",
         "Historical reference scope changed")
    need(receipt["ranking_manifest_sha256"] == sha(ranking / "ranking_manifest.json")
         == predictions["ranking_manifest_sha256"]
         and receipt["predictions"]["sha256"] == sha(ranking / "predictions.json")
         and receipt["inputs_sha256"] == historical["inputs_sha256"] == INPUT_SHA
         and receipt["candidates"]["sha256"] == historical["candidates_sha256"]
         == predictions["candidates_sha256"] == manifest["candidates_sha256"],
         "Historical pre-gold prediction/manifest/input/candidate hash chain changed")
    need(set(receipt["item_sha256"]) == {f"items/{index:04d}.json" for index in range(119)}
         and receipt["record_count"] == 119 and receipt["development_count"] == 20 and receipt["test_count"] == 99
         and historical["record_ids"] == manifest["record_ids"]
         and [row["id"] for row in predictions["records"]] == manifest["record_ids"],
         "Historical shards or frozen prediction population incomplete/reordered")
    need(set(manifest["reference_sha256"]) == {"pre_gold_freeze.json", "ranking_manifest.json", "predictions.json",
                                             *receipt["item_sha256"]}, "Reference hash inventory is incomplete or augmented")
    numeric_sources = {f"src/orbitcall/{name}.py" for name in
                       ("bfcl", "ranking", "serialization", "scoring", "kernels", "trie", "gpu_guard")}
    need(numeric_sources <= historical["source_sha256"].keys()
         and numeric_sources <= manifest["source_sha256"].keys(), "Historical numerical source inventory incomplete")
    items = []
    for relative, digest in historical["source_sha256"].items():
        if not relative.startswith("src/orbitcall/"):
            continue
        previous = child(ranking / "source", relative)
        need(sha(previous) == digest, "Historical numeric source snapshot changed")
        old, new = previous.read_bytes(), child(run / "source", relative).read_bytes()
        if relative == "src/orbitcall/scoring.py":
            needle = b'method="tree", reducer="torch", batch_size=8'
            need(old.count(needle) == 1, "Historical default correction is not unique")
            old = old.replace(needle, b'method="full", reducer="torch", batch_size=8')
        need(old == new, "Historical numeric implementation changed beyond released default")
    for index, (source, generated) in enumerate(zip(rows, candidates["records"], strict=True)):
        relative = f"items/{index:04d}.json"
        path = ranking / relative
        need(sha(path) == receipt["item_sha256"][relative] == manifest["reference_sha256"][relative], "Historical item hash changed")
        item = read(path)
        need(item["valid"] is True and item["prediction_source"] == "full"
             and item["ranking_manifest_sha256"] == receipt["ranking_manifest_sha256"]
             and all(item[field] == source[field] for field in ("id", "split", "schema_group", "category")), "Reference metadata changed")
        frozen = predictions["records"][index]
        need(all(frozen["methods"][method] == result
                 for method, result in item["scoring"]["full"]["predictions"].items()),
             "Frozen predictions differ from their full-reference item")
        validate_families(item, source, generated)
        if item["sequence_token_ids"]:
            validate_layout(item["scoring"]["full"]["scores"], item["sequence_token_ids"], item["prompt_token_ids"], 8)
        independent_scores(item["scoring"]["full"], item["families"], item["sequence_token_ids"], item["prompt_token_ids"])
        items.append(item)
    declared_models = read(run / "source/metadata/public_models.json")
    public = [row for row in declared_models["models"] if row["repo_id"] == manifest["model"]["repo_id"]]
    receipt_file = "model_3b_verified.json" if "3B" in model else "model_1p5b_verified.json"
    public_receipt = read(run / "source/metadata" / receipt_file)
    need(len(public) == 1 and public[0]["repo_id"] == manifest["model"]["repo_id"] == public_receipt["repo_id"]
         and public[0]["revision"] == manifest["model"]["revision"] == public_receipt["revision"]
         and public_receipt["all_public_files_match"] is True
         and manifest["model"]["all_public_files_match"] is True and manifest["model"]["receipt_file"] == receipt_file
         and manifest["model"]["receipt_sha256"] == sha(run / "source/metadata" / receipt_file)
         and manifest["model"]["public_manifest_sha256"] == sha(run / "source/metadata/public_models.json"),
         "Pinned checkpoint receipt/public manifest mismatch")
    declared_files = {row["path"]: row for row in public[0]["files"]}
    verified_files = {row["path"]: row for row in public_receipt["files"]}
    need(len(declared_files) == len(public[0]["files"]) and len(verified_files) == len(public_receipt["files"])
         and set(declared_files) == set(verified_files), "Public checkpoint file inventory duplicated/incomplete")
    expected_files = []
    for declared in public[0]["files"]:
        verified = verified_files[declared["path"]]
        need(verified["matches_public_revision"] is True and verified["bytes"] == declared["size"]
             and (declared["sha256"] is None or declared["sha256"] == verified["sha256"]),
             "Canonical checkpoint receipt disagrees with pinned public file metadata")
        expected_files.append({key: verified[key] for key in ("path", "bytes", "sha256")})
    need(manifest["model"]["files"] == expected_files, "Manifest checkpoint files differ from canonical verified receipt")
    return model, rows, items, selection, discovery_report


def analyze_run(run: Path) -> dict:
    manifest = read(run / "manifest.json")
    model, rows, items, selection, discovery_report = source_and_inputs(run, manifest)
    check_precision(read(run / "setup/native_precision.json"), None)
    summary = read(run / "audit_summary.json")
    need(summary["population"] == 119 and set(summary["arms"]) == set(manifest["audit_arms"]), "Audit population/control set changed")
    checks = {arm: [] for arm in manifest["audit_arms"]}
    zero_ids, audit_costs, layouts = [], {arm: [] for arm in checks}, {}
    for index, (source, item) in enumerate(zip(rows, items, strict=True)):
        prompt, sequences, families = item["prompt_token_ids"], item["sequence_token_ids"], item["families"]
        if not sequences:
            zero = read(run / "audit_zero" / f"{index:04d}.json")
            need(zero["id"] == source["id"] and zero["index"] == index and zero["split"] == source["split"]
                 and zero["zero_family"] is True and zero["included_in_accuracy_denominator"] is False
                 and zero["predictions"] == item["scoring"]["full"]["predictions"], "Null-family disposition changed")
            need(zero["source_item_sha256"] == manifest["reference_sha256"][f"items/{index:04d}.json"],
                 "Null-family historical source receipt changed")
            zero_ids.append(source["id"])
            continue
        for arm, config in manifest["audit_arms"].items():
            raw = read(run / "audit_raw" / arm / f"{index:04d}.json")
            checked = read(run / "audit" / arm / f"{index:04d}.json")
            for stored in (raw, checked):
                need(stored["id"] == source["id"] and stored["split"] == source["split"]
                     and stored["index"] == index and stored["arm"] == arm and stored["zero_family"] is False
                     and stored["source_item_sha256"] == manifest["reference_sha256"][f"items/{index:04d}.json"], "Audit identity/provenance changed")
            need(raw["configuration"] == config and raw["prompt_token_ids"] == prompt
                 and raw["sequence_token_ids"] == sequences and raw["scores"] == checked["alternative"]["scores"], "Pre-gate numerical raw output changed")
            check_precision(raw["effective_precision"], config)
            need(raw["scores"]["method"] == method_for(config), "Audit method relabeled")
            layouts[arm] = validate_layout(raw["scores"], sequences, prompt, 8)
            actual = comparison(item["scoring"]["full"], checked["alternative"], families, sequences, prompt,
                                source["id"], source["split"])
            assert_check(checked["check"], actual)
            checks[arm].append(actual)
            need(raw["elapsed_seconds"] >= 0, "Negative audit cost")
            audit_costs[arm].append(raw["elapsed_seconds"])
    need(summary["zero_family_ids"] == zero_ids and summary["zero_family"] == len(zero_ids)
         and summary["scored"] == 119 - len(zero_ids), "Zero-family denominator changed")
    expected_zero_split = {split: sum(row["id"] in zero_ids for row in rows if row["split"] == split)
                           for split in ("development", "test")}
    need(summary["zero_by_split"] == expected_zero_split, "Zero subgroup count changed")
    audited = {arm: check_scoped_summary(summary["arms"][arm], values, zero_ids, rows) for arm, values in checks.items()}
    need(summary["valid"] == audited["optimized"]["valid"], "Audit acceptance changed")
    for arm, values in checks.items():
        actual_paths = {path.stem for path in (run / "audit" / arm).glob("*.json")}
        raw_paths = {path.stem for path in (run / "audit_raw" / arm).glob("*.json")}
        expected = {f"{index:04d}" for index, item in enumerate(items) if item["sequence_token_ids"]}
        need(actual_paths == raw_paths == expected, "Extra/missing audit shards")
        first = read(run / "setup" / ("first_call_" + arm + ".json"))
        need(first["id"] == values[0]["id"] and first["included_in_audit_elapsed"] is True
             and first["accepted_timing_window"] is False, "First-call cost scope changed")
        near(first["elapsed_seconds"], audit_costs[arm][0], "First-call nested cost")
    actual_zero_paths = {path.stem for path in (run / "audit_zero").glob("*.json")}
    need(actual_zero_paths == {f"{index:04d}" for index, item in enumerate(items) if not item["sequence_token_ids"]},
         "Missing/extra null-family files")
    aliases = manifest.get("control_aliases", {})
    for control, arm in manifest["control_audit_arms"].items():
        need(arm in manifest["audit_arms"] and manifest["control_configurations"][control] == manifest["audit_arms"][arm], "Control alias changed configuration")
    if aliases:
        alias_receipt = read(run / "audit_control_alias.json")
        need(alias_receipt["audit_summary_sha256"] == sha(run / "audit_summary.json"), "Alias summary receipt changed")
        need({row["control"]: row["source_arm"] for row in alias_receipt["aliases"]} == aliases
             and all(row["extra_inference_calls"] == 0 for row in alias_receipt["aliases"]), "Duplicate control work hidden")
    base = {"run": run.name, "model": model, "configuration": manifest["configuration"],
            "selection_sha256": manifest["selection_sha256"], "population": 119,
            "audit": audited, "actual_layout_by_arm": layouts, "control_aliases": aliases,
            "audit_elapsed_seconds": summary["elapsed_seconds"],
            "audit_arm_measured_seconds": {arm: math.fsum(values) for arm, values in audit_costs.items()},
            "first_call_scope": "first/JIT call is included in audit costs; never add it again",
            "fidelity_is_task_accuracy": False, "has_task_quality_evaluation": False,
            "development_selection_review": discovery_report}
    failure = read(run / "failure.json") if (run / "failure.json").exists() else None
    if failure:
        need(failure["answers_accessed"] is False and failure["selection_sha256"] == manifest["selection_sha256"]
             and failure["candidates_sha256"] == manifest["candidates_sha256"]
             and failure["source_sha256"] == manifest["source_sha256"]
             and failure["reference_sha256"] == manifest["reference_sha256"],
             "Failure disposition changed its frozen provenance")
    if not (run / "active_timing_plan.json").exists():
        need(failure is not None, "No final timing plan or failure disposition")
        base.update(accepted_complete_timing=False, failure=failure,
                    total_process_elapsed_seconds=failure["elapsed_seconds"], accepted_pairs=[])
        return base
    return analyze_timing(run, manifest, rows, items, audited, base, failure)


def analyze_timing(run: Path, manifest: dict, rows: list[dict], items: list[dict], audited: dict,
                   base: dict, failure: dict | None) -> dict:
    setup = read(run / "timing_setup.json")
    active = read(run / "active_timing_plan.json")
    need(manifest["timing_seed"] == 271828 and manifest["blocks_per_pair"] == 6, "Timing randomization changed")
    _, _, _, _, all_controls = expected_controls(manifest["configuration"])
    rng, expected_plan = random.Random(271828), []
    for control in all_controls:
        for block in range(6):
            order = [control, "optimized", "optimized", control]
            if rng.randrange(2):
                order = ["optimized", control, control, "optimized"]
            expected_plan.append({"pair": control + "_vs_optimized", "control": control, "block": block, "order": order})
    rng.shuffle(expected_plan)
    need(manifest["timing_plan"] == expected_plan, "Prospective six-block schedule changed")
    need(active["plan"] == [block for block in expected_plan if block["control"] in active["controls"]], "Control filtering reseeded/redrew plan")
    item_map, row_map = ({row["id"]: row for row in items}, {row["id"]: row for row in rows})
    fixtures = [row_map[item_id] for item_id in manifest["fixture_ids"]]
    native, prepared, setup_checks = {}, {}, {}
    for item_id in manifest["fixture_ids"]:
        stored = read(run / "timing_gate_raw/native" / (item_id + ".json"))
        native[item_id] = stored["output"]
        item = item_map[item_id]
        need(native[item_id]["id"] == item_id and native[item_id]["scores"]["method"] == "full"
             and native[item_id]["prompt_ids"] == item["prompt_token_ids"]
             and native[item_id]["sequence_ids"] == item["sequence_token_ids"]
             and native[item_id]["families"] == item["families"], "Fresh whole-family reference paths changed")
        validate_layout(native[item_id]["scores"], item["sequence_token_ids"], item["prompt_token_ids"], len(item["sequence_token_ids"]))
    prepared["native"] = native
    expected_setup_arms = {"optimized", *(control for control in all_controls[1:]
                                          if audited[manifest["control_audit_arms"][control]]["valid"])}
    need(set(setup["summaries"]) == expected_setup_arms and setup["includes_first_whole_family_shape_calls"] is True
         and setup["accepted_timing_window"] is False, "Required whole-family setup comparators or scope changed")
    for arm, saved_summary in setup["summaries"].items():
        need(arm in manifest["audit_arms"], "Undeclared whole-family control")
        prepared[arm], setup_checks[arm] = {}, []
        for item_id in manifest["fixture_ids"]:
            raw = read(run / "timing_gate_raw" / arm / (item_id + ".json"))
            gate = read(run / "timing_gates" / arm / (item_id + ".json"))
            need(raw["output"] == gate["alternative"] and gate["native"] == native[item_id], "Raw whole-family gate changed")
            check_precision(raw["effective_precision"], manifest["audit_arms"][arm])
            result = raw["output"]
            actual = compare_output(native[item_id], result, item_id, "test", method_for(manifest["audit_arms"][arm]), len(result["sequence_ids"]))
            assert_check(gate["check"], actual)
            prepared[arm][item_id] = result
            setup_checks[arm].append(actual)
        check_scoped_summary(saved_summary, setup_checks[arm], [], fixtures, True)
    expected_active, expected_excluded = ["native"], {}
    for control in all_controls[1:]:
        arm = manifest["control_audit_arms"][control]
        if not audited[arm]["valid"]:
            expected_excluded[control] = "failed whole119 approximate gate"
        elif not summarize(setup_checks[control], [], 8, True)["valid"]:
            expected_excluded[control] = "failed whole-family approximate gate"
        else:
            expected_active.append(control)
    need(active["controls"] == expected_active and active["excluded_controls"] == expected_excluded,
         "Control exclusion changed after observation")
    raw_windows = lines(run / "raw_windows.jsonl") if (run / "raw_windows.jsonl").exists() else []
    windows = lines(run / "windows.jsonl") if (run / "windows.jsonl").exists() else []
    need(len(raw_windows) - len(windows) in (0, 1)
         and (len(raw_windows) == len(windows) or failure is not None),
         "Raw/checked prefix differs beyond the one retained failed gate window")
    expected_windows = [{"pair": block["pair"], "block": block["block"], "position": position, "arm": arm}
                        for block in active["plan"] for position, arm in enumerate(block["order"])]
    need(len(raw_windows) <= len(expected_windows), "Unplanned additional timing windows")
    checks_per_window, window_costs = [], []
    for expected, raw, window in zip(expected_windows, raw_windows, windows, strict=False):
        need(all(window[key] == raw[key] == expected[key] for key in expected), "Window order/arm differs from frozen plan")
        need(all(window[key] == value for key, value in raw.items()), "Raw measurements/outputs changed after gate")
        checked_timing(raw, manifest["pid"])
        outputs = window["outputs"]
        need([out["id"] for out in outputs] == manifest["fixture_ids"], "Timed fixture count/order changed")
        calculations, repeats = [], []
        for output in outputs:
            item_id = output["id"]
            method = "full" if window["arm"] == "native" else method_for(manifest["audit_arms"][window["arm"]])
            calculation = compare_output(native[item_id], output, item_id, "test", method, len(output["sequence_ids"]))
            assert_check(window["checks"][item_id], calculation)
            calculations.append(calculation)
            repeated = compare_output(prepared[window["arm"]][item_id], output, item_id, "test", method, len(output["sequence_ids"]))
            assert_check(window["repeatability_checks"][item_id], repeated)
            repeats.append(repeated)
        calculated = check_scoped_summary(window["summary"], calculations, [], fixtures, True)
        repeated_summary = check_scoped_summary(window["repeatability_summary"], repeats, [], fixtures, True)
        need(window["valid"] == calculated["valid"] and window["repeatability_is_diagnostic_only"] is True, "Window gate scope changed")
        checks_per_window.append({**expected, "valid": calculated["valid"],
                                  "numerical_fidelity": calculated, "repeatability_diagnostic": repeated_summary})
        window_costs.append(raw["wall_seconds"])
    raw_only = []
    for expected, raw in zip(expected_windows[len(windows):], raw_windows[len(windows):], strict=False):
        need(all(raw[key] == expected[key] for key in expected), "Raw-only failed window is not the next planned prefix")
        checked_timing(raw, manifest["pid"])
        need([output["id"] for output in raw["outputs"]] == manifest["fixture_ids"],
             "Raw-only failed window changed its eight-fixture population")
        method = "full" if raw["arm"] == "native" else method_for(manifest["audit_arms"][raw["arm"]])
        calculations = [compare_output(native[output["id"]], output, output["id"], "test", method,
                                       len(output["sequence_ids"])) for output in raw["outputs"]]
        raw_only.append({**expected, "accepted": False, "wall_seconds": raw["wall_seconds"],
                         "numerical_fidelity_reconstructed": summarize(calculations, [], 8, True),
                         "disposition": "retained pre-gate raw tail; no completed checked window or accepted ratio"})
        window_costs.append(raw["wall_seconds"])
    complete = read(run / "complete.json") if (run / "complete.json").exists() else None
    if complete is None:
        need(failure is not None, "Incomplete window plan without a failure record")
        base.update(accepted_complete_timing=False, failure=failure, accepted_pairs=[],
                    retained_window_count=len(windows), expected_window_count=len(expected_windows),
                    retained_window_diagnostics=checks_per_window,
                    retained_raw_window_count=len(raw_windows), raw_only_unaccepted_windows=raw_only,
                    measured_windows_wall_seconds=math.fsum(window_costs),
                    total_process_elapsed_seconds=failure["elapsed_seconds"])
        return base
    need(not raw_only and len(raw_windows) == len(windows) == len(expected_windows)
         == complete["raw_windows"] == complete["expected_windows"], "Completed plan missing checked windows")
    need(complete["selection_sha256"] == manifest["selection_sha256"] and complete["configuration"] == manifest["configuration"]
         and complete["limits"] == LIMITS and complete["answers_accessed"] is False, "Completion freeze changed")
    primary_valid = all(row["valid"] for row in checks_per_window if row["arm"] in {"native", "optimized"})
    need(complete["valid"] == primary_valid, "Completion acceptance changed")
    accepted, statuses, exclusions = [], [], dict(expected_excluded)
    for control in active["controls"]:
        pair = control + "_vs_optimized"
        observed = [row for row in windows if row["pair"] == pair]
        valid = primary_valid and len(observed) == 24 and all(row["valid"] for row in observed)
        rejected = [{key: row[key] for key in ("block", "position", "arm")} for row in observed if not row["valid"]]
        statuses.append({"pair": pair, "valid": valid, "windows": len(observed), "rejected_windows": rejected})
        if valid:
            recalculated = paired_blocks(observed, control, "optimized")
            saved = [row for row in complete["pairs"] if row["pair"] == pair]
            need(len(saved) == 1, "Accepted pair summary missing/duplicated")
            saved = saved[0]
            for actual_block, stored_block in zip(recalculated["blocks"], saved["blocks"], strict=True):
                need(actual_block["block"] == stored_block["block"], "Block ID changed")
                for arm in (control, "optimized"):
                    near(stored_block["seconds"][arm], actual_block["seconds"][arm], "block wall mean")
                near(stored_block["control_over_optimized"], actual_block["ratio_a_over_b"], "block ratio")
            near(saved["median_ratio"], recalculated["median_ratio_a_over_b"], "paired median ratio")
            for a, b in zip(saved["range"], recalculated["range"], strict=True):
                near(a, b, "paired ratio range")
            for a, b in zip(saved["descriptive_bootstrap_median_95"], recalculated["producer_empirical_bootstrap_median_95"], strict=True):
                near(a, b, "paired empirical bootstrap")
            accepted.append(recalculated)
        elif control != "native" and any(not row["valid"] for row in observed if row["arm"] == control):
            exclusions[control] = "failed measured-window approximate gate"
    need(complete["pair_status"] == statuses and complete["excluded_controls"] == exclusions
         and {row["pair"] for row in complete["pairs"]} == {row["a"] + "_vs_optimized" for row in accepted}, "Accepted/excluded pairs changed")
    need(complete["elapsed_seconds"] >= math.fsum(window_costs), "Process cost smaller than timed windows")
    base.update(accepted_complete_timing=primary_valid, accepted_pairs=accepted, pair_status=statuses,
                excluded_controls=exclusions, setup_fidelity={arm: summarize(values, [], 8, True) for arm, values in setup_checks.items()},
                retained_window_diagnostics=checks_per_window, retained_window_count=len(windows),
                retained_raw_window_count=len(raw_windows), raw_only_unaccepted_windows=raw_only,
                expected_window_count=len(expected_windows), measured_windows_wall_seconds=math.fsum(window_costs),
                whole_family_setup_elapsed_seconds=setup["elapsed_seconds"],
                total_process_elapsed_seconds=complete["elapsed_seconds"], failure=failure,
                accepted_primary_comparison_scope="native IEEE full versus selected; other comparators have separate pair validity",
                strong_cached_comparison_accepted=any(pair["a"] == "cached" for pair in accepted),
                same_structure_arithmetic_comparison_accepted=any(pair["a"] == "structure" for pair in accepted),
                cost_scope="process interval includes all stage work; audit/window/first-call values are nested and must not be summed again",
                pipeline_speedup_measured=False, serving_throughput_measured=False)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("Numerical artifact analysis must run on the authorized Spark")
    identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"],
                                       text=True, timeout=15).strip().splitlines()
    need(len(identity) == 1 and "GB10" in identity[0] and GPU_UUID in identity[0], "Wrong live analysis host/GPU")
    runs = [(ROOT / path).resolve() if not path.is_absolute() else path.resolve() for path in args.runs]
    need(len(set(runs)) == len(runs) and all(path.is_relative_to(ROOT / "runs") for path in runs), "Duplicate/outside run directories")
    output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    need(not output.exists(), "Analysis output already exists; do not overwrite")
    studies = [analyze_run(run) for run in runs]
    need(len({study["model"] for study in studies}) == len(studies), "Duplicate model summaries")
    need(len({study["selection_sha256"] for study in studies}) == 1, "Models used different development selections")
    both = len(studies) == 2 and {study["model"] for study in studies} == MODELS
    combined = both and all(study["accepted_complete_timing"] and study["audit"]["optimized"]["valid"] for study in studies)
    discovery_costs = {study["development_selection_review"]["run"]: study["development_selection_review"]["elapsed_seconds"]
                       for study in studies}
    confirmation_cost = math.fsum(study["total_process_elapsed_seconds"] for study in studies)
    result = {"format_version": 1, "analysis_source_sha256": sha(Path(__file__)),
              "execution": "authorized Spark CPU-only independent artifact analysis; no CUDA/model/gold loading",
              "live_gpu_identity": identity[0], "limits": LIMITS.copy(), "studies": studies,
              "both_declared_models_present": both, "combined_confirmation_accepted": combined,
              "both_models_strong_cached_comparison_accepted": both and all(study.get("strong_cached_comparison_accepted", False) for study in studies),
              "both_models_same_structure_comparison_accepted": both and all(study.get("same_structure_arithmetic_comparison_accepted", False) for study in studies),
              "source_artifact_sha256": SOURCE_HASHES,
              "provided_confirmation_process_wall_seconds": confirmation_cost,
              "selected_discovery_process_intervals": discovery_costs,
              "selected_discovery_plus_provided_confirmation_wall_seconds": confirmation_cost + math.fsum(discovery_costs.values()),
              "cost_scope": "provided confirmation intervals plus selected discovery once; excludes earlier retired studies and is not pure GPU-busy time; nested first-call/windows are not summed again",
              "task_accuracy_measured": False, "pipeline_speedup_measured": False, "serving_throughput_measured": False,
              "claim_limits": ["winner fidelity is not task correctness or a statistical lower confidence guarantee",
                               "whole119 reference audit and eight-fixture whole-family timing have different batch layouts",
                               "all fixed controls and failures are retained; no fastest held-out comparator selection",
                               "empirical bootstrap resamples are not additional independent trials"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "independent_artifact_analysis_complete", "output_sha256": sha(output),
                      "models": [study["model"] for study in studies], "combined_confirmation_accepted": combined}, indent=2))


if __name__ == "__main__":
    main()
