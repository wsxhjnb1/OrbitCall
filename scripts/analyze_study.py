#!/usr/bin/env python3
"""Spark-only deterministic analysis of the two completed OrbitCall studies.

No model, CUDA context, gold loader, new inference, or parameter tuning is used.
Read completed measurements and frozen reports; write new exclusive artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
MODELS = {"Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct"}
INPUT_SHA256 = "1e8edf278caa6066c1b4c1fab07599cc8acb03fdf0037d3447603ff75916060c"
PRIMARY = {17, 29, 43, 71}
ADDITIONAL = {101, 113, 127, 139}
SCORE_METHODS = ("mass", "canonical_schema", "max_order", "mean_logprob", "length_normalized",
                 "dev_selected_fixed", "fixed_012", "fixed_021", "fixed_102", "fixed_120", "fixed_201", "fixed_210",
                 "vote4", "vote8", "greedy", "freeform_vote")
METHOD_LABELS = {"mass": "Finite-family mass", "canonical_schema": "Schema order", "max_order": "Max order",
                 "mean_logprob": "Mean logprob", "length_normalized": "Length-normalized mean",
                 "dev_selected_fixed": "Dev-selected fixed", "vote4": "Vote 4", "vote8": "Vote 8",
                 "greedy": "Native greedy", "freeform_vote": "Freeform + reformat vote"}
ARM_LABELS = {"full": "FULL Torch", "full_lse": "FULL LSE", "full_ce": "FULL native CE",
              "full_triton": "FULL Triton", "torch": "Torch", "torch_lse": "LSE",
              "torch_ce_anchor": "Native CE", "triton": "Triton"}
AUDIT_AGGREGATORS = ("mass", "max_order", "mean_logprob", "length_normalized", "canonical_schema",
                     "fixed_012", "fixed_021", "fixed_102", "fixed_120", "fixed_201", "fixed_210")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def check_finite_positive(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid positive measurement: {name}")
    return float(value)


def generation_summary(directory: Path, expected_candidate_hash: str, expected_manifest_hash: str) -> dict:
    manifest = read_json(directory / "generation_manifest.json")
    candidates_path = directory / "candidates.json"
    if (sha(candidates_path) != expected_candidate_hash or manifest.get("answers_accessed") is not False
            or candidates_path.name != "candidates.json"
            or sha(directory / "generation_manifest.json") != expected_manifest_hash
            or manifest["inputs_sha256"] != INPUT_SHA256
            or manifest["dtype"] != "bfloat16" or manifest["attention"] != "sdpa"):
        raise ValueError("Generation differs from the pre-gold freeze")
    candidates = read_json(candidates_path)
    ids = manifest["record_ids"]
    if [row["id"] for row in candidates["records"]] != ids or len(set(ids)) != len(ids):
        raise ValueError("Incomplete or duplicate generation population")
    plans = ({("greedy", 0)} | {("direct", seed) for seed in PRIMARY | ADDITIONAL}
             | {(mode, seed) for mode in ("freeform", "reformat") for seed in PRIMARY})
    batch_paths = sorted((directory / "generation").glob("*.json"))
    batches = [read_json(path) for path in batch_paths]
    grouped = defaultdict(list)
    for batch in batches:
        key = (batch["mode"], batch["base_seed"])
        if key not in plans:
            raise ValueError("Unexpected generation stage")
        grouped[key].append(batch)
    if set(grouped) != plans:
        raise ValueError("An entire generation stage is missing")
    expected_batches = math.ceil(len(ids) / manifest["batch_size"])
    stages, batch_measurements = [], []
    for (mode, seed), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["batch_index"])
        if [row["batch_index"] for row in rows] != list(range(expected_batches)):
            raise ValueError("Missing/duplicate generation batch")
        if [sample["id"] for batch in rows for sample in batch["samples"]] != ids:
            raise ValueError("Generation batch population/order changed")
        cap = manifest[f"{mode}_max_tokens"] if mode != "direct" else manifest["direct_max_tokens"]
        for batch in rows:
            check_finite_positive(batch["wall_seconds"], "generation wall")
            if len(batch["input_lengths"]) != len(batch["samples"]):
                raise ValueError("Generation prompt/sample count mismatch")
            if any(len(sample["token_ids"]) > cap for sample in batch["samples"]):
                raise ValueError("Generation exceeded the frozen token cap")
            if abs(batch["ended_epoch"] - batch["started_epoch"] - batch["wall_seconds"]) > 1e-6:
                raise ValueError("Generation timing span differs from recorded wall time")
            output_lengths = [len(sample["token_ids"]) for sample in batch["samples"]]
            batch_measurements.append({"mode": mode, "seed": seed, "batch_index": batch["batch_index"],
                                       "record_ids": [sample["id"] for sample in batch["samples"]],
                                       "wall_seconds": batch["wall_seconds"],
                                       "started_epoch": batch["started_epoch"], "ended_epoch": batch["ended_epoch"],
                                       "logical_prompt_lengths": batch["input_lengths"],
                                       "output_lengths_including_eos": output_lengths,
                                       "output_tokens_including_eos": sum(output_lengths),
                                       "maximum_output_token_budget": len(output_lengths) * cap,
                                       "output_token_cap_per_sample": cap})
        stages.append({"mode": mode, "seed": seed, "batches": len(rows),
                       "samples": sum(len(batch["samples"]) for batch in rows),
                       "measured_generate_seconds": math.fsum(batch["wall_seconds"] for batch in rows),
                       "observed_stage_span_seconds": max(batch["ended_epoch"] for batch in rows)
                       - min(batch["started_epoch"] for batch in rows),
                       "output_tokens_including_eos": sum(len(sample["token_ids"]) for batch in rows for sample in batch["samples"]),
                       "logical_prompt_tokens": sum(sum(batch["input_lengths"]) for batch in rows),
                       "padded_prompt_positions": sum(max(batch["input_lengths"]) * len(batch["input_lengths"]) for batch in rows),
                       "output_token_cap_per_sample": cap})
    allocations = {}
    for control, selector in (
        ("greedy", lambda stage: stage["mode"] == "greedy"),
        ("direct4", lambda stage: stage["mode"] == "direct" and stage["seed"] in PRIMARY),
        ("direct8", lambda stage: stage["mode"] == "direct"),
        ("freeform_plus_reformat", lambda stage: stage["mode"] in {"freeform", "reformat"}),
    ):
        selected = [stage for stage in stages if selector(stage)]
        allocations[control] = {field: math.fsum(stage[field] for stage in selected)
                                for field in ("measured_generate_seconds", "observed_stage_span_seconds")}
        allocations[control].update({field: sum(stage[field] for stage in selected)
                                    for field in ("batches", "samples", "output_tokens_including_eos",
                                                  "logical_prompt_tokens", "padded_prompt_positions")})
        allocations[control]["maximum_output_token_budget"] = sum(
            stage["samples"] * stage["output_token_cap_per_sample"] for stage in selected)
    return {"run": directory.name, "model": manifest["model"], "population": len(ids),
            "batch_count": len(batches), "stage_count": len(stages), "generation_dtype": manifest["dtype"],
            "inputs_sha256": manifest["inputs_sha256"],
            "allocations": allocations, "stages": stages, "batch_measurements": batch_measurements,
            "batch_source_sha256": {path.name: sha(path) for path in batch_paths},
            "measured_generate_seconds_all_stages": math.fsum(batch["wall_seconds"] for batch in batches),
            "observed_generation_span_seconds": max(batch["ended_epoch"] for batch in batches)
            - min(batch["started_epoch"] for batch in batches),
            "cold_checkpoint_load_and_initial_setup_seconds": None,
            "cold_setup_scope": "not instrumented by the frozen generator; never inferred from filesystem timestamps",
            "measurement_scope": "synchronized model.generate windows; excludes prompt tokenization/H2D, output decoding/parsing, disk I/O, checkpoint loading and initial setup",
            "span_scope": "first to last recorded forward window; includes intervening work, excludes initial and final tails",
            "allocation_scope": "recorded costs of executed stages, not randomized generation arm comparisons; direct4 is a subset of direct8 and must not be summed twice"}


def ranking_summary(directory: Path) -> tuple[dict, dict]:
    receipt_path = directory / "pre_gold_freeze.json"
    receipt = read_json(receipt_path)
    manifest = read_json(directory / "ranking_manifest.json")
    quality = read_json(directory / "evaluation.json")
    if (receipt.get("answers_accessed") is not False or receipt.get("stage") != "pre-gold"
            or sha(directory / "ranking_manifest.json") != receipt["ranking_manifest_sha256"]
            or sha(directory / "predictions.json") != receipt["predictions"]["sha256"]
            or sha(receipt_path) != quality["receipt_sha256"]
            or quality["predictions_sha256"] != receipt["predictions"]["sha256"]
            or quality["candidates_sha256"] != receipt["candidates"]["sha256"]
            or receipt["inputs_sha256"] != INPUT_SHA256
            or quality["inputs_sha256"] != receipt["inputs_sha256"]
            or manifest["inputs_sha256"] != receipt["inputs_sha256"]
            or manifest["candidates_sha256"] != receipt["candidates"]["sha256"]):
        raise ValueError("Ranking/evaluation receipt mismatch")
    if (manifest["dtype"] != "float32" or manifest["attention"] != "eager" or manifest["tf32"] is not False
            or quality["model"] != manifest["model"] or manifest["selected_method"] != "full"
            or manifest["token_error_tolerance"] != 1e-3):
        raise ValueError("Scoring precision/model differs from the amended protocol")
    predictions = read_json(directory / "predictions.json")
    if predictions["selected_scoring_method"] != "full":
        raise ValueError("Quality predictions must use the amended full FP32 reference")
    items = []
    for relative, digest in receipt["item_sha256"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != digest:
            raise ValueError("Raw ranking item hash mismatch")
        item = read_json(path)
        if item.get("valid") is not True:
            raise ValueError("A failed ranking item cannot enter a completed study")
        if (type(item.get("fast_path_valid")) is not bool
                or item["fast_path_valid"] != all(row["valid"] is True for row in item["comparisons"])):
            raise ValueError("Fast-path diagnostic flag is missing or inconsistent")
        items.append(item)
    if (len(items) != receipt["record_count"] or len(items) != quality["population"]
            or [item["id"] for item in items] != manifest["record_ids"]
            or [row["id"] for row in predictions["records"]] != manifest["record_ids"]
            or [row["id"] for row in quality["records"]] != manifest["record_ids"]
            or (quality["population"], quality["development_count"], quality["test_count"]) != (119, 20, 99)):
        raise ValueError("Ranking/evaluation population mismatch")
    active = [item for item in items if item["candidate_count"] > 0]
    costs = {}
    for method in ("full", "cached", "tree"):
        rows = [item["scoring"][method] for item in items]
        costs[method] = {"complete_score_wall_seconds": math.fsum(row["complete_score_wall_seconds"] for row in rows),
                         "maximum_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
                         **{field: sum(row["scores"][field] for row in rows) for field in
                            ("prompt_tokens", "independent_suffix_tokens", "scored_edges",
                             "processed_prompt_positions", "processed_query_positions")}}
    comparisons = {}
    for alternative in ("cached", "tree"):
        rows = [next(row for row in item["comparisons"] if row["alternative"] == alternative) for item in active]
        for row in rows:
            if (type(row.get("valid")) is not bool
                    or any(type(row[field]) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0
                           for field in ("maximum_token_error", "maximum_sequence_error"))):
                raise ValueError("Malformed/nonfinite fast-path diagnostic")
        methods = sorted({method for row in rows for method in row["winner_agreement"]})
        agreement = {method: {"agreements": sum(row["winner_agreement"][method] for row in rows),
                              "compared_nonempty_items": len(rows),
                              "fraction": sum(row["winner_agreement"][method] for row in rows) / len(rows) if rows else None}
                     for method in methods}
        comparisons[alternative] = {"reference": "full", "nonempty_items": len(rows),
                                     "maximum_token_error": max((row["maximum_token_error"] for row in rows), default=None),
                                     "maximum_sequence_error": max((row["maximum_sequence_error"] for row in rows), default=None),
                                     "numerical_gate_tolerance": manifest["token_error_tolerance"],
                                     "gate_passed_nonempty_items": sum(row["valid"] for row in rows),
                                     "gate_failed_nonempty_items": sum(not row["valid"] for row in rows),
                                     "gate_failed_ids": [item["id"] for item, row in zip(active, rows, strict=True) if not row["valid"]],
                                     "winner_disagreement_ids": [item["id"] for item, row in zip(active, rows, strict=True)
                                                                 if not all(row["winner_agreement"].values())],
                                     "winner_agreement_by_aggregator": agreement}
    summary = {"run": directory.name, "model": manifest["model"], "population": len(items),
               "scored_nonempty_items": len(active), "zero_family_items": len(items) - len(active),
               "selected_scoring_method": "full",
               "combined_fast_path_gate_failed_nonempty_items": sum(not item["fast_path_valid"] for item in active),
               "combined_fast_path_gate_failed_ids": [item["id"] for item in active if not item["fast_path_valid"]],
               "fast_path_scope": "cached/tree scores are diagnostics, including failed fixed numerical gates; full FP32 reference supplies every quality prediction",
               "zero_family_scope": "retained failures; agreement on null predictions is not a numerical scoring comparison",
               "reducer": manifest["reducer"], "sequence_batch_size": manifest["batch_size"],
               "scoring_dtype": manifest["dtype"], "attention": manifest["attention"], "tf32": manifest["tf32"],
               "costs": costs, "comparisons": comparisons,
               "per_item_measurements": [{"id": item["id"], "split": item["split"],
                   "candidate_count": item["candidate_count"],
                   "fast_path_valid": item["fast_path_valid"],
                   "costs": {method: {"complete_score_wall_seconds": item["scoring"][method]["complete_score_wall_seconds"],
                                      "peak_allocated_bytes": item["scoring"][method]["peak_allocated_bytes"],
                                      **{field: item["scoring"][method]["scores"][field] for field in
                                         ("prompt_tokens", "independent_suffix_tokens", "scored_edges",
                                          "processed_prompt_positions", "processed_query_positions")}}
                             for method in ("full", "cached", "tree")},
                   "comparisons": [{field: row[field] for field in
                                    ("reference", "alternative", "maximum_token_error", "maximum_sequence_error",
                                     "winner_agreement", "reference_winner_margins", "alternative_winner_margins", "valid")}
                                   for row in item["comparisons"]]} for item in items],
               "cost_scope": "descriptive one-pass full-population execution including parse/packing/H2D/extraction/aggregation; not randomized performance evidence",
               "cold_checkpoint_load_and_setup_seconds": None,
               "cold_setup_scope": "outside per-item scoring windows and not separately instrumented",
               "pre_gold_receipt_sha256": sha(receipt_path), "candidates_sha256": receipt["candidates"]["sha256"],
               "generation_manifest_sha256": manifest["generation_manifest_sha256"]}
    return summary, quality


def audit_summary(directory: Path, ranking: Path) -> dict:
    """Check the complete stored-vector FULL/LSE versus fresh FULL/Triton audit."""
    completed = read_json(directory / "completed.json")
    summary_path, manifest_path = directory / "summary.json", directory / "manifest.json"
    summary, manifest = read_json(summary_path), read_json(manifest_path)
    if ((directory / "failure.json").exists() or completed.get("valid") is not True
            or completed.get("population_complete") is not True or completed.get("answers_accessed") is not False
            or completed["summary_sha256"] != sha(summary_path)
            or summary.get("valid") is not True or summary.get("population_complete") is not True
            or summary.get("answers_accessed") is not False or summary["manifest_sha256"] != sha(manifest_path)
            or any(summary.get(field) is not True for field in
                   ("source_unchanged", "reference_unchanged", "model_metadata_unchanged"))
            or summary["record_count"] != 119 or summary["gate_failed_items"] != 0 or summary["gate_failed_ids"]):
        raise ValueError("Whole-population normalizer audit is incomplete or invalid")
    if (manifest.get("answers_accessed") is not False or manifest["inputs_sha256"] != INPUT_SHA256
            or manifest["dtype"] != "float32" or manifest["attention"] != "eager" or manifest["tf32"] is not False
            or GPU_UUID not in manifest["gpu"] or manifest["expected_gpu_uuid"] != GPU_UUID
            or manifest["method"] != "full" or manifest["reference_reducer"] != "torch_lse"
            or manifest["alternative_reducer"] != "triton" or manifest["batch_size"] != 8 or manifest["head_chunk"] != 32
            or manifest["token_error_tolerance"] != 3e-5 or summary["token_error_tolerance"] != 3e-5
            or manifest["record_count"] != 119 or set(manifest["aggregators"]) != set(AUDIT_AGGREGATORS)
            or manifest["ranking_run"] != ranking.name):
        raise ValueError("Audit model/precision/backbone/reducer/batch/threshold contract differs")
    reference_manifest = read_json(ranking / "ranking_manifest.json")
    if (manifest["model"] != reference_manifest["model"] or manifest["record_ids"] != reference_manifest["record_ids"]
            or reference_manifest["selected_method"] != "full" or reference_manifest["reducer"] != "torch_lse"
            or reference_manifest["batch_size"] != 8 or reference_manifest["head_chunk"] != 32):
        raise ValueError("Audit checkpoint differs from its stored full reference")
    sources = {"completed.json": sha(directory / "completed.json"),
               "summary.json": sha(summary_path), "manifest.json": sha(manifest_path)}
    for relative, digest in manifest["source_sha256"].items():
        path = (directory / "source" / relative).resolve()
        if not path.is_relative_to((directory / "source").resolve()) or sha(path) != digest:
            raise ValueError("Audit retained implementation snapshot hash mismatch")
        sources["source/" + relative] = digest
    for name in ("bfcl", "ranking", "serialization", "scoring", "kernels", "trie", "gpu_guard"):
        relative = f"src/orbitcall/{name}.py"
        if (relative not in manifest["source_sha256"] or relative not in reference_manifest["source_sha256"]
                or manifest["source_sha256"][relative] != reference_manifest["source_sha256"][relative]):
            raise ValueError("Audit and stored reference use different runtime implementations")
    required_reference_paths = {"pre_gold_freeze.json", "ranking_manifest.json", "predictions.json",
                                *(f"items/{index:04d}.json" for index in range(119))}
    if set(manifest["reference_sha256"]) != required_reference_paths:
        raise ValueError("Audit retained-reference hash chain is incomplete")
    for relative, digest in manifest["reference_sha256"].items():
        path = (ranking / relative).resolve()
        if not path.is_relative_to(ranking.resolve()) or sha(path) != digest:
            raise ValueError("Audit stored reference artifact hash mismatch")
    checkpoint = manifest["checkpoint"]
    receipt_name = {"Qwen2.5-3B-Instruct": "model_3b_verified.json",
                    "Qwen2.5-1.5B-Instruct": "model_1p5b_verified.json"}[manifest["model"]]
    receipt_path, public_path = ROOT / "metadata" / receipt_name, ROOT / "metadata" / "public_models.json"
    receipt, public = read_json(receipt_path), read_json(public_path)
    declarations = [row for row in public["models"] if row["repo_id"].split("/")[-1] == manifest["model"]]
    if (checkpoint.get("all_public_files_match") is not True or receipt.get("all_public_files_match") is not True
            or checkpoint["receipt_file"] != receipt_name or checkpoint["receipt_sha256"] != sha(receipt_path)
            or checkpoint["public_manifest_sha256"] != sha(public_path) or len(declarations) != 1
            or (checkpoint["repo_id"], checkpoint["revision"]) != (receipt["repo_id"], receipt["revision"])
            or (checkpoint["repo_id"], checkpoint["revision"]) != (declarations[0]["repo_id"], declarations[0]["revision"])):
        raise ValueError("Audit public checkpoint provenance/receipt is inconsistent")
    checkpoint_files = {row["path"]: (row["bytes"], row["sha256"]) for row in checkpoint["files"]}
    receipt_files = {row["path"]: (row["bytes"], row["sha256"]) for row in receipt["files"]}
    if (len(checkpoint_files) != len(checkpoint["files"]) or len(receipt_files) != len(receipt["files"])
            or set(checkpoint_files) != {row["path"] for row in declarations[0]["files"]} or checkpoint_files != receipt_files
            or any(row.get("matches_public_revision") is not True for row in receipt["files"])):
        raise ValueError("Audit verified checkpoint file receipt is incomplete or altered")
    paths = [f"items/{index:04d}.json" for index in range(119)]
    if set(summary["item_sha256"]) != set(paths):
        raise ValueError("Audit raw checkpoints do not cover all119 items")
    rows = []
    for index, relative in enumerate(paths):
        path = directory / relative
        if sha(path) != summary["item_sha256"][relative]:
            raise ValueError("Audit raw item hash mismatch")
        item, reference = read_json(path), read_json(ranking / relative)
        comparison = item["comparison"]
        if (item["id"] != manifest["record_ids"][index] or item["id"] != reference["id"]
                or any(item[field] != reference[field] for field in ("split", "category", "schema_group", "candidate_count"))
                or item["source_item_sha256"] != manifest["reference_sha256"][relative]
                or item["reference"] != reference["scoring"]["full"]
                or item["prompt_token_ids"] != reference["prompt_token_ids"]
                or item["sequence_token_ids"] != reference["sequence_token_ids"] or item["families"] != reference["families"]
                or item.get("valid") is not True or comparison.get("valid") is not True
                or item["alternative"]["reducer"] != "triton" or item["alternative"]["scores"]["method"] != "full"
                or comparison["reference"] != "full_torch_lse" or comparison["alternative"] != "full_triton"
                or comparison["token_error_tolerance"] != 3e-5
                or set(comparison["winner_agreement"]) != set(AUDIT_AGGREGATORS)
                or not all(value is True for value in comparison["winner_agreement"].values())):
            raise ValueError("Audit raw item violates reference integrity or winner agreement")
        if any(type(comparison[field]) not in (int, float) or not math.isfinite(comparison[field]) or comparison[field] < 0
               for field in ("maximum_token_error", "maximum_sequence_error")) or comparison["maximum_token_error"] > 3e-5:
            raise ValueError("Audit numerical error is nonfinite or outside its unchanged threshold")
        sequences = item["sequence_token_ids"]
        if (len(comparison["sequence_errors"]) != len(sequences)
                or comparison["sequence_error_bounds"] != [len(sequence) * 3e-5 for sequence in sequences]
                or any(not math.isfinite(error) or error > bound + 1e-8 for error, bound in
                       zip(comparison["sequence_errors"], comparison["sequence_error_bounds"], strict=True))):
            raise ValueError("Audit complete sequence error exceeds its fixed EOS-inclusive length bound")
        rows.append(item)
        sources[relative] = summary["item_sha256"][relative]
    if (len(rows), sum(row["split"] == "development" for row in rows), sum(row["split"] == "test" for row in rows)) != (119, 20, 99):
        raise ValueError("Audit complete population/split differs from the frozen study")
    active = [row for row in rows if row["candidate_count"] > 0]
    if not active or len(active) != summary["scored_nonempty_items"] or 119 - len(active) != summary["zero_family_items"]:
        raise ValueError("Audit nonempty/null-family denominators are inconsistent")
    for field in ("maximum_token_error", "maximum_sequence_error"):
        if max(row["comparison"][field] for row in active) != summary[field]:
            raise ValueError("Audit maximum error summary differs from raw nonempty items")
    if set(summary["winner_agreement_by_aggregator"]) != set(AUDIT_AGGREGATORS):
        raise ValueError("Audit summary omits a declared same-score aggregator")
    agreements = {}
    for method in AUDIT_AGGREGATORS:
        raw_count = sum(row["comparison"]["winner_agreement"][method] for row in active)
        stored = summary["winner_agreement_by_aggregator"][method]
        if stored != {"agreements": raw_count, "compared_nonempty_items": len(active), "disagreement_ids": []}:
            raise ValueError("Audit winner-count summary differs from actual nonempty items")
        agreements[method] = {**stored, "fraction": raw_count / len(active)}
    elapsed = check_finite_positive(summary["elapsed_seconds"], "audit process elapsed")
    return {"run": directory.name, "model": manifest["model"], "valid": True,
            "population": 119, "development_count": 20, "test_count": 99,
            "scored_nonempty_items": len(active), "zero_family_items": 119 - len(active),
            "zero_family_scope": "retained null outputs, excluded from numerical comparison denominators",
            "maximum_token_error": summary["maximum_token_error"], "maximum_sequence_error": summary["maximum_sequence_error"],
            "winner_agreement_by_aggregator": agreements, "token_error_tolerance": 3e-5,
            "batch_size": 8, "head_chunk": 32, "reference_reducer": "torch_lse", "alternative_reducer": "triton",
            "process_elapsed_seconds": elapsed,
            "descriptive_alternative_score_window_seconds": math.fsum(row["alternative"]["descriptive_score_wall_seconds"] for row in rows),
            "descriptive_scope": "fresh alternative score windows only; stored reference costs and audit cost are not a randomized performance pair",
            "checkpoint": checkpoint, "input_sha256": manifest["inputs_sha256"],
            "pre_gold_receipt_sha256": manifest["reference_sha256"]["pre_gold_freeze.json"],
            "claim_scope": summary["claim_scope"], "source_artifact_sha256": sources}


def failed_ranking_summary(directory: Path) -> dict:
    """Account retained prefix checkpoints without producing quality outputs."""
    manifest_path = directory / "ranking_manifest.json"
    manifest = read_json(manifest_path)
    if ((directory / "pre_gold_freeze.json").exists()
            or manifest.get("answers_accessed") is not False
            or manifest["inputs_sha256"] != INPUT_SHA256
            or manifest["dtype"] != "float32" or manifest["attention"] != "eager"
            or manifest["tf32"] is not False or manifest["token_error_tolerance"] != 1e-3):
        raise ValueError("Failed ranking history is outside the retained pre-gold FP32 protocol")
    paths = sorted((directory / "items").glob("*.json"))
    if not paths or [path.name for path in paths] != [f"{index:04d}.json" for index in range(len(paths))]:
        raise ValueError("Retained failed ranking checkpoints must form the original contiguous prefix")
    items = [read_json(path) for path in paths]
    manifest_hash = sha(manifest_path)
    if any(item["id"] != manifest["record_ids"][index]
           or item["ranking_manifest_sha256"] != manifest_hash
           or type(item.get("valid")) is not bool for index, item in enumerate(items)):
        raise ValueError("Failed ranking checkpoint identity/manifest is inconsistent")
    failures = [item for item in items if not item["valid"]]
    if not failures:
        raise ValueError("--failed-rankings must reference retained failed attempts")
    methods = ("full", "cached", "tree")
    costs = {}
    for method in methods:
        rows = [item["scoring"][method] for item in items if method in item["scoring"]]
        for row in rows:
            check_finite_positive(row["complete_score_wall_seconds"], "failed-attempt scorer wall")
        costs[method] = {"completed_measurement_count": len(rows),
                         "unrecorded_item_measurements": len(items) - len(rows),
                         "measured_wall_seconds": math.fsum(row["complete_score_wall_seconds"] for row in rows)}
    return {"run": directory.name, "model": manifest["model"],
            "retained_item_count": len(items), "successful_item_count": sum(item["valid"] for item in items),
            "failed_item_count": len(failures), "failed_ids": [item["id"] for item in failures],
            "unattempted_or_unretained_population_count": len(manifest["record_ids"]) - len(items),
            "per_method_costs": costs,
            "total_measured_scorer_wall_seconds": math.fsum(row["measured_wall_seconds"] for row in costs.values()),
            "token_error_tolerance": manifest["token_error_tolerance"],
            "selected_method_in_failed_attempt": manifest["selected_method"],
            "quality_outputs_created": False,
            "disposition": "initial attempt stopped at its unchanged1e-3 numerical/winner gate; all prefix checkpoints and failed item retained; no quality derived from this attempt",
            "cost_scope": "synchronized completed scorer windows only; lower bound excludes initial model setup, unrecorded interrupted work, guards, comparison checks and I/O",
            "per_item_measurements": [{"id": item["id"], "valid": item["valid"],
                                       "error": item.get("error"), "candidate_count": item.get("candidate_count"),
                                       "measured_wall_seconds_by_method": {
                                           method: item["scoring"][method]["complete_score_wall_seconds"]
                                           for method in methods if method in item["scoring"]},
                                       "comparisons": item.get("comparisons", [])} for item in items],
            "ranking_manifest_sha256": manifest_hash,
            "item_sha256": {f"items/{path.name}": sha(path) for path in paths}}


def failed_benchmark_summary(directory: Path) -> dict:
    """Retain the actual failed study and cause; never derive timing ratios."""
    manifest_path, failure_path = directory / "manifest.json", directory / "failure.json"
    manifest, failure = read_json(manifest_path), read_json(failure_path)
    fixture = read_json(directory / "fixtures.json")
    if ((directory / "completed.json").exists() or failure.get("status") != "invalid_or_incomplete"
            or failure.get("answers_accessed") is not False or manifest.get("answers_accessed") is not False
            or fixture.get("answers_accessed") is not False or fixture["inputs_sha256"] != INPUT_SHA256
            or manifest["dtype"] != "float32" or manifest["attention"] != "eager"
            or manifest["tf32"] is not False or GPU_UUID not in manifest["gpu"]
            or sha(directory / "fixtures.json") != manifest["fixture_file_sha256"]):
        raise ValueError("Failed benchmark history is outside the frozen pre-gold FP32 protocol")
    duration = check_finite_positive(failure["ended_epoch"] - failure["started_epoch"], "failed benchmark process span")
    windows = {}
    for stage in ("scorer", "reducer"):
        raw_path = directory / f"{stage}_raw_windows.jsonl"
        valid_path = directory / f"{stage}_observations.jsonl"
        raw = read_jsonl(raw_path) if raw_path.exists() else []
        valid = read_jsonl(valid_path) if valid_path.exists() else []
        plan_path = directory / f"{stage}_timing_plan.json"
        planned_count = sum(len(block["order"]) for block in read_json(plan_path)) if plan_path.exists() else None
        strict_count = sum(
            bool(row["validity"]) and all(record.get("valid") is True for record in row["validity"].values())
            if stage == "scorer" else row["validity"].get("valid") is True for row in valid)
        windows[stage] = {"retained_raw_window_count": len(raw), "validated_window_count": len(valid),
                          "strictly_validated_window_count": strict_count,
                          "declared_windows_in_retained_plan": planned_count,
                          "raw_window_wall_seconds": math.fsum(row["wall_seconds"] for row in raw)}
    error = failure["error"]
    kind = manifest.get("study_kind", "packed_scorer_study")
    if kind not in {"packed_scorer_study", "full_reference_normalizer_study"}:
        raise ValueError("Unknown failed benchmark study kind")
    if kind == "full_reference_normalizer_study" and "budget exhausted" in error.lower():
        cause = "runtime_budget_exhausted"
        disposition = ("full-reference normalizer study exceeded its declared runtime budget; preserve every partial raw/strict-valid window and process cost; no incomplete-window ratios are accepted; a separately declared retry keeps fixtures, arms, seeds and numerical thresholds unchanged")
    elif kind == "packed_scorer_study" and error.startswith("Numerical/winner gate failed:"):
        cause = "packed_numerical_gate_failure"
        disposition = "original cache/tree numerical gate failed at the unchanged1e-3 tolerance; preserve the negative and process cost; RQ2 is retired with no cache/tree speed claim"
    else:
        cause = "other_incomplete_attempt"
        disposition = "incomplete attempt of the recorded study kind; preserve the actual error and process cost; no speed ratios or quality are accepted"
    diagnostics = None
    if "{" in error:
        try:
            diagnostics = json.loads(error[error.index("{"):])
        except json.JSONDecodeError:
            pass
    selected = [manifest_path, failure_path, directory / "fixtures.json"]
    selected.extend(sorted((directory / "gate_raw").glob("*.json")))
    selected.extend(path for stage in ("scorer", "reducer")
                    for suffix in ("raw_windows.jsonl", "observations.jsonl", "timing_plan.json")
                    if (path := directory / f"{stage}_{suffix}").exists())
    selected.extend(directory / name for name in ("scorer_gate.json", "scorer_warmup.json", "reducer_gate.json")
                    if (directory / name).exists())
    for relative, digest in manifest["source_sha256"].items():
        path = (directory / "source" / relative).resolve()
        if not path.is_relative_to((directory / "source").resolve()) or sha(path) != digest:
            raise ValueError("Retained failed benchmark source snapshot hash mismatch")
        selected.append(path)
    return {"run": directory.name, "model": manifest["model"], "fixture_ids": manifest["fixture_ids"],
            "study_kind": kind, "failure_cause": cause, "declared_max_seconds": manifest["max_seconds"],
            "candidates_sha256": fixture["candidates_sha256"], "inputs_sha256": fixture["inputs_sha256"],
            "recorded_process_wall_seconds": duration, "started_epoch": failure["started_epoch"],
            "ended_epoch": failure["ended_epoch"], "retained_timing_windows": windows,
            "failed_before_any_timing_window": all(row["retained_raw_window_count"] == 0 for row in windows.values()),
            "error": error, "failed_gate_diagnostics": diagnostics,
            "quality_outputs_created": False, "speed_ratios_accepted": False, "accepted_pair_ratio_count": 0,
            "disposition": disposition,
            "interval_scope": "retained failure start-to-end host wall interval, including this attempt's setup and gate work; not pure GPU-busy time",
            "source_artifact_sha256": {str(path.resolve().relative_to(directory.resolve())).replace("\\", "/"): sha(path) for path in selected}}


def cumulative_component_cost(studies: list[dict], failed_rankings: list[dict], failed_benchmarks: list[dict]) -> dict:
    # Each generation shard is counted once. Direct4 is already contained in
    # direct8, so allocation summaries must not be summed for project cost.
    generation = math.fsum(study["generation"]["measured_generate_seconds_all_stages"] for study in studies)
    final_ranking = math.fsum(row["complete_score_wall_seconds"]
                             for study in studies for row in study["ranking"]["costs"].values())
    failed_ranking = math.fsum(run["total_measured_scorer_wall_seconds"] for run in failed_rankings)
    failed_benchmark = math.fsum(run["recorded_process_wall_seconds"] for run in failed_benchmarks)
    audit = math.fsum(study["audit"]["process_elapsed_seconds"] for study in studies)
    benchmark = math.fsum(study["benchmark"]["total_process_wall_seconds"] for study in studies)
    total = math.fsum((generation, final_ranking, failed_ranking, failed_benchmark, audit, benchmark))
    return {"declared_initial_spark_gpu_budget_hours": 4,
            "declared_initial_spark_gpu_budget_seconds": 4 * 3600,
            "final_generation_all_stages_measured_window_seconds": generation,
            "final_ranking_all_methods_measured_window_seconds": final_ranking,
            "retained_failed_ranking_measured_window_seconds": failed_ranking,
            "retained_failed_benchmark_complete_process_interval_seconds": failed_benchmark,
            "final_benchmark_complete_process_interval_seconds": benchmark,
            "final_full_triton_audit_process_interval_seconds": audit,
            "cumulative_measured_component_seconds_lower_bound": total,
            "cumulative_measured_component_hours_lower_bound": total / 3600,
            "pure_gpu_busy_seconds": None,
            "complete_project_host_or_gpu_allocation_seconds": None,
            "complete_budget_compliance_from_this_subtotal": "not established by an incomplete component subtotal",
            "interval_scope": "actual recorded host wall intervals during sequential Spark GPU stages; synchronized windows and benchmark process spans, not an integral of pure GPU utilization",
            "missing_cost_scope": "excludes downloads, initial generation/ranking cold loads, standalone numerical gates not supplied here, unrecorded interrupted work, interstage/controller overhead, and CPU evaluation/analysis; supplied audits and completed/failed benchmark process spans already include their own setup/gates/warmup/validation/I/O"}


def exact_block_bootstrap(values: list[float]) -> dict:
    """Enumerate the 4^4 empirical resamples; there are still only four trials."""
    import numpy as np

    if len(values) != 4 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Exactly four positive paired block ratios are required")
    medians = [statistics.median(values[index] for index in sample)
               for sample in itertools.product(range(4), repeat=4)]
    return {"block_count": 4, "median_block_ratio": statistics.median(values),
            "geometric_mean_block_ratio": math.exp(math.fsum(math.log(value) for value in values) / 4),
            "observed_block_range": [min(values), max(values)],
            "empirical_block_bootstrap_median_95_interval": np.quantile(medians, [0.025, 0.975]).tolist(),
            "empirical_resample_states": 256,
            "uncertainty_scope": "descriptive exact empirical bootstrap over four observed blocks; 256 resamples are not independent trials"}


def benchmark_pairs(directory: Path, stage: str) -> list[dict]:
    observations_path = directory / f"{stage}_observations.jsonl"
    raw_path = directory / f"{stage}_raw_windows.jsonl"
    observations, raw = read_jsonl(observations_path), read_jsonl(raw_path)
    plan = read_json(directory / f"{stage}_timing_plan.json")
    source_summary = read_json(directory / f"{stage}_summary.json")
    if source_summary.get("valid") is not True:
        raise ValueError("Benchmark stage did not complete validly")
    identity = lambda row: (row["pair"], row["block"], row["position"], row["arm"])
    expected = {(row["pair"], row["block"], index, arm)
                for row in plan for index, arm in enumerate(row["order"])}
    if (len({identity(row) for row in observations}) != len(observations)
            or {identity(row) for row in observations} != expected
            or {identity(row) for row in raw} != expected or len(raw) != len(expected)):
        raise ValueError("Raw/timing-plan/validated observation windows are incomplete")
    raw_map = {identity(row): row for row in raw}
    for row in observations:
        if any(row[field] != raw_map[identity(row)][field] for field in
               ("wall_seconds", "cuda_interval_ms", "repeats", "started_epoch", "ended_epoch")):
            raise ValueError("Validated window differs from its retained raw measurement")
        validity = row["validity"]
        if stage == "scorer":
            valid = all(record["valid"] is True for record in validity.values())
        else:
            valid = validity.get("valid") is True
        if not valid:
            raise ValueError("A failed observation cannot enter timing analysis")
    result = []
    for pair in source_summary["pairs"]:
        name, a, b = pair["pair"], pair["arm_a"], pair["arm_b"]
        blocks = []
        for block in range(4):
            rows = [row for row in observations if row["pair"] == name and row["block"] == block]
            if len(rows) != 4 or any(sum(row["arm"] == arm for row in rows) != 2 for arm in (a, b)):
                raise ValueError("Incomplete four-window ABBA block")
            ordered = sorted(rows, key=lambda row: row["position"])
            if [row["arm"] for row in ordered] not in ([a, b, b, a], [b, a, a, b]):
                raise ValueError("Window order is not ABBA/BAAB")
            medians = {arm: {metric: statistics.median(
                check_finite_positive(row[metric], metric) / check_finite_positive(row["repeats"], "repeats")
                for row in rows if row["arm"] == arm) for metric in ("wall_seconds", "cuda_interval_ms")}
                       for arm in (a, b)}
            blocks.append({"block": block, "order": [row["arm"] for row in ordered],
                           "medians_per_call": medians,
                           "wall_ratio_a_over_b": medians[a]["wall_seconds"] / medians[b]["wall_seconds"],
                           "cuda_ratio_a_over_b": medians[a]["cuda_interval_ms"] / medians[b]["cuda_interval_ms"]})
        wall = exact_block_bootstrap([row["wall_ratio_a_over_b"] for row in blocks])
        cuda = exact_block_bootstrap([row["cuda_ratio_a_over_b"] for row in blocks])
        if (abs(wall["median_block_ratio"] - pair["median_block_wall_ratio_a_over_b"]) > 1e-10
                or abs(cuda["median_block_ratio"] - pair["median_block_cuda_ratio_a_over_b"]) > 1e-10):
            raise ValueError("Recomputed block ratio differs from frozen benchmark summary")
        result.append({"pair": name, "arm_a": a, "arm_b": b, "blocks": blocks,
                       "wall": wall, "cuda": cuda,
                       "ratio_direction": "A time / B time; above 1 means B is faster",
                       "timed_window_count": 16,
                       "maximum_incremental_allocated_bytes_by_arm": {
                           arm: max(row["incremental_peak_allocated_bytes"] for row in observations
                                    if row["pair"] == name and row["arm"] == arm) for arm in (a, b)}})
    return result


def benchmark_summary(directory: Path) -> dict:
    completed = read_json(directory / "completed.json")
    manifest = read_json(directory / "manifest.json")
    if ((directory / "failure.json").exists() or completed.get("status") != "valid_development_benchmark"
            or completed.get("kind") != "full_reference_normalizer_study"
            or completed.get("source_unchanged") is not True or completed.get("candidates_unchanged") is not True
            or manifest["dtype"] != "float32" or manifest["attention"] != "eager"
            or manifest["tf32"] is not False or GPU_UUID not in manifest["gpu"]
            or completed.get("answers_accessed") is not False or manifest.get("answers_accessed") is not False):
        raise ValueError("Benchmark is incomplete, invalid, or outside the declared host/precision")
    scorer = benchmark_pairs(directory, "scorer")
    reducer = benchmark_pairs(directory, "reducer")
    if {row["pair"] for row in scorer} != {"full_torch_vs_lse", "full_torch_vs_ce", "full_torch_vs_triton", "full_lse_vs_triton", "full_ce_vs_triton"}:
        raise ValueError("Missing declared full-reference normalizer comparison")
    if (set(manifest["scorer_arms"]) != {"full", "full_lse", "full_ce", "full_triton"}
            or any(configuration[0] != "full" for configuration in manifest["scorer_arms"].values())):
        raise ValueError("Every timed scorer arm must use the same full reference backbone")
    if {row["pair"] for row in reducer} != {"torch_vs_lse", "torch_vs_ce", "torch_vs_triton", "lse_vs_triton", "ce_vs_triton"}:
        raise ValueError("Missing strong paired reducer comparison")
    windows = read_jsonl(directory / "scorer_observations.jsonl") + read_jsonl(directory / "reducer_observations.jsonl")
    timed = math.fsum(row["wall_seconds"] for row in windows)
    if timed > completed["total_wall_seconds"] + 1e-6:
        raise ValueError("Timed windows exceed total benchmark wall span")
    fixture = read_json(directory / "fixtures.json")
    if (sha(directory / "fixtures.json") != manifest["fixture_file_sha256"]
            or fixture.get("answers_accessed") is not False or fixture["inputs_sha256"] != INPUT_SHA256
            or fixture["fallback"] != "none"):
        raise ValueError("Timing fixture receipt differs from the frozen input-only selection")
    scorer_gate, reducer_gate = read_json(directory / "scorer_gate.json"), read_json(directory / "reducer_gate.json")
    if scorer_gate.get("passed") is not True or reducer_gate.get("passed") is not True:
        raise ValueError("Benchmark correctness gates did not pass")
    reducer_fixture = read_json(directory / "real_reducer_fixture.json")
    if (reducer_fixture.get("synthetic") is not False
            or reducer_fixture.get("backbone_method") != "full"
            or reducer_fixture["row_ids"] != list(range(reducer_fixture["shape"][0]))
            or reducer_fixture["requested_edges"] != reducer_fixture["shape"][0]):
        raise ValueError("Normalizer microbenchmark must use actual full logits with unique parent rows")
    direct_ce_error = reducer_gate.get("direct_ce_max_error", reducer_gate.get("direct_ce_anchor_max_error"))
    if direct_ce_error is None:
        raise ValueError("Missing direct native-CE numerical control")
    return {"run": directory.name, "model": manifest["model"], "kind": completed["kind"],
            "fixture_ids": manifest["fixture_ids"], "selected_fixture_count": manifest["selected_fixture_count"],
            "eligible_development_fixture_count": manifest["eligible_fixture_count"],
            "fixture_selection": fixture["selection"], "fallback": fixture["fallback"],
            "candidate_sha256": fixture["candidates_sha256"], "input_sha256": fixture["inputs_sha256"],
            "total_process_wall_seconds": completed["total_wall_seconds"], "timed_window_wall_seconds": timed,
            "nontimed_setup_gate_warmup_validation_io_and_other_seconds": completed["total_wall_seconds"] - timed,
            "cold_model_load_seconds": None,
            "cold_scope": "benchmark process span includes model load/setup, gates, warmup, validation and I/O; cold load is not separately instrumented",
            "scorer_pairs": scorer, "reducer_pairs": reducer,
            "scorer_cuda_scope": "device timeline brackets include host-induced gaps; not active GPU compute",
            "reducer_scope": "one actual resident LM-head projection chunk; excludes model projection/backbone/generation",
            "reducer_fixture": reducer_fixture,
            "maximum_scorer_gate_token_error": max(record["max_token_error"]
                for row in scorer_gate["records"] for record in row["comparisons"].values()),
            "maximum_reducer_gate_error": max(row["max_error"] for row in reducer_gate["arms"].values()),
            "direct_native_ce_max_error": direct_ce_error,
            "cache_tree_speed_claim": False,
            "normalizer_scope": "all scorer arms use independent full FP32 reference likelihood; only full-vocabulary requested-token reduction changes"}


def percent(value) -> str:
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def interval(values, scale=1.0) -> str:
    return "unavailable" if values is None else f"[{values[0] * scale:.4g}, {values[1] * scale:.4g}]"


def make_markdown(studies: list[dict], runtime: dict, failed_rankings: list[dict], failed_benchmarks: list[dict], cumulative_cost: dict) -> str:
    lines = ["# OrbitCall completed study results", "",
             "This report is generated from frozen artifacts on DGX Spark. It reports both models and all controls; no best-model or test-subset selection is used.", "",
             "Candidate generation uses BF16/SDPA. All likelihood comparisons use FP32/eager with TF32 disabled. Earlier BF16 full/cache disagreement (up to 0.75 nats/token) failed the numerical gate and remains a failure, not a positive result.", "",
             "The initial all-item FP32 fast-path run also failed the unchanged 1e-3 nats/token gate. A prospective amendment before labels or timing makes full FP32 reference scoring the quality-prediction source for both models. Cached/tree comparisons retain their failures as diagnostics; no all-item exact fast-path or cheap production guard is claimed.", "",
             "The original cache/tree development benchmark then failed its strict numerical gate before any timing. RQ2 has no accepted cache/tree speed result. A separate RQ3 normalizer study keeps the same complete full FP32 backbone, frozen development fixtures and four paired blocks while comparing Torch log-softmax, LSE, direct native cross-entropy and Triton requested-token normalization.", "",
             "The score is mass over a restricted 2!/3! JSON argument-order family including one explicit EOS. It is not a complete semantic posterior or a universal accuracy guarantee.", "",
             "## Task quality", "",
             "Accuracy below uses the supported scalar BFCL AST/value scope, not the full official BFCL leaderboard. Unsupported labels and zero-candidate questions remain reported in the frozen population.", ""]
    for study in studies:
        quality = study["quality"]
        test = quality["accuracy"]["test"]
        lines += [f"### {study['model']}", "",
                  f"Frozen population {quality['population']}: {quality['development_count']} development and {quality['test_count']} test; unsupported labels {len(quality['unsupported_ids'])}. Development-selected order: `{quality['selected_fixed_policy']}`.", "",
                  "| Method | Correct / supported test | Accuracy | Question Wilson 95% interval |",
                  "| --- | --- | --- | --- |"]
        for method in SCORE_METHODS:
            row = test[method]
            lines.append(f"| {METHOD_LABELS.get(method, method)} | {row['successes']} / {row['supported']} | {percent(row['supported_accuracy'])} | {interval(row['question_wilson_95_interval'], 100)} percentage points |")
        comparison = quality["primary_heldout_comparison"]
        strongest = quality["strongest_observed_test_equal_score_control"]
        lines += ["", f"The development-selected held-out comparator is `{quality['dev_selected_equal_score_control']}`. Mass-minus-control accuracy difference: {percent(comparison['accuracy_difference'])}; paired question-bootstrap interval {interval(comparison['question_bootstrap_95_interval'], 100)} percentage points, schema-bootstrap interval {interval(comparison['schema_bootstrap_95_interval'], 100)} percentage points.",
                  f"The strongest observed test same-score control is `{strongest}` ({percent(test[strongest]['supported_accuracy'])}); this oracle test comparison is descriptive. All five paired equal-score comparisons and within-model Holm p values are preserved in `summary.json`.", "",
                  f"Primary candidate coverage: {quality['candidate_coverage']['with_primary_candidate']}/{quality['population']} with a valid call; {quality['candidate_coverage']['zero_candidate']} zero-candidate items. JSON-and-schema valid samples: {quality['candidate_coverage']['valid_primary_samples']}/{quality['candidate_coverage']['primary_samples']}. Oracle-correct primary pool coverage: {quality['candidate_coverage']['oracle_correct_primary_candidate']}/{quality['candidate_coverage']['oracle_metric_denominator']} supported items.", ""]
    lines += ["## Numerical agreement and descriptive full-pool costs", "",
              "Null predictions on zero-family items are not counted as actual numerical scoring comparisons. Every nonempty item was compared. The numerical gate remains 1e-3 nats/token; failed fast paths are reported and are not required for accepting the independent full reference output. These one-pass costs are not randomized speedup evidence.", ""]
    for study in studies:
        ranking = study["ranking"]
        lines += [f"### {study['model']}", "", f"Scored nonempty items: {ranking['scored_nonempty_items']}; retained zero-family items: {ranking['zero_family_items']}.", "",
                  "| Alternative vs full | Maximum token error (nats) | Maximum sequence error (nats) | Failed family gates | Mass winner agreement |",
                  "| --- | --- | --- | --- | --- |"]
        for method, row in ranking["comparisons"].items():
            count = row["winner_agreement_by_aggregator"].get("mass", {"agreements": 0, "compared_nonempty_items": 0, "fraction": None})
            lines.append(f"| {method} | {row['maximum_token_error']} | {row['maximum_sequence_error']} | {row['gate_failed_nonempty_items']}/{row['nonempty_items']} | {count['agreements']}/{count['compared_nonempty_items']} ({percent(count['fraction'])}) |")
        lines += ["", f"Combined cached/tree gate failures: {ranking['combined_fast_path_gate_failed_nonempty_items']}/{ranking['scored_nonempty_items']} nonempty families. IDs: {', '.join(ranking['combined_fast_path_gate_failed_ids']) or 'none'}. Quality selection uses full FP32 reference scores for every item."]
        lines += ["", "| Scorer | All-item measured seconds | Prompt positions | Query positions | Scored edges |",
                  "| --- | --- | --- | --- | --- |"]
        for method, row in ranking["costs"].items():
            lines.append(f"| {method} | {row['complete_score_wall_seconds']:.3f} | {row['processed_prompt_positions']} | {row['processed_query_positions']} | {row['scored_edges']} |")
        lines += ["", "All aggregator winner counts, raw query/token counts, reducer choice, peak allocations and source/hash receipts are retained in the JSON and raw item evidence.", ""]
    lines += ["## Whole-population FULL/Triton numerical audit", "",
              "This separate answer-free audit compares stored FULL/Torch-LSE vectors with fresh FULL/Triton vectors at batch8/chunk32, FP32/eager and TF32 disabled. It uses every frozen item, the same complete EOS-inclusive family and an unchanged3e-5 nats/token threshold. Null families remain in the population and are excluded from numerical agreement counts.", "",
              "| Model | Complete population (dev/test) | Nonempty / zero families | Maximum token error (nats) | Maximum sequence error (nats) | Mass winner agreement | Audit elapsed seconds |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for study in studies:
        audit = study["audit"]
        agreement = audit["winner_agreement_by_aggregator"]["mass"]
        lines.append(f"| {audit['model']} | {audit['population']} ({audit['development_count']}/{audit['test_count']}) | {audit['scored_nonempty_items']} / {audit['zero_family_items']} | {audit['maximum_token_error']:.8g} | {audit['maximum_sequence_error']:.8g} | {agreement['agreements']}/{agreement['compared_nonempty_items']} ({percent(agreement['fraction'])}) | {audit['process_elapsed_seconds']:.3f} |")
    lines += ["", "All eleven aggregator agreement counts, verified public checkpoint receipts, raw-item and retained-source hashes are preserved under each study's `audit` in [summary.json](summary.json). Audit elapsed cost is counted once in the cumulative subtotal. Its alternative scoring windows are descriptive and are not compared as a randomized timing pair against stored reference costs. This empirical full-path audit establishes neither bitwise identity, an online production guard, nor cache/tree correctness.", "",
              "## Full-reference normalizer randomized ABBA timing", "",
              "Each pair has **four paired blocks**, with two windows per arm per block. Ratios are A time / B time, so values above one favor B. We enumerate the 256 empirical resamples of those four blocks for a descriptive bootstrap interval; these are not 256 independent trials. The observed block range is also reported. Fixtures are fixed development IDs selected without answers, with no test fallback.", ""]
    for study in studies:
        benchmark = study["benchmark"]
        lines += [f"### {study['model']}", "",
                  f"Fixtures: {', '.join(benchmark['fixture_ids'])}; selected {benchmark['selected_fixture_count']} from {benchmark['eligible_development_fixture_count']} eligible development items.", "",
                  "| Stage / pair (A / B) | Wall median ratio | Wall block range | Descriptive block-bootstrap 95% interval | CUDA interval median ratio |",
                  "| --- | --- | --- | --- | --- |"]
        for stage, pairs in (("complete full scorer", benchmark["scorer_pairs"]), ("resident full-logit reducer", benchmark["reducer_pairs"])):
            for row in pairs:
                lines.append(f"| {stage}: {ARM_LABELS.get(row['arm_a'], row['arm_a'])} / {ARM_LABELS.get(row['arm_b'], row['arm_b'])} | {row['wall']['median_block_ratio']:.4f} | {interval(row['wall']['observed_block_range'])} | {interval(row['wall']['empirical_block_bootstrap_median_95_interval'])} | {row['cuda']['median_block_ratio']:.4f} |")
        lines += ["", "Complete-full-scorer wall time includes parsing/tokenization, packing, transfers, unchanged full-backbone model execution, extraction and aggregation. Its CUDA bracket includes host-induced gaps; it is not active GPU time. Resident full-logit reducer timing uses one target per unique parent row and excludes LM-head projection, backbone computation and generation. Native CE and LSE are strong direct paired controls. The legacy raw arm name `torch_ce_anchor` invokes direct native CE in this full-only study; plots use its actual operation name. No cache/tree speed, serving-throughput or full generation-pipeline speedup is inferred.", "",
                  f"Benchmark process span: {benchmark['total_process_wall_seconds']:.3f}s; summed timed windows: {benchmark['timed_window_wall_seconds']:.3f}s; non-timed setup/gates/warmup/validation/I/O/other: {benchmark['nontimed_setup_gate_warmup_validation_io_and_other_seconds']:.3f}s. Cold model load was not separately measured.", ""]
    lines += ["## Generation cost and scope", "",
              "The fixed generator records synchronized `model.generate` windows. These omit prompt preprocessing/H2D, decoding/parsing, checkpoint setup and disk I/O. First-to-last spans include intervening work but omit initial/final tails. The control costs are actual executed allocations, not randomized generation comparisons; direct4 is a subset of direct8.", ""]
    for study in studies:
        generation = study["generation"]
        lines += [f"### {study['model']}", "", f"Retained generation batches: {generation['batch_count']} across {generation['stage_count']} mode/seed stages.", "",
                  "| Control allocation | Batch windows | Generated stage samples | Measured generate seconds | Actual output tokens incl. EOS | Maximum output-token budget | Logical prompt tokens |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for control, row in generation["allocations"].items():
            lines.append(f"| {control} | {row['batches']} | {row['samples']} | {row['measured_generate_seconds']:.3f} | {row['output_tokens_including_eos']} | {row['maximum_output_token_budget']} | {row['logical_prompt_tokens']} |")
        lines += ["", f"All-stage measured generate sum: {generation['measured_generate_seconds_all_stages']:.3f}s. Observed complete generation span between recorded forward windows: {generation['observed_generation_span_seconds']:.3f}s. Cold checkpoint loading/setup is unmeasured. Summing generation windows and descriptive scoring costs is a measured-component total, not a randomized end-to-end speedup.", ""]
    lines += ["## Retained failed attempts and cumulative cost", "",
              "Failed history contributes measured cost only. It supplies no task-quality predictions and does not change the numerical threshold or final candidate pool.", ""]
    for run in failed_rankings:
        lines += [f"### {run['run']}", "",
                  f"Retained checkpoints: {run['retained_item_count']}; successful items before stopping: {run['successful_item_count']}; failed items: {run['failed_item_count']}; unattempted or unretained population: {run['unattempted_or_unretained_population_count']}. Failed IDs: {', '.join(run['failed_ids'])}.", "",
                  "| Method | Completed measurement windows | Recorded wall seconds |",
                  "| --- | --- | --- |"]
        for method, row in run["per_method_costs"].items():
            lines.append(f"| {method} | {row['completed_measurement_count']} | {row['measured_wall_seconds']:.3f} |")
        diagnostics = [(item, row) for item in run["per_item_measurements"] if not item["valid"]
                       for row in item["comparisons"]]
        if diagnostics:
            lines += ["", "| Failed item | Alternative | Actual maximum token error (nats) | Actual maximum sequence error (nats) | Winner agreements |",
                      "| --- | --- | --- | --- | --- |"]
            for item, row in diagnostics:
                winners = row["winner_agreement"]
                lines.append(f"| {item['id']} | {row['alternative']} | {row['maximum_token_error']:.8g} | {row['maximum_sequence_error']:.8g} | {sum(winners.values())}/{len(winners)} |")
        lines += ["", f"The attempt stopped at the original {run['token_error_tolerance']} nats/token numerical/winner gate. Per-item errors, comparator values and winner margins remain in `summary.json` and raw checkpoints. Its measured scorer subtotal is {run['total_measured_scorer_wall_seconds']:.3f}s; initial loading and unrecorded work are excluded.", ""]
    if not failed_rankings:
        lines += ["No failed ranking directories were supplied; no absent history cost is inferred.", ""]
    for run in failed_benchmarks:
        gate = run["failed_gate_diagnostics"] or {}
        lines += [f"### {run['run']}", "",
                  f"Study kind: `{run['study_kind']}`; failure cause: `{run['failure_cause']}`; declared runtime allowance: {run['declared_max_seconds']}s. Failed process interval: {run['recorded_process_wall_seconds']:.3f}s. Frozen fixtures: {', '.join(run['fixture_ids'])}. Failed before any timed window: {run['failed_before_any_timing_window']}. Accepted pair ratios: {run['accepted_pair_ratio_count']}; no quality output is produced.",
                  run["disposition"], ""]
        for stage, counts in run["retained_timing_windows"].items():
            lines.append(f"Retained {stage} windows: {counts['retained_raw_window_count']}; strictly validated: {counts['strictly_validated_window_count']}; total windows in saved plan: {counts['declared_windows_in_retained_plan']}.")
        if gate:
            lines += [f"Actual retained maximum token error: {gate.get('max_token_error', 'unavailable')} nats; maximum sequence error: {gate.get('max_sequence_error', 'unavailable')} nats. The original 1e-3 nats/token threshold remains unchanged."]
        lines += ["", "RQ2 cache/tree speed remains unestablished. Retained incomplete normalizer windows are not selected into accepted timing ratios.", ""]
    if not failed_benchmarks:
        lines += ["No failed benchmark directories were supplied; no absent history cost is inferred.", ""]
    lines += ["| Recorded component | Seconds |", "| --- | --- |",
              f"| Final generation, every executed stage once | {cumulative_cost['final_generation_all_stages_measured_window_seconds']:.3f} |",
              f"| Final ranking, all three measured scorer paths | {cumulative_cost['final_ranking_all_methods_measured_window_seconds']:.3f} |",
              f"| Supplied failed ranking attempts | {cumulative_cost['retained_failed_ranking_measured_window_seconds']:.3f} |",
              f"| Supplied failed benchmark process intervals | {cumulative_cost['retained_failed_benchmark_complete_process_interval_seconds']:.3f} |",
              f"| Final benchmark complete process intervals | {cumulative_cost['final_benchmark_complete_process_interval_seconds']:.3f} |",
              f"| Final whole-population FULL/Triton audit process intervals | {cumulative_cost['final_full_triton_audit_process_interval_seconds']:.3f} |",
              f"| **Cumulative measured component lower bound** | **{cumulative_cost['cumulative_measured_component_seconds_lower_bound']:.3f}** |", "",
              f"The measured component subtotal is **{cumulative_cost['cumulative_measured_component_hours_lower_bound']:.4f} host/GPU-stage hours, a lower bound**. These are actual host wall intervals on Spark, not pure GPU-busy time. The declared initial budget remains four Spark GPU-hours. An incomplete subtotal cannot establish complete budget compliance. Downloads, initial generation/ranking loads, standalone gates not supplied here, unrecorded work, interstage overhead and CPU evaluation/analysis are not fully accounted; supplied audit and failed/completed benchmark process intervals already include their own setup/gates/warmup/validation/I/O.", "",
              "## Reproducibility and limits", "",
              f"Runtime: Python {runtime['python']}, architecture {runtime['host_architecture']}; package versions and input/report/raw-source hashes are in `summary.json`.", "",
              "Both public checkpoints and the same frozen population are reported. The small held-out dataset, four timing blocks, restricted scalar evaluator, separate generation/scoring precision and finite serialization family limit generalization. All fixed-order predictions were frozen before labels; only the declared development selection uses development answers. Earlier BF16 failures are retained and must not be represented as passing. No publication, official leaderboard, universal quality, or serving speedup claim follows from this report.", ""]
    return "\n".join(lines)


def plot_accuracy(studies, output, plt, np) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 8.5), sharey=True)
    positions = np.arange(len(SCORE_METHODS))
    for axis, study in zip(axes, studies, strict=True):
        rows = study["quality"]["accuracy"]["test"]
        values = [rows[method]["supported_accuracy"] for method in SCORE_METHODS]
        colors = ["#2166ac" if method == "mass" else "#8f9ca8" for method in SCORE_METHODS]
        heights = np.asarray([value if value is not None else np.nan for value in values])
        axis.barh(positions, heights, color=colors, height=0.7)
        for position, method, value in zip(positions, SCORE_METHODS, values, strict=True):
            if value is None:
                continue
            bounds = rows[method]["question_wilson_95_interval"]
            axis.errorbar(value, position, xerr=[[max(0.0, value - bounds[0])], [max(0.0, bounds[1] - value)]],
                          fmt="none", ecolor="#263442", capsize=2, linewidth=0.8)
            axis.text(value + 0.015, position, f"{rows[method]['successes']}/{rows[method]['supported']}", va="center", fontsize=8)
        axis.set_xlim(0, 1.2)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        axis.set_title(study["model"].replace("-Instruct", "") + f"\nTest {study['quality']['test_count']}; unknown {rows['mass']['unsupported']}")
        axis.set_xlabel("Restricted scalar test accuracy (supported items)")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(positions, [METHOD_LABELS.get(method, method.replace("fixed_", "Fixed ")) for method in SCORE_METHODS])
    axes[0].invert_yaxis()
    figure.suptitle("All frozen controls; question Wilson intervals are approximate", fontsize=11)
    figure.tight_layout()
    save_figure(figure, output, "test_accuracy")
    plt.close(figure)


def plot_blocks(studies, stage, output, plt) -> None:
    pairs = studies[0]["benchmark"][stage + "_pairs"]
    names = [row["pair"] for row in pairs]
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)
    colors = ["#2166ac", "#c75728"]
    for metric, axis in zip(("wall", "cuda"), axes, strict=True):
        axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
        for model_index, study in enumerate(studies):
            by_name = {row["pair"]: row for row in study["benchmark"][stage + "_pairs"]}
            for index, name in enumerate(names):
                row = by_name[name]
                ratios = [block[metric + "_ratio_a_over_b"] for block in row["blocks"]]
                center = index + (-0.12 if model_index == 0 else 0.12)
                axis.scatter([center + offset for offset in (-0.045, -0.015, 0.015, 0.045)], ratios,
                             color=colors[model_index], alpha=0.75, s=25,
                             label=study["model"].replace("-Instruct", "") if index == 0 else None)
                median = row[metric]["median_block_ratio"]
                low, high = row[metric]["empirical_block_bootstrap_median_95_interval"]
                axis.errorbar(center, median, yerr=[[max(0.0, median - low)], [max(0.0, high - median)]],
                              fmt="D", color=colors[model_index], markersize=5, capsize=4)
        axis.set_yscale("log")
        axis.set_ylabel(("Wall time" if metric == "wall" else "CUDA interval") + " A / B ratio")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    axes[-1].set_xticks(range(len(pairs)), [ARM_LABELS.get(row["arm_a"], row["arm_a"]) + " /\n"
                                         + ARM_LABELS.get(row["arm_b"], row["arm_b"]) for row in pairs])
    figure.suptitle(("Complete FULL scorer: same backbone, different normalizers" if stage == "scorer" else "Resident FULL-logit requested-token reducer")
                   + ": all four paired blocks; >1 favors denominator", fontsize=11)
    axes[-1].set_xlabel("Diamonds: block medians; whiskers: descriptive four-block empirical bootstrap interval")
    figure.tight_layout()
    save_figure(figure, output, stage + "_block_ratios")
    plt.close(figure)


def save_figure(figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        with (output / f"{stem}.{suffix}").open("xb") as handle:
            metadata = {"Software": "OrbitCall analyze_study"} if suffix == "png" else {
                "Creator": "OrbitCall analyze_study", "CreationDate": None, "ModDate": None}
            figure.savefig(handle, format=suffix, dpi=180, bbox_inches="tight", metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, nargs=2, required=True)
    parser.add_argument("--rankings", type=Path, nargs=2, required=True)
    parser.add_argument("--benchmarks", type=Path, nargs=2, required=True)
    parser.add_argument("--audits", type=Path, nargs=2, required=True,
                        help="Completed whole-population FULL/Triton audit run directories, in matching model order")
    parser.add_argument("--failed-rankings", type=Path, nargs="*", default=[],
                        help="Retained unsuccessful pre-gold ranking attempts; costs only, never quality")
    parser.add_argument("--failed-benchmarks", type=Path, nargs="*", default=[],
                        help="Retained aborted cache/tree benchmarks; errors/costs only, never speed ratios")
    parser.add_argument("--output", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("All numerical/statistical analysis must run on the authorized DGX Spark")
    # nvidia-smi is a read-only identity query; it does not create a CUDA context.
    live_gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    ).strip().splitlines()
    if len(live_gpu) != 1 or "GB10" not in live_gpu[0] or GPU_UUID not in live_gpu[0]:
        raise RuntimeError("Live GPU identity differs from the authorized Spark")
    runtime_path = ROOT / "metadata" / "runtime_current.json"
    if not runtime_path.exists():
        runtime_path = ROOT / "metadata" / "runtime.json"
    runtime = read_json(runtime_path)
    if runtime.get("host_architecture") != "aarch64" or GPU_UUID not in runtime.get("gpu", "") or "GB10" not in runtime["gpu"]:
        raise RuntimeError("Runtime metadata differs from the authorized Spark GPU")
    # Matplotlib and NumPy are loaded only after the host guard. No torch import,
    # CUDA initialization or GPU allocation occurs in this script.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy
    studies = []
    sources = {}
    for generation, ranking, benchmark, audit_directory in zip(args.generations, args.rankings, args.benchmarks, args.audits, strict=True):
        ranked, quality = ranking_summary(ranking)
        generated = generation_summary(generation, ranked["candidates_sha256"], ranked["generation_manifest_sha256"])
        timed = benchmark_summary(benchmark)
        audited = audit_summary(audit_directory, ranking)
        if not generated["model"] == ranked["model"] == timed["model"] == quality["model"]:
            raise ValueError("Generation/ranking/benchmark/evaluation model mismatch")
        if generated["population"] != quality["population"] or timed["candidate_sha256"] != ranked["candidates_sha256"]:
            raise ValueError("Timing fixtures use another candidate pool")
        if timed["input_sha256"] != quality["inputs_sha256"]:
            raise ValueError("Timing/quality frozen input hashes differ")
        if (audited["model"] != ranked["model"] or audited["input_sha256"] != quality["inputs_sha256"]
                or audited["pre_gold_receipt_sha256"] != ranked["pre_gold_receipt_sha256"]):
            raise ValueError("Whole-population audit uses another model, input or full-reference freeze")
        studies.append({"model": quality["model"], "generation": generated, "ranking": ranked,
                        "benchmark": timed, "quality": quality, "audit": audited,
                        "measured_component_seconds_direct4_plus_full":
                        generated["allocations"]["direct4"]["measured_generate_seconds"]
                        + ranked["costs"]["full"]["complete_score_wall_seconds"],
                        "component_total_scope": "quality path uses full reference scoring; incomplete overhead accounting, descriptive component sum, not complete pipeline latency or speedup"})
        sources.update({audited["run"] + "/" + relative: digest for relative, digest in audited["source_artifact_sha256"].items()})
        for directory, names in ((generation, ("generation_manifest.json", "candidates.json")),
                                 (ranking, ("ranking_manifest.json", "predictions.json", "pre_gold_freeze.json", "evaluation.json")),
                                 (benchmark, ("manifest.json", "completed.json", "fixtures.json", "scorer_observations.jsonl",
                                              "scorer_raw_windows.jsonl", "reducer_observations.jsonl", "reducer_raw_windows.jsonl",
                                              "scorer_timing_plan.json", "reducer_timing_plan.json", "scorer_summary.json",
                                              "reducer_summary.json", "scorer_gate.json", "reducer_gate.json", "real_reducer_fixture.json"))):
            for name in names:
                sources[directory.name + "/" + name] = sha(directory / name)
    if {study["model"] for study in studies} != MODELS or len({study["quality"]["inputs_sha256"] for study in studies}) != 1:
        raise ValueError("The report must include both declared models on one frozen population")
    studies.sort(key=lambda row: ("1.5B" in row["model"], row["model"]))
    failed_paths = [directory.resolve() for directory in args.failed_rankings]
    if len(set(failed_paths)) != len(failed_paths) or set(failed_paths) & {directory.resolve() for directory in args.rankings}:
        raise ValueError("Failed ranking cost must not duplicate another supplied run")
    failed_rankings = [failed_ranking_summary(directory) for directory in args.failed_rankings]
    for run in failed_rankings:
        if run["model"] not in MODELS:
            raise ValueError("Failed ranking history must use a declared checkpoint")
        sources[run["run"] + "/ranking_manifest.json"] = run["ranking_manifest_sha256"]
        sources.update({run["run"] + "/" + relative: digest for relative, digest in run["item_sha256"].items()})
    failed_benchmark_paths = [directory.resolve() for directory in args.failed_benchmarks]
    if (len(set(failed_benchmark_paths)) != len(failed_benchmark_paths)
            or set(failed_benchmark_paths) & {directory.resolve() for directory in args.benchmarks}):
        raise ValueError("Failed benchmark cost must not duplicate another supplied run")
    failed_benchmarks = [failed_benchmark_summary(directory) for directory in args.failed_benchmarks]
    by_model = {study["model"]: study for study in studies}
    for run in failed_benchmarks:
        if run["model"] not in by_model:
            raise ValueError("Failed benchmark history must use a declared checkpoint")
        study = by_model[run["model"]]
        if (run["fixture_ids"] != study["benchmark"]["fixture_ids"]
                or run["candidates_sha256"] != study["ranking"]["candidates_sha256"]):
            raise ValueError("Normalizer study must preserve the original failed benchmark fixtures and candidate pool")
        sources.update({run["run"] + "/" + relative: digest for relative, digest in run["source_artifact_sha256"].items()})
    cumulative_cost = cumulative_component_cost(studies, failed_rankings, failed_benchmarks)
    output_names = ["summary.json", "RESULTS.md", "test_accuracy.png", "test_accuracy.pdf",
                    "scorer_block_ratios.png", "scorer_block_ratios.pdf", "reducer_block_ratios.png", "reducer_block_ratios.pdf"]
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in output_names):
        raise FileExistsError("Analysis artifacts already exist; use a fresh output directory")
    summary = {"format_version": 1, "analysis_source_sha256": sha(Path(__file__)),
               "analysis_host": "authorized DGX Spark GB10; CPU-only analysis, no new GPU experiment",
               "runtime_metadata_sha256": sha(runtime_path), "runtime": runtime,
               "live_gpu_identity": live_gpu[0],
               "analysis_packages": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
               "source_artifact_sha256": sources, "studies": studies,
               "failed_rankings": failed_rankings, "failed_benchmarks": failed_benchmarks,
               "cumulative_component_cost": cumulative_cost,
               "scope": "both frozen models; full119 population and fixed development timing fixtures; no test-driven tuning",
               "timing_uncertainty": "exact empirical four-block bootstrap; block_count4, resample_states256, not256 independent trials",
               "failed_bf16_note": "Native BF16 scoring gate failed (up to0.75 nats/token full/cache). Only prospective FP32/eager scoring was accepted; retain failed logs/artifacts.",
               "full_reference_amendment": "Initial FP32 all-item cached/tree gate failed at the unchanged1e-3 tolerance before labels/timing. Quality outputs now use full reference for both models; retain failed run and subsequent diagnostic failures without a global exact fast-path claim.",
               "full_pipeline_speedup_measured": False,
               "cache_tree_speedup_measured": False,
               "accepted_timing_kind": "full_reference_normalizer_study",
               "cold_setup_individually_instrumented": False}
    # Complete all validation/analysis before publication. Every output is new;
    # failures retain partial artifacts and must use a fresh analysis directory.
    write_exclusive(args.output / "summary.json", json_bytes(summary))
    write_exclusive(args.output / "RESULTS.md", make_markdown(studies, runtime, failed_rankings, failed_benchmarks, cumulative_cost).encode("utf-8"))
    plot_accuracy(studies, args.output, plt, np)
    plot_blocks(studies, "scorer", args.output, plt)
    plot_blocks(studies, "reducer", args.output, plt)
    print(json.dumps({"status": "analysis_complete", "models": [study["model"] for study in studies],
                      "artifacts": {name: sha(args.output / name) for name in output_names}}, indent=2))


if __name__ == "__main__":
    main()
