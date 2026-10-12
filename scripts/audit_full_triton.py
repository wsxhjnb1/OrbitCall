#!/usr/bin/env python3
"""Spark-only full-population FULL/Triton agreement audit; never reads answers.

Compare the frozen full-prompt Torch-LSE vectors with a fresh full-prompt pass
using only the Triton normalizer substitution. Keep batch 8 and head chunk 32,
FP32/eager attention, TF32 disabled, and the stored prompt/suffix token arrays.
This is an offline numerical audit, not a speed benchmark or an online guard.
Numerical failures are retained while the remaining population is audited;
any failed item makes the completed audit invalid and exits with status 1.
An execution/integrity exception stops the audit and preserves partial work.

Run on the authorized Spark only:
  PYTHONPATH=src python scripts/audit_full_triton.py --model models/MODEL \
      --ranking runs/REFERENCE_RANKING --run runs/full_triton_audit_v1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
INPUT_SHA256 = "1e8edf278caa6066c1b4c1fab07599cc8acb03fdf0037d3447603ff75916060c"
TOKEN_TOLERANCE = 3e-5
BATCH_SIZE = 8
HEAD_CHUNK = 32
MAX_SECONDS = 1200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: object) -> None:
    """Exclusive atomic publication; interruption retains the partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def source_inventory() -> dict[str, str]:
    paths = [Path(__file__).resolve(), ROOT / "AGENTS.md", ROOT / "PROTOCOL.md"]
    paths.extend(sorted((ROOT / "src" / "orbitcall").glob("*.py")))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def check_budget(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise RuntimeError("Predeclared 1200-second audit budget exhausted")


def safe_child(directory: Path, relative: str) -> Path:
    if type(relative) is not str or Path(relative).is_absolute():
        raise ValueError("Expected a relative retained artifact path")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Retained artifact escapes its ranking directory")
    return path


def verify_model(model_path: Path, read_json) -> tuple[dict, dict[str, tuple[int, int]]]:
    """Recheck every pinned local checkpoint file against its retained receipt."""
    public = read_json(ROOT / "metadata" / "public_models.json")
    matches = [row for row in public["models"]
               if row["repo_id"].split("/")[-1] == model_path.name]
    if len(matches) != 1:
        raise ValueError("Model is not one of the two pinned public checkpoints")
    declared = matches[0]
    receipt_name = {"Qwen2.5-3B-Instruct": "model_3b_verified.json",
                    "Qwen2.5-1.5B-Instruct": "model_1p5b_verified.json"}[model_path.name]
    receipt_path = ROOT / "metadata" / receipt_name
    receipt = read_json(receipt_path)
    if (receipt.get("all_public_files_match") is not True
            or receipt["repo_id"] != declared["repo_id"]
            or receipt["revision"] != declared["revision"]):
        raise ValueError("Checkpoint receipt does not match its public revision")
    actual = {row["path"]: row for row in receipt["files"]}
    if len(actual) != len(receipt["files"]) or set(actual) != {row["path"] for row in declared["files"]}:
        raise ValueError("Checkpoint receipt is incomplete or duplicated")
    records, stats = [], {}
    for record in declared["files"]:
        path = safe_child(model_path, record["path"])
        verified = actual[record["path"]]
        info = path.stat()
        digest = sha256(path)
        if (verified.get("matches_public_revision") is not True
                or info.st_size != verified["bytes"] or info.st_size != record["size"]
                or digest != verified["sha256"]
                or (record["sha256"] is not None and digest != record["sha256"])):
            raise ValueError(f"Pinned checkpoint file changed: {record['path']}")
        if record["sha256"] is None:
            blob = hashlib.sha1(f"blob {info.st_size}\0".encode("ascii"))
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    blob.update(block)
            if blob.hexdigest() != record["git_blob_id"]:
                raise ValueError(f"Public Git blob identity changed: {record['path']}")
        stats[record["path"]] = (info.st_size, info.st_mtime_ns)
        records.append({"path": record["path"], "bytes": info.st_size, "sha256": digest})
    return {"repo_id": declared["repo_id"], "revision": declared["revision"],
            "receipt_file": receipt_name, "receipt_sha256": sha256(receipt_path),
            "public_manifest_sha256": sha256(ROOT / "metadata" / "public_models.json"),
            "all_public_files_match": True, "files": records}, stats


def load_reference(ranking: Path, model_name: str, inventory: dict, read_json,
                   load_inputs) -> tuple[dict, list[dict], list[dict], dict[str, str]]:
    receipt_path = ranking / "pre_gold_freeze.json"
    manifest_path = ranking / "ranking_manifest.json"
    predictions_path = ranking / "predictions.json"
    receipt, manifest, predictions = map(read_json, (receipt_path, manifest_path, predictions_path))
    inputs_path = ROOT / "data" / "frozen_inputs.json"
    if (receipt.get("stage") != "pre-gold" or receipt.get("answers_accessed") is not False
            or manifest.get("answers_accessed") is not False
            or predictions.get("answers_accessed") is not False
            or receipt["ranking_manifest_sha256"] != sha256(manifest_path)
            or receipt["predictions"]["sha256"] != sha256(predictions_path)
            or receipt["inputs_sha256"] != sha256(inputs_path) or sha256(inputs_path) != INPUT_SHA256
            or manifest["inputs_sha256"] != receipt["inputs_sha256"]
            or manifest["candidates_sha256"] != receipt["candidates"]["sha256"]
            or predictions["candidates_sha256"] != receipt["candidates"]["sha256"]
            or predictions["ranking_manifest_sha256"] != receipt["ranking_manifest_sha256"]):
        raise ValueError("Full-reference pre-gold freeze/hash chain is invalid")
    if (manifest["model"] != model_name or manifest["dtype"] != "float32"
            or manifest["attention"] != "eager" or manifest["tf32"] is not False
            or manifest["reducer"] != "torch_lse" or manifest["batch_size"] != BATCH_SIZE
            or manifest["head_chunk"] != HEAD_CHUNK or manifest["selected_method"] != "full"
            or predictions["selected_scoring_method"] != "full"):
        raise ValueError("Reference must be FULL/Torch-LSE, batch8/chunk32, FP32/eager/TF32off")
    for name in ("bfcl", "ranking", "serialization", "scoring", "kernels", "trie", "gpu_guard"):
        relative = f"src/orbitcall/{name}.py"
        if manifest["source_sha256"].get(relative) != inventory[relative]:
            raise ValueError(f"Reference implementation changed: {relative}")
    rows = load_inputs(inputs_path)
    if (len(rows), sum(row["split"] == "development" for row in rows),
            sum(row["split"] == "test" for row in rows)) != (119, 20, 99):
        raise ValueError("Audit requires the unchanged complete 119-item population")
    expected_paths = [f"items/{index:04d}.json" for index in range(len(rows))]
    if (set(receipt["item_sha256"]) != set(expected_paths) or receipt["record_count"] != len(rows)
            or manifest["record_ids"] != [row["id"] for row in rows]
            or [row["id"] for row in predictions["records"]] != manifest["record_ids"]):
        raise ValueError("Full-reference freeze is incomplete, duplicated or reordered")
    hashes = {"pre_gold_freeze.json": sha256(receipt_path),
              "ranking_manifest.json": sha256(manifest_path),
              "predictions.json": sha256(predictions_path)}
    items = []
    for relative, source, predicted in zip(expected_paths, rows, predictions["records"], strict=True):
        path = safe_child(ranking, relative)
        if sha256(path) != receipt["item_sha256"][relative]:
            raise ValueError(f"Full-reference item bytes changed: {relative}")
        item = read_json(path)
        if (item.get("valid") is not True or item.get("prediction_source") != "full"
                or item["ranking_manifest_sha256"] != receipt["ranking_manifest_sha256"]
                or any(item[key] != source[key] for key in ("id", "split", "category", "schema_group"))):
            raise ValueError("Invalid reference prediction provenance or input metadata")
        if any(predicted["methods"][method] != result
               for method, result in item["scoring"]["full"]["predictions"].items()):
            raise ValueError("Frozen predictions differ from their full-reference item")
        hashes[relative] = sha256(path)
        items.append(item)
    return manifest, rows, items, hashes


def validate_vectors(scores: dict, prompt: list, sequences: list) -> None:
    tokens, sums = scores["token_logprobs"], scores["sequence_logprobs"]
    if (scores["method"] != "full" or scores["prompt_tokens"] != len(prompt)
            or len(tokens) != len(sequences) or len(sums) != len(sequences)
            or any(len(path) != len(sequence) for path, sequence in zip(tokens, sequences, strict=True))):
        raise ValueError("Incomplete or misaligned full-reference token vectors")
    if any(type(value) not in (float, int) or not math.isfinite(value)
           for value in [*sums, *(value for path in tokens for value in path)]):
        raise ValueError("Non-finite score vector")
    if any(abs(math.fsum(path) - total) > 1e-5 for path, total in zip(tokens, sums, strict=True)):
        raise ValueError("Sequence score differs from the complete EOS-inclusive sum")


def validate_item(item: dict, source: dict, scorer, aggregate, semantic_key,
                  serialization_orbit, validate_call, methods) -> tuple[list, list, list]:
    prompt, sequences, families = item["prompt_token_ids"], item["sequence_token_ids"], item["families"]
    vocabulary = scorer.model.config.vocab_size
    if (type(prompt) is not list or not prompt or type(sequences) is not list
            or type(families) is not list or item["candidate_count"] != len(families)):
        raise ValueError("Malformed stored prompt, suffixes or family count")
    if any(type(token) is not int or not 0 <= token < vocabulary for token in prompt):
        raise ValueError("Invalid stored prompt token ID")
    configured = scorer.model.generation_config.eos_token_id
    stop_ids = {scorer.end_token, *(configured if isinstance(configured, list) else [configured])}
    for sequence in sequences:
        if (type(sequence) is not list or not sequence or sequence[-1] != scorer.end_token
                or sequence.count(scorer.end_token) != 1
                or any(type(token) is not int or not 0 <= token < vocabulary for token in sequence)
                or any(token in stop_ids for token in sequence[:-1])):
            raise ValueError("Invalid, missing, repeated or interior suffix terminator")
    if len(set(map(tuple, sequences))) != len(sequences):
        raise ValueError("Duplicate stored suffix paths")
    function = source["function"][0]
    keys = tuple(function["parameters"]["properties"])
    orders = list(itertools.permutations(keys))
    offset, texts, seen = 0, [], set()
    for family in families:
        call = validate_call(family["call"], function)
        key = semantic_key(call)
        expected = list(serialization_orbit(call, key_order=keys))
        expected_fixed = {method: orders.index(tuple(keys[index] for index in policy if index < len(keys)))
                          for policy, method in zip(itertools.permutations(range(3)), methods[5:], strict=True)}
        if (family["semantic_key"] != key or key in seen or family["texts"] != expected
                or family["orders"] != [list(order) for order in orders]
                or family["sequence_begin"] != offset
                or family["sequence_end"] != offset + len(expected)
                or family["fixed_indices"] != expected_fixed):
            raise ValueError("Incomplete, reordered or altered finite family")
        seen.add(key)
        offset += len(expected)
        texts.extend(expected)
    if offset != len(sequences):
        raise ValueError("Stored paths do not cover every complete finite family")
    # Integrity checks never substitute these re-encodings into the scored arm.
    if scorer.sequence_ids(texts) != sequences:
        raise ValueError("Stored sequence IDs differ from the pinned tokenizer")
    reference = item["scoring"]["full"]
    validate_vectors(reference["scores"], prompt, sequences)
    recomputed = aggregate(families, reference["scores"]["sequence_logprobs"], list(map(len, sequences)))
    if set(reference["predictions"]) != set(methods) or reference["predictions"] != recomputed:
        raise ValueError("Stored full-reference aggregation/margins are inconsistent")
    return prompt, sequences, families


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.run.exists():
        raise FileExistsError(args.run)
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Numerical execution is restricted to the authorized Spark")

    import torch
    import transformers
    import triton
    from orbitcall.bfcl import load_frozen_inputs, render_messages
    from orbitcall.gpu_guard import gpu_pids, require_idle_gpu, require_owned_gpu
    from orbitcall.ranking import SCORE_METHODS, FIXED_METHODS, aggregate_scores, compare_scoring_results
    from orbitcall.scoring import SequenceScorer
    from orbitcall.serialization import semantic_key, serialization_orbit, strict_json_loads, validate_call

    require_idle_gpu()
    gpu_info = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"
    ], text=True, timeout=15).strip().splitlines()
    if len(gpu_info) != 1 or GPU_UUID not in gpu_info[0] or "GB10" not in gpu_info[0]:
        raise RuntimeError("GPU identity differs from the authorized study Spark")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The study requires the one visible Spark GB10")
    args.run.mkdir(parents=True, exist_ok=False)
    started_epoch, started = time.time(), time.monotonic()
    deadline = started + MAX_SECONDS
    inventory, observations, model_stats = source_inventory(), [], {}
    reference_hashes, manifest, current_item = {}, {}, None
    read_json = lambda path: strict_json_loads(path.read_text(encoding="utf-8"))
    try:
        for relative in inventory:
            destination = args.run / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        _, rows, items, reference_hashes = load_reference(
            args.ranking, args.model.name, inventory, read_json, load_frozen_inputs)
        model_receipt, model_stats = verify_model(args.model, read_json)
        check_budget(deadline)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        methods = (*SCORE_METHODS, *FIXED_METHODS)
        manifest = {"format_version": 1, "started_epoch": started_epoch, "pid": os.getpid(),
                    "hostname": platform.node(), "gpu": gpu_info[0], "expected_gpu_uuid": GPU_UUID,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "triton": triton.__version__, "python": platform.python_version(),
                    "model": args.model.name, "checkpoint": model_receipt,
                    "dtype": "float32", "attention": "eager", "tf32": False,
                    "deterministic_algorithms": True,
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                    "batch_size": BATCH_SIZE, "head_chunk": HEAD_CHUNK,
                    "reference_reducer": "torch_lse", "alternative_reducer": "triton",
                    "method": "full", "token_error_tolerance": TOKEN_TOLERANCE,
                    "sequence_error_bound": "EOS-inclusive length * token tolerance; plus 1e-8 comparison slack",
                    "record_ids": [row["id"] for row in rows], "record_count": len(rows),
                    "ranking_run": args.ranking.name, "reference_sha256": reference_hashes,
                    "inputs_sha256": INPUT_SHA256, "source_sha256": inventory,
                    "max_seconds": MAX_SECONDS, "answers_accessed": False,
                    "aggregators": list(methods), "zero_family_scope": "retained null outputs; not a numerical comparison",
                    "claim_scope": "empirical full-path agreement with retained vectors, not bitwise identity, kernel-only attribution, speed or online certification"}
        write_new(args.run / "manifest.json", manifest)
        scorer = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=HEAD_CHUNK)
        if (scorer.model.config.model_type != "qwen2"
                or getattr(scorer.model.config, "use_sliding_window", False)
                or type(scorer.end_token) is not int or scorer.model.dtype != torch.float32
                or scorer.model.config._attn_implementation != "eager"):
            raise RuntimeError("Loaded checkpoint architecture/precision/attention violates the audit contract")
        require_owned_gpu()
        for index, (source, item) in enumerate(zip(rows, items, strict=True)):
            current_item = {"index": index, "id": item["id"],
                            "source_item_sha256": reference_hashes[f"items/{index:04d}.json"]}
            check_budget(deadline)
            require_owned_gpu()
            prompt, sequences, families = validate_item(
                item, source, scorer, aggregate_scores, semantic_key, serialization_orbit, validate_call, methods)
            if scorer.prompt_ids(render_messages(source)) != prompt:
                raise ValueError("Stored prompt IDs differ from the pinned input/template")
            observation = {"id": item["id"], "split": item["split"], "category": item["category"],
                           "schema_group": item["schema_group"], "candidate_count": len(families),
                           "source_item_sha256": reference_hashes[f"items/{index:04d}.json"],
                           "prompt_token_ids": prompt, "sequence_token_ids": sequences,
                           "families": families, "reference": item["scoring"]["full"],
                           "comparison_scope": "actual nonempty family" if sequences else "zero-family retained null outputs",
                           "started_epoch": time.time(), "compute_pids_before": sorted(gpu_pids())}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if sequences:
                scores = asdict(scorer.score(prompt, sequences, method="full", reducer="triton", batch_size=BATCH_SIZE))
            else:
                scores = {"method": "full", "sequence_logprobs": [], "token_logprobs": [],
                          "prompt_tokens": len(prompt), "independent_suffix_tokens": 0,
                          "scored_edges": 0, "processed_prompt_positions": 0, "processed_query_positions": 0}
            torch.cuda.synchronize()
            elapsed, peak = time.perf_counter() - start, torch.cuda.max_memory_allocated()
            validate_vectors(scores, prompt, sequences)
            for field in ("prompt_tokens", "independent_suffix_tokens", "scored_edges",
                          "processed_prompt_positions", "processed_query_positions"):
                if scores[field] != item["scoring"]["full"]["scores"][field]:
                    raise ValueError(f"Full-backbone workload differs from the retained reference: {field}")
            predictions = aggregate_scores(families, scores["sequence_logprobs"], list(map(len, sequences)))
            alternative = {"scores": scores, "predictions": predictions,
                           "reducer": "triton", "descriptive_score_wall_seconds": elapsed,
                           "peak_allocated_bytes": peak}
            comparison = compare_scoring_results(item["scoring"]["full"], alternative, tolerance=TOKEN_TOLERANCE)
            comparison.update({"reference": "full_torch_lse", "alternative": "full_triton"})
            observation.update({"alternative": alternative, "comparison": comparison,
                                "valid": comparison["valid"], "ended_epoch": time.time()})
            require_owned_gpu()
            observation["compute_pids_after"] = sorted(gpu_pids())
            write_new(args.run / "items" / f"{index:04d}.json", observation)
            observations.append(observation)
            print(json.dumps({"status": "item_audited", "id": item["id"], "valid": observation["valid"],
                              "maximum_token_error": comparison["maximum_token_error"]}), flush=True)
            check_budget(deadline)
        require_owned_gpu()
        if source_inventory() != inventory or any(sha256(safe_child(args.ranking, path)) != digest
                                                for path, digest in reference_hashes.items()):
            raise RuntimeError("Source or frozen full-reference artifacts changed during the audit")
        if sha256(ROOT / "data" / "frozen_inputs.json") != INPUT_SHA256:
            raise RuntimeError("Frozen input bytes changed during the audit")
        if any((safe_child(args.model, relative).stat().st_size,
                safe_child(args.model, relative).stat().st_mtime_ns) != expected
               for relative, expected in model_stats.items()):
            raise RuntimeError("Verified checkpoint file metadata changed during the audit")
        active = [row for row in observations if row["candidate_count"] > 0]
        failed = [row for row in observations if not row["valid"]]
        if not active:
            raise RuntimeError("No nonempty families; a vacuous population is not numerical validation")
        summary = {"format_version": 1, "valid": not failed, "population_complete": True,
                   "record_count": len(observations), "scored_nonempty_items": len(active),
                   "zero_family_items": len(observations) - len(active),
                   "gate_failed_items": len(failed), "gate_failed_ids": [row["id"] for row in failed],
                   "maximum_token_error": max((row["comparison"]["maximum_token_error"] for row in active), default=None),
                   "maximum_sequence_error": max((row["comparison"]["maximum_sequence_error"] for row in active), default=None),
                   "winner_agreement_by_aggregator": {
                       method: {"agreements": sum(row["comparison"]["winner_agreement"][method] for row in active),
                                "compared_nonempty_items": len(active),
                                "disagreement_ids": [row["id"] for row in active if not row["comparison"]["winner_agreement"][method]]}
                       for method in methods},
                   "token_error_tolerance": TOKEN_TOLERANCE, "source_unchanged": True,
                   "reference_unchanged": True, "model_metadata_unchanged": True,
                   "started_epoch": started_epoch, "ended_epoch": time.time(),
                   "elapsed_seconds": time.monotonic() - started, "answers_accessed": False,
                   "manifest_sha256": sha256(args.run / "manifest.json"),
                   "item_sha256": {f"items/{index:04d}.json": sha256(args.run / "items" / f"{index:04d}.json")
                                   for index in range(len(observations))},
                   "claim_scope": manifest["claim_scope"],
                   "cost_scope": "audit execution cost only; retained reference and fresh alternative are not a randomized timing pair"}
        write_new(args.run / "summary.json", summary)
        if failed:
            write_new(args.run / "failure.json", {"stage": "completed_numerical_audit", "valid": False,
                      "error": "At least one unchanged full-path tolerance or winner gate failed",
                      "population_complete": True, "failed_ids": summary["gate_failed_ids"],
                      "summary_sha256": sha256(args.run / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]})
            return 1
        write_new(args.run / "completed.json", {"valid": True, "population_complete": True,
                  "summary_sha256": sha256(args.run / "summary.json"), "answers_accessed": False})
        return 0
    except Exception as exc:
        write_new(args.run / "failure.json", {"valid": False, "population_complete": False,
                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                  "current_item": current_item,
                  "completed_items": len(observations), "completed_ids": [row["id"] for row in observations],
                  "retained_failed_ids": [row["id"] for row in observations if not row["valid"]],
                  "elapsed_seconds": time.monotonic() - started, "answers_accessed": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
