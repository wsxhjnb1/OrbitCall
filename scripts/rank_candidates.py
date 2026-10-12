#!/usr/bin/env python3
"""Spark-only FP32 scoring with immutable item checkpoints and pre-gold freeze."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from orbitcall.bfcl import load_frozen_inputs, render_messages, sha256_file
from orbitcall.ranking import (
    aggregate_scores, build_families, compare_scoring_results, sample_controls,
    validate_sample_plan,
)
from orbitcall.serialization import strict_json_loads


def atomic_json(path: Path, value: object) -> None:
    """Atomic exclusive publication; retain a partial artifact on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    # link fails if a competing writer created the destination. This avoids
    # replace() overwriting any completed checkpoint during a resume race.
    os.link(temporary, path)
    temporary.unlink()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = ["scripts/rank_candidates.py", *[f"src/orbitcall/{name}.py" for name in
             ("ranking", "serialization", "bfcl", "scoring", "kernels", "trie", "gpu_guard")]]
    return {path: sha256_file(root / path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("data/frozen_inputs.json"))
    parser.add_argument("--reducer", choices=("torch", "torch_lse", "triton"), default="torch_lse")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-chunk", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Ranking experiments must run on DGX Spark")
    import torch
    from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
    from orbitcall.scoring import SequenceScorer
    from orbitcall.trie import TokenTrie

    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name():
        raise RuntimeError("Ranking requires the authorized Spark GB10 GPU")
    if args.batch_size < 1 or args.head_chunk < 1:
        raise ValueError("Positive batching/chunk sizes are required")
    candidate_path = args.generation / "candidates.json"
    candidates = strict_json_loads(candidate_path.read_text(encoding="utf-8"))
    generation_manifest = strict_json_loads((args.generation / "generation_manifest.json").read_text(encoding="utf-8"))
    if candidates.get("answers_accessed") is not False or generation_manifest.get("answers_accessed") is not False:
        raise ValueError("Generation must be complete and answer-isolated")
    candidate_rows = candidates["records"]
    rows = load_frozen_inputs(args.inputs)
    if [row["id"] for row in candidate_rows] != [row["id"] for row in rows]:
        raise ValueError("Candidate IDs/order differ from the complete frozen population")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate frozen input IDs")
    if generation_manifest["inputs_sha256"] != sha256_file(args.inputs):
        raise ValueError("Generation/input hash mismatch")
    if generation_manifest["model"] != Path(args.model).name:
        raise ValueError("Scoring checkpoint name differs from generation checkpoint")
    for row, generated in zip(rows, candidate_rows, strict=True):
        if (generated["function"] != row["function"][0]
                or generated["split"] != row["split"] or generated["schema_group"] != row["schema_group"]):
            raise ValueError("Candidate schema/split metadata differs from frozen input")
        validate_sample_plan(generated)
    manifest = {
        "format_version": 1, "model": Path(args.model).name,
        "inputs_sha256": sha256_file(args.inputs), "candidates_sha256": sha256_file(candidate_path),
        "generation_manifest_sha256": sha256_file(args.generation / "generation_manifest.json"),
        "source_sha256": _source_hashes(), "dtype": "float32", "attention": "eager",
        "tf32": False, "reducer": args.reducer, "batch_size": args.batch_size,
        "head_chunk": args.head_chunk, "methods": ["full", "cached", "tree"],
        "token_error_tolerance": 1e-3, "selected_method": "full", "answers_accessed": False,
        "valid_scope": "full-reference prediction provenance; fast_path_valid is separate",
        "record_ids": [row["id"] for row in rows],
        "tie_break": "lexicographic typed semantic_key", "end_token": "tokenizer.eos_token_id; exactly once",
        "fixed_policy": "permutations of schema positions 0,1,2; remove absent position2 for two keys",
        "length_normalized": "arithmetic mean of each EOS-inclusive sequence logprob/token length",
        "timing_role": "descriptive execution costs; separate randomized ABBA benchmark required",
    }
    if args.run.exists() and not args.resume:
        raise FileExistsError(args.run)
    args.run.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run / "ranking_manifest.json"
    if manifest_path.exists():
        if strict_json_loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Refuse to resume changed model/source/input/scoring configuration")
    else:
        atomic_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    prediction_path = args.run / "predictions.json"
    receipt_path = args.run / "pre_gold_freeze.json"
    if receipt_path.exists():
        receipt = strict_json_loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt["candidates"]["sha256"] != sha256_file(candidate_path)
                or receipt["predictions"]["sha256"] != sha256_file(prediction_path)
                or receipt["ranking_manifest_sha256"] != manifest_hash):
            raise RuntimeError("Completed pre-gold freeze does not match files")
        print(json.dumps({"status": "already_frozen", "receipt_sha256": sha256_file(receipt_path)}))
        return
    require_idle_gpu()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    scorer = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=args.head_chunk)
    if type(scorer.end_token) is not int:
        raise RuntimeError("Checkpoint lacks a single explicit EOS token")
    completed = []
    shard_hashes = {}
    for index, (row, generated) in enumerate(zip(rows, candidate_rows, strict=True)):
        shard = args.run / "items" / f"{index:04d}.json"
        if shard.exists():
            item = strict_json_loads(shard.read_text(encoding="utf-8"))
            if item["id"] != row["id"] or item["ranking_manifest_sha256"] != manifest_hash:
                raise RuntimeError("Checkpoint ID/manifest mismatch")
            if item.get("valid") is not True:
                raise RuntimeError("A retained failed item cannot be resumed as successful")
        else:
            item = {"id": row["id"], "split": row["split"], "category": row["category"],
                    "schema_group": row["schema_group"], "ranking_manifest_sha256": manifest_hash,
                    "valid": False, "scoring": {}, "comparisons": []}
            try:
                controls = sample_controls(generated)
                item["sampling_controls"] = controls
                item["primary_sample_count"] = controls["vote4"]["sample_count"]
                item["primary_valid_sample_count"] = controls["vote4"]["valid_sample_count"]
                for method in manifest["methods"]:
                    require_owned_gpu()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    # Include validation, orbit packing, prompt/sequence encoding,
                    # H2D, model execution, CPU score extraction and aggregation.
                    families = build_families(generated)
                    texts = [text for family in families for text in family["texts"]]
                    prompt = scorer.prompt_ids(render_messages(row))
                    sequences = scorer.sequence_ids(texts) if texts else []
                    if any(sequence[-1] != scorer.end_token or sequence.count(scorer.end_token) != 1
                           for sequence in sequences):
                        raise RuntimeError("An orbit path contains a missing/repeated terminator")
                    if sequences:
                        scores = asdict(scorer.score(prompt, sequences, method=method,
                                                    reducer=args.reducer, batch_size=args.batch_size))
                        if (len(scores["sequence_logprobs"]) != len(sequences)
                                or len(scores["token_logprobs"]) != len(sequences)
                                or any(len(path) != len(sequence) for path, sequence in
                                       zip(scores["token_logprobs"], sequences, strict=True))):
                            raise RuntimeError("Scorer returned incomplete EOS-inclusive token paths")
                        predictions = aggregate_scores(families, scores["sequence_logprobs"],
                                                       [len(sequence) for sequence in sequences])
                    else:
                        scores = {"method": method, "sequence_logprobs": [], "token_logprobs": [],
                                  "prompt_tokens": len(prompt), "independent_suffix_tokens": 0,
                                  "scored_edges": 0, "processed_prompt_positions": 0,
                                  "processed_query_positions": 0}
                        predictions = aggregate_scores([], [], [])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    peak = torch.cuda.max_memory_allocated()
                    require_owned_gpu()
                    item["scoring"][method] = {"scores": scores, "predictions": predictions,
                                                "complete_score_wall_seconds": elapsed,
                                                "peak_allocated_bytes": peak}
                    if method == "tree":
                        item["families"] = families
                        item["prompt_token_ids"] = prompt
                        item["sequence_token_ids"] = sequences
                        if sequences:
                            tree = TokenTrie.build(sequences)
                            item["tree"] = {"tokens": tree.tokens, "parents": tree.parents,
                                            "depths": tree.depths, "paths": tree.paths}
                item["candidate_count"] = len(item["families"])
                item["comparisons"] = [compare_scoring_results(item["scoring"]["full"], item["scoring"][method],
                                                               tolerance=manifest["token_error_tolerance"])
                                       for method in ("cached", "tree")]
                item["fast_path_valid"] = all(comparison["valid"] for comparison in item["comparisons"])
                item["prediction_source"] = "full"
                # Shape, finite values and sum checks above still fail closed.
                # Quality uses the complete reference for every item. A fast
                # backend's failed empirical gate is retained and reported;
                # it neither alters the reference prediction nor drops an item.
                item["valid"] = True
                if not item["fast_path_valid"]:
                    item["fast_path_error"] = "Per-token tolerance or same-score winner agreement gate failed"
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                atomic_json(shard, item)
                raise
            atomic_json(shard, item)
            if not item["valid"]:
                raise RuntimeError(f"Retained invalid scoring item: {row['id']}")
        completed.append(item)
        shard_hashes[f"items/{index:04d}.json"] = sha256_file(shard)
        print(json.dumps({"status": "item_complete", "id": item["id"],
                          "candidate_count": item["candidate_count"]}), flush=True)
    require_owned_gpu()
    predictions = {
        "format_version": 1, "answers_accessed": False, "selected_scoring_method": "full",
        "ranking_manifest_sha256": manifest_hash, "candidates_sha256": sha256_file(candidate_path),
        "records": [{"id": item["id"], "split": item["split"], "category": item["category"],
                     "schema_group": item["schema_group"], "candidate_count": item["candidate_count"],
                     "primary_sample_count": item["primary_sample_count"],
                     "primary_valid_sample_count": item["primary_valid_sample_count"],
                     "primary_calls": [family["call"] for family in item["families"]],
                     "methods": {**item["scoring"]["full"]["predictions"], **item["sampling_controls"]}}
                    for item in completed],
    }
    if prediction_path.exists():
        if strict_json_loads(prediction_path.read_text(encoding="utf-8")) != predictions:
            raise RuntimeError("Existing final predictions differ; refusing to overwrite")
    else:
        atomic_json(prediction_path, predictions)
    receipt = {"format_version": 1, "stage": "pre-gold", "answers_accessed": False,
               "candidates": {"file": candidate_path.name, "sha256": sha256_file(candidate_path)},
               "predictions": {"file": prediction_path.name, "sha256": sha256_file(prediction_path)},
               "inputs_sha256": sha256_file(args.inputs), "ranking_manifest_sha256": manifest_hash,
               "item_sha256": shard_hashes, "record_count": len(completed),
               "development_count": sum(item["split"] == "development" for item in completed),
               "test_count": sum(item["split"] == "test" for item in completed)}
    atomic_json(receipt_path, receipt)
    print(json.dumps({"status": "predictions_frozen", "receipt_sha256": sha256_file(receipt_path),
                      "predictions_sha256": receipt["predictions"]["sha256"],
                      "record_count": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
