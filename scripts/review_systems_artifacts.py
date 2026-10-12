#!/usr/bin/env python3
"""Independent CPU-only review of completed OrbitCall systems artifacts.

No model loading, CUDA initialization, timing, answers, or evidence mutation.
Numerical recomputation and CPU tensor inspection run on Spark only. The sole
outputs are a fresh review receipt and log; neither may already exist.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
METHODS = ("mass", "max_order", "mean_logprob", "length_normalized", "canonical_schema",
           "fixed_012", "fixed_021", "fixed_102", "fixed_120", "fixed_201", "fixed_210")
SCORER_PAIRS = (("full_torch_vs_lse", "full", "full_lse"),
                ("full_torch_vs_ce", "full", "full_ce"),
                ("full_torch_vs_triton", "full", "full_triton"),
                ("full_lse_vs_triton", "full_lse", "full_triton"),
                ("full_ce_vs_triton", "full_ce", "full_triton"))
REDUCER_PAIRS = (("torch_vs_lse", "torch", "torch_lse"),
                 ("torch_vs_ce", "torch", "torch_ce_anchor"),
                 ("torch_vs_triton", "torch", "triton"),
                 ("lse_vs_triton", "torch_lse", "triton"),
                 ("ce_vs_triton", "torch_ce_anchor", "triton"))
HASHES: dict[str, str] = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[path.relative_to(ROOT).as_posix()] = value
    return value


def read(path):
    digest(path)
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def jsonl(path):
    digest(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def close(a, b, tolerance=1e-12):
    return type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def sources(directory, manifest):
    for relative, expected in manifest["source_sha256"].items():
        path = (directory / "source" / relative).resolve()
        require(path.is_relative_to((directory / "source").resolve()), "Source path escapes its snapshot")
        require(digest(path) == expected, "Retained source SHA256 mismatch")


def aggregates(families, scores, lengths):
    """Rebuild the eleven frozen likelihood aggregators, without model code."""
    require(len(scores) == len(lengths) == (families[-1]["sequence_end"] if families else 0), "Family vector coverage differs")
    values = {method: [] for method in METHODS}
    for family in families:
        begin, end = family["sequence_begin"], family["sequence_end"]
        group = scores[begin:end]
        require(len(group) == len(family["texts"]) and len(group) in (2, 6), "Incomplete two/three-key orbit")
        maximum = max(group)
        values["mass"].append(maximum + math.log(math.fsum(math.exp(x - maximum) for x in group)))
        values["max_order"].append(maximum)
        values["mean_logprob"].append(math.fsum(group) / len(group))
        values["length_normalized"].append(math.fsum(x / n for x, n in zip(group, lengths[begin:end], strict=True)) / len(group))
        values["canonical_schema"].append(group[0])
        for method in METHODS[5:]:
            values[method].append(group[family["fixed_indices"][method]])
    result = {}
    for method, vector in values.items():
        if not vector:
            result[method] = {"semantic_key": None, "margin": None, "scores": vector}
            continue
        order = sorted(range(len(vector)), key=lambda index: (-vector[index], families[index]["semantic_key"]))
        result[method] = {"semantic_key": families[order[0]]["semantic_key"], "scores": vector,
                          "margin": vector[order[0]] - vector[order[1]] if len(order) > 1 else None}
    return result


def validate_result(result, sequences=None, families=None):
    sequences = result.get("sequence_ids") if sequences is None else sequences
    families = result.get("families") if families is None else families
    scores = result["scores"]
    tokens, totals = scores["token_logprobs"], scores["sequence_logprobs"]
    require(scores["method"] == "full", "A full-only artifact uses another backbone")
    require(len(tokens) == len(totals) == len(sequences), "Score-vector shape differs")
    require(len(set(map(tuple, sequences))) == len(sequences), "Duplicate token path")
    for sequence, path, total in zip(sequences, tokens, totals, strict=True):
        require(sequence and sequence[-1] == 151645 and sequence.count(151645) == 1, "Missing, repeated or nonfinal EOS")
        require(all(type(token) is int and 0 <= token < 151936 for token in sequence), "Invalid model token ID")
        require(len(sequence) == len(path) and all(math.isfinite(value) for value in [*path, total]), "Incomplete/nonfinite token vector")
        require(close(math.fsum(path), total, 1e-5), "Sequence score differs from token sum")
    rebuilt = aggregates(families, totals, list(map(len, sequences)))
    require(set(result["predictions"]) == set(METHODS), "An aggregator is absent")
    for method, expected in rebuilt.items():
        observed = result["predictions"][method]
        require(observed["semantic_key"] == expected["semantic_key"], "Stored winner differs from rebuilt vector")
        require((observed["margin"] is None and expected["margin"] is None)
                or close(observed["margin"], expected["margin"]), "Stored winner margin differs")
        require(len(observed["candidate_scores"]) == len(expected["scores"]), "Candidate score vector differs")
        require(all(close(row["score"], value) for row, value in zip(observed["candidate_scores"], expected["scores"], strict=True)), "Stored aggregation differs")
    return rebuilt


def compare(reference, alternative, sequences, families, tolerance, top_required):
    left_winners = validate_result(reference, sequences, families)
    right_winners = validate_result(alternative, sequences, families)
    left, right = reference["scores"], alternative["scores"]
    errors = [[abs(a - b) for a, b in zip(x, y, strict=True)]
              for x, y in zip(left["token_logprobs"], right["token_logprobs"], strict=True)]
    sequence_errors = [abs(a - b) for a, b in zip(left["sequence_logprobs"], right["sequence_logprobs"], strict=True)]
    maximum = max((value for path in errors for value in path), default=0.0)
    require(maximum < tolerance if top_required else maximum <= tolerance, "Independent numerical gate failed")
    require(all(error <= len(sequence) * tolerance + 1e-8 for error, sequence in zip(sequence_errors, sequences, strict=True)), "Length-scaled sequence bound failed")
    require(all(left_winners[method]["semantic_key"] == right_winners[method]["semantic_key"] for method in METHODS), "An independent aggregator winner differs")
    if sequences and top_required:
        require(max(range(len(sequences)), key=left["sequence_logprobs"].__getitem__)
                == max(range(len(sequences)), key=right["sequence_logprobs"].__getitem__), "Top sequence differs")
    return maximum, max(sequence_errors, default=0.0)


def timing_plan(rng, pairs):
    rows = []
    for name, a, b in pairs:
        for block in range(4):
            order = [a, b, b, a] if rng.randrange(2) == 0 else [b, a, a, b]
            rows.append({"pair": name, "block": block, "order": order})
    rng.shuffle(rows)
    return rows


def timing_stage(directory, stage, manifest, planned, pairs, gate):
    raw, valid = jsonl(directory / f"{stage}_raw_windows.jsonl"), jsonl(directory / f"{stage}_observations.jsonl")
    plan, summary = read(directory / f"{stage}_timing_plan.json"), read(directory / f"{stage}_summary.json")
    require(plan == planned and summary["valid"] is True, "Frozen random timing plan differs")
    flattened = [(block["pair"], block["block"], position, arm)
                 for block in plan for position, arm in enumerate(block["order"])]
    require(len(raw) == len(valid) == len(flattened) == 80, "Missing or extra raw/strict window")
    observed = lambda row: (row["pair"], row["block"], row["position"], row["arm"])
    require([observed(row) for row in raw] == [observed(row) for row in valid] == flattened, "Timing rows are missing, duplicated or reordered")
    references = {row["id"]: row["results"]["full"] for row in gate["records"]} if stage == "scorer" else gate["arms"]["torch"]["values"]
    memory, maximum_error = {}, 0.0
    for source, row in zip(raw, valid, strict=True):
        require(all(row[field] == source[field] for field in source if field not in ("outputs", "values")), "Raw versus validated timing/memory differs")
        for field in ("wall_seconds", "cuda_interval_ms"):
            require(math.isfinite(row[field]) and row[field] > 0, "Nonpositive timing")
        require(row["repeats"] == (1 if stage == "scorer" else 100), "Repeat budget differs")
        for field in ("allocated", "reserved"):
            require(row[f"peak_{field}_bytes"] >= row[f"baseline_{field}_bytes"] >= 0, "Malformed peak/baseline memory")
        require(row["incremental_peak_allocated_bytes"] == row["peak_allocated_bytes"] - row["baseline_allocated_bytes"], "Incremental allocation differs from peak minus baseline")
        require(all(set(row[field]["compute_pids"]) <= {manifest["pid"]} for field in ("guard_before", "guard_after")), "Unrelated GPU context in saved guard")
        memory.setdefault(row["arm"], []).append(row["incremental_peak_allocated_bytes"])
        if stage == "scorer":
            require(row["outputs"] == source["outputs"], "Raw scorer output differs")
            require([output["id"] for output in row["outputs"]] == manifest["fixture_ids"], "Window fixture population differs")
            require(set(row["validity"]) == set(manifest["fixture_ids"]), "Strict window gate omits a fixture")
            for output in row["outputs"]:
                reference = references[output["id"]]
                require(all(output[field] == reference[field] for field in ("prompt_ids", "sequence_ids", "families")), "A timed arm changes token arrays/families")
                error, _ = compare(reference, output, output["sequence_ids"], output["families"], 1e-3, True)
                stored = row["validity"][output["id"]]
                require(stored["valid"] is True and close(error, stored["max_token_error"]), "Strict scorer validity/error differs")
                maximum_error = max(maximum_error, error)
        else:
            values = row["validity"]["values"]
            require(values == source["values"] and len(values) == len(references) == 32, "Raw reducer vector shape differs")
            require(all(math.isfinite(value) for value in values), "Nonfinite reducer output")
            error = max(abs(a - b) for a, b in zip(values, references, strict=True))
            require(error < 3e-5 and row["validity"]["valid"] is True and close(error, row["validity"]["max_error"]), "Strict reducer gate/error differs")
            maximum_error = max(maximum_error, error)
    results = []
    require([row["pair"] for row in summary["pairs"]] == [name for name, _, _ in pairs], "Source summary pair list differs")
    for (name, a, b), reported in zip(pairs, summary["pairs"], strict=True):
        wall_ratios, cuda_ratios = [], []
        for block in range(4):
            selected = [row for row in valid if row["pair"] == name and row["block"] == block]
            require(len(selected) == 4 and all(sum(row["arm"] == arm for row in selected) == 2 for arm in (a, b)), "A four-window paired block is incomplete")
            for field, ratios in (("wall_seconds", wall_ratios), ("cuda_interval_ms", cuda_ratios)):
                medians = {arm: statistics.median(row[field] / row["repeats"] for row in selected if row["arm"] == arm) for arm in (a, b)}
                ratios.append(medians[a] / medians[b])
        require(close(statistics.median(wall_ratios), reported["median_block_wall_ratio_a_over_b"], 1e-10)
                and close(statistics.median(cuda_ratios), reported["median_block_cuda_ratio_a_over_b"], 1e-10), "Independent block-ratio median differs")
        results.append({"pair": name, "wall_ratio_a_over_b": statistics.median(wall_ratios),
                        "cuda_ratio_a_over_b": statistics.median(cuda_ratios),
                        "wall_block_range": [min(wall_ratios), max(wall_ratios)], "blocks": 4})
    return {"raw_windows": len(raw), "strict_valid_windows": len(valid), "maximum_recomputed_token_error": maximum_error,
            "pairs": results, "incremental_peak_allocated_bytes": {
                arm: {"minimum": min(values), "maximum": max(values), "windows": len(values)} for arm, values in memory.items()},
            "memory_scope": "per-window peak allocated minus baseline; excludes resident weights/fixture baseline, not total-device or reserved-memory savings"}


def benchmark(label):
    directory = ROOT / "runs" / f"normalizer_{label}_20261013_budget_v2"
    manifest, completed = read(directory / "manifest.json"), read(directory / "completed.json")
    require(not (directory / "failure.json").exists() and completed["status"] == "valid_development_benchmark", "Benchmark incomplete or failed")
    require(completed["source_unchanged"] is True and completed["candidates_unchanged"] is True, "Benchmark source/input changed")
    require(manifest["study_kind"] == "full_reference_normalizer_study" and manifest["blocks"] == 4
            and manifest["head_chunk"] == 32 and manifest["dtype"] == "float32"
            and manifest["attention"] == "eager" and manifest["tf32"] is False
            and manifest["expected_gpu_uuid"] == GPU_UUID and GPU_UUID in manifest["gpu"]
            and manifest["max_seconds"] == 1200, "Benchmark precision/host/design differs")
    require(manifest["scorer_arms"] == {"full": ["full", "torch"], "full_lse": ["full", "torch_lse"],
                                       "full_ce": ["full", "torch_ce"], "full_triton": ["full", "triton"]}, "A normalizer arm changes full backbone")
    sources(directory, manifest)
    fixtures = read(directory / "fixtures.json")
    require(digest(directory / "fixtures.json") == manifest["fixture_file_sha256"] and fixtures["fallback"] == "none", "Fixture receipt mismatch")
    require(manifest["selected_fixture_count"] == (4 if label == "qwen3b" else 3), "Unexpected selected fixture count")
    scorer_gate, reducer_gate = read(directory / "scorer_gate.json"), read(directory / "reducer_gate.json")
    require(scorer_gate["passed"] is True and reducer_gate["passed"] is True
            and reducer_gate["ce_adapter"] == "direct native full-V CE, one target per parent", "Direct native CE gate/scope differs")
    for fixture in scorer_gate["records"]:
        for arm in manifest["scorer_arms"]:
            output = fixture["results"][arm]
            compare(fixture["results"]["full"], output, output["sequence_ids"], output["families"], 1e-3, True)
    tensor_metadata = read(directory / "real_reducer_fixture.json")
    require(tensor_metadata["backbone_method"] == "full" and tensor_metadata["synthetic"] is False
            and tensor_metadata["shape"] == [32, 151936] and tensor_metadata["dtype"] == "torch.float32"
            and tensor_metadata["stride"] == [151936, 1] and tensor_metadata["row_ids"] == list(range(32))
            and tensor_metadata["requested_edges"] == 32, "Actual full-logit shape or parent identity differs")
    require(digest(directory / "real_reducer_fixture.pt") == tensor_metadata["tensor_sha256"], "Captured CPU tensor bytes changed")
    import torch
    tensors = torch.load(directory / "real_reducer_fixture.pt", map_location="cpu", weights_only=True)
    logits = tensors["logits"]
    require(logits.device.type == "cpu" and tuple(logits.shape) == (32, 151936)
            and logits.dtype == torch.float32 and logits.stride() == (151936, 1)
            and bool(torch.isfinite(logits).all()), "Captured actual logits have wrong shape/dtype/finite values")
    require(tensors["row_ids"].tolist() == list(range(32))
            and tensors["target_ids"].tolist() == tensor_metadata["target_ids"]
            and tensors["anchor_ids"].tolist() == tensor_metadata["target_ids"], "Captured targets/identity rows differ")
    require(logits.device.type == "cpu", "Review unexpectedly uses a GPU tensor")
    rng = random.Random(manifest["random_seed"])
    scorer = timing_stage(directory, "scorer", manifest, timing_plan(rng, SCORER_PAIRS), SCORER_PAIRS, scorer_gate)
    reducer = timing_stage(directory, "reducer", manifest, timing_plan(rng, REDUCER_PAIRS), REDUCER_PAIRS, reducer_gate)
    return {"run": directory.name, "model": manifest["model"], "fixture_ids": manifest["fixture_ids"],
            "scorer": scorer, "reducer": reducer, "native_ce": "direct full-vocabulary F.cross_entropy; legacy raw name torch_ce_anchor",
            "actual_logits": tensor_metadata, "source_sha256": manifest["source_sha256"]}


def population_audit(label):
    directory = ROOT / "runs" / f"audit_full_triton_{label}_20261013_v1"
    ranking = ROOT / "runs" / f"rank_{label}_20261013_reference_v2"
    manifest, summary, completed = read(directory / "manifest.json"), read(directory / "summary.json"), read(directory / "completed.json")
    require(not (directory / "failure.json").exists() and summary["valid"] is True
            and summary["population_complete"] is True and completed["valid"] is True
            and completed["summary_sha256"] == digest(directory / "summary.json")
            and summary["manifest_sha256"] == digest(directory / "manifest.json"), "Whole-population audit receipt differs")
    require(manifest["method"] == "full" and manifest["reference_reducer"] == "torch_lse"
            and manifest["alternative_reducer"] == "triton" and manifest["dtype"] == "float32"
            and manifest["attention"] == "eager" and manifest["tf32"] is False
            and manifest["batch_size"] == 8 and manifest["head_chunk"] == 32
            and manifest["token_error_tolerance"] == 3e-5 and manifest["expected_gpu_uuid"] == GPU_UUID, "Whole-population audit contract differs")
    sources(directory, manifest)
    for relative, expected in manifest["reference_sha256"].items():
        require(digest(ranking / relative) == expected, "Audit retained full-reference hash mismatch")
    require(set(summary["item_sha256"]) == {f"items/{index:04d}.json" for index in range(119)}, "Audit does not cover all119 input IDs")
    errors, sequence_errors, nonempty, zeros, split = [], [], 0, 0, {"development": 0, "test": 0}
    for index in range(119):
        relative = f"items/{index:04d}.json"
        require(digest(directory / relative) == summary["item_sha256"][relative], "Audit raw-item hash differs")
        item, reference = read(directory / relative), read(ranking / relative)
        require(item["id"] == reference["id"] == manifest["record_ids"][index]
                and item["reference"] == reference["scoring"]["full"] and item["valid"] is True
                and all(item[field] == reference[field] for field in ("families", "prompt_token_ids", "sequence_token_ids")), "Raw audit input/reference differs")
        split[item["split"]] += 1
        sequences, families = item["sequence_token_ids"], item["families"]
        nonempty += bool(families)
        zeros += not families
        error, seq_error = compare(item["reference"], item["alternative"], sequences, families, 3e-5, False)
        require(close(error, item["comparison"]["maximum_token_error"])
                and close(seq_error, item["comparison"]["maximum_sequence_error"])
                and item["comparison"]["valid"] is True, "Stored audit comparison differs from rebuilt vectors")
        errors.append(error)
        sequence_errors.append(seq_error)
    require((nonempty, zeros) == ((119, 0) if label == "qwen3b" else (103, 16))
            and split == {"development": 20, "test": 99}, "Nonempty/zero/split denominators differ")
    require(close(max(errors), summary["maximum_token_error"])
            and close(max(sequence_errors), summary["maximum_sequence_error"])
            and summary["gate_failed_items"] == 0 and not summary["gate_failed_ids"], "Audit maximum/failure summary differs")
    require(all(row == {"agreements": nonempty, "compared_nonempty_items": nonempty, "disagreement_ids": []}
                for row in summary["winner_agreement_by_aggregator"].values())
            and set(summary["winner_agreement_by_aggregator"]) == set(METHODS), "Audit winner denominator differs")
    return {"run": directory.name, "population": 119, "split": split, "nonempty": nonempty,
            "zero_family": zeros, "eleven_aggregators_agree": True,
            "maximum_recomputed_token_error": max(errors), "maximum_recomputed_sequence_error": max(sequence_errors)}


def default_method():
    source = ROOT / "src" / "orbitcall" / "scoring.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "score":
            defaults = dict(zip((arg.arg for arg in node.args.args[-len(node.args.defaults):]), node.args.defaults, strict=True))
            return ast.literal_eval(defaults["method"])
    raise ValueError("Public scorer score method not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "metadata" / "systems_artifact_review.json")
    parser.add_argument("--log", type=Path, default=ROOT / "logs" / "system_artifact_review_v1.log")
    args = parser.parse_args()
    require(platform.system() == "Linux" and platform.machine() == "aarch64", "All numerical review must run on Spark CPU")
    require(not args.output.exists() and not args.log.exists(), "Refuse to overwrite a review receipt/log")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with args.log.open("x", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            evidence = read(ROOT / "evidence" / "summary.json")
            for relative, expected in evidence["source_artifact_sha256"].items():
                require(digest(ROOT / "runs" / relative) == expected, "Final evidence artifact hash chain differs")
            studies = []
            for label in ("qwen3b", "qwen1p5b"):
                study = {"benchmark": benchmark(label), "audit": population_audit(label)}
                studies.append(study)
                print(json.dumps({"stage": "reviewed", "model": study["benchmark"]["model"],
                                  "scorer_windows": study["benchmark"]["scorer"]["strict_valid_windows"],
                                  "reducer_windows": study["benchmark"]["reducer"]["strict_valid_windows"]}), flush=True)
            method = default_method()
            result = {"format_version": 1, "valid": True, "review_scope": "independent CPU artifact/source review; no new GPU execution or timing",
                      "host_architecture": platform.machine(), "hostname": platform.node(), "source_sha256": digest(Path(__file__).resolve()),
                      "started_epoch": started, "ended_epoch": time.time(), "studies": studies,
                      "public_default_method": method,
                      "default_recommendation": "Prefer full as public default and make tree/cache explicitly diagnostic; do not rewrite historical frozen sources" if method == "tree" else "Full default avoids silently selecting the retired tree path",
                      "full_pipeline_speedup_established": False, "core_or_evidence_modified": False,
                      "reviewed_artifact_sha256": HASHES}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as target:
                json.dump(result, target, indent=2, ensure_ascii=False, allow_nan=False)
                target.write("\n")
            print(json.dumps({"status": "pass", "receipt": args.output.name}), flush=True)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
