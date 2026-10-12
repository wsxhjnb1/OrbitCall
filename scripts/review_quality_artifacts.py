#!/usr/bin/env python3
"""Independent Spark-only CPU audit of already frozen quality artifacts.

No model, CUDA context, answer parser, or evaluator execution is used. Stored
outcome summaries are recalculated; correctness itself remains the restricted
offline evaluator's judgment. Artifact hashes and source ordering establish
the recorded freeze chain, not a cryptographic proof of wall-clock chronology.
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
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
EXPECTED_ANALYZER = "68184c78c9bf862c4825c4fe33efeda995ff10a41a2c24d3bfba7bedb7bc6941"
EXPECTED_INPUT = "1e8edf278caa6066c1b4c1fab07599cc8acb03fdf0037d3447603ff75916060c"
FIXED = tuple("fixed_" + "".join(map(str, order)) for order in itertools.permutations(range(3)))
SCORED = ("mass", "max_order", "mean_logprob", "length_normalized", "canonical_schema", *FIXED)
SAMPLED = ("vote4", "vote8", "greedy", "freeform_vote")
EQUAL_SCORE = ("max_order", "mean_logprob", "length_normalized", "canonical_schema", "dev_selected_fixed")
ALL_METHODS = (*SCORED, *SAMPLED, "dev_selected_fixed")
PRIMARY = (17, 29, 43, 71)
EXTRA = (101, 113, 127, 139)
CHECKS: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key in {path.name}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def close(actual: float, expected: float, name: str, tolerance: float = 1e-12) -> None:
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= tolerance, f"Numeric summary mismatch: {name}")


def safe_path(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), "Artifact path escaped its root")
    return path


def key(call: dict) -> str:
    require(set(call) == {"name", "arguments"} and type(call["name"]) is str
            and type(call["arguments"]) is dict, "Malformed frozen call")
    return json.dumps(call, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def pool(samples: list[dict], mode: str, seeds: tuple[int, ...]) -> list[dict]:
    selected = [sample for sample in samples if sample["mode"] == mode and sample["seed"] in seeds]
    require(len(selected) == len(seeds) and {sample["seed"] for sample in selected} == set(seeds),
            "Incomplete sampling control")
    calls, counts = {}, Counter()
    for sample in selected:
        if sample["call"] is not None:
            semantic = key(sample["call"])
            require(semantic == sample["semantic_key"], "Frozen generated semantic identity mismatch")
            calls[semantic] = sample["call"]
            counts[semantic] += 1
    return [{"semantic_key": semantic, "call": calls[semantic], "sample_count": counts[semantic]}
            for semantic in sorted(calls)]


def check_family(raw: dict, generated: dict) -> int:
    samples = generated["samples"]
    expected = ({("greedy", 0)} | {("direct", seed) for seed in (*PRIMARY, *EXTRA)}
                | {(mode, seed) for mode in ("freeform", "reformat") for seed in PRIMARY})
    actual = [(sample["mode"], sample["seed"]) for sample in samples]
    require(len(actual) == 17 and set(actual) == expected, "Frozen 17-stage sample plan changed")
    primary = pool(samples, "direct", PRIMARY)
    families = raw["families"]
    require(len(primary) == raw["candidate_count"] == len(families), "Primary family count mismatch")
    names = tuple(generated["function"]["parameters"]["properties"])
    orders = list(itertools.permutations(names))
    begin = 0
    for candidate, family in zip(primary, families, strict=True):
        require(all(candidate[field] == family[field] for field in candidate), "Generated pool changed")
        texts = [json.dumps({"name": candidate["call"]["name"],
                            "arguments": {name: candidate["call"]["arguments"][name] for name in order}},
                            ensure_ascii=False, separators=(",", ":"), allow_nan=False) for order in orders]
        indices = {method: orders.index(tuple(names[index] for index in permutation if index < len(names)))
                   for method, permutation in zip(FIXED, itertools.permutations(range(3)), strict=True)}
        require(family["texts"] == texts and family["orders"] == list(map(list, orders))
                and family["fixed_indices"] == indices and family["sequence_begin"] == begin
                and family["sequence_end"] == begin + len(texts), "Incomplete/source-order orbit")
        begin += len(texts)
    sequences = raw["sequence_token_ids"]
    require(len(sequences) == begin and len(set(map(tuple, sequences))) == begin,
            "Missing or duplicate token paths")
    for sequence in sequences:
        require(sequence and sequence[-1] == 151645 and sequence.count(151645) == 1,
                "Explicit Qwen tokenizer EOS absent/repeated")
    full = raw["scoring"]["full"]
    scores = full["scores"]
    require(len(scores["sequence_logprobs"]) == len(scores["token_logprobs"]) == begin,
            "Malformed full reference scores")
    for sequence, values, total in zip(sequences, scores["token_logprobs"], scores["sequence_logprobs"], strict=True):
        require(len(values) == len(sequence) and all(math.isfinite(value) for value in values),
                "Nonfinite/incomplete full reference path")
        close(total, math.fsum(values), "EOS-inclusive token sum", 1e-5)
    candidate_scores = {method: [] for method in SCORED}
    for family in families:
        lo, hi = family["sequence_begin"], family["sequence_end"]
        values = scores["sequence_logprobs"][lo:hi]
        lengths = list(map(len, sequences[lo:hi]))
        largest = max(values)
        candidate_scores["mass"].append(largest + math.log(math.fsum(math.exp(value - largest) for value in values)))
        candidate_scores["max_order"].append(max(values))
        candidate_scores["mean_logprob"].append(math.fsum(values) / len(values))
        candidate_scores["length_normalized"].append(math.fsum(value / length for value, length in zip(values, lengths, strict=True)) / len(values))
        candidate_scores["canonical_schema"].append(values[0])
        for method in FIXED:
            candidate_scores[method].append(values[family["fixed_indices"][method]])
    for method, values in candidate_scores.items():
        saved = full["predictions"][method]
        if not families:
            require(saved["call"] is None and saved["semantic_key"] is None and saved["score"] is None
                    and saved["candidate_scores"] == [], "Zero-family prediction repaired/dropped")
            continue
        require(len(saved["candidate_scores"]) == len(families), "Missing candidate score")
        for family, value, recorded in zip(families, values, saved["candidate_scores"], strict=True):
            require(recorded["semantic_key"] == family["semantic_key"], "Candidate score identity mismatch")
            close(recorded["score"], value, "independent aggregate", 1e-10)
        winner = min(range(len(families)), key=lambda index: (-values[index], families[index]["semantic_key"]))
        require(saved["semantic_key"] == families[winner]["semantic_key"]
                and saved["call"] == families[winner]["call"], "Incorrect independent aggregate winner")
    for method, mode, seeds in (("vote4", "direct", PRIMARY), ("vote8", "direct", (*PRIMARY, *EXTRA)),
                                ("greedy", "greedy", (0,)), ("freeform_vote", "reformat", PRIMARY)):
        candidates = pool(samples, mode, seeds)
        saved = raw["sampling_controls"][method]
        valid = sum(candidate["sample_count"] for candidate in candidates)
        winner = min(candidates, key=lambda candidate: (-candidate["sample_count"], candidate["semantic_key"])) if candidates else None
        require(saved["semantic_key"] == (winner["semantic_key"] if winner else None)
                and saved["call"] == (winner["call"] if winner else None)
                and saved["valid_sample_count"] == valid and saved["sample_count"] == len(seeds)
                and saved["invalid_sample_count"] == len(seeds) - valid,
                "Changed sampling control or invalid-sample denominator")
    return len(primary)


def check_accuracy(rows: list[dict], summary: dict) -> None:
    supported = [row for row in rows if row["supported"]]
    for method in ALL_METHODS:
        recorded = summary[method]
        success = sum(row["correctness"][method] is True for row in supported)
        require(recorded["population"] == len(rows) and recorded["supported"] == len(supported)
                and recorded["unsupported"] == len(rows) - len(supported)
                and recorded["successes"] == success, "Quality denominator or success mismatch")
        if not rows:
            require(recorded["all_population_accuracy_bounds"] is None, "Nonempty empty-population bounds")
            continue
        for actual, expected in zip(recorded["all_population_accuracy_bounds"],
                                    (success / len(rows), (success + len(rows) - len(supported)) / len(rows)), strict=True):
            close(actual, expected, "unsupported-label population bound")
        if supported:
            close(recorded["supported_accuracy"], success / len(supported), "supported accuracy")
            n, z = len(supported), 1.959963984540054
            p = success / n
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / (1 + z * z / n)
            for actual, expected in zip(recorded["question_wilson_95_interval"],
                                        (max(0., center - radius), min(1., center + radius)), strict=True):
                close(actual, expected, "Wilson interval")


def audit_study(study: dict, input_rows: list[dict], gold_receipt: dict) -> dict:
    ranking = ROOT / "runs" / study["ranking"]["run"]
    generation = ROOT / "runs" / study["generation"]["run"]
    receipt = read(ranking / "pre_gold_freeze.json")
    manifest = read(ranking / "ranking_manifest.json")
    generated_manifest = read(generation / "generation_manifest.json")
    candidates = read(generation / "candidates.json")
    predictions = read(ranking / "predictions.json")
    quality = read(ranking / "evaluation.json")
    source_receipt = read(ranking / "source_receipt.json")
    require(source_receipt["byte_exact_manifest_match"] is True, "Archived source receipt invalid")
    require({row["file"]: row["sha256"] for row in source_receipt["files"]} == manifest["source_sha256"],
            "Rank archived-source receipt differs from manifest")
    for relative, digest in manifest["source_sha256"].items():
        require(sha(safe_path(ranking / "source", relative)) == digest == sha(safe_path(ROOT, relative)),
                "Current/archive/frozen ranking source mismatch: " + relative)
    require(generated_manifest["generation_source_sha256"] == sha(ROOT / "scripts/generate_candidates.py"),
            "Current generation source differs from executed source")
    require(quality["evaluation_source_sha256"] == sha(ROOT / "scripts/evaluate_predictions.py"),
            "Current evaluator source differs from executed source")
    require(quality == study["quality"], "Analyzer quality not byte-content-equivalent to evaluator")
    require(receipt["stage"] == "pre-gold" and receipt["answers_accessed"] is False
            and all(value["answers_accessed"] is False for value in (manifest, generated_manifest, candidates, predictions)),
            "Candidate/rank isolation flag mismatch")
    require(manifest["selected_method"] == predictions["selected_scoring_method"] == "full"
            and manifest["dtype"] == "float32" and manifest["attention"] == "eager" and manifest["tf32"] is False,
            "Quality source is not full FP32 reference")
    require(receipt["inputs_sha256"] == manifest["inputs_sha256"] == generated_manifest["inputs_sha256"]
            == quality["inputs_sha256"] == EXPECTED_INPUT, "Frozen input linkage mismatch")
    require(sha(ranking / "ranking_manifest.json") == receipt["ranking_manifest_sha256"]
            == predictions["ranking_manifest_sha256"], "Manifest freeze mismatch")
    candidate_sha, prediction_sha, receipt_sha = (sha(generation / "candidates.json"),
                                                sha(ranking / "predictions.json"), sha(ranking / "pre_gold_freeze.json"))
    require(candidate_sha == receipt["candidates"]["sha256"] == manifest["candidates_sha256"]
            == predictions["candidates_sha256"] == quality["candidates_sha256"], "Candidate freeze mismatch")
    require(prediction_sha == receipt["predictions"]["sha256"] == quality["predictions_sha256"]
            and receipt_sha == quality["receipt_sha256"], "Prediction freeze mismatch")
    require(sha(generation / "generation_manifest.json") == manifest["generation_manifest_sha256"],
            "Generation-to-ranking source linkage mismatch")
    matching = [row for row in gold_receipt["pre_gold_freezes"] if row["run"] == ranking.name]
    require(len(matching) == 1 and matching[0]["receipt_sha256"] == receipt_sha
            and matching[0]["candidates_sha256"] == candidate_sha and matching[0]["predictions_sha256"] == prediction_sha,
            "Both-model pre-download receipt changed")
    require(quality["gold_source_sha256"] == {row["file"]: row["sha256"] for row in gold_receipt["downloads"]},
            "Gold provenance mismatch; no answers opened by this audit")
    ids = [row["id"] for row in input_rows]
    require(len(ids) == 119 and receipt["record_count"] == 119 and receipt["development_count"] == 20
            and receipt["test_count"] == 99 and len(receipt["item_sha256"]) == 119,
            "Incomplete population freeze")
    for values in (candidates["records"], predictions["records"], quality["records"]):
        require([row["id"] for row in values] == ids, "Missing/reordered frozen population")
    require(quality["evaluator"] == "restricted_scalar_bfcl_ast_value_v1"
            and "restricted scalar evaluator, not the full official BFCL score" in quality["limitations"],
            "Official BFCL scope incorrectly claimed")
    zero_ids, failed_ids = [], []
    for index, (source, generated, prediction, outcome) in enumerate(zip(input_rows, candidates["records"], predictions["records"], quality["records"], strict=True)):
        path = ranking / f"items/{index:04d}.json"
        require(sha(path) == receipt["item_sha256"][f"items/{index:04d}.json"], "Raw pre-gold shard changed")
        raw = read(path)
        require(raw["valid"] is True and raw["prediction_source"] == "full"
                and raw["id"] == source["id"] and generated["function"] == source["function"][0],
                "Raw full-reference provenance mismatch")
        for row in (generated, prediction, outcome, raw):
            require(row["split"] == source["split"] and row["schema_group"] == source["schema_group"],
                    "Post-freeze split/group changed")
        count = check_family(raw, generated)
        require(prediction["candidate_count"] == outcome["candidate_count"] == count,
                "Zero-family/sample count changed")
        for method in SCORED:
            require(prediction["methods"][method] == raw["scoring"]["full"]["predictions"][method],
                    "Quality substituted another scorer")
        for method in SAMPLED:
            require(prediction["methods"][method] == raw["sampling_controls"][method], "Sampling predictions changed")
        require(type(outcome["supported"]) is bool and set(outcome["correctness"]) == set(ALL_METHODS),
                "Outcome methods/support malformed")
        for method in (*SCORED, *SAMPLED):
            evaluation = outcome["evaluation"][method]
            require(evaluation["supported"] is outcome["supported"]
                    and evaluation["correct"] is outcome["correctness"][method], "Outcome summary changed")
            require((type(evaluation["correct"]) is bool) if outcome["supported"] else evaluation["correct"] is None,
                    "Unsupported labels counted as correctness")
        if not count:
            zero_ids.append(source["id"])
            require(not outcome["supported"] or all(outcome["correctness"][method] is False for method in SCORED),
                    "Zero-family likelihood outcome repaired")
        if not raw["fast_path_valid"]:
            failed_ids.append(source["id"])
    rows = quality["records"]
    unsupported = [row["id"] for row in rows if not row["supported"]]
    require(unsupported == quality["unsupported_ids"] and len(unsupported) == 1
            and all(row["split"] == "test" for row in rows if not row["supported"]),
            "Unsupported label dropped or moved")
    dev = [row for row in rows if row["split"] == "development" and row["supported"]]
    selected_fixed = min(FIXED, key=lambda method: (-sum(row["correctness"][method] is True for row in dev), method))
    require(selected_fixed == quality["selected_fixed_policy"] and quality["selected_fixed_policy_stage"] == "development labels only",
            "Fixed-order control selected on another population")
    require(all(row["correctness"]["dev_selected_fixed"] is row["correctness"][selected_fixed] for row in rows),
            "Development-selected policy not applied unchanged to test")
    selected_control = min(EQUAL_SCORE, key=lambda method: (-sum(row["correctness"][method] is True for row in dev), method))
    require(selected_control == quality["dev_selected_equal_score_control"], "Primary comparator selected outside development")
    for split in ("development", "test"):
        check_accuracy([row for row in rows if row["split"] == split], quality["accuracy"][split])
    for category, recorded in quality["category_test_accuracy"].items():
        check_accuracy([row for row in rows if row["split"] == "test" and row["category"] == category], recorded)
    coverage = quality["candidate_coverage"]
    require(coverage["zero_candidate"] == len(zero_ids) and coverage["with_primary_candidate"] == 119 - len(zero_ids)
            and coverage["primary_samples"] == sum(row["primary_sample_count"] for row in rows) == 476
            and coverage["valid_primary_samples"] == sum(row["primary_valid_sample_count"] for row in rows)
            and coverage["oracle_metric_denominator"] == 118
            and coverage["oracle_correct_primary_candidate"] == sum(row["oracle_primary_candidate_correct"] is True for row in rows),
            "Candidate coverage denominator mismatch")
    close(coverage["json_schema_validity"], coverage["valid_primary_samples"] / 476, "primary sample validity")
    supported_test = [row for row in rows if row["split"] == "test" and row["supported"]]
    require(len(supported_test) == 98, "Supported test denominator changed")
    for control in EQUAL_SCORE:
        paired = quality["all_equal_score_paired_test_comparisons"][control]
        wins = sum(row["correctness"]["mass"] is True and row["correctness"][control] is False for row in supported_test)
        losses = sum(row["correctness"]["mass"] is False and row["correctness"][control] is True for row in supported_test)
        require(paired["mass_only_correct"] == wins and paired["control_only_correct"] == losses
                and paired["supported"] == 98 and paired["population"] == 99 and paired["unsupported"] == 1,
                "Paired comparison denominator/count mismatch")
        close(paired["accuracy_difference"], (wins - losses) / 98, "paired difference")
        n, k = wins + losses, min(wins, losses)
        probability = min(1., 2 * sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)) if n else 1.
        close(paired["exact_mcnemar_p"], probability, "exact McNemar")
        if not wins and not losses:
            require(paired["question_bootstrap_95_interval"] == paired["schema_bootstrap_95_interval"] == [0., 0.],
                    "Degenerate zero paired intervals misreported")
    primary = quality["primary_heldout_comparison"]
    require(primary == quality["all_equal_score_paired_test_comparisons"][selected_control], "Primary held-out comparison mismatch")
    require(quality["strongest_test_comparison_scope"].startswith("descriptive oracle"), "Test-selected control used inferentially")
    require(len(zero_ids) == study["ranking"]["zero_family_items"]
            and failed_ids == study["ranking"]["combined_fast_path_gate_failed_ids"],
            "Diagnostic failures omitted by analyzer")
    CHECKS.append(study["model"] + ": full source/freeze, orbit, all controls, support, dev selection and aggregate summaries verified")
    return {"model": study["model"], "population": 119, "development": 20, "test": 99,
            "supported_test": 98, "unsupported_ids": unsupported, "zero_candidate_ids": zero_ids,
            "diagnostic_fast_path_failed_ids": failed_ids, "selected_fixed_policy": selected_fixed,
            "dev_selected_equal_score_control": selected_control, "mass_test": quality["accuracy"]["test"]["mass"],
            "canonical_test": quality["accuracy"]["test"]["canonical_schema"],
            "vote8_test": quality["accuracy"]["test"]["vote8"],
            "mass_advantage_over_dev_selected_control_supported": primary["accuracy_difference"] > 0,
            "receipt_sha256": receipt_sha, "evaluation_sha256": sha(ranking / "evaluation.json"),
            "evaluator_source_sha256": quality["evaluation_source_sha256"],
            "current_archived_frozen_source_match": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "metadata/quality_artifact_review.json")
    parser.add_argument("--log", type=Path, default=ROOT / "logs/quality_artifact_review_v1.log")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("Artifact calculations must run on the authorized DGX Spark")
    identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader,nounits"],
                                       text=True, timeout=15).strip().splitlines()
    require(len(identity) == 1 and "GB10" in identity[0] and GPU_UUID in identity[0], "Wrong live GPU identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists() and not args.log.exists(), "Audit output/log already exists; do not overwrite")
    with args.log.open("x", encoding="utf-8") as log:
        try:
            evidence = read(ROOT / "evidence/summary.json")
            require(evidence["analysis_source_sha256"] == EXPECTED_ANALYZER == sha(ROOT / "scripts/analyze_study.py"),
                    "Analyzer executed/current source mismatch")
            artifact_count = 0
            for relative, digest in evidence["source_artifact_sha256"].items():
                require(sha(safe_path(ROOT / "runs", relative)) == digest, "Analyzer input artifact changed: " + relative)
                artifact_count += 1
            frozen = read(ROOT / "data/frozen_inputs.json")
            require(sha(ROOT / "data/frozen_inputs.json") == EXPECTED_INPUT and frozen["answer_accessed"] is False
                    and frozen["production_input_hashes"] is True, "Input-only freeze/provenance mismatch")
            rows = frozen["records"]
            require(len(rows) == 119 and len(frozen["development_ids"]) == 20 and len(frozen["test_ids"]) == 99,
                    "Actual frozen population mismatch")
            dev_groups = {row["schema_group"] for row in rows if row["split"] == "development"}
            test_groups = {row["schema_group"] for row in rows if row["split"] == "test"}
            require(not dev_groups & test_groups, "Schema group leaked across development/test")
            gold = read(ROOT / "metadata/bfcl_gold_download.json")
            require(gold["revision"] == frozen["gorilla_revision"] and gold["labels_evaluated"] is False
                    and len(gold["pre_gold_freezes"]) == len({row["run"] for row in gold["pre_gold_freezes"]}) == 2,
                    "Both-model answer-download freeze receipt absent")
            for download in gold["downloads"]:
                require("/" + gold["revision"] + "/" in download["url"], "Unpinned answer source")
            studies = [audit_study(study, rows, gold) for study in evidence["studies"]]
            require({study["model"] for study in studies} == {"Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct"}
                    and len(studies) == 2 and not any(study["mass_advantage_over_dev_selected_control_supported"] for study in studies),
                    "Actual no-advantage conclusion mismatch")
            require(evidence["full_pipeline_speedup_measured"] is False and evidence["cache_tree_speedup_measured"] is False,
                    "Unmeasured speed conclusion claimed")
            CHECKS.append(f"All {artifact_count} analyzer input hashes verified, including failed retained runs")
            result = {"status": "accepted_artifact_consistency_review", "script_sha256": sha(Path(__file__)),
                      "live_gpu_identity": identity[0], "execution": "CPU-only audit on authorized DGX Spark; no model/GPU jobs",
                      "evidence_summary_sha256": sha(ROOT / "evidence/summary.json"),
                      "results_markdown_sha256": sha(ROOT / "evidence/RESULTS.md"),
                      "analyzer_source_sha256": EXPECTED_ANALYZER, "verified_analyzer_input_artifacts": artifact_count,
                      "studies": studies, "checks": CHECKS,
                      "limitations": ["Does not re-evaluate gold correctness or run the official BFCL checker",
                                      "Source/hash chain is not a cryptographic wall-clock chronology proof",
                                      "Finite-orbit mass does not outperform the strongest predeclared equal-score controls here",
                                      "Kernel-only speed is not end-to-end inference throughput",
                                      "Full FP32 quality reference is retained despite cached/tree diagnostic failures"]}
            payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
            with args.output.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            log.write(json.dumps({"status": result["status"], "output_sha256": sha(args.output),
                                  "checks": CHECKS}, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"status": result["status"], "output_sha256": sha(args.output)}, indent=2))
        except Exception as exc:
            log.write(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "checks": CHECKS}) + "\n")
            raise


if __name__ == "__main__":
    main()
