#!/usr/bin/env python3
"""Spark-only offline evaluation of predictions frozen before gold access."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

from orbitcall.bfcl import (
    evaluate_call, load_frozen_inputs, load_gold_after_freeze, sha256_file,
)
from orbitcall.ranking import (
    FIXED_METHODS, SCORE_METHODS, SAMPLE_METHODS, candidate_pool, sample_controls,
    validate_sample_plan,
)
from orbitcall.serialization import semantic_key, strict_json_loads


BOOTSTRAP_SEED = 20261013
BOOTSTRAP_REPLICATES = 10000
EQUAL_SCORE_CONTROLS = ("max_order", "mean_logprob", "length_normalized", "canonical_schema", "dev_selected_fixed")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def accuracy_summary(outcomes: list[dict], method: str) -> dict:
    total = len(outcomes)
    supported = [row for row in outcomes if row["supported"]]
    successes = sum(row["correctness"][method] is True for row in supported)
    count = len(supported)
    unknown = total - count
    interval = None
    if count:
        z = 1.959963984540054
        proportion = successes / count
        denominator = 1 + z * z / count
        center = (proportion + z * z / (2 * count)) / denominator
        radius = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
        interval = [max(0.0, center - radius), min(1.0, center + radius)]
    return {"population": total, "supported": count, "unsupported": unknown,
            "successes": successes, "supported_accuracy": successes / count if count else None,
            "question_wilson_95_interval": interval,
            "all_population_accuracy_bounds": [successes / total, (successes + unknown) / total] if total else None,
            "interval_scope": "supported items; question-level independence is an approximation"}


def paired_comparison(outcomes: list[dict], control: str) -> dict:
    import numpy as np
    from scipy.stats import binomtest

    rows = [row for row in outcomes if row["supported"]]
    result = {"method": "mass", "control": control, "population": len(outcomes),
              "supported": len(rows), "unsupported": len(outcomes) - len(rows),
              "metric_scope": "supported gold only; unsupported remains explicitly reported",
              "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES}
    if not rows:
        return {**result, "accuracy_difference": None, "question_bootstrap_95_interval": None,
                "schema_bootstrap_95_interval": None, "exact_mcnemar_p": None}
    left = np.asarray([row["correctness"]["mass"] for row in rows], dtype=np.int64)
    right = np.asarray([row["correctness"][control] for row in rows], dtype=np.int64)
    difference = left - right
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    question_indices = rng.integers(0, len(rows), size=(BOOTSTRAP_REPLICATES, len(rows)))
    question_values = difference[question_indices].mean(axis=1)
    groups = sorted({row["schema_group"] for row in rows})
    totals = np.asarray([sum(int(difference[index]) for index, row in enumerate(rows)
                             if row["schema_group"] == group) for group in groups], dtype=np.float64)
    counts = np.asarray([sum(row["schema_group"] == group for row in rows) for group in groups], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    group_indices = rng.integers(0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
    cluster_values = totals[group_indices].sum(axis=1) / counts[group_indices].sum(axis=1)
    wins = int(((left == 1) & (right == 0)).sum())
    losses = int(((left == 0) & (right == 1)).sum())
    return {**result, "accuracy_difference": float(difference.mean()),
            "question_bootstrap_95_interval": np.quantile(question_values, [0.025, 0.975]).tolist(),
            "schema_bootstrap_95_interval": np.quantile(cluster_values, [0.025, 0.975]).tolist(),
            "schema_count": len(groups), "mass_only_correct": wins, "control_only_correct": losses,
            "both_correct": int(((left == 1) & (right == 1)).sum()),
            "both_incorrect": int(((left == 0) & (right == 0)).sum()),
            "exact_mcnemar_p": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
            "uncertainty_scope": "paired percentile bootstrap; limited question/schema population, not a universal guarantee"}


def _holm_adjust(comparisons: dict[str, dict]) -> None:
    available = sorted((result["exact_mcnemar_p"], method) for method, result in comparisons.items()
                       if result["exact_mcnemar_p"] is not None)
    previous = 0.0
    for index, (probability, method) in enumerate(available):
        previous = max(previous, min(1.0, probability * (len(available) - index)))
        comparisons[method]["holm_p_within_model_equal_score_controls"] = previous


def choose_dev_method(outcomes: list[dict], methods: tuple[str, ...]) -> str:
    development = [row for row in outcomes if row["split"] == "development" and row["supported"]]
    if not development:
        raise ValueError("No supported development labels for a predeclared control selection")
    return min(methods, key=lambda method: (-sum(row["correctness"][method] is True for row in development), method))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("data/frozen_inputs.json"))
    parser.add_argument("--gold", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Evaluation/statistical analysis must run on DGX Spark")
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError(args.output)
    receipt = strict_json_loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("stage") != "pre-gold" or receipt.get("answers_accessed") is not False:
        raise ValueError("An independently saved pre-gold freeze is required")
    if receipt["inputs_sha256"] != sha256_file(args.inputs):
        raise ValueError("Frozen input hash mismatch")
    if (receipt["candidates"]["sha256"] != sha256_file(args.candidates)
            or receipt["predictions"]["sha256"] != sha256_file(args.predictions)):
        raise ValueError("Pre-gold candidate/prediction freeze mismatch")
    ranking_manifest_path = args.receipt.parent / "ranking_manifest.json"
    if sha256_file(ranking_manifest_path) != receipt["ranking_manifest_sha256"]:
        raise ValueError("Ranking manifest mismatch")
    ranking_manifest = strict_json_loads(ranking_manifest_path.read_text(encoding="utf-8"))
    if (ranking_manifest.get("dtype") != "float32" or ranking_manifest.get("attention") != "eager"
            or ranking_manifest.get("tf32") is not False or ranking_manifest.get("answers_accessed") is not False
            or ranking_manifest.get("selected_method") != "full"):
        raise ValueError("Scoring precision/isolation manifest is outside the protocol")
    for relative, digest in receipt["item_sha256"].items():
        path = (args.receipt.parent / relative).resolve()
        if not path.is_relative_to(args.receipt.parent.resolve()) or sha256_file(path) != digest:
            raise ValueError("Raw item checkpoint hash mismatch")
        if strict_json_loads(path.read_text(encoding="utf-8")).get("valid") is not True:
            raise ValueError("A failed raw scoring item cannot be evaluated as valid")
    input_rows = load_frozen_inputs(args.inputs)
    prediction_file = strict_json_loads(args.predictions.read_text(encoding="utf-8"))
    candidate_file = strict_json_loads(args.candidates.read_text(encoding="utf-8"))
    predictions, candidates = prediction_file["records"], candidate_file["records"]
    ids = [row["id"] for row in input_rows]
    if (prediction_file.get("answers_accessed") is not False or candidate_file.get("answers_accessed") is not False
            or prediction_file.get("selected_scoring_method") != "full"
            or prediction_file.get("ranking_manifest_sha256") != receipt["ranking_manifest_sha256"]
            or [row["id"] for row in predictions] != ids or [row["id"] for row in candidates] != ids
            or receipt["record_count"] != len(ids) or len(receipt["item_sha256"]) != len(ids)):
        raise ValueError("Predictions/candidates must cover the complete frozen population")
    expected_methods = {*SCORE_METHODS, *FIXED_METHODS, *SAMPLE_METHODS}
    for index, (source, prediction, generated) in enumerate(zip(input_rows, predictions, candidates, strict=True)):
        validate_sample_plan(generated)
        if (prediction["split"] != source["split"] or prediction["schema_group"] != source["schema_group"]
                or generated["function"] != source["function"][0] or set(prediction["methods"]) != expected_methods):
            raise ValueError("Prediction schema/split/control mismatch")
        pool = candidate_pool(generated)
        keys = {row["semantic_key"] for row in pool}
        if (prediction["candidate_count"] != len(pool)
                or {semantic_key(call) for call in prediction["primary_calls"]} != keys):
            raise ValueError("Primary pool changed after generation")
        raw_item = strict_json_loads((args.receipt.parent / f"items/{index:04d}.json").read_text())
        for method in (*SCORE_METHODS, *FIXED_METHODS):
            result = prediction["methods"][method]
            if raw_item.get("prediction_source") != "full" or result != raw_item["scoring"]["full"]["predictions"][method]:
                raise ValueError("Quality prediction must use the frozen full reference")
            if ((result["call"] is None) != (result["semantic_key"] is None)
                    or (keys and result["semantic_key"] not in keys)
                    or (not keys and result["call"] is not None)):
                raise ValueError("A likelihood prediction is outside the inference-generated pool")
            if result["call"] is not None and semantic_key(result["call"]) != result["semantic_key"]:
                raise ValueError("Prediction call/key mismatch")
        controls = sample_controls(generated)
        if any(prediction["methods"][method] != controls[method] for method in SAMPLE_METHODS):
            raise ValueError("Sampling controls changed after generation")
    # The first and only label-loading operation is below. Both models must have
    # completed their freezes before the parent stages any answer files.
    gold = load_gold_after_freeze(args.gold, candidates_path=args.candidates,
                                  candidates_sha256=receipt["candidates"]["sha256"],
                                  predictions_path=args.predictions,
                                  predictions_sha256=receipt["predictions"]["sha256"])
    if any(item_id not in gold for item_id in ids):
        raise ValueError("Gold is missing an eligible frozen input ID")
    outcomes = []
    for prediction, generated in zip(predictions, candidates, strict=True):
        evaluations = {method: evaluate_call(result["call"], generated["function"], gold[prediction["id"]])
                       for method, result in prediction["methods"].items()}
        supports = {result["supported"] for result in evaluations.values()}
        if len(supports) != 1:
            raise ValueError("Evaluator support unexpectedly depends on the prediction")
        supported = supports.pop()
        oracle = [evaluate_call(call, generated["function"], gold[prediction["id"]])
                  for call in prediction["primary_calls"]]
        outcomes.append({"id": prediction["id"], "split": prediction["split"],
                         "category": prediction["category"], "schema_group": prediction["schema_group"],
                         "supported": supported, "candidate_count": prediction["candidate_count"],
                         "primary_sample_count": prediction["primary_sample_count"],
                         "primary_valid_sample_count": prediction["primary_valid_sample_count"],
                         "oracle_primary_candidate_correct": any(row["correct"] is True for row in oracle) if supported else None,
                         "correctness": {method: result["correct"] for method, result in evaluations.items()},
                         "evaluation": evaluations})
    fixed = choose_dev_method(outcomes, FIXED_METHODS)
    for row in outcomes:
        row["correctness"]["dev_selected_fixed"] = row["correctness"][fixed]
    dev_control = choose_dev_method(outcomes, EQUAL_SCORE_CONTROLS)
    methods = (*SCORE_METHODS, *FIXED_METHODS, *SAMPLE_METHODS, "dev_selected_fixed")
    splits = {split: [row for row in outcomes if row["split"] == split] for split in ("development", "test")}
    accuracy = {split: {method: accuracy_summary(rows, method) for method in methods}
                for split, rows in splits.items()}
    paired = {control: paired_comparison(splits["test"], control) for control in EQUAL_SCORE_CONTROLS}
    _holm_adjust(paired)
    strongest_test = min(EQUAL_SCORE_CONTROLS,
                         key=lambda method: (-accuracy["test"][method]["successes"], method))
    total_samples = sum(row["primary_sample_count"] for row in outcomes)
    report = {
        "format_version": 1, "evaluator": "restricted_scalar_bfcl_ast_value_v1",
        "receipt_sha256": sha256_file(args.receipt),
        "candidates_sha256": receipt["candidates"]["sha256"],
        "predictions_sha256": receipt["predictions"]["sha256"],
        "inputs_sha256": receipt["inputs_sha256"],
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "gold_source_sha256": {path.name: sha256_file(path) for path in args.gold},
        "model": ranking_manifest["model"], "generation_dtype": "bfloat16", "scoring_dtype": "float32",
        "population": len(outcomes), "development_count": len(splits["development"]),
        "test_count": len(splits["test"]),
        "unsupported_ids": [row["id"] for row in outcomes if not row["supported"]],
        "input_category_counts": dict(Counter(row["category"] for row in outcomes)),
        "selected_fixed_policy": fixed, "selected_fixed_policy_stage": "development labels only",
        "dev_selected_equal_score_control": dev_control,
        "primary_heldout_comparison": paired[dev_control],
        "strongest_observed_test_equal_score_control": strongest_test,
        "strongest_test_comparison_scope": "descriptive oracle control selection; no selected-after-test significance claim",
        "all_equal_score_paired_test_comparisons": paired,
        "accuracy": accuracy,
        "category_test_accuracy": {
            category: {method: accuracy_summary([row for row in splits["test"] if row["category"] == category], method)
                       for method in methods}
            for category in sorted({row["category"] for row in outcomes})},
        "candidate_coverage": {
            "with_primary_candidate": sum(row["candidate_count"] > 0 for row in outcomes),
            "zero_candidate": sum(row["candidate_count"] == 0 for row in outcomes),
            "primary_samples": total_samples,
            "valid_primary_samples": sum(row["primary_valid_sample_count"] for row in outcomes),
            "json_schema_validity": sum(row["primary_valid_sample_count"] for row in outcomes) / total_samples if total_samples else None,
            "oracle_correct_primary_candidate": sum(row["oracle_primary_candidate_correct"] is True for row in outcomes),
            "oracle_metric_denominator": sum(row["supported"] for row in outcomes)},
        "limitations": ["restricted scalar evaluator, not the full official BFCL score",
                         "119-item frozen input population; no claim of broad task generalization",
                         "unsupported gold retained; population bounds distinguish unknown labels",
                         "paired intervals are approximate; Holm correction is within model only",
                         "generation/scoring dtype differ by the prospective precision amendment",
                         "ranking execution costs are not the separate randomized ABBA timing study"],
        "records": outcomes,
    }
    atomic_json(args.output, report)
    print(json.dumps({"status": "evaluation_complete", "population": report["population"],
                      "unsupported": len(report["unsupported_ids"]), "selected_fixed_policy": fixed,
                      "output_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
