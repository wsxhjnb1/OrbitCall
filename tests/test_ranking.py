"""Ranking, numerical gates, and held-out selection contracts; Spark only."""

import copy
import json
from pathlib import Path
import runpy

import pytest

from orbitcall.ranking import (
    ADDITIONAL_SEEDS, FIXED_METHODS, PRIMARY_SEEDS, aggregate_scores, build_families,
    candidate_pool, compare_scoring_results, logsumexp, sample_controls, validate_sample_plan,
)
from orbitcall.serialization import semantic_key


def function():
    return {"name": "f", "parameters": {"type": "dict", "properties": {
        "z": {"type": "integer"}, "a": {"type": "string"},
    }, "required": ["z", "a"]}}


def call(number, *, reverse=False):
    arguments = {"a": "x", "z": number} if reverse else {"z": number, "a": "x"}
    return {"name": "f", "arguments": arguments}


def sample(mode, seed, value):
    return {"id": "fixture", "mode": mode, "seed": seed, "call": value,
            "text": json.dumps(value) if value is not None else "invalid JSON",
            "semantic_key": semantic_key(value) if value is not None else None}


def row(samples):
    return {"id": "fixture", "function": function(), "samples": samples}


def full_row():
    samples = [sample("greedy", 0, call(3))]
    samples.extend(sample("direct", seed, call(1 if index < 3 else 2)) for index, seed in enumerate(PRIMARY_SEEDS))
    samples.extend(sample("direct", seed, call(2)) for seed in ADDITIONAL_SEEDS)
    samples.extend(sample("reformat", seed, call(4)) for seed in PRIMARY_SEEDS)
    samples.extend(sample("freeform", seed, None) for seed in PRIMARY_SEEDS)
    return row(samples)


def test_primary_pool_is_isolated_deduplicated_and_typed():
    generated = full_row()
    pool = candidate_pool(generated)
    assert {candidate["call"]["arguments"]["z"] for candidate in pool} == {1, 2}
    assert sorted(candidate["sample_count"] for candidate in pool) == [1, 3]
    assert all(candidate["call"]["arguments"]["z"] != 4 for candidate in pool)
    controls = sample_controls(generated)
    assert controls["vote4"]["call"]["arguments"]["z"] == 1
    assert controls["vote8"]["call"]["arguments"]["z"] == 2
    assert controls["greedy"]["call"]["arguments"]["z"] == 3
    assert controls["freeform_vote"]["call"]["arguments"]["z"] == 4
    assert controls["vote8"]["sample_count"] == 8


def test_zero_candidate_is_a_recorded_failure():
    generated = row([sample("direct", seed, None) for seed in PRIMARY_SEEDS])
    assert build_families(generated) == []
    assert all(prediction["call"] is None for prediction in aggregate_scores([], [], []).values())
    controls = sample_controls(generated)
    assert controls["vote4"]["sample_count"] == 4
    assert controls["vote4"]["valid_sample_count"] == 0
    assert controls["vote4"]["invalid_sample_count"] == 4
    assert controls["vote4"]["call"] is None


def test_input_schema_order_is_consistent_across_output_orders():
    generated = row([sample("direct", 17, call(1, reverse=True)), sample("direct", 29, call(2))])
    families = build_families(generated)
    for family in families:
        assert family["orders"] == [["z", "a"], ["a", "z"]]
        assert list(json.loads(family["texts"][0])["arguments"]) == ["z", "a"]
        assert family["fixed_indices"] == {
            "fixed_012": 0, "fixed_021": 0, "fixed_102": 1,
            "fixed_120": 1, "fixed_201": 0, "fixed_210": 1,
        }
    results = aggregate_scores(families, [-1.0, -10.0, -1.3, -1.3], [10, 10, 10, 10])
    assert results["mass"]["call"]["arguments"]["z"] == 2
    assert results["max_order"]["call"]["arguments"]["z"] == 1
    assert results["canonical_schema"]["call"]["arguments"]["z"] == 1
    assert results["mean_logprob"]["call"]["arguments"]["z"] == 2
    assert results["fixed_102"]["call"]["arguments"]["z"] == 2
    normalized = aggregate_scores(families, [-1.0, -10.0, -1.3, -1.3], [10, 10, 1, 1])
    assert normalized["length_normalized"]["call"]["arguments"]["z"] == 1


def test_three_key_family_has_all_six_consistent_fixed_orders():
    generated = row([sample("direct", 17, {"name": "f", "arguments": {"a": "x", "b": True, "z": 1}})])
    generated["function"]["parameters"]["properties"]["b"] = {"type": "boolean"}
    generated["function"]["parameters"]["required"].append("b")
    family = build_families(generated)[0]
    assert len(family["texts"]) == 6
    assert sorted(family["fixed_indices"].values()) == list(range(6))
    assert family["orders"][0] == ["z", "a", "b"]


def test_mass_is_stable_for_sequences_that_would_underflow_probability():
    assert logsumexp([-10000.0, -10000.0]) == pytest.approx(-10000.0 + 0.6931471805599453)
    assert logsumexp([-10000.0, -20000.0]) == pytest.approx(-10000.0)
    for values in ([], [float("nan")], [float("inf")], [float("-inf")]):
        with pytest.raises(ValueError):
            logsumexp(values)


def test_equal_scores_and_equal_vote_counts_use_same_declared_lexical_tie_break():
    generated = row([sample("direct", 17, call(2)), sample("direct", 29, call(1))])
    families = build_families(generated)
    expected = min(semantic_key(call(1)), semantic_key(call(2)))
    likelihood = aggregate_scores(families, [-2.0] * 4, [3] * 4)
    assert all(result["semantic_key"] == expected for result in likelihood.values())
    assert sample_controls(generated)["vote4"]["semantic_key"] == expected


def test_corrupted_generator_and_incomplete_plan_fail_closed():
    generated = full_row()
    validate_sample_plan(generated)
    incomplete = copy.deepcopy(generated)
    incomplete["samples"].pop()
    with pytest.raises(ValueError, match="generation plan"):
        validate_sample_plan(incomplete)
    corrupted = row([sample("direct", 17, call(1))])
    corrupted["samples"][0]["call"] = call(2)
    with pytest.raises(ValueError, match="mismatch"):
        candidate_pool(corrupted)
    omitted = row([sample("direct", 17, call(1))])
    omitted["samples"][0]["call"] = None
    with pytest.raises(ValueError, match="omitted"):
        candidate_pool(omitted)


def scored(method="full", *, offset=0.0, key="a", margin=0.5):
    paths = [[-1.0 + offset, -2.0], [-3.0, -4.0]]
    return {"scores": {"method": method, "token_logprobs": paths,
                       "sequence_logprobs": [sum(path) for path in paths]},
            "predictions": {"mass": {"semantic_key": key, "margin": margin}}}


def test_comparison_retains_per_edge_sequence_errors_and_margins():
    comparison = compare_scoring_results(scored(), scored("tree", offset=0.0001))
    assert comparison["valid"] is True
    assert comparison["maximum_token_error"] == pytest.approx(0.0001)
    assert comparison["sequence_errors"][0] == pytest.approx(0.0001)
    assert comparison["sequence_error_bounds"] == [0.002, 0.002]
    assert comparison["reference_winner_margins"]["mass"] == 0.5


def test_near_tie_winner_change_is_invalid_even_below_numeric_tolerance():
    comparison = compare_scoring_results(scored(margin=0.00001),
                                         scored("tree", offset=0.0001, key="b", margin=0.00001))
    assert comparison["maximum_token_error"] < 1e-3
    assert comparison["valid"] is False
    assert comparison["ambiguous_or_invalid_winner_controls"] == ["mass"]


@pytest.mark.parametrize("mutation", ["shape", "nan", "sum", "controls"])
def test_malformed_score_sources_rejected_before_zipping(mutation):
    alternative = scored("tree")
    if mutation == "shape":
        alternative["scores"]["token_logprobs"].pop()
    elif mutation == "nan":
        alternative["scores"]["token_logprobs"][0][0] = float("nan")
    elif mutation == "sum":
        alternative["scores"]["sequence_logprobs"][0] += 1.0
    else:
        alternative["predictions"] = {}
    with pytest.raises(ValueError):
        compare_scoring_results(scored(), alternative)


def test_development_selection_cannot_use_test_labels():
    evaluator = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_predictions.py"))
    choose = evaluator["choose_dev_method"]
    rows = [
        {"split": "development", "supported": True,
         "correctness": {method: method == "fixed_102" for method in FIXED_METHODS}},
        {"split": "test", "supported": True,
         "correctness": {method: method == "fixed_012" for method in FIXED_METHODS}},
    ]
    assert choose(rows, FIXED_METHODS) == "fixed_102"
    rows[1]["correctness"] = {method: method != "fixed_102" for method in FIXED_METHODS}
    assert choose(rows, FIXED_METHODS) == "fixed_102"


def test_evaluation_statistics_preserve_unknown_population_bounds_and_pairing():
    evaluator = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_predictions.py"))
    rows = [
        {"supported": True, "schema_group": "g1", "correctness": {"mass": True, "max_order": False}},
        {"supported": True, "schema_group": "g2", "correctness": {"mass": False, "max_order": False}},
        {"supported": False, "schema_group": "g3", "correctness": {"mass": None, "max_order": None}},
    ]
    summary = evaluator["accuracy_summary"](rows, "mass")
    assert summary["population"] == 3
    assert summary["supported"] == 2
    assert summary["supported_accuracy"] == 0.5
    assert summary["all_population_accuracy_bounds"] == [1 / 3, 2 / 3]
    paired = evaluator["paired_comparison"](rows, "max_order")
    assert paired["accuracy_difference"] == 0.5
    assert paired["mass_only_correct"] == 1
    assert paired["control_only_correct"] == 0
    assert paired["schema_count"] == 2
    assert paired["exact_mcnemar_p"] == 1.0
