"""Answer-free finite-family ranking and deterministic sampling controls.

This module contains no model or gold loader. All orders are relative to the
frozen input function's property order, never a generated dictionary's order.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from .serialization import parse_call, semantic_key, serialization_orbit, validate_call


PRIMARY_SEEDS = (17, 29, 43, 71)
ADDITIONAL_SEEDS = (101, 113, 127, 139)
FIXED_POLICIES = tuple(itertools.permutations(range(3)))
SCORE_METHODS = ("mass", "max_order", "mean_logprob", "length_normalized", "canonical_schema")
FIXED_METHODS = tuple("fixed_" + "".join(map(str, order)) for order in FIXED_POLICIES)
SAMPLE_METHODS = ("vote4", "vote8", "greedy", "freeform_vote")


def logsumexp(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Finite, nonempty sequence logprobs are required")
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _parsed_sample(sample: Mapping[str, Any], function: Mapping[str, Any]) -> dict | None:
    if sample.get("call") is None:
        # Re-parsing distinguishes a genuine invalid output from a corrupted
        # generator artifact that omitted a valid parsed call.
        try:
            parse_call(sample["text"], function)
        except ValueError:
            return None
        raise ValueError("Generator artifact omitted a valid parsed call")
    call = validate_call(sample["call"], function)
    parsed = parse_call(sample["text"], function)
    key = semantic_key(call)
    if key != semantic_key(parsed) or sample.get("semantic_key") != key:
        raise ValueError("Generator text/call/semantic-key mismatch")
    return call


def select_samples(row: Mapping[str, Any], mode: str, seeds: Sequence[int]) -> list[dict]:
    selected = [sample for sample in row["samples"]
                if sample["mode"] == mode and sample["seed"] in seeds]
    seen = set()
    for sample in selected:
        if sample["seed"] in seen or sample.get("id", row["id"]) != row["id"]:
            raise ValueError("Duplicate seed or mismatched sample ID")
        seen.add(sample["seed"])
    return sorted(selected, key=lambda sample: seeds.index(sample["seed"]))


def validate_sample_plan(row: Mapping[str, Any]) -> None:
    expected = ({("greedy", 0)} | {("direct", seed) for seed in (*PRIMARY_SEEDS, *ADDITIONAL_SEEDS)}
                | {(mode, seed) for mode in ("freeform", "reformat") for seed in PRIMARY_SEEDS})
    actual = [(sample["mode"], sample["seed"]) for sample in row["samples"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or changed frozen candidate generation plan")


def candidate_pool(
    row: Mapping[str, Any], *, mode: str = "direct", seeds: Sequence[int] = PRIMARY_SEEDS,
) -> list[dict[str, Any]]:
    calls: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    for sample in select_samples(row, mode, seeds):
        call = _parsed_sample(sample, row["function"])
        if call is not None:
            key = semantic_key(call)
            calls[key] = call
            counts[key] += 1
    return [{"semantic_key": key, "call": calls[key], "sample_count": counts[key]}
            for key in sorted(calls)]


def build_families(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = tuple(row["function"]["parameters"]["properties"])
    if len(keys) not in (2, 3):
        raise ValueError("Only frozen two/three-key schemas may be scored")
    orders = tuple(itertools.permutations(keys))
    families = []
    begin = 0
    for candidate in candidate_pool(row):
        texts = serialization_orbit(candidate["call"], key_order=keys)
        fixed_indices = {}
        for policy, method in zip(FIXED_POLICIES, FIXED_METHODS, strict=True):
            order = tuple(keys[index] for index in policy if index < len(keys))
            fixed_indices[method] = orders.index(order)
        families.append({**candidate, "texts": list(texts),
                         "orders": [list(order) for order in orders],
                         "fixed_indices": fixed_indices,
                         "sequence_begin": begin, "sequence_end": begin + len(texts)})
        begin += len(texts)
    return families


def _winner(candidates: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    if not candidates:
        return {"call": None, "semantic_key": None, "score": None, "margin": None,
                "runner_up_key": None, "candidate_scores": []}
    if len(scores) != len(candidates) or any(not math.isfinite(value) for value in scores):
        raise ValueError("Malformed/non-finite candidate score vector")
    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], candidates[index]["semantic_key"]))
    best = order[0]
    second = order[1] if len(order) > 1 else None
    return {"call": candidates[best]["call"], "semantic_key": candidates[best]["semantic_key"],
            "score": scores[best], "margin": scores[best] - scores[second] if second is not None else None,
            "runner_up_key": candidates[second]["semantic_key"] if second is not None else None,
            "candidate_scores": [{"semantic_key": candidate["semantic_key"], "score": score}
                                 for candidate, score in zip(candidates, scores, strict=True)]}


def aggregate_scores(
    families: Sequence[Mapping[str, Any]], sequence_logprobs: Sequence[float],
    token_lengths: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Use exactly the same EOS-inclusive scores for every likelihood control."""
    total = families[-1]["sequence_end"] if families else 0
    if len(sequence_logprobs) != total or len(token_lengths) != total:
        raise ValueError("Sequence score/length/family count mismatch")
    if any(type(length) is not int or length < 1 for length in token_lengths):
        raise ValueError("Every sequence must include an explicit EOS token")
    method_values: dict[str, list[float]] = {method: [] for method in (*SCORE_METHODS, *FIXED_METHODS)}
    for family in families:
        begin, end = family["sequence_begin"], family["sequence_end"]
        values = list(sequence_logprobs[begin:end])
        lengths = token_lengths[begin:end]
        if len(values) != len(family["texts"]):
            raise ValueError("Incomplete finite serialization family")
        method_values["mass"].append(logsumexp(values))
        method_values["max_order"].append(max(values))
        method_values["mean_logprob"].append(math.fsum(values) / len(values))
        method_values["length_normalized"].append(
            math.fsum(value / length for value, length in zip(values, lengths, strict=True)) / len(values)
        )
        method_values["canonical_schema"].append(values[0])
        for method in FIXED_METHODS:
            method_values[method].append(values[family["fixed_indices"][method]])
    return {method: _winner(families, values) for method, values in method_values.items()}


def sample_controls(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    controls = {}
    for method, mode, seeds in (
        ("vote4", "direct", PRIMARY_SEEDS),
        ("vote8", "direct", (*PRIMARY_SEEDS, *ADDITIONAL_SEEDS)),
        ("greedy", "greedy", (0,)),
        ("freeform_vote", "reformat", PRIMARY_SEEDS),
    ):
        samples = select_samples(row, mode, seeds)
        pool = candidate_pool(row, mode=mode, seeds=seeds)
        valid = sum(candidate["sample_count"] for candidate in pool)
        winner = _winner(pool, [float(candidate["sample_count"]) for candidate in pool])
        controls[method] = {**winner, "sample_count": len(samples), "valid_sample_count": valid,
                            "invalid_sample_count": len(samples) - valid,
                            "unique_candidate_count": len(pool),
                            "selection": "plurality; lexical semantic-key tie break"}
    return controls


def compare_scoring_results(
    reference: Mapping[str, Any], alternative: Mapping[str, Any], *, tolerance: float = 1e-3,
) -> dict[str, Any]:
    left, right = reference["scores"]["token_logprobs"], alternative["scores"]["token_logprobs"]
    for result in (reference, alternative):
        scores = result["scores"]
        tokens, sequences = scores["token_logprobs"], scores["sequence_logprobs"]
        if len(tokens) != len(sequences) or any(not path for path in tokens):
            raise ValueError("Malformed sequence/token score shape")
        if any(not math.isfinite(value) for path in tokens for value in path) or any(
            not math.isfinite(value) for value in sequences
        ):
            raise ValueError("Non-finite source scores")
        if any(abs(math.fsum(path) - total) > 1e-5
               for path, total in zip(tokens, sequences, strict=True)):
            raise ValueError("Sequence score does not equal its EOS-inclusive token sum")
    if set(reference["predictions"]) != set(alternative["predictions"]):
        raise ValueError("Different ranking controls in scoring comparison")
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right, strict=True)):
        raise ValueError("Scoring methods returned different token paths")
    errors = [[abs(a - b) for a, b in zip(x, y, strict=True)]
              for x, y in zip(left, right, strict=True)]
    if any(not math.isfinite(error) for path in errors for error in path):
        raise ValueError("Non-finite method comparison")
    sequence_errors = [abs(a - b) for a, b in zip(reference["scores"]["sequence_logprobs"],
                                                alternative["scores"]["sequence_logprobs"], strict=True)]
    agreements = {method: result["semantic_key"] == alternative["predictions"][method]["semantic_key"]
                  for method, result in reference["predictions"].items()}
    maximum = max((error for path in errors for error in path), default=0.0)
    sequence_bounds = [len(path) * tolerance for path in left]
    bounded = all(error <= bound + 1e-8 for error, bound in zip(sequence_errors, sequence_bounds, strict=True))
    return {"reference": reference["scores"]["method"], "alternative": alternative["scores"]["method"],
            "token_error_tolerance": tolerance, "maximum_token_error": maximum,
            "maximum_sequence_error": max(sequence_errors, default=0.0),
            "per_sequence_token_errors": errors, "sequence_errors": sequence_errors,
            "sequence_error_bounds": sequence_bounds,
            "winner_agreement": agreements,
            "reference_winner_margins": {method: result["margin"] for method, result in reference["predictions"].items()},
            "alternative_winner_margins": {method: result["margin"] for method, result in alternative["predictions"].items()},
            "ambiguous_or_invalid_winner_controls": [method for method, agrees in agreements.items() if not agrees],
            "valid": maximum <= tolerance and bounded and all(agreements.values())}
