"""Synthetic input/gold separation tests; execute exclusively on DGX Spark."""

import copy
import json
from pathlib import Path

import pytest

from orbitcall.bfcl import (
    evaluate_call,
    freeze_inputs,
    input_eligibility,
    load_frozen_inputs,
    load_gold_after_freeze,
    render_messages,
    render_tools,
    schema_group_key,
    sha256_file,
    write_frozen_json,
)


def record(item_id="fixture_0", name="book"):
    return {
        "id": item_id,
        "question": [[{"role": "user", "content": "Book two tickets to New York."}]],
        "function": [{
            "name": name, "description": "Book a destination and ticket count.",
            "parameters": {
                "type": "dict", "properties": {
                    "destination": {"type": "string", "description": "Where to go"},
                    "count": {"type": "integer", "description": "Number of tickets"},
                }, "required": ["destination", "count"],
            },
        }],
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_eligibility_depends_only_on_inputs_and_rejects_optional_or_nested():
    row = record()
    assert input_eligibility(row) is None
    optional = copy.deepcopy(row)
    optional["function"][0]["parameters"]["required"].pop()
    assert input_eligibility(optional) == "optional_or_invalid_required_keys"
    nested = copy.deepcopy(row)
    nested["function"][0]["parameters"]["properties"]["destination"]["type"] = "array"
    assert input_eligibility(nested) == "not_flat_supported_scalars"
    multiple = copy.deepcopy(row)
    multiple["function"].append(copy.deepcopy(multiple["function"][0]))
    assert input_eligibility(multiple) == "not_one_function"
    extra = copy.deepcopy(row)
    extra["function"][0]["parameters"]["additionalProperties"] = True
    assert input_eligibility(extra) == "additional_keys_allowed"


def test_answer_fields_are_forbidden_in_input_stage():
    row = record()
    row["ground_truth"] = []
    with pytest.raises(ValueError, match="Answer fields"):
        input_eligibility(row)
    with pytest.raises(ValueError, match="Answer fields"):
        render_messages(row)


def test_schema_grouping_ignores_question_annotations_and_key_order():
    first = record()
    second = record("fixture_1")
    second["question"][0][0]["content"] = "A wholly different request."
    function = second["function"][0]
    function["description"] = "Changed prose"
    parameters = function["parameters"]
    parameters["properties"] = dict(reversed(list(parameters["properties"].items())))
    parameters["properties"]["count"]["description"] = "Changed annotation"
    parameters["required"].reverse()
    assert schema_group_key(first["function"][0]) == schema_group_key(function)
    function["parameters"]["properties"]["count"]["minimum"] = 1
    assert schema_group_key(first["function"][0]) != schema_group_key(function)


def test_frozen_split_is_complete_reproducible_grouped_and_keeps_failures(tmp_path):
    rows = [record(f"fixture_{i}", f"tool_{i // 2}") for i in range(12)]
    excluded = record("excluded")
    excluded["function"][0]["parameters"]["required"] = ["count"]
    input_path = tmp_path / "input.json"
    write_jsonl(input_path, [*rows, excluded])
    hashes = {"fixture": sha256_file(input_path)}
    output = tmp_path / "frozen.json"
    manifest = freeze_inputs({"fixture": input_path}, output, target_dev=4, expected_hashes=hashes)
    assert len(manifest["eligible_ids"]) == 12
    assert len(manifest["development_ids"]) == 4
    assert len(manifest["test_ids"]) == 8
    assert set(manifest["development_ids"]).isdisjoint(manifest["test_ids"])
    assert set(manifest["eligible_ids"]) == {row["id"] for row in rows}
    for ids in manifest["schema_groups"].values():
        assert set(ids) <= set(manifest["development_ids"]) or set(ids) <= set(manifest["test_ids"])
    assert manifest["answer_accessed"] is False
    assert manifest["exclusions"] == [{"id": "excluded", "category": "fixture", "row": 12,
                                       "reason": "optional_or_invalid_required_keys"}]
    assert freeze_inputs({"fixture": input_path}, output, target_dev=4, expected_hashes=hashes) == manifest
    assert len(load_frozen_inputs(output)) == 12
    assert len(load_frozen_inputs(output, "development")) == 4
    # No generated-call filter exists at preparation time: every eligible item
    # remains in the denominator even if a later generator yields zero calls.
    assert evaluate_call(None, rows[0]["function"][0], [{"tool_0": {
        "destination": ["New York"], "count": [2],
    }}]) == {"supported": True, "correct": False, "reason": "no_candidate"}


def test_freeze_rejects_wrong_hash_duplicate_id_and_overwrite(tmp_path):
    input_path = tmp_path / "input.json"
    write_jsonl(input_path, [record()])
    with pytest.raises(ValueError, match="hash mismatch"):
        freeze_inputs({"fixture": input_path}, tmp_path / "freeze.json", expected_hashes={"fixture": "0" * 64})
    write_jsonl(input_path, [record(), record()])
    with pytest.raises(ValueError, match="Duplicate input ID"):
        freeze_inputs({"fixture": input_path}, tmp_path / "freeze.json", expected_hashes={"fixture": sha256_file(input_path)})
    output = tmp_path / "immutable.json"
    write_frozen_json(output, {"a": 1})
    with pytest.raises(ValueError, match="Refusing to change"):
        write_frozen_json(output, {"a": 2})


def test_prompt_is_input_only_and_native_tools_are_standard_json_schema():
    row = record()
    messages = render_messages(row)
    assert messages[-1] == row["question"][0][0]
    assert "name and arguments" in messages[0]["content"]
    assert "Book a destination" in messages[0]["content"]
    assert "ground_truth" not in json.dumps(messages)
    tools = render_tools(row)
    parameters = tools[0]["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["additionalProperties"] is False
    assert row["function"][0]["parameters"]["type"] == "dict"


def test_gold_is_not_opened_until_both_frozen_hashes_pass(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.json"
    predictions = tmp_path / "predictions.json"
    gold = tmp_path / "gold.json"
    candidates.write_text('{"candidates": []}', encoding="utf-8")
    predictions.write_text('{"winner": null}', encoding="utf-8")
    write_jsonl(gold, [{"id": "fixture_0", "ground_truth": [{"book": {
        "destination": ["New York"], "count": [2],
    }}]}])
    real_open = Path.open
    opened = []

    def track_open(self, *args, **kwargs):
        if self == gold:
            opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    with pytest.raises(ValueError, match="freeze mismatch"):
        load_gold_after_freeze([gold], candidates_path=candidates, candidates_sha256="0" * 64,
                               predictions_path=predictions, predictions_sha256=sha256_file(predictions))
    assert opened == []
    with pytest.raises(ValueError, match="freeze mismatch"):
        load_gold_after_freeze([gold], candidates_path=candidates, candidates_sha256=sha256_file(candidates),
                               predictions_path=predictions, predictions_sha256="0" * 64)
    assert opened == []
    loaded = load_gold_after_freeze([gold], candidates_path=candidates, candidates_sha256=sha256_file(candidates),
                                  predictions_path=predictions, predictions_sha256=sha256_file(predictions))
    assert list(loaded) == ["fixture_0"]
    assert opened == [gold]


def test_scalar_ast_value_evaluation_is_separate_from_json_validity():
    schema = record()["function"][0]
    gold = [{"book": {"destination": ["New York", "NYC"], "count": [2]}}]
    correct = {"name": "book", "arguments": {"destination": "NEW-YORK", "count": 2}}
    assert evaluate_call(correct, schema, gold)["correct"] is True
    wrong = {"name": "book", "arguments": {"destination": "Paris", "count": 2}}
    assert evaluate_call(wrong, schema, gold)["correct"] is False
    wrong_type = {"name": "book", "arguments": {"destination": "NYC", "count": True}}
    assert evaluate_call(wrong_type, schema, gold)["correct"] is False
    assert evaluate_call(None, schema, gold)["correct"] is False


@pytest.mark.parametrize("gold", [
    [],
    [{"wrong": {"destination": ["New York"], "count": [2]}}],
    [{"book": {"destination": ["New York"], "count": ["variable_count"]}}],
    [{"book": {"destination": ["New York"], "count": [None]}}],
    [{"book": {"destination": ["New York"]}}],
])
def test_unsupported_gold_is_explicit_and_never_changes_the_input_population(gold):
    result = evaluate_call(None, record()["function"][0], gold)
    assert result == {"supported": False, "correct": None, "reason": "unsupported_gold"}


def test_float_value_semantics_and_identity_have_distinct_roles():
    schema = record()["function"][0]
    schema["parameters"]["properties"]["count"]["type"] = "float"
    gold = [{"book": {"destination": ["NYC"], "count": [2.0]}}]
    integer_call = {"name": "book", "arguments": {"destination": "NYC", "count": 2}}
    assert evaluate_call(integer_call, schema, gold)["correct"] is True
