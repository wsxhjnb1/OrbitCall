"""Original Agent workflow correctness tests; run exclusively on DGX Spark."""

import copy
from collections import Counter
import hashlib
import json

import pytest

from orbitcall.agent_tasks import (
    evaluate_after_freeze, execute_tool, input_provenance, render_messages, tasks, tool_specs,
)
from orbitcall.serialization import serialization_orbit, validate_call


def _task(family="catalog"):
    return next(row for row in tasks() if row["family"] == family)


def _frozen(tmp_path, task, response, trace):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"records": [{"id": task["id"], "final_response": response, "trace": trace}]}), encoding="utf-8")
    receipt = tmp_path / "freeze.json"
    receipt.write_text(json.dumps({"stage": "agent-predictions-frozen", "answers_accessed": False,
                                  "predictions": {"sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()}}), encoding="utf-8")
    return {"predictions_path": predictions, "freeze_receipt_path": receipt}


def _trace(task):
    values = task["input"]
    if task["family"] == "catalog":
        first = {"name": "catalog_lookup", "arguments": {"item_id": values["item_id"], "store": values["store"]}}
        observation = execute_tool(first, task)
        second = {"name": "order_total", "arguments": {"unit_price_cents": observation["result"]["unit_price_cents"],
                  "quantity": values["quantity"], "shipping_cents": observation["result"]["shipping_cents"]}}
    else:
        first = {"name": "calendar_policy", "arguments": {"region": values["region"], "year": values["year"]}}
        observation = execute_tool(first, task)
        second = {"name": "business_date", "arguments": {"start_date": values["start_date"],
                  "business_days": observation["result"]["business_days"], "calendar_id": observation["result"]["calendar_id"]}}
    final = execute_tool(second, task)
    return [{"call": first, "observation": observation}, {"call": second, "observation": final}], json.dumps(final["result"])


def test_task_population_and_group_split_are_input_only_and_repeatable():
    records = tasks()
    assert records == tasks()
    assert len(records) == len({row["id"] for row in records}) == 48
    assert Counter(row["split"] for row in records) == {"development": 12, "test": 36}
    assert Counter(row["family"] for row in records) == {"catalog": 24, "calendar": 24}
    assert len({row["group_id"] for row in records}) == 16
    for group in {row["group_id"] for row in records}:
        assert len({row["split"] for row in records if row["group_id"] == group}) == 1
    assert all(set(row) == {"id", "family", "group_id", "split", "input", "request"} for row in records)
    encoded = json.dumps(records)
    assert "unit_price_cents" not in encoded and "shipping_cents" not in encoded and "business_days" not in encoded
    assert input_provenance()["network_tools"] is False


def test_schema_keys_are_required_flat_and_orbits_complete():
    examples = {
        "catalog_lookup": {"item_id": "item_01", "store": "north"},
        "calendar_policy": {"region": "region_1", "year": 2026},
        "order_total": {"unit_price_cents": 1299, "quantity": 3, "shipping_cents": 450},
        "business_date": {"start_date": "2026-01-01", "business_days": 2, "calendar_id": "region_1_2026"},
    }
    for stage in (1, 2):
        for schema in tool_specs(stage):
            properties = schema["parameters"]["properties"]
            assert schema["parameters"]["required"] == list(properties)
            assert schema["parameters"]["additionalProperties"] is False
            call = {"name": schema["name"], "arguments": examples[schema["name"]]}
            assert validate_call(call, schema) == call
            assert len(serialization_orbit(call, key_order=tuple(properties))) == (2 if stage == 1 else 6)
    assert tool_specs(3) == []
    with pytest.raises(ValueError):
        tool_specs(True)


def test_catalog_and_order_execute_actual_integer_arithmetic():
    task = _task()
    first = execute_tool({"name": "catalog_lookup", "arguments": {"item_id": "item_01", "store": "north"}}, task)
    assert first["result"] == {"item_id": "item_01", "store": "north", "unit_price_cents": 1299, "shipping_cents": 450}
    second = execute_tool({"name": "order_total", "arguments": {"unit_price_cents": 1299, "quantity": 3, "shipping_cents": 450}}, task)
    assert second == {"ok": True, "tool": "order_total", "result": {"total_cents": 4347}}
    other = execute_tool({"name": "catalog_lookup", "arguments": {"item_id": "item_02", "store": "south"}}, task)
    assert other["result"]["unit_price_cents"] == 2375


@pytest.mark.parametrize("arguments,expected", [
    ({"start_date": "2026-01-01", "business_days": 2, "calendar_id": "region_1_2026"}, "2026-01-06"),
    ({"start_date": "2026-01-03", "business_days": 0, "calendar_id": "region_1_2026"}, "2026-01-03"),
    ({"start_date": "2026-03-27", "business_days": 3, "calendar_id": "region_2_2026"}, "2026-04-02"),
])
def test_business_date_respects_start_exclusion_weekends_and_fixture_closures(arguments, expected):
    result = execute_tool({"name": "business_date", "arguments": arguments}, _task("calendar"))
    assert result["ok"] and result["result"] == {"date": expected}


@pytest.mark.parametrize("call", [
    None,
    {"name": "catalog_lookup", "arguments": {"item_id": "missing", "store": "north"}},
    {"name": "order_total", "arguments": {"unit_price_cents": 100, "quantity": True, "shipping_cents": 0}},
    {"name": "order_total", "arguments": {"unit_price_cents": 100.0, "quantity": 1, "shipping_cents": 0}},
    {"name": "business_date", "arguments": {"start_date": "2026-02-30", "business_days": 2, "calendar_id": "region_1_2026"}},
    {"name": "business_date", "arguments": {"start_date": "2026-12-31", "business_days": 1, "calendar_id": "region_1_2026"}},
])
def test_errors_remain_observations_and_are_not_repaired(call):
    assert execute_tool(call, _task())["ok"] is False


def test_prompt_uses_actual_observation_without_reference_execution(monkeypatch):
    task = _task()
    sentinel = {"ok": True, "tool": "catalog_lookup", "result": {"unit_price_cents": 999991, "shipping_cents": 12345}}
    monkeypatch.setattr("orbitcall.agent_tasks.execute_tool", lambda *args: (_ for _ in ()).throw(AssertionError("Prompt must not execute a reference")))
    first = render_messages(task, [], 1)
    second = render_messages(task, [{"call": None, "observation": sentinel}], 2)
    assert "1299" not in json.dumps(first) and "999991" in json.dumps(second) and "12345" in json.dumps(second)
    with pytest.raises(ValueError):
        render_messages(task, [], 2)
    leaked = copy.deepcopy(task)
    leaked["expected_answer"] = 0
    with pytest.raises(ValueError):
        render_messages(leaked, [], 1)


@pytest.mark.parametrize("family", ["catalog", "calendar"])
def test_separate_evaluator_requires_frozen_retained_two_round_trace(tmp_path, family):
    task = _task(family)
    trace, response = _trace(task)
    paths = _frozen(tmp_path, task, response, trace)
    result = evaluate_after_freeze(task, response, trace, **paths)
    assert result["task_success"] and result["round_correct"] == [True, True]
    assert result["observation_consistent"] == [True, True]
    paths["predictions_path"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="byte-frozen"):
        evaluate_after_freeze(task, response, trace, **paths)


def test_evaluator_rejects_unretained_response_before_reference_access(tmp_path, monkeypatch):
    task = _task()
    trace, response = _trace(task)
    paths = _frozen(tmp_path, task, response, trace)
    monkeypatch.setattr("orbitcall.agent_tasks._reference", lambda *args: (_ for _ in ()).throw(AssertionError("Not frozen")))
    with pytest.raises(ValueError, match="retained"):
        evaluate_after_freeze(task, "{}", trace, **paths)


def test_wrong_actual_call_or_fabricated_observation_cannot_pass_with_correct_final(tmp_path):
    task = _task()
    trace, response = _trace(task)
    trace[0]["observation"]["result"]["unit_price_cents"] += 1
    paths = _frozen(tmp_path, task, response, trace)
    result = evaluate_after_freeze(task, response, trace, **paths)
    assert result["final_correct"] and not result["task_success"]
    assert result["observation_consistent"] == [False, True]


def test_duplicate_key_final_json_remains_failure(tmp_path):
    task = _task()
    trace, _ = _trace(task)
    response = '{"total_cents":false,"total_cents":0}'
    paths = _frozen(tmp_path, task, response, trace)
    result = evaluate_after_freeze(task, response, trace, **paths)
    assert not result["final_json_valid"] and not result["task_success"]


def test_null_candidate_pool_retains_two_round_errors_and_task_failure(tmp_path):
    task = _task()
    trace = [{"call": None, "observation": execute_tool(None, task)} for _ in range(2)]
    response = '{"error":"tool_workflow_failed"}'
    paths = _frozen(tmp_path, task, response, trace)
    result = evaluate_after_freeze(task, response, trace, **paths)
    assert result["complete_trace"] and result["observation_consistent"] == [True, True]
    assert result["round_correct"] == [False, False] and not result["task_success"]


def test_evaluation_cannot_change_input_under_a_frozen_episode_id(tmp_path, monkeypatch):
    task = _task()
    trace, response = _trace(task)
    paths = _frozen(tmp_path, task, response, trace)
    task["input"]["quantity"] += 1
    monkeypatch.setattr("orbitcall.agent_tasks._reference", lambda *args: (_ for _ in ()).throw(AssertionError("Changed input")))
    with pytest.raises(ValueError, match="original input"):
        evaluate_after_freeze(task, response, trace, **paths)
