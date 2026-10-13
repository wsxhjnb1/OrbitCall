"""Original, deterministic local fixtures for a two-round Agent workflow.

The catalog and calendars are MIT-licensed test data, not real merchant prices
or government holidays. No network, sleep, model, or gold asset is used here.
``tasks`` and prompt rendering expose inputs only. Reference task outcomes are
computed only by the separately gated ``evaluate_after_freeze`` entry point.
Run input construction, tools, tests and evaluation exclusively on Spark.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

from .serialization import CallValidationError, scalar_equal, strict_json_loads, validate_call


FORMAT_VERSION = 1
SPLIT_SALT = "orbitcall-local-agent-input-only-v1"
PROVENANCE = {
    "kind": "original deterministic local test fixtures",
    "license": "MIT",
    "author": "Yuhao Xie / OrbitCall",
    "external_assets": [],
    "network_tools": False,
    "artificial_tool_latency": False,
    "scope": "two round-specific tools and final JSON; not an official Agent leaderboard",
}

# Fixture-only catalog prices, in integer cents, and actual local shipping lookup.
_PRICES = {
    "item_01": {"north": 1299, "south": 1399},
    "item_02": {"north": 2475, "south": 2375},
    "item_03": {"north": 3600, "south": 3750},
    "item_04": {"north": 825, "south": 975},
    "item_05": {"north": 4999, "south": 4799},
    "item_06": {"north": 1575, "south": 1675},
    "item_07": {"north": 6850, "south": 6950},
    "item_08": {"north": 2125, "south": 2025},
}
_SHIPPING = {"north": 450, "south": 650}
_LEADS = {"region_1": 2, "region_2": 3, "region_3": 5, "region_4": 7}
_YEARS = (2026, 2027)
# These closure dates are invented local test fixtures, explicitly not holidays.
_CLOSURES = {
    "region_1": ("01-02", "04-06", "11-23"),
    "region_2": ("01-05", "03-30", "11-24"),
    "region_3": ("01-06", "03-31", "11-25"),
    "region_4": ("01-07", "04-01", "11-26"),
}


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _function(name: str, description: str, properties: dict) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}


def tool_specs(stage: int) -> list[dict]:
    """Return BFCL-style schemas for both tools available in the named round.

    Source ``properties`` order fixes the serialization orbit. Stages are 1/2
    for actual tool rounds and 3 for the final response (no tools). These are
    plain function schemas; a caller may wrap them as OpenAI ``type=function``.
    """
    if type(stage) is not int or stage not in (1, 2, 3):
        raise ValueError("Agent stage must be integer 1, 2, or 3")
    if stage == 1:
        return [
            _function("catalog_lookup", "Look up one item in the local fixture store. Returns unit_price_cents and shipping_cents.", {
                "item_id": {"type": "string", "enum": list(_PRICES)},
                "store": {"type": "string", "enum": list(_SHIPPING)},
            }),
            _function("calendar_policy", "Look up a local fixture region/year. Returns business_days and calendar_id.", {
                "region": {"type": "string", "enum": list(_LEADS)},
                "year": {"type": "integer", "enum": list(_YEARS)},
            }),
        ]
    if stage == 2:
        return [
            _function("order_total", "Compute quantity*unit_price_cents+shipping_cents using the actual lookup result. All amounts are integer cents.", {
                "unit_price_cents": {"type": "integer", "minimum": 0, "maximum": 1000000},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
                "shipping_cents": {"type": "integer", "minimum": 0, "maximum": 1000000},
            }),
            _function("business_date", "Advance by business_days strictly AFTER start_date. Saturdays, Sundays and fixture closure dates are excluded. Zero days returns start_date. Use the actual calendar_policy result.", {
                "start_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
                "business_days": {"type": "integer", "minimum": 0, "maximum": 31},
                "calendar_id": {"type": "string", "enum": [f"{region}_{year}" for region in _LEADS for year in _YEARS]},
            }),
        ]
    return []


def tasks() -> list[dict]:
    """Construct 48 input-only tasks with deterministic 12-dev/36-test groups.

    Each family has eight groups and three input variants per group. The first
    two salted-hash groups per family are development. No execution output,
    correct call, reference answer, or price/calendar data is placed in a task.
    """
    records = []
    for item_id in _PRICES:
        for variant, (store, quantity) in enumerate((("north", 1), ("south", 3), ("north", 7))):
            records.append({"id": f"catalog_{item_id}_{variant}", "family": "catalog",
                            "group_id": f"catalog:{item_id}", "split": None,
                            "input": {"item_id": item_id, "store": store, "quantity": quantity},
                            "request": f"Find the total in integer cents for {quantity} units of {item_id} from the {store} fixture store, including its shipping fee. First look up the item/store, then calculate the order total. Finally respond with exactly {{\"total_cents\":integer}}."})
    for region in _LEADS:
        for year in _YEARS:
            for variant, month_day in enumerate(("01-01", "03-27", "11-20")):
                start = f"{year}-{month_day}"
                records.append({"id": f"calendar_{region}_{year}_{variant}", "family": "calendar",
                                "group_id": f"calendar:{region}:{year}", "split": None,
                                "input": {"region": region, "year": year, "start_date": start},
                                "request": f"For the {region} local fixture calendar in {year}, find the scheduled date starting strictly after {start}. First look up its policy, then use the returned business-day lead and calendar ID to calculate the date. Finally respond with exactly {{\"date\":\"YYYY-MM-DD\"}}."})
    for family in ("catalog", "calendar"):
        groups = {row["group_id"] for row in records if row["family"] == family}
        ordered = sorted(groups, key=lambda group: (hashlib.sha256(f"{SPLIT_SALT}:{group}".encode()).hexdigest(), group))
        development = set(ordered[:2])
        for row in records:
            if row["family"] == family:
                row["split"] = "development" if row["group_id"] in development else "test"
    return sorted(records, key=lambda row: row["id"])


def _validate_task(task: dict) -> None:
    if (type(task) is not dict or set(task) != {"id", "family", "group_id", "split", "input", "request"}
            or task["family"] not in {"catalog", "calendar"}
            or task["split"] not in {"development", "test"}
            or any(type(task[key]) is not str or not task[key] for key in ("id", "group_id", "request"))
            or type(task["input"]) is not dict):
        raise ValueError("Expected one input-only local Agent task")
    values = task["input"]
    if task["family"] == "catalog":
        if (set(values) != {"item_id", "store", "quantity"} or values["item_id"] not in _PRICES
                or values["store"] not in _SHIPPING or type(values["quantity"]) is not int
                or not 1 <= values["quantity"] <= 100):
            raise ValueError("Malformed catalog task inputs")
    elif (set(values) != {"region", "year", "start_date"} or values["region"] not in _LEADS
          or type(values["year"]) is not int or values["year"] not in _YEARS
          or _iso_date(values["start_date"]).year != values["year"]):
        raise ValueError("Malformed calendar task inputs")


def render_messages(task: dict, history: list[dict], stage: int) -> list[dict]:
    """Render actual feedback without executing tools or consulting references.

    ``history`` consists of {call, observation} events, in execution order.
    Extra trace metadata is not rendered. Observations are encoded unchanged,
    including errors. Plain assistant/user content deliberately implements the
    compact-JSON protocol; it does not imply native chat-template tool parsing.
    """
    _validate_task(task)
    schemas = tool_specs(stage)
    if type(history) is not list or len(history) != stage - 1:
        raise ValueError("Each next stage requires the actual preceding tool events")
    system = ("You are a local tool-using assistant. Fixture data is not a live external service. "
              "Use only the actual tool observations; do not guess lookup values. ")
    if stage < 3:
        system += ("Return exactly one compact JSON object with name and arguments, no explanation or Markdown. "
                   "Choose a tool available in this round; every argument is required and no extra keys are allowed. "
                   f"Current round: {stage}. Tool schemas: {_compact(schemas)}")
    else:
        system += ("Both tool rounds have finished. Return only the final JSON object requested by the user. "
                   "Use the actual second tool result. If the workflow failed, return exactly {\"error\":\"tool_workflow_failed\"}.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task["request"]}]
    for event in history:
        if type(event) is not dict or not {"call", "observation"} <= set(event):
            raise ValueError("History requires call and actual observation")
        messages.append({"role": "assistant", "content": _compact(event["call"])})
        messages.append({"role": "user", "content": "Actual local tool observation: " + _compact(event["observation"])})
    return messages


def _iso_date(text: str) -> date:
    if type(text) is not str:
        raise ValueError("Date must be a string")
    parsed = date.fromisoformat(text)
    if parsed.isoformat() != text:
        raise ValueError("Date must be canonical YYYY-MM-DD")
    return parsed


def _calendar(calendar_id: str) -> tuple[str, int, frozenset[date]]:
    region, separator, year_text = calendar_id.rpartition("_")
    if not separator or region not in _LEADS or year_text not in {str(year) for year in _YEARS}:
        raise ValueError("Unknown fixture calendar")
    year = int(year_text)
    return region, year, frozenset(_iso_date(f"{year}-{part}") for part in _CLOSURES[region])


def execute_tool(call: dict | None, task: dict) -> dict:
    """Execute the selected call as given; never correct it to the task intent.

    A wrong but valid item, store, region, year or arithmetic argument therefore
    produces its real local result. Errors are observations, not hidden repairs.
    Timing belongs to the caller and includes this function's real CPU work.
    """
    _validate_task(task)
    if call is None:
        return {"ok": False, "tool": None, "error": {"code": "no_valid_call", "message": "No valid generated candidate was available."}}
    tool = call.get("name") if type(call) is dict else None
    try:
        specs = {schema["name"]: schema for stage in (1, 2) for schema in tool_specs(stage)}
        if tool not in specs:
            raise CallValidationError("Unknown local tool")
        validated = validate_call(call, specs[tool])
        args = validated["arguments"]
        if tool == "catalog_lookup":
            result = {"item_id": args["item_id"], "store": args["store"],
                      "unit_price_cents": _PRICES[args["item_id"]][args["store"]],
                      "shipping_cents": _SHIPPING[args["store"]]}
        elif tool == "order_total":
            result = {"total_cents": args["unit_price_cents"] * args["quantity"] + args["shipping_cents"]}
        elif tool == "calendar_policy":
            result = {"region": args["region"], "year": args["year"],
                      "business_days": _LEADS[args["region"]],
                      "calendar_id": f"{args['region']}_{args['year']}"}
        else:
            _, year, closures = _calendar(args["calendar_id"])
            current = _iso_date(args["start_date"])
            if current.year != year:
                raise ValueError("Start date is outside the selected fixture calendar year")
            remaining = args["business_days"]
            while remaining:
                current += timedelta(days=1)
                if current.year != year:
                    raise ValueError("Result is outside the selected fixture calendar year")
                if current.weekday() < 5 and current not in closures:
                    remaining -= 1
            result = {"date": current.isoformat()}
        return {"ok": True, "tool": tool, "result": result}
    except (CallValidationError, ValueError, TypeError, OverflowError) as exc:
        return {"ok": False, "tool": tool if type(tool) is str else None,
                "error": {"code": "invalid_tool_call", "message": str(exc)}}


def _reference(task: dict) -> tuple[list[dict], dict]:
    """Evaluator-only reference construction; never called by prompt rendering."""
    values = task["input"]
    if task["family"] == "catalog":
        first = {"name": "catalog_lookup", "arguments": {"item_id": values["item_id"], "store": values["store"]}}
        observation = execute_tool(first, task)["result"]
        second = {"name": "order_total", "arguments": {"unit_price_cents": observation["unit_price_cents"],
                  "quantity": values["quantity"], "shipping_cents": observation["shipping_cents"]}}
    else:
        first = {"name": "calendar_policy", "arguments": {"region": values["region"], "year": values["year"]}}
        observation = execute_tool(first, task)["result"]
        second = {"name": "business_date", "arguments": {"start_date": values["start_date"],
                  "business_days": observation["business_days"], "calendar_id": observation["calendar_id"]}}
    return [first, second], execute_tool(second, task)["result"]


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_after_freeze(task: dict, final_response: str, trace: list[dict], *,
                          predictions_path: Path, freeze_receipt_path: Path) -> dict:
    """Evaluate one retained prediction only after verifying its immutable file.

    Receipt: {stage:'agent-predictions-frozen', answers_accessed:false,
    predictions:{sha256:...}}. Prediction file: {records:[{id,final_response,
    trace,...},...]}. A matching saved row must exist before any reference is
    constructed. Duplicate episode IDs across arms/windows are allowed, but
    the evaluated response/trace must exactly match retained JSON.
    """
    _validate_task(task)
    receipt = strict_json_loads(Path(freeze_receipt_path).read_text(encoding="utf-8"))
    digest = receipt.get("predictions", {}).get("sha256") if type(receipt) is dict else None
    if (type(receipt) is not dict or receipt.get("stage") != "agent-predictions-frozen" or receipt.get("answers_accessed") is not False
            or type(digest) is not str or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or _file_sha(Path(predictions_path)) != digest):
        raise ValueError("Agent predictions are not byte-frozen; references remain unopened")
    saved = strict_json_loads(Path(predictions_path).read_text(encoding="utf-8"))
    if (type(saved) is not dict or type(saved.get("records")) is not list
            or not any(row.get("id") == task["id"] and row.get("final_response") == final_response
                       and scalar_equal(row.get("trace"), trace) for row in saved["records"] if type(row) is dict)):
        raise ValueError("Evaluated response/trace is not a retained frozen prediction")
    frozen_inputs = [row for row in tasks() if row["id"] == task["id"]]
    if len(frozen_inputs) != 1 or not scalar_equal(task, frozen_inputs[0]):
        raise ValueError("Evaluated task differs from the original input-only fixture")
    expected_calls, expected_final = _reference(task)
    try:
        actual_final = strict_json_loads(final_response)
        valid_final = (type(actual_final) is dict and set(actual_final) == set(expected_final)
                       and all(type(actual_final[key]) is type(value) for key, value in expected_final.items()))
        if valid_final and task["family"] == "calendar":
            _iso_date(actual_final["date"])
        elif valid_final:
            valid_final = 0 <= actual_final["total_cents"] <= 101000000
    except (CallValidationError, ValueError):
        actual_final, valid_final = None, False
    round_correct, observation_consistent = [], []
    if type(trace) is list:
        for index, event in enumerate(trace[:2]):
            if type(event) is dict and {"call", "observation"} <= set(event):
                round_correct.append(scalar_equal(event["call"], expected_calls[index]))
                observation_consistent.append(scalar_equal(event["observation"], execute_tool(event["call"], task)))
            else:
                round_correct.append(False)
                observation_consistent.append(False)
    round_correct += [False] * (2 - len(round_correct))
    observation_consistent += [False] * (2 - len(observation_consistent))
    complete_trace = type(trace) is list and len(trace) == 2
    final_correct = valid_final and scalar_equal(actual_final, expected_final)
    return {"id": task["id"], "split": task["split"], "family": task["family"],
            "reference_access_stage": "after-prediction-byte-freeze", "predictions_sha256": digest,
            "final_json_valid": valid_final, "final_correct": final_correct,
            "round_correct": round_correct, "observation_consistent": observation_consistent,
            "complete_trace": complete_trace,
            "task_success": complete_trace and final_correct and all(round_correct) and all(observation_consistent)}


def input_provenance() -> dict:
    """Return independent provenance without producing task reference outputs."""
    return copy.deepcopy(PROVENANCE)
