"""Input-only BFCL staging, immutable schema splits, and isolated evaluation.

Input preparation does not import/load gold. Gold is opened only by the explicit
offline loader after candidate and prediction byte hashes have been verified.
The scalar evaluator is independently implemented, not a vendored BFCL checker;
its restricted scope and string normalization are documented in RELATED_WORK.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from orbitcall.serialization import (
    CallValidationError,
    SCALAR_TYPES,
    strict_json_loads,
    validate_call,
    validate_scalar_schema,
)


GORILLA_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
GORILLA_BASE = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{GORILLA_REVISION}"
DATA_BASE = f"{GORILLA_BASE}/berkeley-function-call-leaderboard/bfcl_eval/data"
SOURCE_FILES = {
    "simple_python": "BFCL_v4_simple_python.json",
    "live_simple": "BFCL_v4_live_simple.json",
}
SOURCE_SHA256 = {
    "simple_python": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
    "live_simple": "1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4",
}
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
SPLIT_SEED = "orbitcall-bfcl-input-only-v1"
_ANNOTATIONS = {"description", "title", "examples", "default", "deprecated"}
_ANSWER_FIELDS = {"ground_truth", "possible_answer", "answer", "answers", "label", "labels"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_frozen_json(path: str | Path, value: Any) -> str:
    """Create exclusively; an existing freeze may only be replayed byte-identically."""
    path = Path(path)
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to change frozen artifact: {path.name}")
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """The upstream files have a .json suffix but contain one record per line."""
    result: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = strict_json_loads(line)
            except CallValidationError as exc:
                raise ValueError(f"{Path(path).name}:{line_number}: {exc}") from exc
            if type(record) is not dict:
                raise ValueError(f"{Path(path).name}:{line_number}: expected an object")
            result.append(record)
    return result


def _conversation(record: Mapping[str, Any]) -> list[dict[str, str]]:
    question = record.get("question")
    if (type(question) is not list or len(question) != 1
            or type(question[0]) is not list or not question[0]):
        raise ValueError("Expected exactly one nonempty BFCL conversation")
    result: list[dict[str, str]] = []
    for message in question[0]:
        if (type(message) is not dict or message.get("role") not in {"user", "assistant", "system"}
                or type(message.get("content")) is not str):
            raise ValueError("Unsupported conversation message")
        # These are public input/history fields, never a gold call.
        result.append({"role": message["role"], "content": message["content"]})
    return result


def input_eligibility(record: Mapping[str, Any]) -> str | None:
    """Return an input-only exclusion reason, or None for a declared eligible item."""
    if type(record) is not dict:
        return "not_an_object"
    if set(record) & _ANSWER_FIELDS:
        raise ValueError("Answer fields are forbidden in the input stage")
    if type(record.get("id")) is not str or not record["id"]:
        return "invalid_id"
    functions = record.get("function")
    if type(functions) is not list or len(functions) != 1:
        return "not_one_function"
    function = functions[0]
    if type(function) is not dict or type(function.get("name")) is not str or not function["name"]:
        return "invalid_function"
    parameters = function.get("parameters")
    if type(parameters) is not dict or parameters.get("type") not in {"dict", "object"}:
        return "not_object_parameters"
    properties = parameters.get("properties")
    required = parameters.get("required")
    if type(properties) is not dict or len(properties) not in (2, 3):
        return "not_two_or_three_parameters"
    if (type(required) is not list or any(type(key) is not str for key in required)
            or len(required) != len(properties) or set(required) != set(properties)):
        return "optional_or_invalid_required_keys"
    if parameters.get("additionalProperties", False) is not False:
        return "additional_keys_allowed"
    unsupported_container = set(parameters) - {"type", "properties", "required", "additionalProperties"} - _ANNOTATIONS
    if unsupported_container:
        return "unsupported_object_constraints"
    for key, schema in properties.items():
        if type(key) is not str or not key or type(schema) is not dict or schema.get("type") not in SCALAR_TYPES:
            return "not_flat_supported_scalars"
        try:
            validate_scalar_schema(schema)
        except CallValidationError:
            return "unsupported_scalar_constraints"
    try:
        _conversation(record)
    except ValueError:
        return "unsupported_conversation"
    return None


def schema_group_key(function: Mapping[str, Any]) -> str:
    """Group identical named scalar schemas, ignoring descriptive annotations.

    Property/required order is ignored, while names, scalar types and supported
    constraints remain. Questions and candidate outputs cannot affect grouping.
    """
    parameters = function["parameters"]
    structural = {
        "name": function["name"],
        "properties": {
            key: {field: value for field, value in schema.items() if field not in _ANNOTATIONS}
            for key, schema in parameters["properties"].items()
        },
        "required": sorted(parameters["required"]),
        "additionalProperties": False,
    }
    payload = json.dumps(structural, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _development_groups(groups: Mapping[str, Sequence[str]], target: int) -> set[str]:
    """Deterministic subset-sum closest to target, keeping each schema intact.

    Hash-sorted groups, first-reached subset per count, and ties favoring smaller
    development size are declared rules. No question text or gold is consulted.
    """
    if type(target) is not int or target < 0:
        raise ValueError("target_dev must be a nonnegative integer")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(
        f"{SPLIT_SEED}:{key}".encode("utf-8")).hexdigest())
    total = sum(len(ids) for ids in groups.values())
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        for count, chosen in list(reachable.items()):
            new_count = count + len(groups[group])
            if new_count not in reachable:
                reachable[new_count] = (*chosen, group)
    # Keep a nonempty test set whenever the population is nonempty.
    possibilities = [count for count in reachable if count < total or total == 0]
    best = min(possibilities, key=lambda count: (abs(count - target), count))
    return set(reachable[best])


def freeze_inputs(
    source_paths: Mapping[str, str | Path], output_path: str | Path, *,
    target_dev: int = 20, expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze all eligible inputs and exclusions before any answers are loaded.

    Production defaults require the exact two pinned input byte hashes. Tests
    may pass explicit fixture hashes; this exception is recorded in the manifest.
    Run this preparation on Spark under the project's execution policy.
    """
    expected = SOURCE_SHA256 if expected_hashes is None else dict(expected_hashes)
    if set(source_paths) != set(expected):
        raise ValueError("Source categories must exactly match the expected hash set")
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    sources: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {}
    for category in sorted(source_paths):
        path = Path(source_paths[category])
        digest = sha256_file(path)
        if digest != expected[category]:
            raise ValueError(f"Input hash mismatch for {category}")
        inputs = read_jsonl(path)
        sources.append({"category": category, "file": path.name, "sha256": digest,
                        "url": f"{DATA_BASE}/{SOURCE_FILES[category]}" if category in SOURCE_FILES else None,
                        "record_count": len(inputs)})
        for row_index, record in enumerate(inputs):
            item_id = record.get("id")
            if type(item_id) is str:
                if item_id in seen:
                    raise ValueError(f"Duplicate input ID: {item_id}")
                seen.add(item_id)
            reason = input_eligibility(record)
            if reason is not None:
                exclusions.append({"id": str(item_id), "category": category,
                                   "row": row_index, "reason": reason})
                continue
            group = schema_group_key(record["function"][0])
            groups.setdefault(group, []).append(item_id)
            records.append({"id": item_id, "category": category, "schema_group": group,
                            "question": record["question"], "function": record["function"]})
    dev_groups = _development_groups(groups, target_dev)
    records.sort(key=lambda row: (row["category"], row["id"]))
    for record in records:
        record["split"] = "development" if record["schema_group"] in dev_groups else "test"
    manifest = {
        "format_version": 1, "protocol": "orbitcall-v1", "gorilla_revision": GORILLA_REVISION,
        "production_input_hashes": expected == SOURCE_SHA256,
        "answer_accessed": False, "split_seed": SPLIT_SEED, "target_development_count": target_dev,
        "grouping": "function-name + annotation-free scalar schema; order-independent",
        "split_algorithm": "hash-ordered deterministic subset sum; closest size; smaller-count ties",
        "sources": sources, "eligible_ids": [row["id"] for row in records],
        "development_ids": [row["id"] for row in records if row["split"] == "development"],
        "test_ids": [row["id"] for row in records if row["split"] == "test"],
        "counts_by_category": dict(sorted(Counter(row["category"] for row in records).items())),
        "schema_groups": {key: sorted(ids) for key, ids in sorted(groups.items())},
        "records": records, "exclusions": exclusions,
    }
    write_frozen_json(output_path, manifest)
    return manifest


def load_frozen_inputs(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    manifest = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("gorilla_revision") != GORILLA_REVISION or manifest.get("answer_accessed") is not False:
        raise ValueError("Not an answer-isolated OrbitCall input freeze")
    if split not in (None, "development", "test"):
        raise ValueError("split must be development or test")
    return [row for row in manifest["records"] if split is None or row["split"] == split]


def render_tools(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert input-only BFCL scalar schemas to the native tool-template form."""
    reason = input_eligibility(record)
    if reason is not None:
        raise ValueError(f"Ineligible prompt input: {reason}")
    function = strict_json_loads(json.dumps(record["function"][0], ensure_ascii=False))
    function["parameters"]["type"] = "object"
    function["parameters"]["additionalProperties"] = False
    for schema in function["parameters"]["properties"].values():
        if schema["type"] == "float":
            schema["type"] = "number"
        elif schema["type"] == "bool":
            schema["type"] = "boolean"
    return [{"type": "function", "function": function}]


def render_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """An answer-free prompt with the exact fixed JSON envelope being scored.

    Use these messages alone with apply_chat_template; render_tools is an
    alternative native-template input, not a second required schema insertion.
    """
    reason = input_eligibility(record)
    if reason is not None:
        raise ValueError(f"Ineligible prompt input: {reason}")
    tool = render_tools(record)[0]["function"]
    system = (
        "Call the supplied function to answer the user's request. Return exactly one "
        "compact JSON object with fields name and arguments, in that outer order. "
        "arguments must contain every declared parameter and no other parameter. "
        "Use literal JSON values of the declared types. Do not add explanations, "
        "Markdown fences, or tool_call tags.\nFunction schema:\n"
        + json.dumps(tool, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )
    return [{"role": "system", "content": system}, *_conversation(record)]


def load_gold_after_freeze(
    gold_paths: Iterable[str | Path], *, candidates_path: str | Path,
    candidates_sha256: str, predictions_path: str | Path, predictions_sha256: str,
) -> dict[str, Any]:
    """Explicit offline gold stage; both exact artifact hashes are checked first.

    Expected hashes must come from a separately saved immutable pre-gold receipt,
    not be recomputed by the caller after reading labels. No network is used.
    """
    for path, expected in ((candidates_path, candidates_sha256),
                           (predictions_path, predictions_sha256)):
        if (type(expected) is not str or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("A lowercase SHA256 freeze receipt is required")
        if sha256_file(path) != expected:
            raise ValueError("Candidate/prediction freeze mismatch; gold remains unopened")
    gold: dict[str, Any] = {}
    for path in gold_paths:
        for row in read_jsonl(path):
            if type(row.get("id")) is not str or "ground_truth" not in row:
                raise ValueError("Unsupported BFCL answer record")
            if row["id"] in gold:
                raise ValueError(f"Duplicate gold ID: {row['id']}")
            gold[row["id"]] = row["ground_truth"]
    return gold


def _normalized_string(value: str) -> str:
    # BFCL's scalar-string policy, implemented independently: remove the stated
    # punctuation/ASCII spaces, lowercase, and normalize single quotation marks.
    ignored = frozenset(" ,./-_*^")
    return "".join(char for char in value if char not in ignored).lower().replace("'", '"')


def evaluate_call(
    call: Mapping[str, Any] | None, function_schema: Mapping[str, Any], ground_truth: Any,
) -> dict[str, Any]:
    """Restricted BFCL scalar AST/value semantics, including string normalization.

    Support is label-defined but never used to alter the frozen population. Any
    unsupported item is retained and counted explicitly. The number/bool aliases
    are documented extensions, not a claim of running the official full checker.
    """
    unsupported = {"supported": False, "correct": None, "reason": "unsupported_gold"}
    if type(ground_truth) is not list or len(ground_truth) != 1:
        return unsupported
    expected_call = ground_truth[0]
    name = function_schema.get("name")
    if type(expected_call) is not dict or set(expected_call) != {name}:
        return unsupported
    allowed = expected_call[name]
    properties = function_schema["parameters"]["properties"]
    if type(allowed) is not dict or set(allowed) != set(properties):
        return unsupported
    for key, schema in properties.items():
        values = allowed[key]
        if type(values) is not list or not values:
            return unsupported
        expected_type = schema["type"]
        for value in values:
            valid = {
                "string": type(value) is str,
                "integer": type(value) is int,
                "float": type(value) in (int, float),
                "number": type(value) in (int, float),
                "boolean": type(value) is bool,
                "bool": type(value) is bool,
            }.get(expected_type, False)
            if not valid:
                # Variables, nested objects, null and optional sentinels for
                # numeric arguments are unsupported, rather than repaired.
                return unsupported
    if call is None:
        return {"supported": True, "correct": False, "reason": "no_candidate"}
    try:
        validated = validate_call(call, function_schema)
    except CallValidationError as exc:
        return {"supported": True, "correct": False, "reason": f"invalid_call: {exc}"}
    for key, value in validated["arguments"].items():
        if properties[key]["type"] == "string":
            matched = _normalized_string(value) in {_normalized_string(x) for x in allowed[key]}
        else:
            # Type checks above exclude Python's bool == int ambiguity. BFCL
            # permits integer literals under float schemas; numeric values may
            # match across int/float for float/number only, without coercing calls.
            matched = any(value == answer for answer in allowed[key])
        if not matched:
            return {"supported": True, "correct": False, "reason": f"wrong_value:{key}"}
    return {"supported": True, "correct": True, "reason": "scalar_ast_value_match"}
