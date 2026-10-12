"""Strict tool-call identity and a finite JSON argument-order orbit.

Strings returned here do not include an EOS token. The model tokenizer must add
exactly one explicit terminator to each scored sequence and reject token-level
duplicates. This module never reads answers or constructs argument values.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from fractions import Fraction
from typing import Any, Iterable, Mapping


class CallValidationError(ValueError):
    """A generated call is outside the declared JSON/schema contract."""


SCALAR_TYPES = frozenset({"string", "integer", "float", "number", "boolean", "bool"})
_ANNOTATIONS = frozenset({"description", "title", "examples", "default", "deprecated"})
_CONSTRAINTS = frozenset(
    {"type", "enum", "const", "minimum", "maximum", "exclusiveMinimum",
     "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern"}
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CallValidationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CallValidationError(f"Non-finite JSON constant: {value}")


def strict_json_loads(text: str) -> Any:
    """Read one complete JSON value, rejecting duplicate keys and non-finites."""
    if type(text) is not str:
        raise CallValidationError("JSON input must be a string")
    try:
        value = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, CallValidationError):
            raise
        raise CallValidationError(f"Invalid JSON: {exc}") from exc
    _validate_json_value(value)
    return value


def _validate_json_value(value: Any) -> None:
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CallValidationError("Non-finite number")
        return
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise CallValidationError("Unpaired Unicode surrogate") from exc
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CallValidationError("JSON object keys must be strings")
            _validate_json_value(key)
            _validate_json_value(item)
        return
    raise CallValidationError(f"Unsupported JSON value type: {type(value).__name__}")


def _compact(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        sort_keys=sort_keys,
    )


def scalar_equal(left: Any, right: Any) -> bool:
    """Typed literal equality; bool/int and int/float do not collapse."""
    _validate_json_value(left)
    _validate_json_value(right)
    return type(left) is type(right) and _compact(left, sort_keys=True) == _compact(
        right, sort_keys=True
    )


def validate_scalar_schema(schema: Mapping[str, Any]) -> None:
    """Fail closed on constraints outside the small supported scalar schema."""
    if type(schema) is not dict or schema.get("type") not in SCALAR_TYPES:
        raise CallValidationError("Argument schema must declare a supported scalar type")
    unsupported = set(schema) - _ANNOTATIONS - _CONSTRAINTS
    if unsupported:
        raise CallValidationError(f"Unsupported scalar constraints: {sorted(unsupported)}")
    if "enum" in schema:
        if type(schema["enum"]) is not list or not schema["enum"]:
            raise CallValidationError("enum must be a nonempty list")
        for value in schema["enum"]:
            if type(value) not in (str, int, float, bool) and value is not None:
                raise CallValidationError("enum must contain scalar JSON values")
            _validate_json_value(value)
    if "const" in schema:
        _validate_json_value(schema["const"])
        if type(schema["const"]) not in (str, int, float, bool) and schema["const"] is not None:
            raise CallValidationError("const must be a scalar JSON value")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if key in schema:
            value = schema[key]
            if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
                raise CallValidationError(f"{key} must be a finite numeric bound")
            if key == "multipleOf" and value <= 0:
                raise CallValidationError("multipleOf must be positive")
    for key in ("minLength", "maxLength"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise CallValidationError(f"{key} must be a nonnegative integer")
    if "pattern" in schema:
        if type(schema["pattern"]) is not str:
            raise CallValidationError("pattern must be a string")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise CallValidationError("Invalid Python regular expression") from exc


def _validate_scalar(value: Any, schema: Mapping[str, Any], key: str) -> None:
    validate_scalar_schema(schema)
    expected = schema["type"]
    valid_type = {
        "string": type(value) is str,
        "integer": type(value) is int,
        "float": type(value) in (int, float),
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "bool": type(value) is bool,
    }[expected]
    if not valid_type:
        raise CallValidationError(f"Argument {key!r} violates type {expected!r}")
    if "enum" in schema and not any(scalar_equal(value, x) for x in schema["enum"]):
        raise CallValidationError(f"Argument {key!r} violates enum")
    if "const" in schema and not scalar_equal(value, schema["const"]):
        raise CallValidationError(f"Argument {key!r} violates const")
    if type(value) in (int, float):
        for bound, operator in (
            ("minimum", lambda v, b: v >= b),
            ("maximum", lambda v, b: v <= b),
            ("exclusiveMinimum", lambda v, b: v > b),
            ("exclusiveMaximum", lambda v, b: v < b),
        ):
            if bound in schema and not operator(value, schema[bound]):
                raise CallValidationError(f"Argument {key!r} violates {bound}")
        if "multipleOf" in schema:
            if Fraction(str(value)) % Fraction(str(schema["multipleOf"])) != 0:
                raise CallValidationError(f"Argument {key!r} violates multipleOf")
    if type(value) is str:
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise CallValidationError(f"Argument {key!r} violates minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise CallValidationError(f"Argument {key!r} violates maxLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise CallValidationError(f"Argument {key!r} violates pattern")


def validate_call(
    call: Any, function_schema: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the exact name/arguments envelope without repairing a candidate.

    With a BFCL function schema, require every declared key and no extra key.
    No values are coerced. A float/number schema accepts an integer literal, but
    identity still distinguishes that literal from a floating-point literal.
    """
    _validate_json_value(call)
    if type(call) is not dict or set(call) != {"name", "arguments"}:
        raise CallValidationError("Expected exactly name and arguments fields")
    if type(call["name"]) is not str or not call["name"]:
        raise CallValidationError("name must be a nonempty string")
    if type(call["arguments"]) is not dict:
        raise CallValidationError("arguments must be a JSON object")
    if function_schema is not None:
        if type(function_schema) is not dict:
            raise CallValidationError("Function schema must be a JSON object")
        if call["name"] != function_schema.get("name"):
            raise CallValidationError("Wrong function name")
        parameters = function_schema.get("parameters", {})
        if type(parameters) is not dict or parameters.get("type") not in {"dict", "object"}:
            raise CallValidationError("Function parameters must declare an object schema")
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if (type(properties) is not dict or type(required) is not list
                or any(type(key) is not str for key in required)
                or set(required) != set(properties) or len(required) != len(properties)):
            raise CallValidationError("Only schemas with all arguments required are supported")
        if set(call["arguments"]) != set(properties):
            raise CallValidationError("Missing or additional argument keys")
        for key, value in call["arguments"].items():
            _validate_scalar(value, properties[key], key)
    # Return a fresh JSON-shaped object; never alias mutable generator output.
    return strict_json_loads(_compact(call))


def parse_call(
    text: str, function_schema: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Parse bare JSON or exactly one native Qwen tool_call envelope.

    Markdown fences, natural-language prefixes, concatenated calls, stringified
    arguments, and incomplete wrappers are invalid; there is no extraction or
    repair step that could silently change the generated candidate pool.
    """
    if type(text) is not str:
        raise CallValidationError("Generated output must be a string")
    stripped = text.strip()
    if stripped.startswith("<tool_call>"):
        if not stripped.endswith("</tool_call>"):
            raise CallValidationError("Incomplete tool_call wrapper")
        stripped = stripped[len("<tool_call>") : -len("</tool_call>")].strip()
    return validate_call(strict_json_loads(stripped), function_schema)


def canonical_call(call: Mapping[str, Any]) -> str:
    """Fixed outer envelope and lexicographic argument order, preserving types."""
    validated = validate_call(call)
    arguments = {key: validated["arguments"][key] for key in sorted(validated["arguments"])}
    return _compact({"name": validated["name"], "arguments": arguments})


def semantic_key(call: Mapping[str, Any]) -> str:
    """Order-independent object identity; array order and all JSON types matter."""
    validated = validate_call(call)
    # Recursive sorting is for identity only. The orbit below is flat and does
    # not change arrays, nesting, spelling, numbers, or the envelope's order.
    return _compact(validated, sort_keys=True)


def serialization_orbit(
    call: Mapping[str, Any], key_order: Iterable[str] | None = None
) -> tuple[str, ...]:
    """Return the complete two- or three-key permutation family, without EOS.

    key_order is the schema order for the first member. If omitted, sorted order
    is used. Enumeration order is deterministic; it does not affect the mass.
    """
    validated = validate_call(call)
    arguments = validated["arguments"]
    if len(arguments) not in (2, 3):
        raise CallValidationError("The declared orbit requires exactly two or three keys")
    if any(type(value) not in (str, bool, int, float) and value is not None
           for value in arguments.values()):
        raise CallValidationError("The declared orbit supports only flat scalar values")
    keys = tuple(sorted(arguments) if key_order is None else key_order)
    if len(keys) != len(arguments) or set(keys) != set(arguments):
        raise CallValidationError("key_order must contain each argument key exactly once")
    result = tuple(
        _compact({"name": validated["name"],
                  "arguments": {key: arguments[key] for key in permutation}})
        for permutation in itertools.permutations(keys)
    )
    if len(set(result)) != len(result):
        raise CallValidationError("Duplicate string in a supposedly complete orbit")
    return result
