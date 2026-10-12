"""Contract tests; run on DGX Spark, never on the Windows staging host."""

import itertools
import json

import pytest

from orbitcall.serialization import (
    CallValidationError,
    canonical_call,
    parse_call,
    scalar_equal,
    semantic_key,
    serialization_orbit,
    strict_json_loads,
    validate_call,
)


def function_schema():
    return {
        "name": "example.call",
        "parameters": {
            "type": "dict",
            "properties": {"count": {"type": "integer"}, "text": {"type": "string"}},
            "required": ["count", "text"],
        },
    }


def test_native_qwen_and_bare_call_have_same_identity():
    text = '{"name":"example.call","arguments":{"text":"hello","count":2}}'
    bare = parse_call(text, function_schema())
    wrapped = parse_call(f" \n<tool_call>\n{text}\n</tool_call> \n", function_schema())
    assert semantic_key(bare) == semantic_key(wrapped)
    assert canonical_call(bare) == '{"name":"example.call","arguments":{"count":2,"text":"hello"}}'


@pytest.mark.parametrize("text", [
    '{"name":"f","name":"g","arguments":{}}',
    '{"name":"f","arguments":{"x":1,"x":2}}',
    '{"name":"f","arguments":{"x":{"z":1,"z":2}}}',
    '{"name":"f","arguments":{"x":NaN}}',
    '{"name":"f","arguments":{"x":Infinity}}',
    '{"name":"f","arguments":{"x":1e999}}',
    '{"name":"f","arguments":{"x":"\\ud800"}}',
    '{"name":"f","arguments":"{}"}',
    '{"name":"f","arguments":{},"extra":0}',
    '[{"name":"f","arguments":{}}]',
    'Here is the call: {"name":"f","arguments":{}}',
    '```json\n{"name":"f","arguments":{}}\n```',
    '{"name":"f","arguments":{}} trailing',
    '<tool_call>{"name":"f","arguments":{}}',
    '<tool_call>{"name":"f","arguments":{}}</tool_call><tool_call>{}</tool_call>',
])
def test_invalid_generation_is_never_repaired(text):
    with pytest.raises(CallValidationError):
        parse_call(text)


@pytest.mark.parametrize("value", [True, 1, 1.0, "1", None])
def test_scalar_types_remain_distinct(value):
    key = semantic_key({"name": "f", "arguments": {"x": value}})
    for other in (True, 1, 1.0, "1", None):
        other_key = semantic_key({"name": "f", "arguments": {"x": other}})
        assert (key == other_key) is (type(value) is type(other))


def test_missing_null_and_array_order_are_not_collapsed():
    calls = [
        {"name": "f", "arguments": {}},
        {"name": "f", "arguments": {"x": None}},
        {"name": "f", "arguments": {"x": [1, 2]}},
        {"name": "f", "arguments": {"x": [2, 1]}},
    ]
    assert len({semantic_key(call) for call in calls}) == len(calls)
    assert semantic_key({"name": "f", "arguments": {"x": {"b": 2, "a": 1}}}) == semantic_key(
        {"arguments": {"x": {"a": 1, "b": 2}}, "name": "f"}
    )


@pytest.mark.parametrize("arguments", [
    {"a": None, "b": True},
    {"alpha": 1, "beta": 1.0, "unicode": 'a"b\\c\n雪'},
])
def test_orbit_is_complete_flat_and_has_no_terminator(arguments):
    call = {"name": "example.call", "arguments": arguments}
    schema_order = tuple(reversed(arguments))
    strings = serialization_orbit(call, schema_order)
    expected_orders = list(itertools.permutations(schema_order))
    assert len(strings) == len(expected_orders)
    assert len(set(strings)) == len(strings)
    for text, order in zip(strings, expected_orders, strict=True):
        decoded = strict_json_loads(text)
        assert list(decoded) == ["name", "arguments"]
        assert tuple(decoded["arguments"]) == order
        assert semantic_key(decoded) == semantic_key(call)
        assert text == json.dumps(decoded, ensure_ascii=False, separators=(",", ":"))
        assert not text.endswith("</tool_call>")
        assert not text.endswith("<|im_end|>")


@pytest.mark.parametrize("arguments,order", [
    ({"a": 1}, None),
    ({"a": 1, "b": 2, "c": 3, "d": 4}, None),
    ({"a": [1], "b": 2}, None),
    ({"a": 1, "b": 2}, ["a", "a"]),
    ({"a": 1, "b": 2}, ["a", "c"]),
])
def test_orbit_rejects_unsupported_or_incomplete_families(arguments, order):
    with pytest.raises(CallValidationError):
        serialization_orbit({"name": "f", "arguments": arguments}, order)


@pytest.mark.parametrize("call", [
    {"name": "wrong", "arguments": {"count": 1, "text": "x"}},
    {"name": "example.call", "arguments": {"count": 1}},
    {"name": "example.call", "arguments": {"count": 1, "text": "x", "extra": 1}},
    {"name": "example.call", "arguments": {"count": True, "text": "x"}},
    {"name": "example.call", "arguments": {"count": 1.0, "text": "x"}},
    {"name": "example.call", "arguments": {"count": 1, "text": None}},
])
def test_schema_validation_does_not_coerce_or_ignore_constraints(call):
    with pytest.raises(CallValidationError):
        validate_call(call, function_schema())


def test_validated_calls_are_independent_snapshots():
    call = {"name": "example.call", "arguments": {"count": 2, "text": "x"}}
    validated = validate_call(call, function_schema())
    call["arguments"]["count"] = 99
    assert validated["arguments"]["count"] == 2


def test_float_schema_accepts_int_without_changing_identity():
    schema = function_schema()
    schema["parameters"]["properties"]["count"]["type"] = "float"
    integer = validate_call({"name": "example.call", "arguments": {"count": 1, "text": "x"}}, schema)
    floating = validate_call({"name": "example.call", "arguments": {"count": 1.0, "text": "x"}}, schema)
    assert type(integer["arguments"]["count"]) is int
    assert semantic_key(integer) != semantic_key(floating)


def test_supported_constraints_and_typed_enum():
    schema = function_schema()
    schema["parameters"]["properties"]["count"].update({"minimum": 1, "maximum": 3, "enum": [1, 2, 3]})
    schema["parameters"]["properties"]["text"].update({"minLength": 2, "maxLength": 3, "pattern": "^[A-Z]+$"})
    validate_call({"name": "example.call", "arguments": {"count": 2, "text": "AB"}}, schema)
    for count, text in ((0, "AB"), (2, "aB"), (2, "ABCD"), (4, "AB")):
        with pytest.raises(CallValidationError):
            validate_call({"name": "example.call", "arguments": {"count": count, "text": text}}, schema)
    assert not scalar_equal(True, 1)
    assert not scalar_equal(1, 1.0)


def test_decimal_multiple_is_checked_without_float_modulo():
    schema = function_schema()
    schema["parameters"]["properties"]["count"] = {"type": "number", "multipleOf": 0.1}
    validate_call({"name": "example.call", "arguments": {"count": 0.3, "text": "x"}}, schema)
    with pytest.raises(CallValidationError):
        validate_call({"name": "example.call", "arguments": {"count": 0.31, "text": "x"}}, schema)


def test_unknown_constraint_fails_closed():
    schema = function_schema()
    schema["parameters"]["properties"]["text"]["format"] = "date"
    with pytest.raises(CallValidationError, match="Unsupported scalar constraints"):
        validate_call({"name": "example.call", "arguments": {"count": 2, "text": "x"}}, schema)
