"""Tests for deterministic key generation."""

from tirastore._keys import canonical_json, make_input_json, make_key


def test_canonical_json_sorted_keys():
    """Keys must be sorted regardless of insertion order."""
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_canonical_json_no_whitespace():
    result = canonical_json({"key": "value"})
    assert " " not in result
    assert result == '{"key":"value"}'


def test_make_key_deterministic():
    k1 = make_key("prog", "src", "sched")
    k2 = make_key("prog", "src", "sched")
    assert k1 == k2
    assert len(k1) == 64  # SHA-256 hex


def test_make_key_differs_for_different_inputs():
    k1 = make_key("prog", "src", "sched_a")
    k2 = make_key("prog", "src", "sched_b")
    assert k1 != k2


def test_make_input_json_round_trip():
    import json

    text = make_input_json("blur", "code", "schedule")
    obj = json.loads(text)
    assert obj["program_name"] == "blur"
    assert obj["program_tiramisu_source_code"] == "code"
    assert obj["tiralib_schedule_string"] == "schedule"
