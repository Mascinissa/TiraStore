"""Tests for deterministic key generation."""

import json

from tirastore._keys import canonical_json, make_input_json, make_key

_SCHED = "R(L0,comps=['c1'])"


def test_canonical_json_sorted_keys():
    """Keys must be sorted regardless of insertion order."""
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_canonical_json_no_whitespace():
    result = canonical_json({"key": "value"})
    assert " " not in result
    assert result == '{"key":"value"}'


def test_make_key_deterministic():
    k1 = make_key("prog", "src", _SCHED)
    k2 = make_key("prog", "src", _SCHED)
    assert k1 == k2
    assert len(k1) == 64  # SHA-256 hex


def test_make_key_differs_for_different_inputs():
    k1 = make_key("prog", "src", "R(L0,comps=['a'])")
    k2 = make_key("prog", "src", "R(L0,comps=['b'])")
    assert k1 != k2


def test_make_input_json_round_trip():
    text = make_input_json("blur", "code", _SCHED)
    obj = json.loads(text)
    assert obj["program_name"] == "blur"
    assert obj["program_tiramisu_source_code"] == "code"
    assert obj["tiralib_schedule_string"] == _SCHED


def test_make_key_normalizes_schedule():
    """Equivalent schedules with different whitespace/quoting produce the same key."""
    k1 = make_key("p", "c", "R(L0,comps=['comp1'])")
    k2 = make_key("p", "c", ' R( L0 , comps=["comp1"] ) ')
    assert k1 == k2


def test_make_input_json_stores_normalized():
    """The stored input JSON uses the normalized schedule."""
    text = make_input_json("p", "c", ' R( L0, comps=["comp1"] ) ')
    obj = json.loads(text)
    assert obj["tiralib_schedule_string"] == "R(L0,comps=['comp1'])"
