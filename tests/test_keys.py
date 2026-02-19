"""Tests for deterministic key and hash generation."""

from tirastore._keys import canonical_json, make_key, make_program_hash

_SCHED = "R(L0,comps=['c1'])"


def test_canonical_json_sorted_keys():
    """Keys must be sorted regardless of insertion order."""
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_canonical_json_no_whitespace():
    result = canonical_json({"key": "value"})
    assert " " not in result
    assert result == '{"key":"value"}'


def test_make_program_hash_deterministic():
    h1 = make_program_hash("void foo() {}")
    h2 = make_program_hash("void foo() {}")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_make_program_hash_normalizes():
    """Same code with different comments/whitespace produces the same hash."""
    src_a = "void foo() { int x = 1; }"
    src_b = "void  foo()  {  int  x  =  1;  }"
    src_c = "// comment\nvoid foo() { int x = 1; }"
    src_d = "/* block */\nvoid foo() { int x = 1; }"
    src_e = '#include <stdio.h>\nvoid foo() { int x = 1; }'
    assert make_program_hash(src_a) == make_program_hash(src_b)
    assert make_program_hash(src_a) == make_program_hash(src_c)
    assert make_program_hash(src_a) == make_program_hash(src_d)
    assert make_program_hash(src_a) == make_program_hash(src_e)


def test_make_program_hash_differs_for_different_code():
    assert make_program_hash("void foo() {}") != make_program_hash("void bar() {}")


def test_make_key_deterministic():
    h = make_program_hash("src")
    k1 = make_key(h, _SCHED)
    k2 = make_key(h, _SCHED)
    assert k1 == k2
    assert len(k1) == 64


def test_make_key_differs_for_different_schedule():
    h = make_program_hash("src")
    k1 = make_key(h, "R(L0,comps=['a'])")
    k2 = make_key(h, "R(L0,comps=['b'])")
    assert k1 != k2


def test_make_key_differs_for_different_program():
    h1 = make_program_hash("void foo() {}")
    h2 = make_program_hash("void bar() {}")
    k1 = make_key(h1, _SCHED)
    k2 = make_key(h2, _SCHED)
    assert k1 != k2


def test_make_key_normalizes_schedule():
    """Equivalent schedules with different whitespace/quoting produce the same key."""
    h = make_program_hash("code")
    k1 = make_key(h, "R(L0,comps=['comp1'])")
    k2 = make_key(h, ' R( L0 , comps=["comp1"] ) ')
    assert k1 == k2
