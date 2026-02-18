"""Tests for schedule normalization and validation."""

import pytest

from tirastore._schedule import normalize_schedule, validate_schedule


# ------------------------------------------------------------------
# normalize_schedule
# ------------------------------------------------------------------


class TestNormalize:
    def test_empty_string(self):
        assert normalize_schedule("") == ""

    def test_whitespace_removal(self):
        s = "S( L0 , L1 , 4 , 8 , comps=['c1'] )"
        assert " " not in normalize_schedule(s)

    def test_comp_names_single_quoted(self):
        s = 'S(L0,L1,4,8,comps=[c1])'
        assert normalize_schedule(s) == "S(L0,L1,4,8,comps=['c1'])"

    def test_double_quotes_to_single(self):
        s = 'S(L0,L1,4,8,comps=["c1","c2"])'
        assert normalize_schedule(s) == "S(L0,L1,4,8,comps=['c1','c2'])"

    def test_already_single_quoted(self):
        s = "S(L0,L1,4,8,comps=['comp_a','comp_b'])"
        assert normalize_schedule(s) == s

    def test_mixed_quotes(self):
        s = """S(L0,L1,4,8,comps=['c1',"c2",c3])"""
        assert normalize_schedule(s) == "S(L0,L1,4,8,comps=['c1','c2','c3'])"

    def test_multi_transform(self):
        s = "S(L0,L1,4,8,comps=[c1]) | I(L0,L1,comps=[c2])"
        result = normalize_schedule(s)
        assert result == "S(L0,L1,4,8,comps=['c1'])|I(L0,L1,comps=['c2'])"

    def test_none_returns_empty(self):
        assert normalize_schedule(None) == ""


# ------------------------------------------------------------------
# validate_schedule
# ------------------------------------------------------------------


class TestValidate:
    def test_empty_is_valid(self):
        ok, msg = validate_schedule("")
        assert ok
        assert msg == ""

    # -- Valid transformations --

    def test_skew(self):
        ok, _ = validate_schedule("S(L0,L1,4,8,comps=['c1'])")
        assert ok

    def test_interchange(self):
        ok, _ = validate_schedule("I(L0,L1,comps=['c1'])")
        assert ok

    def test_reversal(self):
        ok, _ = validate_schedule("R(L0,comps=['c1'])")
        assert ok

    def test_parallelize(self):
        ok, _ = validate_schedule("P(L0,comps=['c1'])")
        assert ok

    def test_tile2(self):
        ok, _ = validate_schedule("T2(L0,L1,32,64,comps=['c1'])")
        assert ok

    def test_tile3(self):
        ok, _ = validate_schedule("T3(L0,L1,L2,8,16,32,comps=['c1'])")
        assert ok

    def test_unroll(self):
        ok, _ = validate_schedule("U(L0,4,comps=['c1'])")
        assert ok

    def test_fuse(self):
        ok, _ = validate_schedule("F(L0,comps=['c1','c2'])")
        assert ok

    def test_multiple_comps(self):
        ok, _ = validate_schedule("S(L0,L1,4,8,comps=['a','b','c'])")
        assert ok

    def test_pipe_separated_multiple(self):
        ok, _ = validate_schedule("S(L0,L1,4,8,comps=['c1'])|I(L0,L1,comps=['c2'])|R(L0,comps=['c3'])")
        assert ok

    def test_unquoted_comp_names(self):
        ok, _ = validate_schedule("R(L0,comps=[comp_a])")
        assert ok

    def test_double_quoted_comp_names(self):
        ok, _ = validate_schedule('R(L0,comps=["comp_a"])')
        assert ok

    # -- Invalid transformations --

    def test_unknown_transformation(self):
        ok, msg = validate_schedule("X(L0,comps=['c1'])")
        assert not ok
        assert "Unknown transformation" in msg

    def test_malformed_skew_missing_arg(self):
        ok, msg = validate_schedule("S(L0,L1,4,comps=['c1'])")
        assert not ok
        assert "Malformed" in msg

    def test_empty_segment(self):
        ok, msg = validate_schedule("S(L0,L1,4,8,comps=['c1'])||R(L0,comps=['c2'])")
        assert not ok
        assert "Empty segment" in msg

    def test_trailing_pipe(self):
        ok, msg = validate_schedule("R(L0,comps=['c1'])|")
        assert not ok

    def test_leading_pipe(self):
        ok, msg = validate_schedule("|R(L0,comps=['c1'])")
        assert not ok

    def test_no_comps(self):
        ok, msg = validate_schedule("R(L0)")
        assert not ok

    def test_garbage(self):
        ok, msg = validate_schedule("not a schedule at all")
        assert not ok

    def test_lowercase_transformation(self):
        ok, msg = validate_schedule("r(L0,comps=['c1'])")
        assert not ok
