"""Integration tests for the TiraStore high-level API."""

import json
import os
from unittest import mock

import pytest

from tirastore import LookupResult, TiraStore

# A few valid schedule strings for use in tests
_SCHED_A = "R(L0,comps=['c1'])"
_SCHED_B = "I(L0,L1,comps=['c1'])"
_EMPTY = ""


@pytest.fixture
def store(tmp_path):
    """Create a TiraStore instance in a temp directory."""
    db = tmp_path / "test.db"
    return TiraStore(db, source_project="test_project", cpu_model="TestCPU", slurm_cpus="4")


# ------------------------------------------------------------------
# Basic record / lookup
# ------------------------------------------------------------------


def test_record_and_lookup(store):
    wrote = store.record(
        program_name="blur",
        program_source_code="void blur() {}",
        tiralib_schedule_string="S(L0,L1,4,8,comps=['c1'])",
        is_legal=True,
        execution_times=[0.042, 0.039, 0.041],
    )
    assert wrote

    result = store.lookup("blur", "void blur() {}", "S(L0,L1,4,8,comps=['c1'])")
    assert result is not None
    assert isinstance(result, LookupResult)
    assert result.is_legal is True
    assert result.execution_times == [0.042, 0.039, 0.041]
    assert result.source_project == "test_project"


def test_lookup_missing(store):
    result = store.lookup("nonexistent", "code", _EMPTY)
    assert result is None


def test_record_illegal(store):
    wrote = store.record(
        program_name="bad",
        program_source_code="code",
        tiralib_schedule_string=_SCHED_A,
        is_legal=False,
        execution_times=None,
    )
    assert wrote

    result = store.lookup("bad", "code", _SCHED_A)
    assert result is not None
    assert result.is_legal is False
    assert result.execution_times is None


def test_record_legal_requires_times(store):
    with pytest.raises(ValueError, match="execution_times must be provided"):
        store.record("p", "c", _EMPTY, is_legal=True, execution_times=None)


def test_record_legal_requires_nonempty_times(store):
    with pytest.raises(ValueError, match="execution_times must be provided"):
        store.record("p", "c", _EMPTY, is_legal=True, execution_times=[])


# ------------------------------------------------------------------
# Schedule validation
# ------------------------------------------------------------------


def test_record_rejects_invalid_schedule(store):
    with pytest.raises(ValueError, match="Invalid schedule string"):
        store.record("p", "c", "BOGUS(stuff)", is_legal=False)


def test_record_rejects_malformed_schedule(store):
    with pytest.raises(ValueError, match="Invalid schedule string"):
        store.record("p", "c", "S(L0,comps=['c'])", is_legal=False)


def test_record_accepts_empty_schedule(store):
    wrote = store.record("p", "c", "", is_legal=False)
    assert wrote


def test_lookup_normalizes_schedule(store):
    """A record stored with one whitespace/quote variant is found via another."""
    store.record("p", "c", "R(L0,comps=['comp1'])", is_legal=False)

    # Lookup with extra whitespace and double quotes — should still match
    result = store.lookup("p", "c", ' R( L0 , comps=["comp1"] ) ')
    assert result is not None


def test_contains_normalizes_schedule(store):
    store.record("p", "c", "R(L0,comps=['comp1'])", is_legal=False)
    assert store.contains("p", "c", ' R(L0 , comps=["comp1"]) ')


# ------------------------------------------------------------------
# Program deduplication
# ------------------------------------------------------------------


def test_program_dedup_same_source(store):
    """Multiple records for the same program share one programs row."""
    src = "void blur() { /* implementation */ }"
    store.record("blur", src, _SCHED_A, is_legal=False)
    store.record("blur", src, _SCHED_B, is_legal=False)

    assert store.count() == 2
    assert store.program_count() == 1


def test_program_dedup_different_programs(store):
    """Different programs get separate rows."""
    store.record("blur", "void blur() {}", _SCHED_A, is_legal=False)
    store.record("edge", "void edge() {}", _SCHED_A, is_legal=False)

    assert store.count() == 2
    assert store.program_count() == 2


def test_program_source_normalization(store):
    """Same code with different comments/whitespace matches on lookup."""
    src_v1 = "void blur() { int x = 1; }"
    src_v2 = "// header comment\nvoid blur() { int x = 1; }  // trailing"
    src_v3 = "/* block */  void  blur()  {  int  x  =  1;  }"

    store.record("blur", src_v1, _SCHED_A, is_legal=True, execution_times=[0.1])

    # Lookup with cosmetically different source — same program
    result = store.lookup("blur", src_v2, _SCHED_A)
    assert result is not None
    assert result.execution_times == [0.1]

    result = store.lookup("blur", src_v3, _SCHED_A)
    assert result is not None

    # Only one program stored
    assert store.program_count() == 1


def test_readable_source_stored(store):
    """The original (readable) source code is stored, not the normalized form."""
    original = "// My blur program\nvoid blur() {\n    int x = 1;\n}"
    store.record("blur", original, _SCHED_A, is_legal=False)

    key = store.keys()[0]
    row = store.get(key)
    assert row["source_code"] == original


def test_stats_includes_program_count(store):
    store.record("blur", "void blur() {}", _SCHED_A, is_legal=False)
    store.record("blur", "void blur() {}", _SCHED_B, is_legal=False)
    store.record("edge", "void edge() {}", _SCHED_A, is_legal=False)

    s = store.stats()
    assert s["total_records"] == 3
    assert s["total_programs"] == 2


# ------------------------------------------------------------------
# Overwrite behavior
# ------------------------------------------------------------------


def test_no_overwrite_by_default(store):
    store.record("p", "c", _EMPTY, is_legal=False)
    wrote = store.record("p", "c", _EMPTY, is_legal=True, execution_times=[0.1])
    assert not wrote  # Did not overwrite

    result = store.lookup("p", "c", _EMPTY)
    assert result.is_legal is False


def test_overwrite_when_requested(store):
    store.record("p", "c", _EMPTY, is_legal=False)
    wrote = store.record("p", "c", _EMPTY, is_legal=True, execution_times=[0.1], overwrite=True)
    assert wrote

    result = store.lookup("p", "c", _EMPTY)
    assert result.is_legal is True
    assert result.execution_times == [0.1]


# ------------------------------------------------------------------
# contains / count / stats
# ------------------------------------------------------------------


def test_contains(store):
    assert not store.contains("p", "c", _EMPTY)
    store.record("p", "c", _EMPTY, is_legal=False)
    assert store.contains("p", "c", _EMPTY)


def test_count(store):
    assert store.count() == 0
    store.record("p1", "void p1() {}", _SCHED_A, is_legal=False)
    store.record("p2", "void p2() {}", _SCHED_A, is_legal=False)
    assert store.count() == 2


def test_stats(store):
    store.record("p1", "c", _SCHED_A, is_legal=True, execution_times=[0.1])
    store.record("p2", "c", _SCHED_B, is_legal=False)
    s = store.stats()
    assert s["total_records"] == 2
    assert s["legal_records"] == 1
    assert s["illegal_records"] == 1


# ------------------------------------------------------------------
# keys / get / delete
# ------------------------------------------------------------------


def test_keys(store):
    store.record("p1", "void p1() {}", _SCHED_A, is_legal=False)
    store.record("p2", "void p2() {}", _SCHED_A, is_legal=False)
    k = store.keys()
    assert len(k) == 2
    # Each key should be a 64-char hex string
    for key in k:
        assert len(key) == 64


def test_get_by_key(store):
    store.record("p", "c", _EMPTY, is_legal=False)
    k = store.keys()[0]
    row = store.get(k)
    assert row is not None
    assert row["key"] == k
    # Joined data should be present
    assert "program_name" in row
    assert "source_code" in row


def test_delete_by_key(store):
    store.record("p", "c", _EMPTY, is_legal=False)
    k = store.keys()[0]
    assert store.delete(k)
    assert store.count() == 0


# ------------------------------------------------------------------
# CPU mismatch handling
# ------------------------------------------------------------------


def test_cpu_mismatch_blocks_writes(tmp_path):
    db = tmp_path / "test.db"
    # Create the DB with a specific CPU
    store1 = TiraStore(db, cpu_model="Intel Xeon Gold 6248", slurm_cpus="8")
    store1.record("p", "c", _EMPTY, is_legal=False)

    # Connect from a "different CPU"
    store2 = TiraStore(db, cpu_model="AMD EPYC 7742", slurm_cpus="8")
    assert not store2.writes_allowed

    # Lookup should still work
    result = store2.lookup("p", "c", _EMPTY)
    assert result is not None

    # Write should fail
    with pytest.raises(PermissionError, match="CPU metadata mismatch"):
        store2.record("p2", "c", _EMPTY, is_legal=False)


def test_cpu_mismatch_override(tmp_path):
    db = tmp_path / "test.db"
    TiraStore(db, cpu_model="CPU_A", slurm_cpus="4")

    store = TiraStore(db, cpu_model="CPU_B", slurm_cpus="4", allow_cpu_mismatch=True)
    assert store.writes_allowed
    # Should be able to write
    store.record("p", "c", _EMPTY, is_legal=False)


# ------------------------------------------------------------------
# Metadata properties
# ------------------------------------------------------------------


def test_cpu_model_property(store):
    assert store.cpu_model == "TestCPU"


def test_slurm_cpus_property(store):
    assert store.slurm_cpus == "4"


# ------------------------------------------------------------------
# LookupResult
# ------------------------------------------------------------------


def test_lookup_result_to_dict(store):
    store.record("p", "c", _SCHED_A, is_legal=True, execution_times=[0.5])
    result = store.lookup("p", "c", _SCHED_A)
    d = result.to_dict()
    assert d["is_legal"] is True
    assert d["execution_times"] == [0.5]
    assert "hostname" in d
    assert "username" in d


# ------------------------------------------------------------------
# Repr
# ------------------------------------------------------------------


def test_repr(store):
    r = repr(store)
    assert "TiraStore" in r
    assert "test_project" in r
