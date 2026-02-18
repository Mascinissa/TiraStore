"""Integration tests for the TiraStore high-level API."""

import json
import os
from unittest import mock

import pytest

from tirastore import LookupResult, TiraStore


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
        tiralib_schedule_string="L0,L1,L2",
        is_legal=True,
        execution_times=[0.042, 0.039, 0.041],
    )
    assert wrote

    result = store.lookup("blur", "void blur() {}", "L0,L1,L2")
    assert result is not None
    assert isinstance(result, LookupResult)
    assert result.is_legal is True
    assert result.execution_times == [0.042, 0.039, 0.041]
    assert result.source_project == "test_project"


def test_lookup_missing(store):
    result = store.lookup("nonexistent", "code", "sched")
    assert result is None


def test_record_illegal(store):
    wrote = store.record(
        program_name="bad",
        program_source_code="code",
        tiralib_schedule_string="sched",
        is_legal=False,
        execution_times=None,
    )
    assert wrote

    result = store.lookup("bad", "code", "sched")
    assert result is not None
    assert result.is_legal is False
    assert result.execution_times is None


def test_record_legal_requires_times(store):
    with pytest.raises(ValueError, match="execution_times must be provided"):
        store.record("p", "c", "s", is_legal=True, execution_times=None)


def test_record_legal_requires_nonempty_times(store):
    with pytest.raises(ValueError, match="execution_times must be provided"):
        store.record("p", "c", "s", is_legal=True, execution_times=[])


# ------------------------------------------------------------------
# Overwrite behavior
# ------------------------------------------------------------------


def test_no_overwrite_by_default(store):
    store.record("p", "c", "s", is_legal=False)
    wrote = store.record("p", "c", "s", is_legal=True, execution_times=[0.1])
    assert not wrote  # Did not overwrite

    result = store.lookup("p", "c", "s")
    assert result.is_legal is False


def test_overwrite_when_requested(store):
    store.record("p", "c", "s", is_legal=False)
    wrote = store.record("p", "c", "s", is_legal=True, execution_times=[0.1], overwrite=True)
    assert wrote

    result = store.lookup("p", "c", "s")
    assert result.is_legal is True
    assert result.execution_times == [0.1]


# ------------------------------------------------------------------
# contains / count / stats
# ------------------------------------------------------------------


def test_contains(store):
    assert not store.contains("p", "c", "s")
    store.record("p", "c", "s", is_legal=False)
    assert store.contains("p", "c", "s")


def test_count(store):
    assert store.count() == 0
    store.record("p1", "c", "s", is_legal=False)
    store.record("p2", "c", "s", is_legal=False)
    assert store.count() == 2


def test_stats(store):
    store.record("p1", "c", "s1", is_legal=True, execution_times=[0.1])
    store.record("p2", "c", "s2", is_legal=False)
    s = store.stats()
    assert s["total_records"] == 2
    assert s["legal_records"] == 1
    assert s["illegal_records"] == 1


# ------------------------------------------------------------------
# keys / get / delete
# ------------------------------------------------------------------


def test_keys(store):
    store.record("p1", "c", "s", is_legal=False)
    store.record("p2", "c", "s", is_legal=False)
    k = store.keys()
    assert len(k) == 2
    # Each key should be a 64-char hex string
    for key in k:
        assert len(key) == 64


def test_get_by_key(store):
    store.record("p", "c", "s", is_legal=False)
    k = store.keys()[0]
    row = store.get(k)
    assert row is not None
    assert row["key"] == k


def test_delete_by_key(store):
    store.record("p", "c", "s", is_legal=False)
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
    store1.record("p", "c", "s", is_legal=False)

    # Connect from a "different CPU"
    store2 = TiraStore(db, cpu_model="AMD EPYC 7742", slurm_cpus="8")
    assert not store2.writes_allowed

    # Lookup should still work
    result = store2.lookup("p", "c", "s")
    assert result is not None

    # Write should fail
    with pytest.raises(PermissionError, match="CPU metadata mismatch"):
        store2.record("p2", "c", "s", is_legal=False)


def test_cpu_mismatch_override(tmp_path):
    db = tmp_path / "test.db"
    TiraStore(db, cpu_model="CPU_A", slurm_cpus="4")

    store = TiraStore(db, cpu_model="CPU_B", slurm_cpus="4", allow_cpu_mismatch=True)
    assert store.writes_allowed
    # Should be able to write
    store.record("p", "c", "s", is_legal=False)


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
    store.record("p", "c", "s", is_legal=True, execution_times=[0.5])
    result = store.lookup("p", "c", "s")
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
