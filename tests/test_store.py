"""Tests for the low-level SQLite storage layer."""

import json

from tirastore._store import Store


def test_init_and_meta(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("Intel Xeon E5-2680", "4")

    assert store.get_cpu_model() == "Intel Xeon E5-2680"
    assert store.get_slurm_cpus() == "4"
    assert store.get_meta("schema_version") == "1"


def test_put_get_contains(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("cpu", "2")

    key = "abc123"
    input_json = json.dumps({"program_name": "blur"})
    result_json = json.dumps({"is_legal": True, "execution_times": [0.01]})

    assert not store.contains(key)
    wrote = store.put(key, input_json, result_json, "node01", "alice", "proj")
    assert wrote
    assert store.contains(key)

    row = store.get(key)
    assert row is not None
    assert row["key"] == key
    assert row["hostname"] == "node01"
    assert row["username"] == "alice"
    assert row["source_project"] == "proj"
    assert json.loads(row["result_json"])["is_legal"] is True


def test_put_no_overwrite(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("cpu", "2")

    key = "abc123"
    store.put(key, "{}", '{"is_legal":false}', "n1", "u1", "p1")
    wrote = store.put(key, "{}", '{"is_legal":true}', "n2", "u2", "p2", overwrite=False)
    assert not wrote
    # Original data preserved
    row = store.get(key)
    assert json.loads(row["result_json"])["is_legal"] is False


def test_put_overwrite(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("cpu", "2")

    key = "abc123"
    store.put(key, "{}", '{"is_legal":false}', "n1", "u1", "p1")
    wrote = store.put(key, "{}", '{"is_legal":true}', "n2", "u2", "p2", overwrite=True)
    assert wrote
    row = store.get(key)
    assert json.loads(row["result_json"])["is_legal"] is True
    assert row["hostname"] == "n2"


def test_delete(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("cpu", "2")

    store.put("k1", "{}", "{}", "n", "u", "p")
    assert store.delete("k1")
    assert not store.contains("k1")
    assert not store.delete("k1")  # Already gone


def test_count_and_keys(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("cpu", "2")

    for i in range(5):
        store.put(f"k{i}", "{}", "{}", "n", "u", "p")

    assert store.count() == 5
    assert len(store.keys()) == 5
    assert len(store.keys(limit=3)) == 3


def test_stats(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("cpu", "2")

    store.put("k1", "{}", '{"is_legal":true,"execution_times":[0.1]}', "n", "alice", "proj_a")
    store.put("k2", "{}", '{"is_legal":false,"execution_times":null}', "n", "bob", "proj_b")

    s = store.stats()
    assert s["total_records"] == 2
    assert s["legal_records"] == 1
    assert s["illegal_records"] == 1
    assert set(s["users"]) == {"alice", "bob"}
    assert set(s["source_projects"]) == {"proj_a", "proj_b"}
