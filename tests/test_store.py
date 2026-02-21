"""Tests for the low-level SQLite storage layer."""

import json

from tirastore._store import Store


def _init_store(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("cpu", "2")
    return store


def test_init_and_meta(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    store.init_db("Intel Xeon E5-2680", "4")

    assert store.get_cpu_model() == "Intel Xeon E5-2680"
    assert store.get_slurm_cpus() == "4"
    assert store.get_meta("schema_version") == "2"


# ------------------------------------------------------------------
# Programs table
# ------------------------------------------------------------------


def test_put_program(tmp_path):
    store = _init_store(tmp_path)
    inserted = store.put_program("hash1", "blur", "void blur() {}")
    assert inserted
    assert store.program_count() == 1


def test_put_program_idempotent(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("hash1", "blur", "void blur() {}")
    inserted = store.put_program("hash1", "blur", "void blur() {}")
    assert not inserted
    assert store.program_count() == 1


def test_get_program(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("hash1", "blur", "void blur() {}")
    prog = store.get_program("hash1")
    assert prog is not None
    assert prog["program_name"] == "blur"
    assert prog["source_code"] == "void blur() {}"


def test_get_program_missing(tmp_path):
    store = _init_store(tmp_path)
    assert store.get_program("nonexistent") is None


# ------------------------------------------------------------------
# Records table
# ------------------------------------------------------------------


def test_put_get_contains(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("phash", "blur", "void blur() {}")

    key = "abc123"
    result_json = json.dumps({"is_legal": True, "execution_times": [0.01]})

    assert not store.contains(key)
    wrote = store.put(key, "phash", "R(L0,comps=['c1'])", result_json, "node01", "alice", "proj")
    assert wrote
    assert store.contains(key)

    row = store.get(key)
    assert row is not None
    assert row["key"] == key
    assert row["program_hash"] == "phash"
    assert row["schedule"] == "R(L0,comps=['c1'])"
    assert row["hostname"] == "node01"
    assert row["username"] == "alice"
    assert row["source_project"] == "proj"
    # Joined program data
    assert row["program_name"] == "blur"
    assert row["source_code"] == "void blur() {}"
    assert json.loads(row["result_json"])["is_legal"] is True


def test_put_no_overwrite(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph", "p", "code")

    key = "abc123"
    store.put(key, "ph", "", '{"is_legal":false}', "n1", "u1", "p1")
    wrote = store.put(key, "ph", "", '{"is_legal":true}', "n2", "u2", "p2", overwrite=False)
    assert not wrote
    row = store.get(key)
    assert json.loads(row["result_json"])["is_legal"] is False


def test_put_overwrite(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph", "p", "code")

    key = "abc123"
    store.put(key, "ph", "", '{"is_legal":false}', "n1", "u1", "p1")
    wrote = store.put(key, "ph", "", '{"is_legal":true}', "n2", "u2", "p2", overwrite=True)
    assert wrote
    row = store.get(key)
    assert json.loads(row["result_json"])["is_legal"] is True
    assert row["hostname"] == "n2"


def test_delete(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph", "p", "code")

    store.put("k1", "ph", "", "{}", "n", "u", "p")
    assert store.delete("k1")
    assert not store.contains("k1")
    assert not store.delete("k1")  # Already gone


def test_count_and_keys(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph", "p", "code")

    for i in range(5):
        store.put(f"k{i}", "ph", "", "{}", "n", "u", "p")

    assert store.count() == 5
    assert len(store.keys()) == 5
    assert len(store.keys(limit=3)) == 3


def test_stats(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph1", "blur", "code1")
    store.put_program("ph2", "edge", "code2")

    store.put("k1", "ph1", "", '{"is_legal":true,"execution_times":[0.1]}', "n", "alice", "proj_a")
    store.put("k2", "ph2", "", '{"is_legal":false,"execution_times":null}', "n", "bob", "proj_b")

    s = store.stats()
    assert s["total_records"] == 2
    assert s["legal_records"] == 1
    assert s["illegal_records"] == 1
    assert s["total_programs"] == 2
    assert set(s["users"]) == {"alice", "bob"}
    assert set(s["source_projects"]) == {"proj_a", "proj_b"}


def test_program_dedup(tmp_path):
    """Multiple records sharing the same program should not duplicate it."""
    store = _init_store(tmp_path)
    store.put_program("ph", "blur", "void blur() {}")

    store.put("k1", "ph", "sched_a", '{"is_legal":true}', "n", "u", "p")
    store.put("k2", "ph", "sched_b", '{"is_legal":false}', "n", "u", "p")

    assert store.count() == 2
    assert store.program_count() == 1


# ------------------------------------------------------------------
# get_programs_by_name
# ------------------------------------------------------------------


def test_get_programs_by_name(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("h1", "blur", "void blur_v1() {}")
    store.put_program("h2", "blur", "void blur_v2() {}")
    store.put_program("h3", "edge", "void edge() {}")

    results = store.get_programs_by_name("blur")
    assert len(results) == 2
    hashes = {r["program_hash"] for r in results}
    assert hashes == {"h1", "h2"}


def test_get_programs_by_name_empty(tmp_path):
    store = _init_store(tmp_path)
    assert store.get_programs_by_name("nonexistent") == []


# ------------------------------------------------------------------
# put_many
# ------------------------------------------------------------------


def test_put_many(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph", "blur", "void blur() {}")

    rows = [
        ("k1", "sched_a", '{"is_legal":true}'),
        ("k2", "sched_b", '{"is_legal":false}'),
        ("k3", "sched_c", '{"is_legal":true}'),
    ]
    written = store.put_many(rows, "ph", "node01", "alice", "proj")
    assert written == 3
    assert store.count() == 3


def test_put_many_no_overwrite(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph", "blur", "void blur() {}")
    store.put("k1", "ph", "sched_a", '{"is_legal":false}', "n", "u", "p")

    rows = [
        ("k1", "sched_a", '{"is_legal":true}'),
        ("k2", "sched_b", '{"is_legal":false}'),
    ]
    written = store.put_many(rows, "ph", "n", "u", "p", overwrite=False)
    assert written == 1  # k1 skipped, k2 written
    assert store.count() == 2


def test_put_many_overwrite(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph", "blur", "void blur() {}")
    store.put("k1", "ph", "sched_a", '{"is_legal":false}', "n", "u", "p")

    rows = [
        ("k1", "sched_a", '{"is_legal":true}'),
    ]
    written = store.put_many(rows, "ph", "n", "u", "p", overwrite=True)
    assert written == 1
    row = store.get("k1")
    assert json.loads(row["result_json"])["is_legal"] is True


def test_put_many_empty(tmp_path):
    store = _init_store(tmp_path)
    written = store.put_many([], "ph", "n", "u", "p")
    assert written == 0


# ------------------------------------------------------------------
# get_records_by_program_hash
# ------------------------------------------------------------------


def test_get_records_by_program_hash(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph1", "blur", "void blur() {}")
    store.put_program("ph2", "edge", "void edge() {}")

    store.put("k1", "ph1", "sched_a", '{"is_legal":true}', "n", "u", "p")
    store.put("k2", "ph1", "sched_b", '{"is_legal":false}', "n", "u", "p")
    store.put("k3", "ph2", "sched_a", '{"is_legal":true}', "n", "u", "p")

    results = store.get_records_by_program_hash("ph1")
    assert len(results) == 2
    keys = {r["key"] for r in results}
    assert keys == {"k1", "k2"}
    # Should include joined program data
    assert results[0]["program_name"] == "blur"


def test_get_records_by_program_hash_empty(tmp_path):
    store = _init_store(tmp_path)
    assert store.get_records_by_program_hash("nonexistent") == []


# ------------------------------------------------------------------
# get_all_programs_with_records
# ------------------------------------------------------------------


def test_get_all_programs_with_records(tmp_path):
    store = _init_store(tmp_path)
    store.put_program("ph1", "blur", "void blur() {}")
    store.put_program("ph2", "edge", "void edge() {}")

    store.put("k1", "ph1", "sched_a", '{"is_legal":true}', "n", "u", "p")
    store.put("k2", "ph1", "sched_b", '{"is_legal":false}', "n", "u", "p")
    store.put("k3", "ph2", "sched_a", '{"is_legal":true}', "n", "u", "p")

    result = store.get_all_programs_with_records()
    assert len(result) == 2
    names = {r["program_name"] for r in result}
    assert names == {"blur", "edge"}

    blur = [r for r in result if r["program_name"] == "blur"][0]
    assert len(blur["records"]) == 2
    assert blur["source_code"] == "void blur() {}"

    edge = [r for r in result if r["program_name"] == "edge"][0]
    assert len(edge["records"]) == 1


def test_get_all_programs_with_records_empty(tmp_path):
    store = _init_store(tmp_path)
    assert store.get_all_programs_with_records() == []
