# TiraStore

A distributed lookup table for caching [Tiramisu](https://tiramisu-compiler.org/) program execution measurements on HPC clusters. Designed for Slurm-based clusters with Lustre shared filesystems where standard file locking (`fcntl`/`flock`) is unreliable across nodes.

Multiple users and up to 64+ concurrent worker nodes can safely read and write to the same database without coordination beyond the library itself.

## Installation

```bash
pip install -e .
```

The only runtime dependency is [`py-cpuinfo`](https://pypi.org/project/py-cpuinfo/). For running unit tests, install with `pip install -e ".[dev]"`.

## Quick Start

```python
from tirastore import TiraStore

# Connect (creates the DB if it doesn't exist)
store = TiraStore(
    "/lustre/shared/tiramisu_cache.db",
    source_project="autoscheduler-v2",
)

# Record a measurement
store.record(
    program_name="blur",
    program_source_code="void blur() { ... }",
    tiralib_schedule_string="S(L0,L1,4,8,comps=['c1'])",
    is_legal=True,
    execution_times=[0.042, 0.039, 0.041],
)

# Look up a previous measurement
result = store.lookup("blur", "void blur() { ... }", "S(L0,L1,4,8,comps=['c1'])")
if result is not None:
    print(result.is_legal)          # True
    print(result.execution_times)   # [0.042, 0.039, 0.041]
    print(result.hostname)          # node042
    print(result.source_project)    # autoscheduler-v2
```

## Typical Slurm Worker Pattern

```python
from tirastore import TiraStore

store = TiraStore("/lustre/shared/cache.db", source_project="my_experiment")

def process(program_name, source_code, schedule):
    # Check cache first
    result = store.lookup(program_name, source_code, schedule)
    if result is not None:
        return result  # Cache hit — skip computation

    # Cache miss — run the actual measurement
    is_legal, times = run_tiramisu_measurement(source_code, schedule)

    # Store for future lookups
    store.record(program_name, source_code, schedule, is_legal, times)
    return store.lookup(program_name, source_code, schedule)
```

## API Reference

### `TiraStore(db_path, ...)`

Constructor parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | `str` or `Path` | *(required)* | Path to the SQLite database file. Created if it doesn't exist. |
| `source_project` | `str` | `""` | Project name stored as metadata on every new record. |
| `cpu_model` | `str` or `None` | `None` | CPU model for the DB. Auto-detected if not provided. |
| `slurm_cpus` | `str` or `None` | `None` | `SLURM_CPUS_PER_TASK`. Read from environment if not provided. |
| `allow_cpu_mismatch` | `bool` | `False` | Allow writes even if the local CPU doesn't match the DB's CPU metadata. |
| `stale_lock_timeout` | `float` | `600.0` | Seconds before a held lock is considered stale (holder crashed). |

### Main Methods

#### `store.lookup(program_name, program_source_code, tiralib_schedule_string)`

Returns a `LookupResult` if the record exists, otherwise `None`.

`LookupResult` fields: `is_legal`, `execution_times`, `hostname`, `username`, `creation_date`, `update_date`, `source_project`. Call `.to_dict()` to get a plain dictionary.

#### `store.record(program_name, program_source_code, tiralib_schedule_string, is_legal, execution_times, overwrite=False)`

Stores a measurement result. Returns `True` if a write occurred, `False` if the record already existed (and `overwrite` was `False`).

- If `is_legal=True`, `execution_times` must be a non-empty list of floats.
- If `is_legal=False`, `execution_times` can be `None`.
- Set `overwrite=True` to replace an existing record.
- The schedule string is validated before recording. Invalid schedules raise `ValueError`.

#### `store.contains(program_name, program_source_code, tiralib_schedule_string)`

Returns `True` if a record exists for the given input.

### Utility Methods

| Method | Description |
|---|---|
| `store.count()` | Total number of records in the database. |
| `store.program_count()` | Total number of distinct programs. |
| `store.stats()` | Summary dict: total/legal/illegal counts, program count, distinct users, projects, CPU info. |
| `store.keys(limit=0, offset=0)` | List of record SHA-256 keys (with optional pagination). |
| `store.get(key)` | Retrieve a raw record dict (joined with program data) by its SHA-256 key. |
| `store.put(key, program_hash, schedule, result_json, overwrite=False)` | Low-level insert/update by raw key (admin use). |
| `store.delete(key)` | Delete a record by its SHA-256 key. |

### Properties

| Property | Description |
|---|---|
| `store.writes_allowed` | `bool` — whether writes are permitted on this connection. |
| `store.cpu_model` | CPU model string stored in the database metadata. |
| `store.slurm_cpus` | `SLURM_CPUS_PER_TASK` value stored in the database metadata. |

### Standalone Utilities

These are also importable from `tirastore` for direct use:

| Function | Description |
|---|---|
| `normalize_schedule(schedule)` | Normalize a schedule string (strip whitespace, unify quote style). |
| `validate_schedule(schedule)` | Validate a schedule string. Returns `(bool, reason_string)`. |
| `normalize_program(source)` | Normalize a program source string for hashing (strip comments, includes, whitespace). |
| `make_program_hash(source)` | Compute the SHA-256 hash of the normalized program source. |

## How It Works

### Storage

All records are stored in a single SQLite file on the shared Lustre filesystem. This keeps the file count minimal (important for HPC storage quotas).

**Program deduplication:** Program source code is stored in a separate `programs` table, keyed by the SHA-256 hash of its normalized form. The `records` table references programs by hash. This means if you record 100 different optimization sequences for the same program, the source code is stored only once.

**Program normalization:** Before hashing, program source code is normalized by stripping block comments (`/* ... */`), single-line comments (`// ...`), `#include` directives, and all whitespace. This ensures that cosmetically different versions of the same program (different formatting, comments, includes) produce the same hash and share a single programs-table entry. The original readable source code is always what gets stored; normalization is only used for hash computation.

**Schedule normalization:** Schedule strings are normalized (whitespace removed, comp names single-quoted) before hashing and storing. This means `R( L0 , comps=["c1"] )` and `R(L0,comps=['c1'])` are treated as identical.

**Schedule validation:** Schedule strings are validated against the grammar of known Tiramisu transformations (S, I, R, P, T2, T3, U, F) before recording. Invalid schedules are rejected with a descriptive error.

**Record keying:** The record key is the SHA-256 hex digest of the canonical JSON of `{program_hash, normalized_schedule}`. Identical logical inputs always produce the same key regardless of cosmetic differences in the source code or schedule formatting.

**SQLite settings for Lustre:**
- `journal_mode=DELETE` — WAL mode uses shared memory/mmap which breaks on Lustre.
- `synchronous=FULL` — ensure durability.
- `busy_timeout=0` — we rely on our own lock, not SQLite's internal locking.

### Distributed Locking

Since `fcntl`/`flock` don't work reliably across Lustre nodes, TiraStore uses an atomic hard-link based mutex:

1. **Acquire:** Create a temp file with a unique name (hostname, PID, timestamp), then `link()` it to the lock path. `link()` is atomic on POSIX/Lustre — it fails if the target already exists. On failure, retry with exponential backoff + random jitter.
2. **Release:** `unlink()` the lock file.
3. **Stale detection:** If a lock is older than the stale timeout (default 10 minutes), assume the holder crashed and break the lock.

Every database operation follows this protocol:

```
acquire lock  ->  open SQLite connection  ->  read/write  ->  close connection  ->  release lock
```

### CPU Metadata Validation

The database stores the CPU model and `SLURM_CPUS_PER_TASK` at creation time because execution times are only comparable across identical hardware configurations.

When connecting to an existing database:
- If the local machine's CPU matches the DB metadata: full read/write access.
- If there's a mismatch: **writes are blocked** (lookups still work) and a warning is printed. Override with `allow_cpu_mismatch=True` for admin tasks like importing data.

### Multi-User Sharing

The database file is created with mode `666` (world-readable/writable) and the parent directory is set to mode `1777` (sticky bit). No common Unix group is required.

## Running Tests

### Unit Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Multi-Node Concurrency Test

A test suite is included to validate locking and concurrency across actual Slurm cluster nodes. Each worker runs as a separate `srun` job to ensure execution on different nodes.

```bash
# Set your Lustre path and cluster partition
export TIRASTORE_TEST_DIR=/lustre/scratch/$USER/tirastore_test
export SLURM_PARTITION=batch

# Launch: 8 workers, 50 unique records each, 20 contested records
bash tests/multinode/launch_test.sh 8 50 20
```

The test runs four phases per worker:

1. **Unique writes** — each worker writes records only it owns.
2. **Contested writes** — all workers attempt to write the same records. Only one writer should win per record.
3. **Cross-reads** — each worker reads its own records, contested records, and samples from other workers.
4. **Overwrite test** — one worker overwrites a contested record.

A verification script runs automatically at the end, checking record counts, data integrity, error-free execution, and that workers actually ran on distinct nodes.

## Database Schema

### `db_meta` table

Stores database-level configuration:

| Key | Description |
|---|---|
| `schema_version` | Schema version (currently `2`). |
| `cpu_model` | CPU model string of the machine that created the DB. |
| `slurm_cpus` | `SLURM_CPUS_PER_TASK` at DB creation time. |
| `created_at` | ISO 8601 timestamp of DB creation. |

### `programs` table

Stores program source code, deduplicated by content hash:

| Column | Type | Description |
|---|---|---|
| `program_hash` | `TEXT PRIMARY KEY` | SHA-256 hex digest of the normalized source code. |
| `program_name` | `TEXT` | Human-readable program name. |
| `source_code` | `TEXT` | Original (readable) program source code. |

### `records` table

| Column | Type | Description |
|---|---|---|
| `key` | `TEXT PRIMARY KEY` | SHA-256 hex digest of `{program_hash, normalized_schedule}`. |
| `program_hash` | `TEXT` | Foreign key referencing `programs(program_hash)`. |
| `schedule` | `TEXT` | Normalized schedule string. |
| `result_json` | `TEXT` | JSON of `{is_legal, execution_times}`. |
| `hostname` | `TEXT` | Node that recorded this entry. |
| `username` | `TEXT` | User that recorded this entry. |
| `creation_date` | `TEXT` | ISO 8601 timestamp of initial creation. |
| `update_date` | `TEXT` | ISO 8601 timestamp of last update. |
| `source_project` | `TEXT` | Project name provided at connection time. |

## Project Structure

```
tirastore/
    __init__.py       # Public API: TiraStore, LookupResult, normalize/validate helpers
    tirastore.py      # Main TiraStore class
    _lock.py          # HardLinkLock — distributed mutex via atomic link()
    _store.py         # SQLite storage backend (programs + records tables)
    _keys.py          # SHA-256 key/hash generation
    _schedule.py      # Schedule + program normalization and validation

tests/
    test_keys.py      # Key and hash generation tests
    test_lock.py      # Locking tests
    test_schedule.py  # Schedule/program normalization and validation tests
    test_store.py     # SQLite storage tests
    test_tirastore.py # Integration tests
    multinode/
        launch_test.sh       # Slurm launcher for concurrency test
        test_worker.py       # Per-node worker script
        verify_results.py    # Post-test verification
```
