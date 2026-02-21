"""Low-level SQLite storage backend for TiraStore.

Handles schema creation, reads, writes, and metadata management.  All public
methods in this module assume the caller already holds the distributed lock;
they do **not** perform any locking themselves.

Critical SQLite settings for Lustre compatibility:
- journal_mode = DELETE  (WAL uses shared memory / mmap which breaks on Lustre)
- busy_timeout = 0       (we rely on our own lock, not SQLite's)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCHEMA_VERSION = 2

_CREATE_META_TABLE = """\
CREATE TABLE IF NOT EXISTS db_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_PROGRAMS_TABLE = """\
CREATE TABLE IF NOT EXISTS programs (
    program_hash    TEXT PRIMARY KEY,
    program_name    TEXT NOT NULL,
    source_code     TEXT NOT NULL
);
"""

_CREATE_RECORDS_TABLE = """\
CREATE TABLE IF NOT EXISTS records (
    key                TEXT PRIMARY KEY,
    program_hash       TEXT NOT NULL REFERENCES programs(program_hash),
    schedule           TEXT NOT NULL,
    result_json        TEXT NOT NULL,
    hostname           TEXT NOT NULL,
    username           TEXT NOT NULL,
    creation_date      TEXT NOT NULL,
    update_date        TEXT NOT NULL,
    source_project     TEXT NOT NULL DEFAULT ''
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Thin wrapper around an on-disk SQLite database.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    # ------------------------------------------------------------------
    # Connection helpers (short-lived, opened inside the lock)
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA busy_timeout=0;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Schema / init
    # ------------------------------------------------------------------

    def init_db(self, cpu_model: str, slurm_cpus: str) -> None:
        """Create tables and set database-level metadata.

        Should be called exactly once when the database file is first created.
        """
        old_umask = os.umask(0)
        try:
            conn = self._connect()
            try:
                conn.execute(_CREATE_META_TABLE)
                conn.execute(_CREATE_PROGRAMS_TABLE)
                conn.execute(_CREATE_RECORDS_TABLE)
                conn.execute(
                    "INSERT OR IGNORE INTO db_meta (key, value) VALUES (?, ?)",
                    ("schema_version", str(_SCHEMA_VERSION)),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO db_meta (key, value) VALUES (?, ?)",
                    ("cpu_model", cpu_model),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO db_meta (key, value) VALUES (?, ?)",
                    ("slurm_cpus", slurm_cpus),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO db_meta (key, value) VALUES (?, ?)",
                    ("created_at", _now_iso()),
                )
                conn.commit()
            finally:
                conn.close()
        finally:
            os.umask(old_umask)

        # Ensure the file itself is world-readable/writable
        try:
            os.chmod(str(self.db_path), 0o666)
        except OSError:
            pass

    def ensure_tables(self) -> None:
        """Idempotent: create tables if they don't exist (for safety)."""
        conn = self._connect()
        try:
            conn.execute(_CREATE_META_TABLE)
            conn.execute(_CREATE_PROGRAMS_TABLE)
            conn.execute(_CREATE_RECORDS_TABLE)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # DB-level metadata
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value FROM db_meta WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None
        finally:
            conn.close()

    def set_meta(self, key: str, value: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO db_meta (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def get_cpu_model(self) -> Optional[str]:
        return self.get_meta("cpu_model")

    def get_slurm_cpus(self) -> Optional[str]:
        return self.get_meta("slurm_cpus")

    # ------------------------------------------------------------------
    # Programs CRUD
    # ------------------------------------------------------------------

    def put_program(
        self, program_hash: str, program_name: str, source_code: str
    ) -> bool:
        """Insert a program if it does not already exist.  Returns True if inserted."""
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT 1 FROM programs WHERE program_hash = ?", (program_hash,)
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO programs (program_hash, program_name, source_code) VALUES (?, ?, ?)",
                (program_hash, program_name, source_code),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_program(self, program_hash: str) -> Optional[dict[str, Any]]:
        """Return a program row as a dict, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM programs WHERE program_hash = ?", (program_hash,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_programs_by_name(self, program_name: str) -> list[dict[str, Any]]:
        """Return all program rows matching a given name."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM programs WHERE program_name = ? ORDER BY program_hash",
                (program_name,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def program_count(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM programs").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Record CRUD
    # ------------------------------------------------------------------

    def contains(self, key: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM records WHERE key = ?", (key,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return a record joined with its program, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                """\
                SELECT r.*, p.program_name, p.source_code
                  FROM records r
                  JOIN programs p ON r.program_hash = p.program_hash
                 WHERE r.key = ?
                """,
                (key,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def put(
        self,
        key: str,
        program_hash: str,
        schedule: str,
        result_json: str,
        hostname: str,
        username: str,
        source_project: str,
        overwrite: bool = False,
    ) -> bool:
        """Insert or update a record.  Returns True if a write occurred.

        The referenced program must already exist in the programs table.
        """
        now = _now_iso()
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT 1 FROM records WHERE key = ?", (key,)
            ).fetchone()
            if existing and not overwrite:
                return False
            if existing and overwrite:
                conn.execute(
                    """\
                    UPDATE records
                       SET program_hash   = ?,
                           schedule       = ?,
                           result_json    = ?,
                           hostname       = ?,
                           username       = ?,
                           update_date    = ?,
                           source_project = ?
                     WHERE key = ?
                    """,
                    (program_hash, schedule, result_json, hostname, username, now, source_project, key),
                )
            else:
                conn.execute(
                    """\
                    INSERT INTO records
                        (key, program_hash, schedule, result_json, hostname,
                         username, creation_date, update_date, source_project)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key, program_hash, schedule, result_json, hostname,
                     username, now, now, source_project),
                )
            conn.commit()
            return True
        finally:
            conn.close()

    def put_many(
        self,
        rows: list[tuple[str, str, str, str]],
        program_hash: str,
        hostname: str,
        username: str,
        source_project: str,
        overwrite: bool = False,
    ) -> int:
        """Insert multiple records in a single transaction.

        Parameters
        ----------
        rows : list of (key, schedule, result_json) tuples
            Each entry is (record_key, normalized_schedule, result_json).
        program_hash : str
            The program hash shared by all rows.
        overwrite : bool
            If True, overwrite existing records.

        Returns the number of rows actually written.
        """
        now = _now_iso()
        written = 0
        conn = self._connect()
        try:
            for key, schedule, result_json in rows:
                existing = conn.execute(
                    "SELECT 1 FROM records WHERE key = ?", (key,)
                ).fetchone()
                if existing and not overwrite:
                    continue
                if existing and overwrite:
                    conn.execute(
                        """\
                        UPDATE records
                           SET program_hash   = ?,
                               schedule       = ?,
                               result_json    = ?,
                               hostname       = ?,
                               username       = ?,
                               update_date    = ?,
                               source_project = ?
                         WHERE key = ?
                        """,
                        (program_hash, schedule, result_json, hostname,
                         username, now, source_project, key),
                    )
                else:
                    conn.execute(
                        """\
                        INSERT INTO records
                            (key, program_hash, schedule, result_json, hostname,
                             username, creation_date, update_date, source_project)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (key, program_hash, schedule, result_json, hostname,
                         username, now, now, source_project),
                    )
                written += 1
            conn.commit()
            return written
        finally:
            conn.close()

    def get_records_by_program_hash(
        self, program_hash: str
    ) -> list[dict[str, Any]]:
        """Return all records (joined with program) for a given program hash."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """\
                SELECT r.*, p.program_name, p.source_code
                  FROM records r
                  JOIN programs p ON r.program_hash = p.program_hash
                 WHERE r.program_hash = ?
                 ORDER BY r.creation_date
                """,
                (program_hash,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete(self, key: str) -> bool:
        """Delete a record by key.  Returns True if it existed."""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM records WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM records").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the database."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) AS cnt FROM records").fetchone()["cnt"]
            legal = conn.execute(
                "SELECT COUNT(*) AS cnt FROM records WHERE json_extract(result_json, '$.is_legal') = 1"
            ).fetchone()["cnt"]
            illegal = conn.execute(
                "SELECT COUNT(*) AS cnt FROM records WHERE json_extract(result_json, '$.is_legal') = 0"
            ).fetchone()["cnt"]
            programs = conn.execute(
                "SELECT COUNT(*) AS cnt FROM programs"
            ).fetchone()["cnt"]
            projects = conn.execute(
                "SELECT DISTINCT source_project FROM records"
            ).fetchall()
            users = conn.execute(
                "SELECT DISTINCT username FROM records"
            ).fetchall()
            return {
                "total_records": total,
                "legal_records": legal,
                "illegal_records": illegal,
                "total_programs": programs,
                "source_projects": [r["source_project"] for r in projects],
                "users": [r["username"] for r in users],
                "cpu_model": self.get_cpu_model(),
                "slurm_cpus": self.get_slurm_cpus(),
            }
        finally:
            conn.close()

    def keys(self, limit: int = 0, offset: int = 0) -> list[str]:
        """Return record keys with optional pagination."""
        conn = self._connect()
        try:
            query = "SELECT key FROM records ORDER BY creation_date"
            params: list[Any] = []
            if limit > 0:
                query += " LIMIT ? OFFSET ?"
                params = [limit, offset]
            rows = conn.execute(query, params).fetchall()
            return [r["key"] for r in rows]
        finally:
            conn.close()
