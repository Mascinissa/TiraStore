"""TiraStore — a distributed lookup table for Tiramisu program measurements.

Designed for HPC / Slurm clusters with a Lustre shared filesystem where
fcntl/flock are unreliable across nodes.  Uses SQLite for storage and
atomic hard-link creation for distributed locking.

Usage
-----
::

    from tirastore import TiraStore

    store = TiraStore("/shared/my_store.db", source_project="autoscheduler-v2")

    # Record a measurement
    store.record(
        program_name="blur",
        program_source_code="...",
        tiralib_schedule_string="...",
        is_legal=True,
        execution_times=[0.042, 0.039, 0.041],
    )

    # Look up a previous measurement
    result = store.lookup("blur", "...", "...")
    if result is not None:
        print(result.execution_times)
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tirastore._keys import make_key, make_program_hash
from tirastore._lock import HardLinkLock
from tirastore._schedule import normalize_schedule, validate_schedule
from tirastore._store import Store


def _get_cpu_model() -> str:
    """Return the CPU model string for the current machine.

    Uses py-cpuinfo if available, otherwise falls back to
    reading /proc/cpuinfo on Linux.
    """
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", info.get("brand", "unknown"))
    except Exception:
        pass
    # Fallback: parse /proc/cpuinfo (Linux)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _get_slurm_cpus() -> str:
    """Return SLURM_CPUS_PER_TASK or 'N/A'."""
    return os.environ.get("SLURM_CPUS_PER_TASK", "N/A")


@dataclass
class LookupResult:
    """Returned by :meth:`TiraStore.lookup` when a record is found."""

    is_legal: bool
    execution_times: Optional[list[float]]
    schedule: str
    # record metadata
    hostname: str
    username: str
    creation_date: str
    update_date: str
    source_project: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_legal": self.is_legal,
            "execution_times": self.execution_times,
            "schedule": self.schedule,
            "hostname": self.hostname,
            "username": self.username,
            "creation_date": self.creation_date,
            "update_date": self.update_date,
            "source_project": self.source_project,
        }


class TiraStore:
    """High-level interface to the distributed lookup table.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Created if it does not exist.
    source_project : str
        Name of the project/experiment that owns this connection.
        Stored as metadata on every new record.
    cpu_model : str or None
        CPU model string for the database.  If *None* and the database is
        being created for the first time, it is auto-detected.
    slurm_cpus : str or None
        ``SLURM_CPUS_PER_TASK`` value.  If *None* and the database is
        being created, it is read from the environment.
    allow_cpu_mismatch : bool
        If *False* (default) and the connecting machine's CPU does not
        match the database's CPU metadata, write operations are blocked
        and a warning is printed.  Set to *True* to allow writes anyway
        (useful for admin/migration tasks).
    stale_lock_timeout : float
        Seconds before a held lock is considered stale (default 600).
    """

    def __init__(
        self,
        db_path: str | Path,
        source_project: str = "",
        cpu_model: Optional[str] = None,
        slurm_cpus: Optional[str] = None,
        allow_cpu_mismatch: bool = False,
        stale_lock_timeout: float = 600.0,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.source_project = source_project
        self.allow_cpu_mismatch = allow_cpu_mismatch

        self._lock = HardLinkLock(
            self.db_path.with_suffix(".db.lock"),
            stale_timeout=stale_lock_timeout,
        )
        self._store = Store(self.db_path)

        # Detect local machine info (done once at init)
        self._local_cpu_model = cpu_model or _get_cpu_model()
        self._local_slurm_cpus = slurm_cpus or _get_slurm_cpus()
        self._hostname = socket.gethostname()
        self._username = getpass.getuser()

        # Initialize or validate
        self._writes_allowed = True
        self._init_or_validate(cpu_model, slurm_cpus)

    # ------------------------------------------------------------------
    # Initialization / validation
    # ------------------------------------------------------------------

    def _init_or_validate(
        self,
        cpu_model_arg: Optional[str],
        slurm_cpus_arg: Optional[str],
    ) -> None:
        """Create the database if needed; validate CPU metadata otherwise."""
        with self._lock:
            if not self.db_path.exists():
                self._create_db(cpu_model_arg, slurm_cpus_arg)
            else:
                self._validate_cpu()

    def _create_db(
        self,
        cpu_model_arg: Optional[str],
        slurm_cpus_arg: Optional[str],
    ) -> None:
        """First-time database creation."""
        # Ensure the parent directory exists and is world-writable
        parent = self.db_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            parent.chmod(0o1777)
        except OSError:
            pass

        db_cpu = cpu_model_arg if cpu_model_arg else _get_cpu_model()
        db_slurm = slurm_cpus_arg if slurm_cpus_arg else _get_slurm_cpus()

        self._store.init_db(db_cpu, db_slurm)

    def _validate_cpu(self) -> None:
        """Check that the local CPU matches the database's CPU metadata."""
        db_cpu = self._store.get_cpu_model()
        db_slurm = self._store.get_slurm_cpus()

        mismatch_parts: list[str] = []
        if db_cpu and db_cpu != self._local_cpu_model:
            mismatch_parts.append(
                f"  CPU model: DB={db_cpu!r}  local={self._local_cpu_model!r}"
            )
        if db_slurm and db_slurm != "N/A" and db_slurm != self._local_slurm_cpus:
            mismatch_parts.append(
                f"  SLURM_CPUS_PER_TASK: DB={db_slurm!r}  local={self._local_slurm_cpus!r}"
            )

        if mismatch_parts and not self.allow_cpu_mismatch:
            msg = (
                "[TiraStore] CPU metadata mismatch — write operations disabled.\n"
                + "\n".join(mismatch_parts)
                + "\n  Set allow_cpu_mismatch=True to override."
            )
            print(msg)
            self._writes_allowed = False

    def _check_writes(self) -> None:
        if not self._writes_allowed:
            raise PermissionError(
                "Write operations are disabled due to CPU metadata mismatch. "
                "Initialize with allow_cpu_mismatch=True to override."
            )

    # ------------------------------------------------------------------
    # Public API — lookup & record
    # ------------------------------------------------------------------

    def lookup(
        self,
        program_name: str,
        program_source_code: str,
        tiralib_schedule_string: str,
    ) -> Optional[LookupResult]:
        """Look up a previously recorded result.

        Returns a :class:`LookupResult` if found, otherwise *None*.
        """
        prog_hash = make_program_hash(program_source_code)
        key = make_key(prog_hash, tiralib_schedule_string)
        with self._lock:
            row = self._store.get(key)
        if row is None:
            return None
        result = json.loads(row["result_json"])
        return LookupResult(
            is_legal=result["is_legal"],
            execution_times=result.get("execution_times"),
            schedule=row["schedule"],
            hostname=row["hostname"],
            username=row["username"],
            creation_date=row["creation_date"],
            update_date=row["update_date"],
            source_project=row["source_project"],
        )

    def record(
        self,
        program_name: str,
        program_source_code: str,
        tiralib_schedule_string: str,
        is_legal: bool,
        execution_times: Optional[list[float]] = None,
        overwrite: bool = False,
    ) -> bool:
        """Store a measurement result.

        Parameters
        ----------
        is_legal : bool
            Whether the optimized program is legal.
        execution_times : list of float or None
            Measured execution times.  **Required** if ``is_legal`` is True.
        overwrite : bool
            If *True*, overwrite an existing record with the same key.

        Returns
        -------
        bool
            *True* if a write occurred, *False* if the record already existed
            and ``overwrite`` was *False*.

        Raises
        ------
        ValueError
            If ``is_legal`` is True but ``execution_times`` is None or empty,
            or if the schedule string is invalid.
        PermissionError
            If writes are disabled due to CPU mismatch.
        """
        self._check_writes()

        if is_legal and (execution_times is None or len(execution_times) == 0):
            raise ValueError(
                "execution_times must be provided (non-empty list) when is_legal is True."
            )

        # Validate the schedule (on the normalized form)
        normalized_sched = normalize_schedule(tiralib_schedule_string)
        valid, reason = validate_schedule(normalized_sched)
        if not valid:
            raise ValueError(f"Invalid schedule string: {reason}")

        prog_hash = make_program_hash(program_source_code)
        key = make_key(prog_hash, tiralib_schedule_string)

        result_obj = {
            "is_legal": is_legal,
            "execution_times": execution_times,
        }
        result_json = json.dumps(result_obj, separators=(",", ":"), ensure_ascii=True)

        with self._lock:
            # Ensure the program is stored (insert-if-absent)
            self._store.put_program(prog_hash, program_name, program_source_code)
            return self._store.put(
                key=key,
                program_hash=prog_hash,
                schedule=normalized_sched,
                result_json=result_json,
                hostname=self._hostname,
                username=self._username,
                source_project=self.source_project,
                overwrite=overwrite,
            )

    def record_many(
        self,
        program_name: str,
        program_source_code: str,
        schedules: list[dict],
        overwrite: bool = False,
    ) -> int:
        """Record multiple schedules for the same program in one operation.

        Parameters
        ----------
        program_name : str
            Name of the program.
        program_source_code : str
            Source code of the program.
        schedules : list of dict
            Each dict must have keys:
            - ``tiralib_schedule_string`` (str)
            - ``is_legal`` (bool)
            - ``execution_times`` (list of float or None)
        overwrite : bool
            If *True*, overwrite existing records.

        Returns
        -------
        int
            Number of records actually written.

        Raises
        ------
        ValueError
            If any schedule is invalid, or if ``is_legal`` is True but
            ``execution_times`` is None or empty.
        PermissionError
            If writes are disabled due to CPU mismatch.
        """
        self._check_writes()

        prog_hash = make_program_hash(program_source_code)

        # Validate all entries before writing anything
        prepared: list[tuple[str, str, str]] = []
        for i, entry in enumerate(schedules):
            sched = entry["tiralib_schedule_string"]
            is_legal = entry["is_legal"]
            exec_times = entry.get("execution_times")

            if is_legal and (exec_times is None or len(exec_times) == 0):
                raise ValueError(
                    f"schedules[{i}]: execution_times must be provided "
                    f"(non-empty list) when is_legal is True."
                )

            normalized_sched = normalize_schedule(sched)
            valid, reason = validate_schedule(normalized_sched)
            if not valid:
                raise ValueError(f"schedules[{i}]: Invalid schedule string: {reason}")

            key = make_key(prog_hash, sched)
            result_obj = {
                "is_legal": is_legal,
                "execution_times": exec_times,
            }
            result_json = json.dumps(
                result_obj, separators=(",", ":"), ensure_ascii=True
            )
            prepared.append((key, normalized_sched, result_json))

        with self._lock:
            self._store.put_program(prog_hash, program_name, program_source_code)
            return self._store.put_many(
                rows=prepared,
                program_hash=prog_hash,
                hostname=self._hostname,
                username=self._username,
                source_project=self.source_project,
                overwrite=overwrite,
            )

    # ------------------------------------------------------------------
    # Public API — convenience / admin
    # ------------------------------------------------------------------

    def contains(
        self,
        program_name: str,
        program_source_code: str,
        tiralib_schedule_string: str,
    ) -> bool:
        """Check if a record exists for the given input."""
        prog_hash = make_program_hash(program_source_code)
        key = make_key(prog_hash, tiralib_schedule_string)
        with self._lock:
            return self._store.contains(key)

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Retrieve a raw record (joined with program data) by its SHA-256 key."""
        with self._lock:
            return self._store.get(key)

    def put(
        self,
        key: str,
        program_hash: str,
        schedule: str,
        result_json: str,
        overwrite: bool = False,
    ) -> bool:
        """Low-level insert/update by raw key (admin use).

        The referenced program must already exist in the programs table.
        """
        self._check_writes()
        with self._lock:
            return self._store.put(
                key=key,
                program_hash=program_hash,
                schedule=schedule,
                result_json=result_json,
                hostname=self._hostname,
                username=self._username,
                source_project=self.source_project,
                overwrite=overwrite,
            )

    def get_program_source(self, program_name: str) -> list[dict[str, str]]:
        """Retrieve source code for a program by name.

        Returns a list of dicts, each with ``program_hash`` and
        ``source_code``.  Multiple entries are returned if different
        source versions share the same name.
        """
        with self._lock:
            rows = self._store.get_programs_by_name(program_name)
        return [
            {
                "program_hash": r["program_hash"],
                "source_code": r["source_code"],
            }
            for r in rows
        ]

    def get_program_records(
        self,
        program_name: str,
        program_source_code: str,
    ) -> list[LookupResult]:
        """Retrieve all records for a specific program.

        Returns a list of :class:`LookupResult` objects — one per
        schedule that has been recorded for this program.
        """
        prog_hash = make_program_hash(program_source_code)
        with self._lock:
            rows = self._store.get_records_by_program_hash(prog_hash)
        results: list[LookupResult] = []
        for row in rows:
            result = json.loads(row["result_json"])
            results.append(
                LookupResult(
                    is_legal=result["is_legal"],
                    execution_times=result.get("execution_times"),
                    schedule=row["schedule"],
                    hostname=row["hostname"],
                    username=row["username"],
                    creation_date=row["creation_date"],
                    update_date=row["update_date"],
                    source_project=row["source_project"],
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public API — backup & export
    # ------------------------------------------------------------------

    def backup(self, backup_path: str | Path | None = None) -> Path:
        """Create a snapshot copy of the database file.

        Parameters
        ----------
        backup_path : str, Path, or None
            Destination path for the backup.  If *None*, a timestamped
            file is created next to the database
            (e.g. ``store_20260221T153012Z.db``).

        Returns
        -------
        Path
            The path to the backup file.
        """
        if backup_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            stem = self.db_path.stem
            backup_path = self.db_path.with_name(f"{stem}_{ts}.db")
        backup_path = Path(backup_path).resolve()

        with self._lock:
            shutil.copy2(str(self.db_path), str(backup_path))

        return backup_path

    def export(
        self,
        output_path: str | Path,
        fmt: str = "json",
    ) -> Path:
        """Export the entire database to JSON or JSONL.

        Parameters
        ----------
        output_path : str or Path
            Destination file path.
        fmt : str
            ``"json"`` (default) or ``"jsonl"`` (one program per line).

        Returns
        -------
        Path
            The path to the exported file.

        The exported structure groups records by program name::

            {
              "blur": {
                "Tiramisu_cpp": "<source code>",
                "schedules_list": [
                  {
                    "schedule_str": "S(L0,L1,4,8,comps=['c1'])",
                    "is_legal": true,
                    "execution_times": [0.04]
                  },
                  ...
                ],
                "program_name": "blur"
              },
              ...
            }

        If the same program name has multiple source-code versions, keys
        are suffixed with ``_v1``, ``_v2``, etc.
        """
        if fmt not in ("json", "jsonl"):
            raise ValueError(f"fmt must be 'json' or 'jsonl', got {fmt!r}")

        output_path = Path(output_path).resolve()

        with self._lock:
            all_data = self._store.get_all_programs_with_records()

        # Group by program_name, detect duplicates needing _vN suffix
        by_name: dict[str, list[dict[str, Any]]] = {}
        for entry in all_data:
            name = entry["program_name"]
            by_name.setdefault(name, []).append(entry)

        export_dict: dict[str, dict[str, Any]] = {}
        for name, versions in by_name.items():
            needs_suffix = len(versions) > 1
            for idx, entry in enumerate(versions):
                result = json.loads(entry["records"][0]["result_json"]) if entry["records"] else {}
                schedules_list = []
                for rec in entry["records"]:
                    r = json.loads(rec["result_json"])
                    schedules_list.append({
                        "schedule_str": rec["schedule"],
                        "is_legal": r["is_legal"],
                        "execution_times": r.get("execution_times"),
                    })

                key = f"{name}_v{idx + 1}" if needs_suffix else name
                export_dict[key] = {
                    "Tiramisu_cpp": entry["source_code"],
                    "schedules_list": schedules_list,
                    "program_name": name,
                }

        with open(output_path, "w", encoding="utf-8") as f:
            if fmt == "json":
                json.dump(export_dict, f, indent=2, ensure_ascii=False)
                f.write("\n")
            else:  # jsonl
                for key, value in export_dict.items():
                    line_obj = {key: value}
                    f.write(json.dumps(line_obj, ensure_ascii=False))
                    f.write("\n")

        return output_path

    def delete(self, key: str) -> bool:
        """Delete a record by its SHA-256 key."""
        self._check_writes()
        with self._lock:
            return self._store.delete(key)

    def count(self) -> int:
        """Return the total number of records."""
        with self._lock:
            return self._store.count()

    def program_count(self) -> int:
        """Return the total number of distinct programs."""
        with self._lock:
            return self._store.program_count()

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the database."""
        with self._lock:
            return self._store.stats()

    def keys(self, limit: int = 0, offset: int = 0) -> list[str]:
        """Return record keys with optional pagination."""
        with self._lock:
            return self._store.keys(limit=limit, offset=offset)

    @property
    def writes_allowed(self) -> bool:
        """Whether write operations are permitted on this connection."""
        return self._writes_allowed

    @property
    def cpu_model(self) -> Optional[str]:
        """The CPU model stored in the database metadata."""
        with self._lock:
            return self._store.get_cpu_model()

    @property
    def slurm_cpus(self) -> Optional[str]:
        """The SLURM_CPUS_PER_TASK value stored in the database metadata."""
        with self._lock:
            return self._store.get_slurm_cpus()

    def __repr__(self) -> str:
        return (
            f"TiraStore(db_path={str(self.db_path)!r}, "
            f"source_project={self.source_project!r}, "
            f"writes_allowed={self._writes_allowed})"
        )
