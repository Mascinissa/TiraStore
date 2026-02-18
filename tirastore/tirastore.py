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
        print(result["execution_times"])
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tirastore._keys import make_input_json, make_key
from tirastore._lock import HardLinkLock
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
        key = make_key(program_name, program_source_code, tiralib_schedule_string)
        with self._lock:
            row = self._store.get(key)
        if row is None:
            return None
        result = json.loads(row["result_json"])
        return LookupResult(
            is_legal=result["is_legal"],
            execution_times=result.get("execution_times"),
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
            If ``is_legal`` is True but ``execution_times`` is None or empty.
        PermissionError
            If writes are disabled due to CPU mismatch.
        """
        self._check_writes()

        if is_legal and (execution_times is None or len(execution_times) == 0):
            raise ValueError(
                "execution_times must be provided (non-empty list) when is_legal is True."
            )

        key = make_key(program_name, program_source_code, tiralib_schedule_string)
        input_json = make_input_json(
            program_name, program_source_code, tiralib_schedule_string
        )
        result_obj = {
            "is_legal": is_legal,
            "execution_times": execution_times,
        }
        result_json = json.dumps(result_obj, separators=(",", ":"), ensure_ascii=True)

        with self._lock:
            return self._store.put(
                key=key,
                input_json=input_json,
                result_json=result_json,
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
        key = make_key(program_name, program_source_code, tiralib_schedule_string)
        with self._lock:
            return self._store.contains(key)

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Retrieve a raw record by its SHA-256 key."""
        with self._lock:
            return self._store.get(key)

    def put(
        self,
        key: str,
        input_json: str,
        result_json: str,
        overwrite: bool = False,
    ) -> bool:
        """Low-level insert/update by raw key (admin use)."""
        self._check_writes()
        with self._lock:
            return self._store.put(
                key=key,
                input_json=input_json,
                result_json=result_json,
                hostname=self._hostname,
                username=self._username,
                source_project=self.source_project,
                overwrite=overwrite,
            )

    def delete(self, key: str) -> bool:
        """Delete a record by its SHA-256 key."""
        self._check_writes()
        with self._lock:
            return self._store.delete(key)

    def count(self) -> int:
        """Return the total number of records."""
        with self._lock:
            return self._store.count()

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
