"""Atomic hard-link based distributed mutex for Lustre filesystems.

Since fcntl/flock do not work reliably across nodes on Lustre, this module
implements a distributed lock using the atomic link() system call.

Protocol
--------
1. Create a temporary file with a unique name containing hostname, PID,
   and timestamp, plus a JSON payload with the same info.
2. Attempt to hard-link this temp file to the well-known lock path.
   - link() is atomic on POSIX/Lustre: it fails if the target exists.
   - Success  → lock acquired.
   - Failure  → another holder exists; retry with backoff.
3. To release: unlink() the lock file, then unlink the temp file.

Stale lock detection
--------------------
If the lock file is older than ``stale_timeout`` seconds, assume the holder
crashed and break the lock by unlinking it, then retry.
"""

from __future__ import annotations

import json
import os
import random
import socket
import tempfile
import time
from pathlib import Path
from typing import Optional


class HardLinkLock:
    """Distributed mutex using atomic hard-link creation on Lustre.

    Parameters
    ----------
    lock_path : str or Path
        Path to the lock file (e.g. ``/shared/store.db.lock``).
    stale_timeout : float
        Seconds after which a held lock is considered stale (default 600 = 10 min).
    retry_limit : int
        Maximum number of acquire attempts before raising ``TimeoutError``.
    base_delay : float
        Initial backoff delay in seconds.
    max_delay : float
        Maximum backoff delay in seconds.
    """

    def __init__(
        self,
        lock_path: str | Path,
        stale_timeout: float = 600.0,
        retry_limit: int = 120,
        base_delay: float = 0.05,
        max_delay: float = 5.0,
    ) -> None:
        self.lock_path = Path(lock_path)
        self.stale_timeout = stale_timeout
        self.retry_limit = retry_limit
        self.base_delay = base_delay
        self.max_delay = max_delay

        self._tmp_path: Optional[Path] = None
        self._held = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Acquire the lock, blocking with exponential backoff + jitter."""
        delay = self.base_delay

        for attempt in range(self.retry_limit):
            # Create a fresh temp file for this attempt
            self._create_temp_file()
            try:
                os.link(str(self._tmp_path), str(self.lock_path))
                self._held = True
                return
            except OSError:
                # link() failed → lock is held by someone else
                self._remove_temp_file()
                self._try_break_stale_lock()

            # Exponential backoff with jitter
            jitter = random.uniform(0, delay * 0.5)
            time.sleep(delay + jitter)
            delay = min(delay * 2, self.max_delay)

        raise TimeoutError(
            f"Could not acquire lock {self.lock_path} after {self.retry_limit} attempts"
        )

    def release(self) -> None:
        """Release the lock by unlinking the lock file and temp file."""
        if not self._held:
            return
        try:
            os.unlink(str(self.lock_path))
        except FileNotFoundError:
            pass
        self._remove_temp_file()
        self._held = False

    def __enter__(self) -> "HardLinkLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lock_info(self) -> dict:
        """Payload written into the temp file identifying the lock holder."""
        return {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "timestamp": time.time(),
        }

    def _create_temp_file(self) -> None:
        """Create a uniquely-named temp file next to the lock path."""
        self._remove_temp_file()  # clean up any previous attempt
        lock_dir = self.lock_path.parent
        prefix = f".lock_{socket.gethostname()}_{os.getpid()}_"
        fd, path = tempfile.mkstemp(prefix=prefix, dir=lock_dir)
        try:
            info = json.dumps(self._lock_info())
            os.write(fd, info.encode("utf-8"))
        finally:
            os.close(fd)
        self._tmp_path = Path(path)

    def _remove_temp_file(self) -> None:
        if self._tmp_path is not None:
            try:
                os.unlink(str(self._tmp_path))
            except FileNotFoundError:
                pass
            self._tmp_path = None

    def _try_break_stale_lock(self) -> None:
        """If the current lock is stale (holder likely crashed), break it."""
        try:
            data = self.lock_path.read_text(encoding="utf-8")
            info = json.loads(data)
            lock_time = info.get("timestamp", 0)
            if (time.time() - lock_time) > self.stale_timeout:
                # Stale lock detected — break it
                try:
                    os.unlink(str(self.lock_path))
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            # Lock file disappeared or is unreadable — will retry naturally
            pass
