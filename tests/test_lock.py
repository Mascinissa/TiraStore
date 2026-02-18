"""Tests for the hard-link based distributed lock."""

import os
import tempfile
from pathlib import Path

from tirastore._lock import HardLinkLock


def test_acquire_release(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock = HardLinkLock(lock_path, stale_timeout=60)

    lock.acquire()
    assert lock._held
    assert lock_path.exists()

    lock.release()
    assert not lock._held
    assert not lock_path.exists()


def test_context_manager(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock = HardLinkLock(lock_path, stale_timeout=60)

    with lock:
        assert lock._held
        assert lock_path.exists()

    assert not lock._held
    assert not lock_path.exists()


def test_double_release_is_safe(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock = HardLinkLock(lock_path, stale_timeout=60)

    lock.acquire()
    lock.release()
    lock.release()  # Should not raise


def test_stale_lock_broken(tmp_path):
    """A lock older than stale_timeout should be broken by a new acquirer."""
    import json
    import time

    lock_path = tmp_path / "test.lock"

    # Simulate a stale lock (timestamp far in the past)
    stale_info = {"hostname": "dead-node", "pid": 99999, "timestamp": time.time() - 9999}
    lock_path.write_text(json.dumps(stale_info))

    lock = HardLinkLock(lock_path, stale_timeout=5, retry_limit=5, base_delay=0.01)
    lock.acquire()  # Should break the stale lock and succeed
    assert lock._held
    lock.release()


def test_contention_second_acquirer_waits(tmp_path):
    """Second lock attempt on same path should fail when lock is held."""
    lock_path = tmp_path / "test.lock"
    lock1 = HardLinkLock(lock_path, stale_timeout=600)
    lock2 = HardLinkLock(lock_path, stale_timeout=600, retry_limit=3, base_delay=0.01)

    lock1.acquire()

    try:
        lock2.acquire()
        assert False, "Should have raised TimeoutError"
    except TimeoutError:
        pass
    finally:
        lock1.release()
