"""Exclusive-lock behavior: mutual exclusion, stale detection, release."""

from __future__ import annotations

import multiprocessing
import os
import time
from multiprocessing.queues import Queue as _MPQueue
from pathlib import Path

import pytest

from dlm.store import lock
from dlm.store.errors import LockHeldError, StaleLockError

# Module-level worker fns so `spawn` context can pickle them.


def _child_attempt(path: str, queue: _MPQueue) -> None:
    try:
        with lock.exclusive(Path(path), timeout=0.0):
            queue.put("acquired")
    except LockHeldError as exc:
        queue.put(f"held:{exc.holder_pid}")
    except Exception as exc:  # pragma: no cover — diagnostics only
        queue.put(f"error:{type(exc).__name__}:{exc}")


def _holder(path: str, hold_seconds: float) -> None:
    with lock.exclusive(Path(path)):
        time.sleep(hold_seconds)


class TestAcquireRelease:
    def test_clean_acquire_and_release(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        with lock.exclusive(lock_path) as info:
            assert info.pid == os.getpid()
            assert lock_path.exists()
        assert not lock_path.exists()

    def test_second_acquire_after_release(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        for _ in range(3):
            with lock.exclusive(lock_path):
                pass
        assert not lock_path.exists()


class TestMutualExclusion:
    def test_nested_same_process_fails_immediately(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        with (
            lock.exclusive(lock_path),
            pytest.raises(LockHeldError),
            lock.exclusive(lock_path, timeout=0.0),
        ):
            pass

    def test_concurrent_process_cannot_acquire(self, tmp_path: Path) -> None:
        """Spin a child process; it must see the lock as held."""
        lock_path = tmp_path / "test.lock"

        # Parent acquires and releases in sequence; we hold inside the `with`
        # block long enough for the child to try + fail.
        ctx = multiprocessing.get_context("spawn")
        outcome: _MPQueue = ctx.Queue()

        with lock.exclusive(lock_path):
            proc = ctx.Process(target=_child_attempt, args=(str(lock_path), outcome))
            proc.start()
            proc.join(timeout=10)
            assert proc.exitcode == 0, "child process did not exit cleanly"
            result = outcome.get_nowait()

        assert result.startswith("held:"), f"child should have been blocked, got {result!r}"
        # holder PID in the child's error message is our PID.
        holder = int(result.split(":", 1)[1])
        assert holder == os.getpid()


class TestTimeoutBehavior:
    def test_timeout_acquires_after_release(self, tmp_path: Path) -> None:
        """If a concurrent holder releases mid-wait, a timeout acquire succeeds."""
        lock_path = tmp_path / "test.lock"
        ctx = multiprocessing.get_context("spawn")

        proc = ctx.Process(target=_holder, args=(str(lock_path), 0.5))
        proc.start()
        # Give the child time to acquire first.
        deadline = time.monotonic() + 3.0
        while not lock_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock_path.exists(), "child failed to acquire within 3s"

        with lock.exclusive(lock_path, timeout=5.0, poll_interval=0.05):
            pass

        proc.join(timeout=5)
        assert proc.exitcode == 0


class TestStaleLock:
    def test_dead_pid_triggers_stale_error(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        # Forge a lockfile with an impossibly-large PID.
        lock_path.write_text(f"99999999\nhostx\n{time.time()}\n", encoding="utf-8")
        with pytest.raises(StaleLockError), lock.exclusive(lock_path, timeout=0.0):
            pass

    def test_break_lock_removes_file(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("123\nhost\n0\n", encoding="utf-8")
        assert lock.break_lock(lock_path) is True
        assert not lock_path.exists()
        assert lock.break_lock(lock_path) is False

    def test_malformed_lockfile_is_stale(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("garbage", encoding="utf-8")
        with pytest.raises(StaleLockError) as exc, lock.exclusive(lock_path, timeout=0.0):
            pass
        assert exc.value.holder_pid is None
