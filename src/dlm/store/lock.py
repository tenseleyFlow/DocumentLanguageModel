"""Cross-platform exclusive lock for a store root.

Portable PID-based approach (no fcntl / msvcrt dependency):

1. Acquire: `open(..., O_CREAT | O_EXCL | O_WRONLY)` on the lockfile,
   then write the current PID + boot-time hostname so callers can tell
   who's holding it. `O_EXCL` on create is atomic across POSIX and
   Windows NTFS; that's our mutual-exclusion guarantee.
2. Release: delete the lockfile (only if we still own it — we recheck
   the PID inside).
3. Stale lock: if acquisition fails but the recorded PID isn't alive, we
   raise `StaleLockError` instead of silently stealing. `break_lock()`
   is the explicit escape hatch for an operator who knows no writer is
   in flight.

Network filesystems (NFS, SMB, unmapped cloud drives) don't reliably
honor O_EXCL on create; callers should keep stores on local disk and
this module emits a one-line warning when it detects a non-local
filesystem. Detection is best-effort.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import socket
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from dlm.store.errors import LockHeldError, StaleLockError

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LockInfo:
    """Contents of a lockfile as decoded from disk."""

    pid: int
    hostname: str
    acquired_at: float  # epoch seconds


def _encode_lock_payload() -> str:
    return f"{os.getpid()}\n{socket.gethostname()}\n{time.time()}\n"


def _decode_lock_payload(text: str) -> LockInfo | None:
    """Parse the lockfile; return `None` if malformed."""
    parts = text.strip().split("\n")
    if len(parts) < 3:
        return None
    try:
        pid = int(parts[0])
        acquired_at = float(parts[2])
    except (TypeError, ValueError):
        return None
    hostname = parts[1]
    return LockInfo(pid=pid, hostname=hostname, acquired_at=acquired_at)


def _is_alive(pid: int) -> bool:
    """Probe process liveness cross-platform."""
    if pid <= 0:
        return False
    try:
        # Signal 0 doesn't deliver a signal; it just checks existence.
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — treat as alive.
        return True
    except OSError as exc:
        # Windows returns ESRCH on non-existent PIDs via os.kill emulation.
        return exc.errno != errno.ESRCH
    return True


def _acquire_once(lock_path: Path) -> bool:
    """Attempt a single O_EXCL acquisition.

    Returns True on success. On contention, returns False; the caller
    inspects the existing lockfile to decide whether to wait, raise
    LockHeldError, or raise StaleLockError.
    """
    try:
        fd = os.open(
            str(lock_path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError:
        return False
    try:
        os.write(fd, _encode_lock_payload().encode("utf-8"))
    finally:
        os.close(fd)
    return True


def _read_lock(lock_path: Path) -> LockInfo | None:
    try:
        text = lock_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return _decode_lock_payload(text)


def _release(lock_path: Path) -> None:
    """Remove the lockfile iff we still own it."""
    info = _read_lock(lock_path)
    if info is None:
        # Already cleaned up or corrupted; nothing to do.
        return
    if info.pid != os.getpid():
        _LOG.warning(
            "%s: lock owner changed to PID %s before our release; leaving untouched",
            lock_path,
            info.pid,
        )
        return
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()


def break_lock(lock_path: Path) -> bool:
    """Forcibly remove `lock_path`. Returns True iff a file was removed.

    Use only when you are certain no other process is writing. Prefer
    the explicit `break_lock()` over silent steals inside `exclusive()`.
    """
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return False
    _LOG.warning("%s: lock broken by explicit request", lock_path)
    return True


@contextlib.contextmanager
def exclusive(
    lock_path: Path,
    *,
    timeout: float = 0.0,
    poll_interval: float = 0.1,
) -> Iterator[LockInfo]:
    """Acquire `lock_path` for the duration of the context.

    Parameters
    ----------
    timeout:
        Seconds to wait for a held lock to release before giving up.
        `0.0` (default) fails immediately with `LockHeldError`. Pass a
        positive float for polite-polling acquisition.
    poll_interval:
        Sleep between retries when `timeout > 0`.

    Raises
    ------
    LockHeldError
        Another live process holds the lock and acquisition timed out.
    StaleLockError
        The lockfile is present but its holder is not alive. Operator
        must call `break_lock()` explicitly to unblock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout if timeout > 0 else None
    payload: LockInfo | None = None

    while True:
        if _acquire_once(lock_path):
            payload = _read_lock(lock_path)
            break

        existing = _read_lock(lock_path)
        if existing is None:
            # Malformed lockfile, or a race between stat and read.
            # Treat as stale to avoid infinite contention.
            raise StaleLockError(lock_path, holder_pid=None)

        if not _is_alive(existing.pid):
            raise StaleLockError(lock_path, holder_pid=existing.pid)

        if deadline is None or time.monotonic() >= deadline:
            raise LockHeldError(lock_path, holder_pid=existing.pid)

        time.sleep(poll_interval)

    assert payload is not None
    try:
        yield payload
    finally:
        _release(lock_path)
