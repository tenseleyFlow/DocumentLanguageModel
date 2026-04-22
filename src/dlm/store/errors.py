"""Typed errors raised by `dlm.store`."""

from __future__ import annotations

from pathlib import Path


class StoreError(Exception):
    """Base class for all `dlm.store` errors."""


class ManifestCorruptError(StoreError):
    """`manifest.json` is present but unreadable / schema-invalid.

    Carries the offending path and the underlying reason so callers can
    render an actionable diagnostic.
    """

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: manifest is corrupt: {reason}")


class ManifestVersionError(ManifestCorruptError):
    """`manifest.json` parses but its `schema_version` needs migration.

    Structurally a `ManifestCorruptError` subclass so callers that only
    catch the parent still see version mismatches; semantically distinct
    for callers that promote rather than
    reject.
    """

    def __init__(self, path: Path, found_version: int, expected_version: int) -> None:
        self.found_version = found_version
        self.expected_version = expected_version
        reason = (
            f"schema_version {found_version} requires migration to {expected_version} "
            "before this store can be used"
        )
        super().__init__(path, reason)


class LockHeldError(StoreError):
    """Another process currently holds the store's exclusive lock.

    `holder_pid` may be unknown (None) if the lockfile was malformed, in
    which case `StaleLockError` is probably a better fit.
    """

    def __init__(self, lock_path: Path, holder_pid: int | None) -> None:
        self.lock_path = lock_path
        self.holder_pid = holder_pid
        owner = f"PID {holder_pid}" if holder_pid is not None else "unknown owner"
        super().__init__(f"{lock_path}: already locked by {owner}")


class StaleLockError(StoreError):
    """A lockfile exists, its owner PID is dead, but we refuse to steal.

    Raised when `exclusive()` detects a lock whose holder isn't alive.
    Callers must invoke `break_lock()` explicitly — silent steals would
    risk a concurrent-writer race if we misread the liveness probe.
    """

    def __init__(self, lock_path: Path, holder_pid: int | None) -> None:
        self.lock_path = lock_path
        self.holder_pid = holder_pid
        super().__init__(
            f"{lock_path}: lock is stale (PID {holder_pid} not alive). "
            "Use `break_lock()` to remove if you're sure no other process is writing.",
        )


class OrphanedStoreError(StoreError):
    """The store's recorded source `.dlm` file no longer exists or doesn't match.

    Raised by inspection helpers when strict mode is requested; the
    non-strict inspection surfaces an `orphaned` flag instead.
    """

    def __init__(self, store_path: Path, source_path: Path | None) -> None:
        self.store_path = store_path
        self.source_path = source_path
        where = str(source_path) if source_path is not None else "<unknown>"
        super().__init__(f"{store_path}: source .dlm at {where} missing or mismatched")


class UnknownStoreError(StoreError):
    """The requested `dlm_id` has no store on disk."""

    def __init__(self, dlm_id: str, store_path: Path) -> None:
        self.dlm_id = dlm_id
        self.store_path = store_path
        super().__init__(f"no store for dlm_id {dlm_id!r} at {store_path}")
