"""Typed errors raised by `dlm.lock`."""

from __future__ import annotations

from pathlib import Path


class LockError(Exception):
    """Base class for `dlm.lock` failures."""


class LockSchemaError(LockError):
    """`dlm.lock` is present but unreadable or schema-invalid.

    Distinct from `LockValidationError` — this fires when the on-disk
    file can't be parsed at all. The validator only runs after parse
    succeeds.
    """

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class LockWriteError(LockError):
    """Programmer error on the write path.

    Distinct from `LockSchemaError` (which means "on-disk file can't
    be parsed") so callers can tell a write-refusal apart from a
    read-refusal. Raised by `write_lock` when the caller passes a
    `DlmLock` with a lock_version the writer doesn't support.
    """

    def __init__(self, *, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: write refused: {reason}")


class LockValidationError(LockError):
    """Prior and current lock disagree at `error` severity.

    Carries the list of mismatched fields so the CLI can render a
    human-actionable report (`--update-lock` / `--ignore-lock` /
    investigate). Warn-severity mismatches don't raise.
    """

    def __init__(self, *, path: Path, reasons: list[str]) -> None:
        self.path = path
        self.reasons = list(reasons)
        joined = "; ".join(reasons)
        super().__init__(f"{path}: lock validation failed ({joined})")


class GoldenIndexSchemaError(LockError):
    """Repo-level determinism-golden index is unreadable or schema-invalid."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class GoldenIndexWriteError(LockError):
    """Programmer error on the repo-level determinism-golden index write path."""

    def __init__(self, *, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: write refused: {reason}")
